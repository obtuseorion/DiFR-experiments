"""
Lossy speculative decoding: the repair loop.

Attack: draft with a cheap model, audit once with the declared model, repair
only the tokens whose Token-DiFR margin would betray the substitution, and
re-draft the suffix from each repair point.

THE CRITICAL CONSTRAINT
-----------------------
token_difr_vllm.verify_vllm_gumbel_max() seeds one generator and draws one
Exponential(1) vector per position, in order. So the noise at generated
position t is the t-th draw from a single stream.

If we repair at position t and restart generation with a fresh sampler, the
sampler draws noise[0] where the verifier expects noise[t]. Every suffix token
then mismatches, whichever model produced it. So we precompute the whole noise
matrix once and index it by ABSOLUTE position. Never reset mid-sequence.

WHAT THIS MEASURES
------------------
The threshold sweep held the cheap trajectory fixed and assumed repairs are
free. They are not: repairing position t invalidates the KV cache and the
drafted suffix, so each repair costs a re-draft plus a re-audit of what
remains. This loop instruments that honestly:

    cheap_decode_tokens     M_c decode steps, summed over all draft attempts
    declared_prefill_tokens M_d positions scored, summed over all audits
    declared_decode_tokens  M_d decode steps (repairs are argmax lookups: 0)
    n_repairs, n_iterations

Compare against the honest provider, who pays declared_decode_tokens = T.

Note the repaired token is a LOOKUP, not a search: the audit pass already
computed the declared model's argmax at every position, so substitution costs
nothing beyond the pass we already ran.
"""

import argparse
import json
import time
from dataclasses import dataclass, field, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Reuse the repo's scoring so the attack and the verifier agree by construction.
from token_difr_vllm import apply_top_k_top_p, compute_margin_batch

VERIFIER_CLIP = 0.5  # from plot_classification.py get_features()


@dataclass
class Costs:
    cheap_decode_tokens: int = 0
    declared_prefill_tokens: int = 0
    declared_decode_tokens: int = 0
    n_repairs: int = 0
    n_iterations: int = 0
    wall_seconds: float = 0.0
    final_statistic: float = 0.0
    repair_positions: list[int] = field(default_factory=list)


def precompute_noise(seed: int, n_positions: int, vocab: int, device: str) -> torch.Tensor:
    """Exponential(1) noise, one row per generated position.

    Draw order must match verify_vllm_gumbel_max exactly: seed once, then one
    V-vector per position via .exponential_(), float32.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rows = []
    for _ in range(n_positions):
        v = torch.empty(vocab, dtype=torch.float32, device=device)
        v.exponential_(generator=gen)
        rows.append(v)
    return torch.stack(rows, dim=0)


def gumbel_pick(logits_V: torch.Tensor, noise_V: torch.Tensor,
                temperature: float, top_k: int, top_p: float) -> int:
    """Gumbel-Max token choice, mirroring the repo's probs/noise formulation."""
    x = logits_V.float() / max(temperature, 1e-8)
    x = apply_top_k_top_p(x[None, :], top_k, top_p).squeeze()
    probs = torch.softmax(x, dim=-1, dtype=torch.float32)
    return int(torch.argmax(probs / noise_V))


@torch.no_grad()
def draft(model, prompt_ids: list[int], start_pos: int, n_tokens: int,
          noise: torch.Tensor, temperature: float, top_k: int, top_p: float,
          costs: Costs) -> list[int]:
    """Decode n_tokens with the cheap model, using noise rows [start_pos:].

    prompt_ids is the full context (original prompt + any already-fixed prefix).
    """
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)
    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits_V = out.logits[0, -1]

    generated = []
    for i in range(n_tokens):
        tok = gumbel_pick(logits_V, noise[start_pos + i], temperature, top_k, top_p)
        generated.append(tok)
        costs.cheap_decode_tokens += 1
        if i == n_tokens - 1:
            break
        step = torch.tensor([[tok]], device=device)
        out = model(step, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits_V = out.logits[0, -1]
    return generated


@torch.no_grad()
def audit(model, prompt_ids: list[int], gen_ids: list[int], noise: torch.Tensor,
          temperature: float, top_k: int, top_p: float, costs: Costs):
    """One teacher-forced prefill of the declared model over prompt+generation.

    Returns (margins, declared_argmax_tokens) for each generated position.
    This is the self-audit: it yields both the score and the repair token.
    """
    device = next(model.parameters()).device
    seq = prompt_ids + gen_ids
    ids = torch.tensor([seq], device=device)
    logits = model(ids, use_cache=False).logits[0]
    costs.declared_prefill_tokens += len(seq)

    # Position t of the generation is predicted by the logits at index
    # len(prompt) + t - 1.
    lo = len(prompt_ids) - 1
    hi = lo + len(gen_ids)
    logits_JV = logits[lo:hi].float()

    J, V = logits_JV.shape
    noise_JV = noise[:J]

    x = logits_JV / max(temperature, 1e-8)
    filtered = apply_top_k_top_p(x.clone(), top_k, top_p)
    neg_inf_mask = ~torch.isfinite(filtered)
    probs = torch.softmax(filtered, dim=-1, dtype=torch.float32)

    declared_tokens = torch.argmax(probs / noise_JV, dim=-1)

    gold = torch.tensor(gen_ids, device=device)
    margins = compute_margin_batch(
        logits_JV, noise_JV, neg_inf_mask_JV=neg_inf_mask,
        temperature=temperature, gold_idx_J=gold,
    )
    return margins, declared_tokens


def statistic(margins: torch.Tensor, clip: float) -> float:
    """Verifier's batch statistic: mean of clipped margins (inf -> clip)."""
    m = margins.float().clone()
    m[~torch.isfinite(m)] = clip
    return float(torch.clamp(m, 0.0, clip).mean())


def repair_loop(cheap, declared, prompt_ids: list[int], n_tokens: int,
                noise: torch.Tensor, tau: float, budget: float,
                temperature: float, top_k: int, top_p: float,
                max_iterations: int = 64) -> tuple[list[int], Costs]:
    """Draft, audit, repair the earliest violation, re-draft the suffix.

    tau     per-token margin above which a token must be repaired
    budget  batch statistic the final sequence must stay under
    """
    costs = Costs()
    t0 = time.time()

    gen = draft(cheap, prompt_ids, 0, n_tokens, noise,
                temperature, top_k, top_p, costs)

    for _ in range(max_iterations):
        costs.n_iterations += 1
        margins, declared_tokens = audit(
            declared, prompt_ids, gen, noise, temperature, top_k, top_p, costs
        )

        s = statistic(margins, VERIFIER_CLIP)
        m = margins.float().clone()
        m[~torch.isfinite(m)] = float("inf")
        violations = (m > tau).nonzero().flatten()

        if s <= budget and len(violations) == 0:
            costs.final_statistic = s
            break

        if len(violations) == 0:
            # Statistic over budget but nothing exceeds tau: lower tau to the
            # largest observed margin so we make progress.
            violations = torch.tensor([int(torch.argmax(m))], device=m.device)

        t = int(violations[0])
        gen = gen[:t] + [int(declared_tokens[t])]
        costs.n_repairs += 1
        costs.repair_positions.append(t)

        remaining = n_tokens - len(gen)
        if remaining > 0:
            # Re-draft the suffix from absolute position len(gen). The prefix is
            # now fixed, so the cheap model conditions on the repaired context.
            gen = gen + draft(cheap, prompt_ids + gen, len(gen), remaining,
                              noise, temperature, top_k, top_p, costs)
        else:
            costs.final_statistic = s

    costs.wall_seconds = time.time() - t0
    return gen, costs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--declared", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--cheap", default=None, help="defaults to --declared in 4-bit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.14)
    ap.add_argument("--budget", type=float, default=0.0015,
                    help="batch statistic ceiling; calibrate from your honest run")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--out", default="repair_loop_costs.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.declared)

    declared = AutoModelForCausalLM.from_pretrained(
        args.declared, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    if args.cheap:
        cheap = AutoModelForCausalLM.from_pretrained(
            args.cheap, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
    else:
        cheap = AutoModelForCausalLM.from_pretrained(
            args.declared,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="cuda",
        ).eval()

    vocab = declared.config.vocab_size
    noise = precompute_noise(args.seed, args.tokens, vocab, "cuda")

    prompts = [
        "Explain why cities are often built near rivers.",
        "What is the difference between precision and recall?",
        "Describe an unusual hobby and the equipment it needs.",
        "What makes a scientific hypothesis a good one?",
        "Tell me about anglerfish.",
        "Why study classical languages?",
        "Explain compound interest with an analogy.",
        "Invent a name for a coffee shop and justify it.",
    ][: args.n_prompts]

    results = []
    for p in prompts:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=True,
        )
        gen, costs = repair_loop(
            cheap, declared, list(ids), args.tokens, noise,
            args.tau, args.budget, args.temperature, args.top_k, args.top_p,
        )
        row = asdict(costs)
        row["prompt"] = p
        row["text"] = tok.decode(gen)
        results.append(row)
        print(f"[{p[:40]:40s}] repairs {costs.n_repairs:3d}  "
              f"iters {costs.n_iterations:3d}  "
              f"cheap-decode {costs.cheap_decode_tokens:5d}  "
              f"declared-prefill {costs.declared_prefill_tokens:6d}  "
              f"S {costs.final_statistic:.5f}  {costs.wall_seconds:.1f}s")

    n = len(results)
    print()
    print(f"per sequence of {args.tokens} tokens, averaged over {n} prompts:")
    print(f"  repairs                 {sum(r['n_repairs'] for r in results) / n:8.2f}"
          f"  ({100 * sum(r['n_repairs'] for r in results) / (n * args.tokens):.2f}% of positions)")
    print(f"  cheap decode tokens     {sum(r['cheap_decode_tokens'] for r in results) / n:8.1f}"
          f"  (honest pays 0)")
    print(f"  declared prefill tokens {sum(r['declared_prefill_tokens'] for r in results) / n:8.1f}"
          f"  (honest pays 0)")
    print(f"  declared decode tokens  {0:8.1f}  (honest pays {args.tokens})")
    print()
    print("Attack is profitable iff cheap_decode + declared_prefill/asymmetry")
    print(f"costs less than {args.tokens} declared decode steps. Prefill is 3-5x")
    print("cheaper per token than decode under production batching.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
