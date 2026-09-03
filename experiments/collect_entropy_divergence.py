"""Collect paired (cheap-side features, declared Token-DiFR margin) per token.

Tests whether a cheating provider can PREDICT which served tokens will fail the
declared model's Token-DiFR check, using only quantities it computes for free
while drafting with the cheap model. If it can, the selective cheat (declared
only on predicted-risky tokens, cheap elsewhere) beats honesty; if it can't,
divergence is incompressible on the token channel.

TWO PASSES, both teacher-forced HF (vLLM's serving path is not needed to measure
the predictor):

  1. CHEAP DRAFT (int4 cheap model): autoregressively generate tokens under the
     SHARED Gumbel-Max seed, indexing noise by ABSOLUTE position (the
     repair_loop.py constraint: one Exponential(1) stream, never reset). Record
     per position the served token t*_cheap and the ATTACKER-VISIBLE features
     (all from the cheap forward pass, no declared logits):
         H_cheap      entropy of cheap filtered softmax
         max_p_cheap  top prob under cheap model
         gap12_cheap  p1 - p2 under cheap model
         nucleus_size # tokens surviving top-k/top-p under cheap model
         position     absolute generated-token index

  2. DECLARED SCORING (bf16 declared model = the verifier's ground truth, which
     the attacker never runs): one teacher-forced pass over prompt + cheap
     tokens; compute the true Token-DiFR margin at each position with the same
     seed, exactly as verify_vllm_gumbel_max / compute_margin_batch does. This is
     the LABEL we predict from step 1's features.

The consistency is exact: the served token whose margin we score is the cheap
model's own Gumbel pick, and the declared margin is computed on the served
sequence (the consistency trap, token-channel version).

Run on the GPU box (uses /root/difr-env):
    /root/difr-env/bin/python experiments/collect_entropy_divergence.py 2>&1
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "difr_org"))
# Reuse the repo's own filtering + margin so attacker and verifier agree by
# construction (same functions token_difr_vllm.py uses).
from token_difr_vllm import apply_top_k_top_p, compute_margin_batch  # noqa: E402


def _load_hf_token():
    import os
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in (Path.home() / ".huggingface_token", Path.home() / ".cache/huggingface/token"):
        if p.exists():
            t = p.read_text().strip()
            if t:
                os.environ["HF_TOKEN"] = t
                return t
    return None


HF_TOKEN = _load_hf_token()


def filter_row(logits_V, top_k, top_p):
    """apply_top_k_top_p wants [B,V] logits with per-row k/p tensors."""
    x = logits_V[None, :].clone()
    k = torch.full((1,), top_k, dtype=torch.long, device=x.device)
    p = torch.full((1,), top_p, dtype=torch.float32, device=x.device)
    return apply_top_k_top_p(x, k, p).squeeze(0)


def precompute_noise(seed, n_positions, vocab, device):
    """Exponential(1) noise, one row per absolute position. Draw order matches
    verify_vllm_gumbel_max: seed once, one V-vector per position, float32."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rows = []
    for _ in range(n_positions):
        v = torch.empty(vocab, dtype=torch.float32, device=device)
        v.exponential_(generator=gen)
        rows.append(v)
    return torch.stack(rows, dim=0)


def cheap_features(logits_V, temperature, top_k, top_p):
    """Attacker-visible scalars from the cheap model's filtered distribution."""
    x = logits_V.float() / max(temperature, 1e-8)
    filt = filter_row(x, top_k, top_p)
    finite = torch.isfinite(filt)
    nucleus = int(finite.sum())
    probs = torch.softmax(filt, dim=-1, dtype=torch.float32)
    p = probs[finite]
    ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
    top2 = torch.topk(probs, k=min(2, probs.numel())).values
    max_p = float(top2[0])
    gap12 = float(top2[0] - top2[1]) if top2.numel() > 1 else float(top2[0])
    return {
        "H_cheap": ent,
        "max_p_cheap": max_p,
        "gap12_cheap": gap12,
        "nucleus_size": nucleus,
    }


def gumbel_pick(logits_V, noise_V, temperature, top_k, top_p):
    """Gumbel-Max token choice on the cheap logits under the shared noise row.
    Mirrors the repo's probs/noise formulation (argmax of probs / exp-noise)."""
    x = logits_V.float() / max(temperature, 1e-8)
    filt = filter_row(x, top_k, top_p)
    probs = torch.softmax(filt, dim=-1, dtype=torch.float32)
    return int(torch.argmax(probs / noise_V))


@torch.no_grad()
def cheap_draft(cheap, prompt_ids, n_tokens, noise, temperature, top_k, top_p):
    """Autoregressive cheap-model generation under the shared seed. Returns the
    served token ids and the per-position attacker features."""
    device = next(cheap.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)
    out = cheap(ids, use_cache=True)
    past = out.past_key_values
    logits_V = out.logits[0, -1]

    served, feats = [], []
    for i in range(n_tokens):
        feats.append(cheap_features(logits_V, temperature, top_k, top_p))
        tok = gumbel_pick(logits_V, noise[i], temperature, top_k, top_p)
        served.append(tok)
        if i == n_tokens - 1:
            break
        step = torch.tensor([[tok]], device=device)
        out = cheap(step, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits_V = out.logits[0, -1]
    return served, feats


@torch.no_grad()
def declared_margins(declared, prompt_ids, served, noise, temperature, top_k, top_p):
    """Teacher-forced declared pass over prompt+served; true Token-DiFR margin
    per served position (the verifier's ground truth, clipped at +inf->inf)."""
    device = next(declared.parameters()).device
    seq = prompt_ids + served
    logits = declared(torch.tensor([seq], device=device), use_cache=False).logits[0]

    lo = len(prompt_ids) - 1
    hi = lo + len(served)
    logits_JV = logits[lo:hi].float()
    J = logits_JV.shape[0]
    noise_JV = noise[:J]

    x = logits_JV / max(temperature, 1e-8)
    k = torch.full((J,), top_k, dtype=torch.long, device=device)
    p = torch.full((J,), top_p, dtype=torch.float32, device=device)
    filtered = apply_top_k_top_p(x.clone(), k, p)
    neg_inf_mask = ~torch.isfinite(filtered)

    gold = torch.tensor(served, device=device)
    margins = compute_margin_batch(
        logits_JV, noise_JV, neg_inf_mask_JV=neg_inf_mask,
        temperature=temperature, gold_idx_J=gold,
    )
    # gold filtered out by declared top-k/top-p -> margin +inf (verifier clips).
    gold_filtered = neg_inf_mask[torch.arange(J, device=device), gold]
    m = margins.float()
    m[gold_filtered] = float("inf")
    return m.cpu().tolist()


def build_prompts(tok, n_prompts, max_ctx):
    """Unique English LMSYS prompts ending in a user turn (matches the paper's
    corpus). Falls back to a fixed list if the dataset is unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
    except Exception as e:
        print(f"[warn] LMSYS unavailable ({type(e).__name__}); using fallback prompts")
        base = [
            "Explain why cities are often built near rivers.",
            "What is the difference between precision and recall?",
            "Describe an unusual hobby and the equipment it needs.",
            "What makes a scientific hypothesis a good one?",
            "Tell me about anglerfish.",
            "Why study classical languages?",
            "Explain compound interest with an analogy.",
            "Invent a name for a coffee shop and justify it.",
        ]
        outs, seen = [], set()
        i = 0
        while len(outs) < n_prompts:
            content = base[i % len(base)] + ("" if i < len(base) else f" (variant {i})")
            ids = tok.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True, tokenize=True,
            )
            if tuple(ids) not in seen and len(ids) <= max_ctx:
                seen.add(tuple(ids))
                outs.append(ids)
            i += 1
        return outs

    outs, seen = [], set()
    idx = 0
    while len(outs) < n_prompts and idx < len(ds):
        row = ds[idx]; idx += 1
        if row.get("language", "").lower() != "english":
            continue
        conv = list(row["conversation"])
        while conv and conv[-1].get("role") == "assistant":
            conv = conv[:-1]
        if not conv or conv[-1].get("role") != "user":
            continue
        ids = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
        if len(ids) <= max_ctx and tuple(ids) not in seen:
            seen.add(tuple(ids))
            outs.append(ids)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--declared", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--cheap", default=None,
                    help="cheap model; default = declared loaded in 4-bit (bnb)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--n-prompts", type=int, default=256)
    ap.add_argument("--max-ctx", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default="results/entropy_divergence.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs the GPU box"
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(args.declared, token=HF_TOKEN)
    prompts = build_prompts(tok, args.n_prompts, args.max_ctx)
    print(f"prompts: {len(prompts)} (target {args.n_prompts}), "
          f"{args.tokens} tokens each -> ~{len(prompts) * args.tokens} paired tokens")

    print("loading declared (bf16)...")
    declared = AutoModelForCausalLM.from_pretrained(
        args.declared, torch_dtype=torch.bfloat16, device_map="cuda", token=HF_TOKEN
    ).eval()
    vocab = declared.config.vocab_size

    print("loading cheap...")
    if args.cheap:
        cheap = AutoModelForCausalLM.from_pretrained(
            args.cheap, torch_dtype=torch.bfloat16, device_map="cuda", token=HF_TOKEN
        ).eval()
        cheap_desc = args.cheap
    else:
        cheap = AutoModelForCausalLM.from_pretrained(
            args.declared,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="cuda", token=HF_TOKEN,
        ).eval()
        cheap_desc = f"{args.declared} [int4-bnb]"
    print(f"cheap = {cheap_desc}")

    noise = precompute_noise(args.seed, args.tokens, vocab, "cuda")

    records = []
    for pi, prompt_ids in enumerate(prompts):
        served, feats = cheap_draft(
            cheap, list(prompt_ids), args.tokens, noise,
            args.temperature, args.top_k, args.top_p,
        )
        margins = declared_margins(
            declared, list(prompt_ids), served, noise,
            args.temperature, args.top_k, args.top_p,
        )
        toks = []
        for j in range(len(served)):
            row = dict(feats[j])
            row["position"] = j
            row["margin"] = margins[j]
            row["served_token"] = served[j]
            toks.append(row)
        records.append({"prompt_idx": pi, "tokens": toks})
        if (pi + 1) % 16 == 0 or pi == len(prompts) - 1:
            finite = [t["margin"] for r in records for t in r["tokens"]
                      if math.isfinite(t["margin"])]
            n_inf = sum(1 for r in records for t in r["tokens"]
                        if not math.isfinite(t["margin"]))
            mean_m = sum(finite) / max(1, len(finite))
            print(f"[{pi + 1:4d}/{len(prompts)}] tokens={sum(len(r['tokens']) for r in records)}  "
                  f"finite-margin-mean={mean_m:.4f}  inf(gold-filtered)={n_inf}")

    del cheap
    gc.collect(); torch.cuda.empty_cache()

    out = {
        "config": {
            "declared": args.declared, "cheap": cheap_desc, "seed": args.seed,
            "tokens": args.tokens, "n_prompts": len(prompts),
            "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
        },
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {out_path}  "
          f"({sum(len(r['tokens']) for r in records)} paired tokens)")


if __name__ == "__main__":
    main()
