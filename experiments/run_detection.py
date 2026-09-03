"""Multi-checkpoint Activation-DiFR on a real model (GPU).

Provers (each generates its own tokens and ships per-token fingerprints at
every checkpoint, taken from its own incremental-decode forward passes):

    honest_vllm   vLLM bf16, the declared configuration
    fp8kv_vllm    vLLM with fp8 KV cache (upstream DiFR's stock misconfig;
                  corrupts attention at EVERY layer, so every checkpoint sees it)
    honest_hf     HF bf16 incremental decode (noise floor for the HF family)
    suffix_hf     HF, honest through layer CHEAT_FRAC*L, int4 per-channel
                  quantized after — the coverage-gap cheater: only checkpoints
                  past the cheat point can see it

Verifier: the true HF model, one teacher-forced pass per sequence, reading
output_hidden_states at the checkpoints. (hidden_states[c] does not depend on
layers > c, so distances are identical to a true partial forward; the partial
forward is purely the compute optimization and stays on the porting
checklist. Compute fractions below are the analytic d/L.)

Thresholds are calibrated per prover family (vllm vs hf) since the honest
cross-implementation noise floor differs.

Run on the GPU box:  python experiments/run_detection.py 2>&1
"""

import copy
import gc
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from difr_mid.fingerprint import fingerprint, fp_distance, make_projections


def _load_hf_token():
    """Read a HuggingFace token for gated models (e.g. Llama). Checks HF_TOKEN,
    then ~/.huggingface_token, then ~/.cache/huggingface/token. Returns None if
    none present (fine for open models like Qwen)."""
    import os
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in (Path.home() / ".huggingface_token", Path.home() / ".cache/huggingface/token"):
        if p.exists():
            t = p.read_text().strip()
            if t:
                os.environ["HF_TOKEN"] = t  # so vLLM subprocess sees it too
                return t
    return None


HF_TOKEN = _load_hf_token()

# Override the model from the environment: DIFR_MODEL=meta-llama/Llama-3.1-8B-Instruct
import os
MODEL_NAME = os.environ.get("DIFR_MODEL", "Qwen/Qwen3-1.7B")
FP_K = 32
MAX_TOKENS = 64
CHEAT_FRAC = 0.75  # suffix cheater honest through floor(0.75 * L) layers

# The audit unit is a SEQUENCE: the verifier recomputes one teacher-forced
# pass to depth d and compares the MEAN fingerprint distance over its tokens
# at each checkpoint <= d. Per-token maxima are too heavy-tailed to threshold
# (bf16 noise occasionally spikes on single tokens); sequence means separate
# cleanly (measured on the first GPU run: fp8-kv sits 2.6-3x above the honest
# max at every checkpoint with zero overlap).
SEQ_THRESHOLD_MARGIN = 2.0

PROMPTS = [
    "Explain why cities are often built near rivers.",
    "What is the difference between precision and recall?",
    "Describe an unusual hobby and the equipment it needs.",
    "What makes a scientific hypothesis a good one?",
    "Tell me about anglerfish.",
    "Why study classical languages?",
    "Explain compound interest with an analogy.",
    "Invent a name for a coffee shop and justify it.",
]


def quantize_int4_per_channel_(module: torch.nn.Module) -> None:
    """Snap every 2D weight to a 16-level grid per output channel (row);
    1D params to a per-tensor grid. Simulates int4 weight quantization."""
    with torch.no_grad():
        for p in module.parameters():
            w = p.float()
            if w.dim() >= 2:
                flat = w.reshape(w.shape[0], -1)
                lo = flat.min(dim=1, keepdim=True).values
                hi = flat.max(dim=1, keepdim=True).values
                scale = torch.clamp((hi - lo) / 15, min=1e-12)
                q = torch.round((flat - lo) / scale) * scale + lo
                p.copy_(q.reshape_as(w).to(p.dtype))
            else:
                lo, hi = w.min(), w.max()
                scale = torch.clamp((hi - lo) / 15, min=1e-12)
                p.copy_((torch.round((w - lo) / scale) * scale + lo).to(p.dtype))


@torch.inference_mode()
def prover_generate_hf(model, prompt_token_ids, checkpoints, projs, max_tokens):
    """Greedy incremental decode with per-step hidden states; fingerprints
    come from the prover's own decode passes (mirrors the vLLM tap)."""
    device = next(model.parameters()).device
    results = []
    for ids in prompt_token_ids:
        inp = torch.tensor([ids], device=device)
        out = model.generate(
            inp,
            attention_mask=torch.ones_like(inp),
            max_new_tokens=max_tokens,
            min_new_tokens=max_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        gen_ids = out.sequences[0, len(ids):].tolist()
        assert len(gen_ids) == max_tokens
        # out.hidden_states: tuple over steps; step 0 covers the prompt
        # (shape [1, prompt_len, D]), later steps [1, 1, D]. Entry [c] of each
        # step's tuple is the hidden state after layer c (0 = embeddings).
        assert len(out.hidden_states) == max_tokens
        fps = []
        for j in range(max_tokens):
            step = out.hidden_states[j]
            fps.append(
                {c: fingerprint(step[c][0, -1], projs[c]).cpu().tolist() for c in checkpoints}
            )
        results.append({"prompt_token_ids": list(ids), "token_ids": gen_ids, "fps": fps})
    return results


@torch.inference_mode()
def verifier_distances(true_model, request, checkpoints, projs):
    """Teacher-forced pass of the true model over prompt+generation; returns
    per-token dicts of checkpoint -> fingerprint distance."""
    device = next(true_model.parameters()).device
    seq = request["prompt_token_ids"] + request["token_ids"]
    out = true_model(
        torch.tensor([seq], device=device), output_hidden_states=True, use_cache=False
    )
    plen = len(request["prompt_token_ids"])
    dists = []
    for j, tok_fp in enumerate(request["fps"]):
        pos = plen + j - 1  # hidden state that predicted generated token j
        d = {}
        for c in checkpoints:
            v = fingerprint(out.hidden_states[c][0, pos], projs[c])
            d[c] = fp_distance(v, torch.tensor(tok_fp[c], device=v.device))
        dists.append(d)
    return dists


def main():
    assert torch.cuda.is_available(), "this experiment needs the GPU box"
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    # enable_thinking is a Qwen-only chat-template kwarg; Llama's template
    # rejects unknown kwargs, so only pass it for models that accept it.
    ct_kwargs = {"enable_thinking": False} if "qwen" in MODEL_NAME.lower() else {}
    prompt_ids = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
            **ct_kwargs,
        )
        for p in PROMPTS
    ]

    # Checkpoints at quarters of the depth
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    L = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    checkpoints = sorted({max(1, round(L * f)) for f in (0.25, 0.5, 0.75, 1.0)})
    cheat_after = math.floor(L * CHEAT_FRAC)
    print(f"{MODEL_NAME}: L={L}, d_model={d_model}, checkpoints={checkpoints}, "
          f"suffix cheat after layer {cheat_after}")

    projs = make_projections(d_model, checkpoints, FP_K, device=device)

    provers = {}

    # ---- vLLM provers (import lazily so HF-only debugging works anywhere) ----
    from difr_mid.tap_vllm import prover_generate_vllm

    print("\n[1/4] honest_vllm: generating...")
    provers["honest_vllm"] = prover_generate_vllm(
        MODEL_NAME, prompt_ids, checkpoints, FP_K, MAX_TOKENS
    )
    print("[2/4] fp8kv_vllm: generating...")
    provers["fp8kv_vllm"] = prover_generate_vllm(
        MODEL_NAME, prompt_ids, checkpoints, FP_K, MAX_TOKENS,
        vllm_args={"kv_cache_dtype": "fp8", "calculate_kv_scales": True},
    )

    # ---- HF provers ----
    print("[3/4] honest_hf: generating...")
    true_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device, token=HF_TOKEN
    ).eval()
    provers["honest_hf"] = prover_generate_hf(
        true_model, prompt_ids, checkpoints, projs, MAX_TOKENS
    )

    print("[4/4] suffix_hf: generating...")
    cheat = copy.deepcopy(true_model)
    for blk in cheat.model.layers[cheat_after:]:
        quantize_int4_per_channel_(blk)
    quantize_int4_per_channel_(cheat.model.norm)
    # Only quantize lm_head if it is NOT tied to the input embedding. When tied
    # (Qwen3-1.7B), quantizing it would corrupt the embedding and thus every
    # checkpoint, not just the suffix. Llama-3.1-8B is untied, so quantize it.
    if not getattr(cfg, "tie_word_embeddings", False):
        quantize_int4_per_channel_(cheat.lm_head)
    provers["suffix_hf"] = prover_generate_hf(
        cheat, prompt_ids, checkpoints, projs, MAX_TOKENS
    )
    del cheat
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Verify everything with the true model ----
    print("\nverifying...")
    dists = {}  # name -> list per sequence of list per token of {ckpt: dist}
    for name, reqs in provers.items():
        dists[name] = [verifier_distances(true_model, req, checkpoints, projs) for req in reqs]

    # ---- Analysis (audit unit = sequence, statistic = mean over tokens) ----
    def seq_means(name, c):
        return [sum(t[c] for t in seq) / len(seq) for seq in dists[name]]

    family = {"honest_vllm": "vllm", "fp8kv_vllm": "vllm",
              "honest_hf": "hf", "suffix_hf": "hf"}
    thresholds = {
        fam: {c: SEQ_THRESHOLD_MARGIN * max(seq_means(h, c)) for c in checkpoints}
        for fam, h in (("vllm", "honest_vllm"), ("hf", "honest_hf"))
    }

    names = list(provers)
    n_seqs = len(dists[names[0]])
    print(f"\nsequences per prover: {n_seqs} x {MAX_TOKENS} tokens")
    print(f"\nper-sequence mean fingerprint distance [min..max] per checkpoint "
          f"(threshold = {SEQ_THRESHOLD_MARGIN}x max honest, per family):")
    print(f"{'ckpt':>5} {'thr(vllm)':>11} {'thr(hf)':>11} " + " ".join(f"{n:>21}" for n in names))
    for c in checkpoints:
        cells = []
        for n in names:
            sm = seq_means(n, c)
            cells.append(f"[{min(sm):8.3f}..{max(sm):8.3f}]")
        print(f"{c:>5} {thresholds['vllm'][c]:>11.3e} {thresholds['hf'][c]:>11.3e} "
              + " ".join(f"{cell:>21}" for cell in cells))

    def flag_rate(name, d):
        """Fraction of sequences flagged when audited at depth d."""
        thr = thresholds[family[name]]
        flags = [
            any(sum(t[c] for t in seq) / len(seq) > thr[c] for c in checkpoints if c <= d)
            for seq in dists[name]
        ]
        return sum(flags) / len(flags)

    print("\nper-sequence flag rate by audit depth d (checks all checkpoints <= d):")
    print(f"{'d':>5} " + " ".join(f"{n:>13}" for n in names))
    for d in checkpoints:
        print(f"{d:>5} " + " ".join(f"{flag_rate(n, d):>13.3f}" for n in names))

    c1, c2, c3, c4 = checkpoints
    policies = {
        f"fixed d={c2}": {c2: 1.0},
        f"fixed d={c4} (full)": {c4: 1.0},
        "uniform random": {c: 1.0 / len(checkpoints) for c in checkpoints},
        "shallow-heavy": {c1: 0.4, c2: 0.3, c3: 0.2, c4: 0.1},
    }

    print("\naudit policies (expected compute = E[d]/L of a full verifier pass):")
    print(f"{'policy':>20} {'E[compute]':>11} "
          + " ".join(f"{'P(flag) ' + n:>21}" for n in names)
          + f" {'seq to 99% (suffix)':>20}")
    policy_results = {}
    for pname, pi in policies.items():
        e_compute = sum(p * d for d, p in pi.items()) / L
        row = {n: sum(p * flag_rate(n, d) for d, p in pi.items()) for n in names}
        p_suf = row["suffix_hf"]
        if p_suf >= 1.0:
            t99 = 1.0
        elif p_suf <= 0:
            t99 = math.inf
        else:
            t99 = math.log(0.01) / math.log(1 - p_suf)
        policy_results[pname] = {"e_compute": e_compute, "p_flag": row,
                                 "seqs_to_99_suffix": t99}
        t99s = "inf" if math.isinf(t99) else f"{t99:.1f}"
        print(f"{pname:>20} {e_compute:>11.3f} "
              + " ".join(f"{row[n]:>21.3f}" for n in names) + f" {t99s:>20}")

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = {
        "config": {
            "model": MODEL_NAME, "n_layers": L, "d_model": d_model,
            "checkpoints": checkpoints, "fp_k": FP_K, "max_tokens": MAX_TOKENS,
            "cheat_after": cheat_after, "threshold_margin": SEQ_THRESHOLD_MARGIN,
            "n_prompts": len(PROMPTS),
        },
        "thresholds": thresholds,
        "distances": {
            n: [[{str(c): t[c] for c in checkpoints} for t in seq] for seq in dists[n]]
            for n in names
        },
        "flag_rate_by_depth": {n: {d: flag_rate(n, d) for d in checkpoints} for n in names},
        "policies": policy_results,
    }
    # Model-tagged filename so Qwen and Llama results coexist.
    slug = MODEL_NAME.split("/")[-1].replace(".", "-")
    out_path = out_dir / f"detection_results_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
