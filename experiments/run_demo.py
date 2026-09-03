"""Toy-model validation of randomized-depth multi-checkpoint Activation-DiFR.

Three providers generate completions with a tiny GPT and ship per-token
fingerprints at every checkpoint layer:

    honest   the declared model (prover runs fp32; the verifier recomputes in
             fp64, so honest distances are the cross-implementation noise
             floor, small but nonzero — same situation as vLLM-vs-HF bf16)
    quant    int4-quantized weights at EVERY layer (upstream DiFR's cheater)
    suffix   honest through layer CHEAT_AFTER, quantized after — the cheater
             that a fixed shallow audit can never see

The verifier teacher-forces each provider's tokens through the true model up
to an audit depth d and checks all checkpoints <= d against calibrated
thresholds. We then compare audit policies on detection-vs-compute.

The claim under test: a fixed audit at depth L/2 pays half the compute but is
structurally blind to the suffix cheater; drawing d at random restores
coverage of every segment at the same expected compute.

Runs on CPU in well under a minute.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from difr_mid.fingerprint import fingerprint, fp_distance, make_projections

VOCAB = 256
D_MODEL = 64
N_HEAD = 4
N_LAYER = 8
CHECKPOINTS = [2, 4, 6, 8]  # audit depths; 8 = full
FP_K = 16                   # fingerprint width
CHEAT_AFTER = 6             # suffix cheater is honest through this layer
PROMPT_LEN = 16
GEN_LEN = 32
N_PROMPTS = 24
SEED = 0

# Accept threshold per checkpoint: this multiple of the max honest distance.
THRESHOLD_MARGIN = 5.0


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEAD, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL)
        )

    def forward(self, x):
        h = self.ln1(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device, dtype=x.dtype)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(PROMPT_LEN + GEN_LEN + 1, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward_with_taps(self, ids, max_depth: int = N_LAYER):
        """Run embed + layers 1..max_depth. Returns (logits or None, taps)
        where taps maps checkpoint layer -> hidden states [B, T, D].
        Logits only exist at full depth."""
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        taps = {}
        for i, blk in enumerate(self.blocks[:max_depth], start=1):
            x = blk(x)
            if i in CHECKPOINTS:
                taps[i] = x
        logits = self.head(self.ln_f(x)) if max_depth == N_LAYER else None
        return logits, taps


def quantize_(model: nn.Module) -> None:
    """Simulate 4-bit weight quantization: snap every weight to a 16-level
    per-tensor grid."""
    with torch.no_grad():
        for p in model.parameters():
            lo, hi = p.min(), p.max()
            scale = (hi - lo) / 15
            if float(scale) > 0:
                p.copy_(torch.round((p - lo) / scale) * scale + lo)


def make_suffix_cheat(honest: TinyGPT) -> TinyGPT:
    """Honest through layer CHEAT_AFTER, quantized after (incl. final head)."""
    import copy

    m = copy.deepcopy(honest)
    with torch.no_grad():
        for blk in m.blocks[CHEAT_AFTER:]:
            quantize_(blk)
        quantize_(m.ln_f)
        quantize_(m.head)
    return m


@torch.no_grad()
def provider_generate(model: TinyGPT, prompt: torch.Tensor, projs):
    """Greedy-decode GEN_LEN tokens; fingerprint every checkpoint at each
    generated position (from the provider's own forward pass, like the vLLM
    hook does). Returns (tokens, fps) with fps[j][c] the fingerprint of
    the hidden state that predicted generated token j."""
    ids = prompt.clone()
    fps = []
    for _ in range(GEN_LEN):
        logits, taps = model.forward_with_taps(ids[None, :])
        fps.append({c: fingerprint(taps[c][0, -1], projs[c]) for c in CHECKPOINTS})
        nxt = int(logits[0, -1].argmax())
        ids = torch.cat([ids, torch.tensor([nxt])])
    return ids[len(prompt):], fps


@torch.no_grad()
def verifier_distances(true_model: TinyGPT, prompt, gen, provider_fps, projs):
    """Teacher-force prompt+gen through the TRUE model (fp64, so even honest
    providers show a numerical noise floor) at FULL depth once, and return
    per-token per-checkpoint distances. Policies then decide which
    checkpoints they would actually have computed/looked at."""
    seq = torch.cat([prompt, gen])[None, :]
    _, taps = true_model.forward_with_taps(seq)
    plen = len(prompt)
    out = []
    for j in range(len(gen)):
        pos = plen + j - 1  # hidden state that predicted gen[j]
        out.append(
            {c: fp_distance(fingerprint(taps[c][0, pos], projs[c]), provider_fps[j][c])
             for c in CHECKPOINTS}
        )
    return out


def main():
    torch.manual_seed(SEED)
    honest = TinyGPT().eval()

    import copy
    quant = copy.deepcopy(honest)
    quantize_(quant)
    suffix = make_suffix_cheat(honest)

    projs = make_projections(D_MODEL, CHECKPOINTS, FP_K)

    verifier = copy.deepcopy(honest).double().eval()

    gen_prompts = torch.Generator().manual_seed(SEED + 1)
    prompts = [torch.randint(0, VOCAB, (PROMPT_LEN,), generator=gen_prompts) for _ in range(N_PROMPTS)]

    providers = {"honest": honest, "quant": quant, "suffix": suffix}
    dists = {name: [] for name in providers}  # flat list over tokens of {c: dist}

    for prompt in prompts:
        for name, model in providers.items():
            gen, fps = provider_generate(model, prompt, projs)
            dists[name].extend(verifier_distances(verifier, prompt, gen, fps, projs))

    # ------------------------------------------------------------------
    # Per-checkpoint distance summary and thresholds from the honest floor
    # ------------------------------------------------------------------
    def col(name, c):
        return torch.tensor([d[c] for d in dists[name]])

    thresholds = {c: THRESHOLD_MARGIN * float(col("honest", c).max()) for c in CHECKPOINTS}

    print(f"tokens per provider: {len(dists['honest'])}")
    print(f"\nmean fingerprint distance per checkpoint (threshold = {THRESHOLD_MARGIN}x max honest):")
    hdr = f"{'ckpt':>5} {'threshold':>12} " + " ".join(f"{n:>12}" for n in providers)
    print(hdr)
    for c in CHECKPOINTS:
        row = f"{c:>5} {thresholds[c]:>12.2e} "
        row += " ".join(f"{float(col(n, c).mean()):>12.2e}" for n in providers)
        print(row)

    # Per-token flag rate if the verifier audits AT depth d (checks all ckpts <= d)
    def flag_rate(name, d):
        flagged = [
            any(tok[c] > thresholds[c] for c in CHECKPOINTS if c <= d)
            for tok in dists[name]
        ]
        return sum(flagged) / len(flagged)

    print("\nper-token flag rate by audit depth d (checks all checkpoints <= d):")
    print(f"{'d':>5} " + " ".join(f"{n:>12}" for n in providers))
    for d in CHECKPOINTS:
        print(f"{d:>5} " + " ".join(f"{flag_rate(n, d):>12.3f}" for n in providers))

    # ------------------------------------------------------------------
    # Audit policies: fixed depths vs randomized
    # ------------------------------------------------------------------
    policies = {
        "fixed d=4": {4: 1.0},
        "fixed d=8 (full)": {8: 1.0},
        "uniform random": {c: 0.25 for c in CHECKPOINTS},
        "shallow-heavy": {2: 0.4, 4: 0.3, 6: 0.2, 8: 0.1},
    }

    print("\naudit policies (expected compute = E[d]/L of a full verifier pass):")
    print(f"{'policy':>18} {'E[compute]':>11} "
          + " ".join(f"{'P(flag) ' + n:>16}" for n in providers)
          + f" {'tok to 99% (suffix)':>20}")
    results = {}
    for pname, pi in policies.items():
        e_compute = sum(p * d for d, p in pi.items()) / N_LAYER
        row = {}
        for name in providers:
            row[name] = sum(p * flag_rate(name, d) for d, p in pi.items())
        p_suf = row["suffix"]
        if p_suf >= 1.0:
            t99 = 1.0
        elif p_suf <= 0:
            t99 = math.inf
        else:
            t99 = math.log(0.01) / math.log(1 - p_suf)
        results[pname] = {"e_compute": e_compute, "p_flag": row, "tokens_to_99_suffix": t99}
        t99s = "inf" if math.isinf(t99) else f"{t99:.1f}"
        print(f"{pname:>18} {e_compute:>11.3f} "
              + " ".join(f"{row[n]:>16.3f}" for n in providers)
              + f" {t99s:>20}")

    print("\nreading: 'fixed d=4' and 'shallow-heavy' pay about the same compute,")
    print("but only the randomized policy ever flags the suffix cheater.")

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = {
        "config": {
            "checkpoints": CHECKPOINTS, "fp_k": FP_K, "cheat_after": CHEAT_AFTER,
            "n_layer": N_LAYER, "d_model": D_MODEL, "n_prompts": N_PROMPTS,
            "gen_len": GEN_LEN, "threshold_margin": THRESHOLD_MARGIN, "seed": SEED,
        },
        "thresholds": thresholds,
        "mean_distance": {n: {c: float(col(n, c).mean()) for c in CHECKPOINTS} for n in providers},
        "flag_rate_by_depth": {n: {d: flag_rate(n, d) for d in CHECKPOINTS} for n in providers},
        "policies": results,
    }
    with open(out_dir / "demo_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_dir / 'demo_results.json'}")


if __name__ == "__main__":
    main()
