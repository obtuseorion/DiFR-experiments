"""
Day-2 go/no-go: is there an exploitable tolerance band in Token-DiFR?

Inputs: two verification JSONs produced by token_difr_vllm.py
  - honest run  (declared model generates, declared model verifies)   -> H0
  - cheap run   (misconfigured generator, declared model verifies)    -> H1 source

TWO VIEWS OF EACH MARGIN, and keeping them apart matters:

  raw     the true post-Gumbel margin. The ATTACKER sees this (it ran the
          declared model in its self-audit pass) and decides accept/repair on it.
  scored  the verifier's feature: min(margin, 0.5), with +inf also mapped to 0.5.
          This clip is taken from plot_classification.py get_features().

So a token whose true margin is 0.51 and one whose margin is +inf (emitted
outside the declared model's top-k/top-p nucleus) cost the verifier's statistic
exactly the same. Above the clip the attacker gains nothing from selectivity.

Attack at threshold tau: keep cheap tokens with raw margin <= tau, repair the
rest. A repaired token IS the declared model's argmax, so its margin is 0.
tau = 0 means "repair every genuine divergence" -- the fully honest-looking
attacker, whose repair rate is the cheap model's raw divergence rate.

Metric is TPR at 1% FPR, one-sided (the verifier flags large statistics):
    TPR ~ 0.01  ->  flagged no more often than honest traffic: PASSES
    TPR -> 1.0  ->  flagged reliably: CAUGHT
Read the largest tau still passing; its repair% is what the attack costs.
"""

import argparse
import json

import numpy as np
from sklearn.metrics import roc_curve

# From plot_classification.py get_features(): margins are clipped at 0.5 and
# +inf is mapped to 0.5 before being fed to the classifier.
VERIFIER_CLIP = 0.5

BATCH_SIZES = [100, 500, 1000, 5000]

TARGET_FPR = 0.01


def load_margins(path: str, clip: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw margins with inf preserved, verifier-clipped margins)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = np.array(
        [tok["margin"] for seq in data["scores"] for tok in seq], dtype=np.float64
    )
    raw = np.nan_to_num(raw, nan=np.inf, posinf=np.inf)
    scored = np.where(np.isinf(raw), clip, np.minimum(raw, clip))
    return raw, scored


def batch_statistic(scored: np.ndarray, batch_size: int) -> np.ndarray:
    """Mean clipped margin over consecutive non-overlapping token batches."""
    n = (len(scored) // batch_size) * batch_size
    if n == 0:
        return np.empty(0)
    return scored[:n].reshape(-1, batch_size).mean(axis=1)


def tpr_at_fpr(h0: np.ndarray, h1: np.ndarray, max_fpr: float) -> float:
    if len(h0) < 2 or len(h1) < 2:
        return float("nan")
    y = np.concatenate([np.zeros(len(h0)), np.ones(len(h1))])
    s = np.concatenate([h0, h1])
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(max_fpr, fpr, tpr))


def describe(name: str, raw: np.ndarray, scored: np.ndarray, clip: float) -> None:
    n = len(raw)
    nz = raw[raw > 0]
    n_inf = int(np.isinf(raw).sum())
    n_over = int((raw > clip).sum())
    print(f"{name:7s} n={n:6d}  scored-mean {scored.mean():.5f}  "
          f"exact-zero {(raw == 0).mean():.4f}")
    print(f"{'':7s} nonzero {len(nz):5d} ({100 * len(nz) / n:.2f}%)   "
          f"above-clip {n_over:5d} ({100 * n_over / n:.2f}%)   "
          f"filtered-out(inf) {n_inf}")
    if len(nz):
        finite = nz[np.isfinite(nz)]
        if len(finite):
            q = np.percentile(finite, [50, 90, 99])
            print(f"{'':7s} finite nonzero: p50 {q[0]:.3f}  p90 {q[1]:.3f}  p99 {q[2]:.3f}")
    # How much of the statistic comes from tokens pinned at the clip?
    print(f"{'':7s} share of scored-mean from above-clip tokens: "
          f"{100 * n_over * clip / (scored.mean() * n):.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("honest")
    ap.add_argument("cheap")
    ap.add_argument("--clip", type=float, default=VERIFIER_CLIP,
                    help="verifier's margin clip (repo default 0.5)")
    args = ap.parse_args()

    honest_raw, honest_scored = load_margins(args.honest, args.clip)
    cheap_raw, cheap_scored = load_margins(args.cheap, args.clip)

    describe("honest", honest_raw, honest_scored, args.clip)
    describe("cheap", cheap_raw, cheap_scored, args.clip)
    print()

    counts = "  ".join(f"{b}:{len(honest_raw) // b}" for b in BATCH_SIZES)
    print(f"batches available per size -> {counts}")
    print("(TPR from fewer than ~20 batches is very coarse; treat with suspicion)")
    print()

    # Dense below the clip, where selectivity actually changes the cost;
    # sparse above it, where every kept token costs the same.
    taus = np.unique(np.round(np.concatenate([
        np.linspace(0.0, args.clip, 26),
        np.array([0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]),
    ]), 6))
    taus = np.append(taus, np.inf)

    header = (f"{'tau':>8} {'repair%':>9} "
              + " ".join(f"{'tpr@' + str(b):>10}" for b in BATCH_SIZES))
    print(header)
    print("-" * len(header))

    still_passing = True
    for tau in taus:
        repaired = cheap_raw > tau                       # attacker sees raw
        attack = np.where(repaired, 0.0, cheap_scored)   # verifier sees clipped
        cells, tprs = [], []
        for b in BATCH_SIZES:
            t = tpr_at_fpr(batch_statistic(honest_scored, b),
                           batch_statistic(attack, b), TARGET_FPR)
            tprs.append(t)
            cells.append(f"{t:10.3f}")
        ref = tprs[BATCH_SIZES.index(1000)]
        flag = ""
        if still_passing and ref > 0.05:
            flag = "  <-- crossing"
            still_passing = False
        label = "  inf" if np.isinf(tau) else f"{tau:8.3f}"
        print(f"{label:>8} {100 * repaired.mean():9.2f} " + " ".join(cells) + flag)

    print()
    print(f"clip = {args.clip}. Above it, keeping a token costs the same whatever")
    print("its true margin, so the curve should flatten -- selectivity stops paying.")
    print("tau=inf row is the un-repaired cheap model (the paper's static cheater).")
    print()
    print("CAVEAT: this holds the cheap trajectory fixed. Real repairs change the")
    print("prefix, so downstream margins shift. Validate with the actual repair")
    print("loop before quoting these numbers as an attack result.")


if __name__ == "__main__":
    main()