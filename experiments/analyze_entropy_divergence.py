"""Analyze paired (cheap features, declared margin) data: can the attacker
predict Token-DiFR divergence from cheap-side state alone?

Runs locally (no GPU). Reads results/entropy_divergence.json.

Produces:
  Plot 1  predictor ROC (P0 entropy-only, P1 logistic, P2 gradient boosting)
          for the binary label 1[margin > tau_tok]; report AUC. Sweep tau_tok.
  Plot 2  the attack: for the best predictor, the minimal risky-fraction r that
          keeps the cheap-served batch statistic under the honest 1%-FPR
          threshold, and the resulting attacker cost C_attack/C_hon under two
          accountings (optimistic declared-resume-free vs honest declared-prefix).

The honest batch threshold tau_batch is derived from the honest-vs-honest margin
distribution if an honest run is supplied (--honest); otherwise from the served
data's own low-margin tokens as a stand-in (documented as such).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import setup, COLORS

VERIFIER_CLIP = 0.5  # plot_classification.py get_features(): margin clipped, inf->clip
FEATURES = ["H_cheap", "max_p_cheap", "gap12_cheap", "nucleus_size", "position"]


def load(path):
    with open(path) as f:
        data = json.load(f)
    recs = data["records"]
    prompt_idx, X, raw_margin = [], [], []
    for r in recs:
        for t in r["tokens"]:
            prompt_idx.append(r["prompt_idx"])
            X.append([t[k] for k in FEATURES])
            raw_margin.append(t["margin"])
    return (data["config"], np.array(prompt_idx), np.array(X, dtype=np.float64),
            np.array(raw_margin, dtype=np.float64))


def scored(raw, clip):
    """Verifier's per-token feature: min(margin, clip), inf -> clip."""
    return np.where(np.isinf(raw), clip, np.minimum(raw, clip))


def prompt_split(prompt_idx, frac=0.5, seed=0):
    """Split by PROMPT (not token) to avoid train==test leakage from the
    deterministic-decode trap. Returns boolean train/test masks over tokens."""
    uniq = np.unique(prompt_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_train = int(len(uniq) * frac)
    train_prompts = set(uniq[:n_train].tolist())
    train_mask = np.array([p in train_prompts for p in prompt_idx])
    return train_mask, ~train_mask


def batch_statistic(scored_vals, batch_size):
    n = (len(scored_vals) // batch_size) * batch_size
    if n == 0:
        return np.empty(0)
    return scored_vals[:n].reshape(-1, batch_size).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/entropy_divergence.json")
    ap.add_argument("--honest", default=None,
                    help="optional honest-vs-honest json for tau_batch calibration")
    ap.add_argument("--clip", type=float, default=VERIFIER_CLIP)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--c-ratio", type=float, default=0.30,
                    help="C_cheap/C_hon: ~0.30 for int4-of-declared, "
                         "~0.155 for 1B-vs-8B (param ratio)")
    ap.add_argument("--out-prefix", default="results/figures/fig_entropy_divergence")
    args = ap.parse_args()

    setup()
    cfg, prompt_idx, X, raw = load(args.data)
    sc = scored(raw, args.clip)
    n = len(raw)
    n_inf = int(np.isinf(raw).sum())
    print(f"loaded {n} tokens from {len(np.unique(prompt_idx))} prompts")
    print(f"cheap = {cfg['cheap']}  declared = {cfg['declared']}")
    print(f"raw margin: mean(finite)={raw[np.isfinite(raw)].mean():.4f}  "
          f"gold-filtered(inf)={n_inf} ({100*n_inf/n:.2f}%)  "
          f"exact-zero={100*(raw==0).mean():.2f}%")

    train_mask, test_mask = prompt_split(prompt_idx, frac=0.5, seed=0)

    # ---- Plot 1: predictor ROC across tau_tok ----
    taus = [0.05, 0.14, 0.3, args.clip]
    ent = X[:, FEATURES.index("H_cheap")]
    fig1, axes = plt.subplots(1, len(taus), figsize=(4 * len(taus), 3.6), squeeze=False)
    auc_table = {}
    for ax, tau in zip(axes[0], taus):
        y = (raw > tau).astype(int)  # divergence label (inf counts as > tau)
        ytr, yte = y[train_mask], y[test_mask]
        row = {"tau": tau, "pos_rate": float(y.mean())}
        if yte.sum() == 0 or yte.sum() == len(yte):
            ax.set_title(f"tau={tau}: degenerate\n(pos rate {y.mean():.3f})")
            auc_table[tau] = row
            continue

        # P0: entropy threshold only (score = H_cheap)
        auc0 = roc_auc_score(yte, ent[test_mask])
        fpr0, tpr0, _ = roc_curve(yte, ent[test_mask])
        ax.plot(fpr0, tpr0, color=COLORS[0], label=f"P0 entropy (AUC {auc0:.3f})")

        # P1: logistic on all features (standardized — raw nucleus_size/position
        # overflow the unscaled matmul).
        p1 = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        p1.fit(X[train_mask], ytr)
        s1 = p1.predict_proba(X[test_mask])[:, 1]
        auc1 = roc_auc_score(yte, s1)
        fpr1, tpr1, _ = roc_curve(yte, s1)
        ax.plot(fpr1, tpr1, color=COLORS[1], label=f"P1 logistic (AUC {auc1:.3f})")

        # P2: gradient boosting on all features
        p2 = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        p2.fit(X[train_mask], ytr)
        s2 = p2.predict_proba(X[test_mask])[:, 1]
        auc2 = roc_auc_score(yte, s2)
        fpr2, tpr2, _ = roc_curve(yte, s2)
        ax.plot(fpr2, tpr2, color=COLORS[2], label=f"P2 GBT (AUC {auc2:.3f})")

        ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=0.8)
        ax.set_title(f"tau_tok={tau}  (divergence rate {y.mean():.3f})")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(loc="lower right")
        row.update({"auc_P0_entropy": auc0, "auc_P1_logistic": auc1, "auc_P2_gbt": auc2})
        auc_table[tau] = row

    fig1.suptitle(
        f"Can cheap-side state predict Token-DiFR divergence?  "
        f"cheap={cfg['cheap']}", y=1.02)
    fig1.tight_layout()
    p1_path = args.out_prefix + "_roc.png"
    Path(p1_path).parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(p1_path)
    print(f"wrote {p1_path}")

    # ---- honest batch threshold ----
    if args.honest:
        _, _, _, hraw = load(args.honest)
        hsc = scored(hraw, args.clip)
        hbatch = batch_statistic(hsc, args.batch_size)
        tau_batch = float(np.quantile(hbatch, 1 - args.target_fpr))
        tau_src = f"honest run {args.honest}"
    else:
        # Stand-in: honest traffic would be dominated by margin-0 tokens; use the
        # served data's exact-honest (margin==0) tokens as a proxy honest batch.
        honest_like = sc[raw == 0.0]
        if len(honest_like) >= args.batch_size:
            hbatch = batch_statistic(honest_like, args.batch_size)
            tau_batch = float(np.quantile(hbatch, 1 - args.target_fpr))
        else:
            tau_batch = float(sc[raw == 0.0].mean() if (raw == 0).any() else 0.0)
        tau_src = "STAND-IN (margin==0 tokens; supply --honest for real calibration)"
    print(f"tau_batch @ {args.target_fpr:.0%} FPR = {tau_batch:.5f}  [{tau_src}]")

    # ---- Plot 2: the attack. Best predictor ranks tokens by risk; the attacker
    # repairs the top-r fraction (declared-served, margin 0), serves cheap on the
    # rest. Find minimal r keeping the cheap-served batch statistic <= tau_batch.
    # Use P2 test-set scores as the risk ranking; evaluate on the test tokens.
    y_any = (raw > 0.0).astype(int)
    p2 = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    if y_any[train_mask].sum() > 0:
        p2.fit(X[train_mask], y_any[train_mask])
        risk = p2.predict_proba(X)[:, 1]
    else:
        risk = ent  # fallback

    sc_test = sc[test_mask]
    risk_test = risk[test_mask]
    order = np.argsort(-risk_test)  # highest risk first
    ranked_scored = sc_test[order]

    rs = np.linspace(0.0, 1.0, 101)
    stats = []
    for r in rs:
        n_repair = int(round(r * len(ranked_scored)))
        attacked = ranked_scored.copy()
        attacked[:n_repair] = 0.0  # declared-served token has margin 0
        b = batch_statistic(attacked, min(args.batch_size, len(attacked)))
        stats.append(float(b.mean()) if len(b) else float("nan"))
    stats = np.array(stats)

    below = np.where(stats <= tau_batch)[0]
    r_min = float(rs[below[0]]) if len(below) else 1.0

    # ---- Cost accounting under TWO models of what a declared re-check costs. ----
    # C_cheap/C_hon: ~0.3 for int4-of-declared, ~0.155 for 1B-vs-8B (param ratio).
    c_ratio = args.c_ratio
    T = int(cfg["tokens"])

    # OPTIMISTIC (declared decode resumes for free from a maintained state):
    # the attacker pays cheap on every token + declared decode ONLY on the
    # repaired r fraction.  C_attack = C_cheap + r * C_hon.
    c_opt = c_ratio + r_min * 1.0
    break_even_r_opt = 1.0 - c_ratio  # r below which optimistic attack profits

    # HONEST (no maintained declared KV cache; a sparse re-check at position t
    # needs a declared PREFILL over the whole prefix 0..t). A repaired position
    # averages T/2 of prefix; prefill is ~PF x cheaper per token than decode.
    # Per SERVED token, declared work (in decode-equivalents) =
    #   r * (avg_prefill_len / PF) = r * (T/2 / PF).
    # C_attack_honest = C_cheap + [r * (T/2)/PF] * C_hon.
    PF = 4.0  # prefill throughput advantage (paper: 3-5x); use 4x
    declared_frac_honest = r_min * (T / 2.0) / PF
    c_honest = c_ratio + declared_frac_honest
    # r below which the honest attack profits: c_ratio + r*(T/2)/PF < 1
    break_even_r_honest = (1.0 - c_ratio) / ((T / 2.0) / PF)

    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    axL.plot(rs, stats, color=COLORS[0], label="cheap-served batch statistic")
    axL.axhline(tau_batch, color=COLORS[1], linestyle="--",
                label=f"honest tau @ {args.target_fpr:.0%} FPR = {tau_batch:.4f}")
    axL.axvline(r_min, color=COLORS[2], linestyle=":",
                label=f"min repair r = {r_min:.2f}")
    axL.set_xlabel("repair fraction r (declared-served, ranked by predicted risk)")
    axL.set_ylabel("batch statistic (mean clipped margin)")
    axL.set_title("Does predicted-risk repair pass the audit?")
    axL.legend(loc="upper right")

    # Right panel: attacker cost vs honest baseline (=1.0) under both accountings.
    bars = ["honest\nbaseline", "cheap\n(caught)", f"attack\noptimistic\nr={r_min:.2f}",
            f"attack\nhonest KV\nr={r_min:.2f}"]
    vals = [1.0, c_ratio, c_opt, c_honest]
    colors = ["0.5", COLORS[3], COLORS[2], COLORS[1]]
    axR.bar(bars, vals, color=colors)
    axR.axhline(1.0, color="0.4", linestyle="--", linewidth=0.8)
    for i, v in enumerate(vals):
        axR.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    axR.set_ylabel("compute per served token (C / C_hon)")
    axR.set_title("Attack cost vs honesty (lower = cheaper than honest)")
    axR.set_ylim(0, max(1.3, max(vals) * 1.15))
    fig2.tight_layout()
    p2_path = args.out_prefix + "_attack.png"
    fig2.savefig(p2_path)
    print(f"wrote {p2_path}")

    print(f"\n--- cost accounting (T={T}, C_cheap/C_hon={c_ratio}, prefill adv PF={PF}) ---")
    print(f"OPTIMISTIC (declared decode resumes free): C_attack/C_hon = {c_opt:.3f}"
          f"  {'< 1 PROFITS' if c_opt < 1 else '>= 1 no gain'}")
    print(f"HONEST (declared prefill per re-check):    C_attack/C_hon = {c_honest:.3f}"
          f"  {'< 1 PROFITS' if c_honest < 1 else '>= 1 no gain'}"
          f"   (declared work alone = {declared_frac_honest:.2f} C_hon)")

    verdict = ("ATTACK VIABLE under optimistic accounting"
               if r_min < break_even_r_opt else
               "ATTACK NOT VIABLE (min repair >= break-even; honesty dominates)")
    print(f"\nmin repair r to pass = {r_min:.3f}   break-even r (optimistic) = "
          f"{break_even_r_opt:.3f}   break-even r (honest) = {break_even_r_honest:.3f}"
          f"   -> {verdict}")

    summary = {
        "config": cfg,
        "n_tokens": n, "n_prompts": int(len(np.unique(prompt_idx))),
        "gold_filtered_frac": n_inf / n,
        "auc_by_tau": auc_table,
        "tau_batch": tau_batch, "tau_batch_source": tau_src,
        "r_min_pass": r_min,
        "break_even_r_optimistic": break_even_r_opt,
        "break_even_r_honest": break_even_r_honest,
        "c_ratio_assumed": c_ratio, "prefill_advantage": PF, "tokens": T,
        "c_attack_optimistic": c_opt,
        "c_attack_honest_kv": c_honest,
        "declared_frac_honest": declared_frac_honest,
        "verdict": verdict,
    }
    out_json = Path(args.data).parent / "entropy_divergence_analysis.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
