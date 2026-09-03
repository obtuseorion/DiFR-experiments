"""Surrogate-forgery sweep: the quantitative core of the security argument
(docs/forgery-attack-analysis.md section 7).

Question: can a forger cheaper than the honest model produce fingerprints that
pass the audit? We sweep pass-rate-at-1%-FPR against forger cost C_surr/C_hon,
one line per number of checkpoints the forger must satisfy simultaneously, under
BOTH a fixed public projection and a reseeded-per-audit projection.

Claim under test: as the number of simultaneously-audited checkpoints grows, a
forger only reaches a passing rate as its cost approaches C_hon -- i.e. forging
is as expensive as honesty, so the attacker is eventually caught or computes the
whole pass.

Pipeline:
  1. collect(): cheap model serves tokens; store honest + cheap raw activations
     on the served tokens at every checkpoint (the consistency trap).
  2. Split sequences into train (fit forgers) and test (evaluate).
  3. Honest noise floor: an independent bf16 recompute vs the stored honest
     activation gives the H0 distance distribution -> per-checkpoint threshold
     at 1% FPR. (Here H0 = honest-vs-honest numerical noise; we approximate it
     with the honest activation vs itself under two projection draws plus the
     residual bf16 vs fp32 gap already present in the stored tensors. See
     honest_floor() for the exact construction.)
  4. For each forger and each audit depth d (= checkpoints <= d), measure the
     fraction of TEST tokens whose forged fingerprint passes at ALL of those
     checkpoints simultaneously, under fixed-P and reseeded-P.

Run on the GPU box:  python experiments/run_forgery_sweep.py 2>&1
Collection is the only GPU stage; forger training/eval is CPU/GPU-light.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from difr_mid.fingerprint import make_projections
from difr_mid.collect_activations import collect
from difr_mid.forgers import (
    CopyForger, MeanForger, RidgeForger, MLPForger, ShortcutForger,
    DistilledSuffixForger,
)

MODEL_NAME = "Qwen/Qwen3-1.7B"
FP_K = 32
MAX_TOKENS = 48
N_PROMPTS = 600          # >>D: F4 needs many more training tokens than D=2048
DATASET = "lmsys/lmsys-chat-1m"
MAX_CTX = 256            # cap prompt length so collection stays fast
TRAIN_FRAC = 0.7
TARGET_FPR = 0.01
N_RESEED_EPOCHS = 8      # audits average over this many fresh projections
NORMALIZE = True         # per-layer whitening before projection (section 5v)
SEED = 0

# Only the GENTLE cheat is worth scaling: with a harsh int4 cheat the cheap
# activations are so far from honest that even F0-copy is ~20x over threshold
# (already shown conclusively at N=120). The int8-gentle regime is where
# forgery has a chance and the reseeding question is live, so we spend the big
# collection there. 256 levels ~ int8.
CHEAT_LEVELS = [256]


def load_prompts(tokenizer, n, max_ctx):
    """First-turn user prompts from LMSYS chat (English), rendered with the
    chat template -> token ids. Falls back to a small built-in list if the
    dataset can't be reached."""
    fallback = [
        "Explain why cities are often built near rivers.",
        "What is the difference between precision and recall?",
        "How does a suspension bridge stay up?",
        "Explain recursion to a ten-year-old.",
        "Why is the sky blue at noon and red at sunset?",
        "How does GPS know where you are?",
    ]
    prompts = []
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET, split="train", streaming=True)
        seen = set()
        for row in ds:
            if row.get("language", "").lower() != "english":
                continue
            conv = row.get("conversation") or []
            if not conv or conv[0].get("role") != "user":
                continue
            text = conv[0]["content"].strip()
            if not text or text in seen or len(text) < 12:
                continue
            seen.add(text)
            prompts.append(text)
            if len(prompts) >= n * 2:  # oversample; length filter drops some
                break
    except Exception as e:  # noqa: BLE001
        print(f"dataset load failed ({e}); using fallback prompts")
        prompts = (fallback * (n // len(fallback) + 1))[:n]

    ids = []
    for p in prompts:
        toks = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=True, enable_thinking=False,
        )
        if len(toks) <= max_ctx:
            ids.append(toks)
        if len(ids) >= n:
            break
    return ids


def stack_positions(sequences, key, checkpoints):
    """Concatenate all tokens across sequences -> {c: Tensor[N_tok, D]}."""
    return {c: torch.cat([s[key][c] for s in sequences], dim=0) for c in checkpoints}


def fit_normalizer(target_train, checkpoints):
    """Per-checkpoint mean/std over honest activations (for whitening before
    projection). Whitening is applied to BOTH the honest reference and the
    forged guess, so it doesn't change honest distances to zero -- it removes
    the massive-activation shortcut (section 5v) by equalizing dimensions."""
    norm = {}
    for c in checkpoints:
        X = target_train[c]
        norm[c] = (X.mean(dim=0, keepdim=True), X.std(dim=0, keepdim=True) + 1e-6)
    return norm


def whiten(x, c, norm):
    if norm is None:
        return x
    mu, sd = norm[c]
    return (x - mu) / sd


def proj_distances(guess_c, truth_c, proj_c, c, norm):
    """Per-token L2 distance between projected (whitened) guess and truth,
    normalized by sqrt(k)."""
    g = whiten(guess_c, c, norm) @ proj_c
    t = whiten(truth_c, c, norm) @ proj_c
    return torch.linalg.vector_norm(g - t, dim=1) / (proj_c.shape[1] ** 0.5)


def honest_floor_distances(truth_c, truth_alt_c, proj_c, c, norm):
    """H0 distances: the honest bf16 activation vs an independent honest fp32
    recompute of the SAME model on the SAME served tokens, projected. This is
    the genuine benign cross-precision noise the verifier tolerates -- the real
    floor, not a synthetic one."""
    return proj_distances(truth_alt_c, truth_c, proj_c, c, norm)


def run_one(cheat_levels, prompt_ids, checkpoints, L, tok):
    """One full sweep for a given cheat severity. Returns the results dict."""
    print(f"\n{'#'*70}\n# CHEAT LEVELS = {cheat_levels} "
          f"({'int4-harsh' if cheat_levels <= 16 else 'int8-gentle'})\n{'#'*70}")
    print("collecting activations (cheap serves, honest recomputes on served tokens)...")
    data = collect(MODEL_NAME, prompt_ids, checkpoints, MAX_TOKENS, cheat_levels=cheat_levels)
    d_model = data["d_model"]
    seqs = data["sequences"]

    n_train = int(len(seqs) * TRAIN_FRAC)
    train, test = seqs[:n_train], seqs[n_train:]
    print(f"sequences: {len(train)} train / {len(test)} test "
          f"({len(test) * MAX_TOKENS} test tokens)")

    tgt_tr = stack_positions(train, "target", checkpoints)
    feat_tr = stack_positions(train, "feat", checkpoints)
    tgt_te = stack_positions(test, "target", checkpoints)
    tgt32_te = stack_positions(test, "target_fp32", checkpoints)
    feat_te = stack_positions(test, "feat", checkpoints)

    norm = fit_normalizer(tgt_tr, checkpoints) if NORMALIZE else None

    # ------------------------------------------------------------------
    # Build forgers. F3 needs an honest shallow feature (layer j); we use the
    # shallowest checkpoint as the truncation depth the attacker pays for.
    # ------------------------------------------------------------------
    j_short = checkpoints[0]
    forgers = [
        MeanForger(),
        CopyForger(),
        RidgeForger(lam=1.0),
        MLPForger(hidden=64, device="cuda"),
        MLPForger(hidden=512, device="cuda"),
    ]
    # F3 shortcut is special-cased (different feature) below.

    # Fit each forger per checkpoint with an INDEPENDENT instance (stateful
    # forgers -- ridge W, MLP weights, stored mean -- must not share state
    # across checkpoints). feature = cheap activation at that checkpoint.
    fitted = {}
    def fresh(proto, c):
        if isinstance(proto, MeanForger):
            return MeanForger().fit(feat_tr[c], tgt_tr[c])
        if isinstance(proto, CopyForger):
            return CopyForger().fit(feat_tr[c], tgt_tr[c])
        if isinstance(proto, RidgeForger):
            return RidgeForger(lam=proto.lam).fit(feat_tr[c], tgt_tr[c])
        if isinstance(proto, MLPForger):
            m = MLPForger(hidden=proto.hidden, device=proto.device)
            return m.fit(feat_tr[c], tgt_tr[c])
        raise TypeError(proto)
    for f in forgers:
        fitted[f.name] = {c: fresh(f, c) for c in checkpoints}
        print(f"  fitted {f.name}")

    # F3 shortcut: predict checkpoint c from honest layer-j activation.
    shortcut = {}
    for c in checkpoints:
        if c <= j_short:
            shortcut[c] = None  # no shortcut needed at/below the truncation depth
        else:
            shortcut[c] = ShortcutForger(j=j_short, lam=1.0).fit(tgt_tr[j_short], tgt_tr[c])
    print(f"  fitted F3-shortcut@{j_short}")

    # F3-strong: distilled thin-suffix forgers at two cost points (1 and 3
    # transformer blocks). Trained on per-sequence honest layer-j -> deep
    # honest activations; predicts all deeper checkpoints at once.
    def seq_list(sequences, key, c):
        return [s[key][c] for s in sequences]

    distilled = {}
    for n_blocks in (1, 3):
        f = DistilledSuffixForger(
            j=j_short, targets=checkpoints, n_blocks=n_blocks, n_head=8,
            epochs=150, lr=1e-3, device="cuda", n_layers=L,
        )
        f.fit(seq_list(train, "target", j_short),
              {c: seq_list(train, "target", c) for c in f.targets})
        distilled[f.name] = f
        print(f"  fitted {f.name}")

    # Precompute distilled guesses on the test set, concatenated to match the
    # token ordering of tgt_te / feat_te (stack_positions cat order).
    distilled_guess = {}  # name -> {c: [N_tok, D]}
    for name, f in distilled.items():
        per_c = {c: [] for c in f.targets}
        for s in test:
            out = f.predict_seq(s["target"][j_short])
            for c in f.targets:
                per_c[c].append(out[c])
        # for checkpoints <= j the forger computed honestly: exact = tgt_te
        distilled_guess[name] = {}
        for c in checkpoints:
            if c in f.targets:
                distilled_guess[name][c] = torch.cat(per_c[c], dim=0)
            else:
                distilled_guess[name][c] = tgt_te[c]

    # ------------------------------------------------------------------
    # Evaluate. For each projection regime, audit depth d, and forger:
    # pass = forged fingerprint within threshold at ALL checkpoints <= d.
    # Thresholds calibrated per (checkpoint, regime) at 1% FPR on H0.
    # ------------------------------------------------------------------
    def guesses_for(forger_name, c):
        """Test-set guessed honest activation at checkpoint c for a forger."""
        if forger_name == "F3-shortcut":
            if shortcut[c] is None:
                return tgt_te[c]  # attacker computed honestly to j >= c: exact
            return shortcut[c].predict(tgt_te[j_short])
        if forger_name in distilled_guess:
            return distilled_guess[forger_name][c]
        return fitted[forger_name][c].predict(feat_te[c])

    def cost_for(forger_name, cks_upto_d):
        """C_surr/C_hon = the deepest honest work the forger pays for among the
        audited checkpoints (it must satisfy the deepest one)."""
        if forger_name == "F3-shortcut":
            return j_short / L
        if forger_name == "F4-projaware":
            return 0.0  # reads only cheap activations
        if forger_name == "F4-oracle":
            return j_short / L  # runs honestly to layer j, then fingerprint map
        if forger_name in distilled:
            return distilled[forger_name].cost_fraction(checkpoints[-1], L)
        # To satisfy the deepest audited checkpoint the forger must read the
        # cheap activation at that depth, so its cost is the max c/L over the
        # audited checkpoints (0 for the mean forger).
        return max(fitted[forger_name][c].cost_fraction(c, L) for c in cks_upto_d)

    # F4 projection-aware: the forger the reseeding defense targets. It learns
    # cheap_act -> whitened honest FINGERPRINT for a SPECIFIC projection. We fit
    # it on epoch 0's projection at each checkpoint; the sweep then evaluates it
    # against the audit epoch's projection (same P for fixed, fresh P for
    # reseeded). Whitening applied to the target so units match the others.
    from difr_mid.forgers import ProjectionAwareForger
    proj0 = make_projections(d_model, checkpoints, FP_K, device="cpu", epoch=0)

    def fit_projaware(feat_train_w, tgt_train, proj_c):
        """Fit F4 with lambda chosen on a held-out validation slice, so the
        forger is as strong as the data allows (not arbitrarily regularized).
        Selection is on fixed-P fingerprint error -- the attacker's best case."""
        n = feat_train_w.shape[0]
        n_val = max(1, n // 5)
        fv, ft = feat_train_w[:n_val], feat_train_w[n_val:]
        tv, tt = tgt_train[:n_val], tgt_train[n_val:]
        true_fp_val = tv.float() @ proj_c
        best, best_err = None, float("inf")
        for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
            m = ProjectionAwareForger(lam=lam).fit(ft, tt, proj_c)
            err = torch.linalg.vector_norm(m.predict_fp(fv) - true_fp_val, dim=1).median().item()
            if err < best_err:
                best_err, best = err, lam
        return ProjectionAwareForger(lam=best).fit(feat_train_w, tgt_train, proj_c), best

    # F4-cheap: reads cheap activations. F4-oracle: reads honest layer-j
    # activations (the attacker paid j/L honest work) -- the STRONGEST possible
    # fixed-P attacker, the real stress test for the reseeding defense.
    projaware, projaware_oracle = {}, {}
    lam_pick = {}
    for c in checkpoints:
        projaware[c], la = fit_projaware(whiten(feat_tr[c], c, norm), tgt_tr[c], proj0[c])
        projaware_oracle[c], lo = fit_projaware(
            whiten(tgt_tr[j_short], j_short, norm), tgt_tr[c], proj0[c])
        lam_pick[c] = (la, lo)
    print("  fitted F4-projaware / F4-oracle; lambda picks (cheap,oracle): "
          + ", ".join(f"c{c}:{lam_pick[c]}" for c in checkpoints))

    def projaware_distances(c, proj_c, oracle=False):
        """Distance between F4's forged fingerprint and the true whitened honest
        fingerprint under proj_c. F4 was trained to hit epoch-0's projection;
        under fixed-P proj_c==proj0 (on-target), under reseeded-P proj_c is
        fresh (stale shadow). oracle=True reads honest layer-j feats."""
        if oracle:
            forged_fp = projaware_oracle[c].predict_fp(whiten(tgt_te[j_short], j_short, norm))
        else:
            forged_fp = projaware[c].predict_fp(whiten(feat_te[c], c, norm))
        # F4 always outputs its epoch-0-trained fingerprint; the audit compares
        # against proj_c. Under fixed-P proj_c == proj0 and the fit is on-target;
        # under reseeded-P proj_c is fresh and the fixed fingerprint is stale.
        true_fp = whiten(tgt_te[c], c, norm) @ proj_c
        # forged_fp is in epoch-0 fingerprint coordinates; to audit under proj_c
        # the forger must commit ONE fingerprint per token ahead of the reveal.
        # That committed value is its epoch-0 prediction, compared to proj_c's
        # true fingerprint -- exactly the stale-shadow situation section 5i
        # describes.
        return torch.linalg.vector_norm(forged_fp - true_fp, dim=1) / (FP_K ** 0.5)

    forger_names = ([f.name for f in forgers] + ["F3-shortcut"]
                    + list(distilled.keys()) + ["F4-projaware", "F4-oracle"])

    # Diagnostic: median projected distance per checkpoint (epoch 0), so the
    # pass/fail table is interpretable. Honest floor vs each forger's error.
    print("\nMedian projected distance per checkpoint (epoch 0), and 1% FPR "
          "threshold from the honest floor:")
    W = max(14, max(len(fn) for fn in forger_names) + 1)
    print(f"{'checkpoint':>12} {'honest-floor':>13} {'threshold':>11} "
          + " ".join(f"{fn:>{W}}" for fn in forger_names))
    diag_projs = make_projections(d_model, checkpoints, FP_K, device="cpu", epoch=0)
    for c in checkpoints:
        h0 = honest_floor_distances(tgt_te[c], tgt32_te[c], diag_projs[c], c, norm)
        thr = torch.quantile(h0, 1 - TARGET_FPR).item()
        cells = []
        for fn in forger_names:
            if fn == "F4-projaware":
                dm = projaware_distances(c, diag_projs[c]).median().item()
            elif fn == "F4-oracle":
                dm = projaware_distances(c, diag_projs[c], oracle=True).median().item()
            else:
                dm = proj_distances(guesses_for(fn, c), tgt_te[c], diag_projs[c], c, norm).median().item()
            cells.append(f"{dm:>{W}.4f}")
        print(f"{c:>12} {h0.median().item():>13.4f} {thr:>11.4f} " + " ".join(cells))

    regimes = ["fixed", "reseeded"]
    results = {r: {fn: {} for fn in forger_names} for r in regimes}

    for regime in regimes:
        epochs = [0] if regime == "fixed" else list(range(1, N_RESEED_EPOCHS + 1))
        for fn in forger_names:
            for d in checkpoints:
                cks = [c for c in checkpoints if c <= d]
                pass_flags = None  # per test token, passes all audited ckpts
                # average pass rate over audit epochs (fresh P each, for reseeded)
                epoch_rates = []
                for ep in epochs:
                    projs = make_projections(d_model, cks, FP_K, device="cpu", epoch=ep)
                    per_ck_pass = []
                    for c in cks:
                        # threshold at 1% FPR from the real honest-vs-honest floor
                        h0 = honest_floor_distances(tgt_te[c], tgt32_te[c], projs[c], c, norm)
                        thr = torch.quantile(h0, 1 - TARGET_FPR).item()
                        if fn == "F4-projaware":
                            # forged fingerprint (bound to epoch-0 P) vs the audit
                            # epoch's true fingerprint: on-target when projs==proj0
                            # (fixed), stale when projs is fresh (reseeded).
                            dist = projaware_distances(c, projs[c])
                        elif fn == "F4-oracle":
                            dist = projaware_distances(c, projs[c], oracle=True)
                        else:
                            g = guesses_for(fn, c)
                            dist = proj_distances(g, tgt_te[c], projs[c], c, norm)
                        per_ck_pass.append(dist <= thr)
                    all_pass = torch.stack(per_ck_pass, dim=0).all(dim=0)
                    epoch_rates.append(all_pass.float().mean().item())
                rate = sum(epoch_rates) / len(epoch_rates)
                results[regime][fn][d] = {
                    "pass_rate": rate,
                    "cost": cost_for(fn, cks),
                    "n_checkpoints": len(cks),
                }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\nPass rate at {int(TARGET_FPR*100)}% FPR, forged fingerprints must pass "
          f"ALL checkpoints <= d simultaneously.")
    NW = max(16, max(len(fn) for fn in forger_names) + 1)
    for regime in regimes:
        print(f"\n=== projection: {regime} ===")
        print(f"{'forger':>{NW}} {'cost/C_hon':>11} " + " ".join(f"{'d='+str(d):>8}" for d in checkpoints)
              + "   (#ckpts: " + ",".join(str(len([c for c in checkpoints if c<=d])) for d in checkpoints) + ")")
        for fn in forger_names:
            cost_full = results[regime][fn][checkpoints[-1]]["cost"]
            cells = " ".join(f"{results[regime][fn][d]['pass_rate']:8.3f}" for d in checkpoints)
            print(f"{fn:>{NW}} {cost_full:>11.3f} {cells}")

    # The reseeding test (section 5i): does a fresh projection catch a forger
    # that fixed-P would let through, at the full-depth audit?
    d_full = checkpoints[-1]
    print(f"\nReseeding effect at full-depth audit (d={d_full}), pass rate:")
    print(f"{'forger':>{NW}} {'cost':>7} {'fixed-P':>9} {'reseeded-P':>11} {'delta':>8}")
    for fn in forger_names:
        pf = results['fixed'][fn][d_full]['pass_rate']
        pr = results['reseeded'][fn][d_full]['pass_rate']
        print(f"{fn:>{NW}} {results['fixed'][fn][d_full]['cost']:>7.2f} "
              f"{pf:>9.3f} {pr:>11.3f} {pf - pr:>+8.3f}")

    print("\nReading: a forger is 'caught' when its full-depth pass rate is ~0 unless")
    print("its cost approaches 1.0 (C_hon). A positive fixed->reseeded delta is direct")
    print("evidence that reseeding the projection (section 5i) does work the fixed")
    print("projection cannot -- only visible if a forger gets close under fixed-P.")

    # Also record the diagnostic median-distance table for plotting.
    diag_table = {}
    for c in checkpoints:
        h0 = honest_floor_distances(tgt_te[c], tgt32_te[c], diag_projs[c], c, norm)
        row = {"floor": h0.median().item(),
               "thr": torch.quantile(h0, 1 - TARGET_FPR).item()}
        for fn in forger_names:
            if fn == "F4-projaware":
                row[fn] = projaware_distances(c, diag_projs[c]).median().item()
            elif fn == "F4-oracle":
                row[fn] = projaware_distances(c, diag_projs[c], oracle=True).median().item()
            else:
                row[fn] = proj_distances(guesses_for(fn, c), tgt_te[c], diag_projs[c], c, norm).median().item()
        diag_table[c] = row

    return {
        "config": {
            "model": MODEL_NAME, "n_layers": L, "d_model": d_model,
            "checkpoints": checkpoints, "fp_k": FP_K, "max_tokens": MAX_TOKENS,
            "n_prompts": len(prompt_ids), "n_train": len(train), "n_test": len(test),
            "train_frac": TRAIN_FRAC, "cheat_levels": cheat_levels,
            "target_fpr": TARGET_FPR, "n_reseed_epochs": N_RESEED_EPOCHS,
            "normalize": NORMALIZE, "shortcut_j": j_short,
        },
        "results": results,
        "diagnostic": diag_table,
    }


def main():
    torch.manual_seed(SEED)
    assert torch.cuda.is_available(), "collection needs the GPU box"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_ids = load_prompts(tok, N_PROMPTS, MAX_CTX)
    print(f"loaded {len(prompt_ids)} prompts")

    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    L = cfg.num_hidden_layers
    checkpoints = sorted({max(1, round(L * f)) for f in (0.25, 0.5, 0.75, 1.0)})
    print(f"{MODEL_NAME}: L={L}, checkpoints={checkpoints}, k={FP_K}, normalize={NORMALIZE}")

    all_runs = {}
    for lv in CHEAT_LEVELS:
        all_runs[str(lv)] = run_one(lv, prompt_ids, checkpoints, L, tok)

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "forgery_sweep_results.json", "w") as f:
        json.dump({"runs": all_runs}, f, indent=2)
    print(f"\nwrote {out_dir / 'forgery_sweep_results.json'}")


if __name__ == "__main__":
    main()
