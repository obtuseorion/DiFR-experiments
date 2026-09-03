# Experiment design: does cheap-model entropy predict Token-DiFR divergence?

**Status:** BUILT AND RUNNING (2026-08-20). Decisions locked (user said "go ahead"
without picking options, so my stated leans stand):
- cheap model = declared int4 (bnb) — strongest-attacker / extends repair_loop.
  Small distinct model (Llama-3.2-1B) is the ablation, not yet run.
- honest batch-tau calibrated from a real honest-vs-honest run: declared bf16
  serving itself (`--cheap` = declared), scored by declared. 128 prompts.
- attack run: 256 prompts x 200 tokens ~= 51k paired tokens.
- Scripts: experiments/collect_entropy_divergence.py (GPU, two passes),
  experiments/analyze_entropy_divergence.py (local, predictors + plots).

**Gumbel consistency verified (important):** the cheap draft picks the served
token via argmax(probs/exp_noise) with probs=softmax(logits/T); the verifier's
margin (compute_margin_batch) uses argmax(logits + T*gumbel), gumbel=-log(exp).
These are the SAME ordering: argmax(probs/exp) = argmax(log_softmax(logits/T) +
gumbel) = argmax(logits/T + gumbel) = argmax(logits + T*gumbel) since logsumexp
is constant across vocab. So served-token selection and margin scoring are
consistent by construction (smoke run confirms: ~89% exact-zero margin, not ~0).

## The claim under test

Token-DiFR's security rests on an unstated assumption: that a cheating provider
cannot cheaply tell *which* tokens it is about to get caught on. If it could, it
would run the declared model only there and serve the cheap model everywhere
else — a *selective cheat* (§7.2 of the paper, named but never quantified) that
never pays for a full declared audit and so is **not** dominated by honesty.

The whole selective cheat hinges on one predictor being good:

> **H:** the cheap model's own per-token entropy (free to the attacker while
> drafting) predicts whether that token will *fail* the declared model's
> Token-DiFR check under the shared Gumbel seed.

- If **H holds** (predictor AUC high): selective cheating is real. The attacker
  spends declared compute on the small predicted-risky fraction and lands under
  τ, saving `(1 − risky_fraction)·(C_hon − C_cheap)` per token. Token-DiFR is
  *compressible* — the token-channel analog of the activation break we could not
  find.
- If **H fails** (predictor AUC ≈ chance): divergence is unpredictable from
  cheap-side state, so there is no cheap way to know where to spend the declared
  pass. This is the **incompressibility** result on the token channel, matching
  the activation finding — a unified security story.

Either way the outcome is a clean, publishable binary. That is why this is the
first experiment.

## Getting the divergence event *exactly* right

The naive event "argmax of cheap logits ≠ argmax of declared logits" is **wrong**
for this scheme. The verifier does not compare argmaxes; it recomputes the
declared model's **Gumbel-Max** choice under the shared seed and scores the
provider's served token against it (`verify_vllm_gumbel_max` in
`token_difr_vllm.py`). So the event that costs the statistic is:

    served token t*  vs  t_hat = argmax_i ( declared_logits[i] + T · gumbel_seed[i] )

and the per-token cost is the clipped margin
`min( z[t_hat] − z[t*], Δmax )`, where `z = declared_logits + T·gumbel`, and it
is `+∞ → Δmax` if `t*` falls outside declared top-k/top-p.

In the selective-cheat scenario the served token `t*` is the **cheap model's**
Gumbel-Max pick on the same seed and same position (the attacker drafts with the
cheap model). So the divergence label we must predict is:

    y_pos = 1[ margin_declared(t*_cheap, seed) > tau_tok ]

for a per-token repair threshold `tau_tok`, and — because the verifier's real
decision is on the batch mean — we ALSO care about the continuous margin itself,
not just the binary. Two targets, reported side by side:

  - **binary:** `y = 1[margin > tau_tok]`  → ROC / AUC of the predictor.
  - **continuous:** `margin` (clipped)      → does the predictor rank tokens by
    how much they cost the batch statistic?

## The predictor (attacker-side, must be free)

Everything the predictor reads must be computable by the attacker **without**
running the declared model — otherwise we have smuggled the honest pass back in
and reproduced the repair_loop failure. Allowed features, all from the cheap
draft pass:

  - `H_cheap` = entropy of the cheap model's (temp/top-k/top-p-filtered) softmax
    at that position. **Primary feature.** One scalar/token, already computed.
  - `max_p_cheap` = top probability under cheap model (peakiness proxy).
  - `gap12_cheap` = p1 − p2 (margin between cheap model's top two).
  - `nucleus_size` = # tokens surviving top-k/top-p under cheap model.
  - position index (cheap tokens late in a sequence may drift).

Predictors, in increasing power (report all — the ladder is the result):

  - **P0:** threshold on `H_cheap` alone (the pure hypothesis; one number).
  - **P1:** logistic regression on the 4–5 scalar features above.
  - **P2:** small gradient-boosted tree on the same features (captures
    nonlinearity without needing the declared pass).

If even P0 gives high AUC, H is strongly confirmed and the attack is cheap. If
P2 is needed, the attacker pays a tiny bit more (train a cheap classifier
offline) but the online cost is still just cheap-side scalars.

## Cost accounting (the thing that decides if it's an attack)

Let `r = risky_fraction` = fraction of tokens the predictor flags to run
declared-honestly at operating point (chosen to keep batch statistic < τ).
Online cost per served token, attacker:

    C_attack = C_cheap            (draft every token)
             + r · C_hon          (declared pass only on flagged tokens)

Honest baseline is `C_hon` per token. Attack is profitable iff
`C_cheap + r·C_hon < C_hon`, i.e. `r < 1 − C_cheap/C_hon`. For int4 cheap on an
8B declared, `C_cheap/C_hon ≈ 0.25–0.4` → attacker needs `r ≲ 0.6–0.75` while
still passing. **The experiment's headline number is the smallest `r` that keeps
the batch statistic under τ at 1% FPR** — read off the predictor's ROC composed
with the margin distribution. Two caveats to model honestly, not hand-wave:

  1. **The declared pass on flagged tokens must be a real forward pass in
     context**, so on a flagged position the attacker pays declared *prefill*
     for the whole prefix up to there unless it maintains a declared KV cache in
     parallel (which costs declared compute on every token → kills the attack).
     So the realistic attacker only gets `r·C_hon` if it can run declared as a
     *sparse* re-check, which in autoregressive serving means it must have the
     declared hidden state at that position — i.e. it pays declared for the
     prefix. **This is the crux and the place the attack can still die.** The
     design must measure `r` under the *honest* accounting where a flagged token
     costs a declared prefill to its position, then also report the optimistic
     "declared is free-to-resume" bound. If the attack only works under the
     optimistic bound, that is itself the finding (it says the KV-cache
     dependency, not entropy-unpredictability, is what saves Token-DiFR).
  2. **Repairing changes the downstream trajectory** (same lesson as
     repair_loop). Here we sidestep it in v1 by measuring the *predictor's*
     quality on a fixed honest-vs-cheap paired corpus (no repair), which cleanly
     isolates H. The repair-dynamics cost is a v2 concern only if v1 says H holds.

## Data collection (reuses existing machinery)

On the GPU, for N LMSYS/UltraChat prompts (target ~50k–100k tokens, matching the
paper's batch-size regime), with declared = Llama-3.1-8B-Instruct, cheap = same
in int4 (bitsandbytes), shared seed 42, temp 1.0, top-k 50, top-p 0.95:

  1. **Cheap draft pass** (vLLM, cheap model): generate tokens with the shared
     seed. Record, per position: the served token `t*_cheap`, and the cheap
     features (`H_cheap, max_p_cheap, gap12, nucleus_size`). This is the only
     pass the *attacker* is allowed.
  2. **Declared scoring pass** (the verifier's job, and OUR ground truth):
     teacher-force the declared model over prompt+cheap-tokens, run
     `verify_vllm_gumbel_max` to get `margin_declared` per position. This is what
     we are trying to *predict from step 1*; the attacker never runs it.

Both are single prefill passes — cheap, minutes. Save paired
`(features_from_cheap, margin_from_declared)` to JSON, rsync down, analyze
locally.

## Analysis (local, no GPU)

  - Fit P0/P1/P2 on a train split, evaluate on held-out prompts (unique prompts —
    watch the greedy-determinism train==test trap from the activation work).
  - **Plot 1 (the hypothesis):** ROC of each predictor for `y = 1[margin>tau_tok]`.
    Report AUC. Sweep `tau_tok`.
  - **Plot 2 (the attack):** for each operating point of the best predictor,
    compute `r` (fraction flagged declared) and the resulting batch statistic on
    the un-flagged (cheap-served) remainder; overlay honest τ at 1% FPR. The
    x-intercept where the attacked statistic crosses τ gives the minimal `r`.
    Convert to `C_attack/C_hon` and compare to the `1 − C_cheap/C_hon` break-even.
  - **Plot 3 (both accountings):** minimal-`r` attack cost under (a) declared-
    resume-is-free optimistic bound and (b) declared-costs-prefix honest bound.
    The gap between them localizes *what* actually protects Token-DiFR.

## What would falsify the attack

  - P0/P1/P2 all sit near AUC 0.5 → entropy does not predict divergence →
    incompressibility on the token channel. (Report as the positive security
    result.)
  - H holds but minimal `r` under honest accounting still exceeds
    `1 − C_cheap/C_hon` → the predictor is good but the KV-cache/prefix
    dependency makes sparse declared re-checks too expensive → Token-DiFR is
    saved by serving structure, not unpredictability. (A different, also
    interesting, finding.)

## Deliverables

  - `experiments/run_entropy_divergence.py` (GPU: two passes, saves paired data)
  - `experiments/analyze_entropy_divergence.py` (local: predictors + 3 plots)
  - `results/entropy_divergence.json`, figures under `results/figures/`
  - a short write-up section slotting into WRITEUP.md as the Token-DiFR analog of
    the activation incompressibility result.

## Open questions for you before I build

  1. **Cheap model choice.** int4 bnb (matches repair_loop) vs a genuinely
     smaller draft model (e.g. Llama-3.2-1B as the cheap serve). int4 is the
     conservative/strong-attacker case (cheap is *close* to declared, so
     divergence is rare and maybe more predictable); a small distinct model is
     the realistic-latency-cheat case. I lean **int4 first** (directly extends
     repair_loop, strongest case for the attacker) then a small model as ablation.
  2. **tau_tok vs batch-τ.** I plan to derive the honest batch-τ at 1% FPR from
     an honest-vs-honest run (bf16 vLLM vs HF, as in the detection work) and
     treat `tau_tok` as a swept knob. Confirm that matches how you want the
     operating point defined.
  3. **Scale.** 50k tokens is enough to fit predictors and get a stable batch-τ;
     the paper uses ~1M/config for its AUC curves. Start at 50k, scale only if
     the predictor AUC is borderline. OK?

---

## RESULT (2026-08-20)

**Data:** Llama-3.1-8B-Instruct declared, int4-bnb cheap, shared seed 42, temp 1.0,
top-k 50, top-p 0.95. Attack corpus 256 prompts x 200 tok = 51,200 paired tokens.
Honest calibration (bf16 self-serve) 128 prompts = 25,600 tokens.

**1. The hypothesis holds strongly.** Cheap-model entropy predicts
`1[declared margin > tau_tok]` at **AUC ~0.79**, stable across tau_tok in
{0.05, 0.14, 0.3, 0.5}. Entropy-only (P0), logistic (P1), and gradient boosting
(P2) are identical to 3 decimals — **entropy IS the signal**; the other features
(max_p, gap12, nucleus_size, position) add nothing. 88.3% of served tokens have
exact-zero margin (cheap and declared agree under the shared seed); only ~10%
carry divergence, and that 10% is predictable. So the token channel is
**compressible** — unlike the deep activation trajectory.

**2. But viability is entirely a cost-accounting question, and the two bounds
diverge 16x.** Selectively repairing the top-r predicted-risky fraction (declared-
served, margin 0) drops the batch statistic under the honest 1%-FPR threshold
(tau_batch = 0.0029) at **r = 0.51**.
  - **Optimistic** (declared decode resumes free from a maintained state):
    C_attack = C_cheap + r = 0.30 + 0.51 = **0.81 C_hon** -> attack profits.
  - **Honest** (no maintained declared KV cache; each sparse re-check needs a
    declared PREFILL over the prefix, avg length T/2 = 100, prefill 4x cheaper):
    declared work alone = 0.51 * (100/4) = 12.75 C_hon -> C_attack = **13.05
    C_hon** -> no gain whatsoever. Break-even r under honest accounting ~= 0.028,
    ~18x below the 0.51 the predictor needs.

**3. The real finding.** Token-DiFR is **not** protected by divergence being
unpredictable — it is very predictable (AUC 0.79). It is protected by
**autoregressive serving structure**: you cannot evaluate the declared model at
position t without its hidden state up to t, so "run declared only on the risky
tokens" is not actually sparse — each re-check drags in a full prefix prefill.
This is a sharper security statement than the paper's ("seed leaves no room")
and different from the activation incompressibility result.

**4. The open crack.** The one regime where sparse declared *decode* IS cheap is
if the attacker maintains the declared KV cache — but that costs declared on
every token, so it is dominated by honesty... UNLESS a cheap surrogate can supply
the declared KV/hidden state on the fly. That loops straight back to the
activation-forgery incompressibility question: is the declared hidden state
recoverable from cheap-side state? We already have evidence it is not (linear
surrogates fail at depth). So the two channels close each other's escape routes.

**Figures:** results/figures/fig_entropy_divergence_roc.png (predictor AUC),
fig_entropy_divergence_attack.png (attack pass-curve + dual cost accounting).
**Summary JSON:** results/entropy_divergence_analysis.json.

**Caveats / next:** (a) int4-bnb only; the small-distinct-model cheap (Llama-3.2-1B)
ablation is unrun and would have a lower C_cheap/C_hon and possibly different
predictability. (b) The honest bound assumes no declared KV maintenance; the
maintained-KV regime is the crack above and deserves its own accounting. (c)
tau_batch uses bf16-self-serve as the honest floor (cheap-draft-vs-teacher-forced
numerical gap); deployment-real is vLLM-vs-HF, same TODO as the activation work.

---

## ABLATION: distinct small model cheap (Llama-3.2-1B-Instruct), 2026-08-20

Replaces the int4-of-declared cheap model with a genuinely smaller, distinct model
(unsloth/Llama-3.2-1B-Instruct — ungated mirror, vocab 128256 identical to the 8B
declared, 16 layers / hidden 2048). Same 256 prompts x 200 tokens = 51,200 paired
tokens; honest calibration reused (independent of cheap model). C_cheap/C_hon =
0.155 (1.24B/8.03B param ratio).

| quantity | int4-of-8B | 1B-distinct |
|---|---|---|
| predictor AUC (entropy P0) | 0.79 | 0.80 |
| exact-zero margin tokens | 88.3% | 75.6% |
| gold-filtered (inf) tokens | 1.9% | 5.7% |
| divergence rate (risky @ tau=0.05) | 11% | 24% |
| finite-margin mean | 0.072 | 0.280 |
| min repair r to pass audit | 0.51 | 0.82 |
| C_cheap/C_hon | 0.30 | 0.155 |
| C_attack optimistic | 0.81 | 0.975 |
| C_attack honest (prefix prefill) | 13.05 | 20.66 |

**The ablation strengthens the conclusion.** Two effects that both hurt the attacker:

1. **Entropy predicts divergence regardless of cheap-model type** (AUC ~0.80 either
   way). The predictability finding is not an int4 artifact — it is a property of
   the token channel. Entropy-only still ties gradient boosting; entropy IS the
   signal.
2. **A cheaper cheap model diverges MORE, and that erases its savings.** The 1B is
   0.155 C_hon to run (vs int4's 0.30), but it disagrees with the 8B on 24% of
   tokens (vs 10%), so it must repair r=0.82 (vs 0.51). Even under the optimistic
   free-resume accounting the attack costs 0.975 C_hon — it barely breaks even —
   and under honest accounting it is 20.66 C_hon, worse than int4.

So across the cheap-model ladder the selective-cheat attack is **at best marginal**
(optimistic bound, and only for a cheap model close enough to the declared one that
divergences stay rare) and **hopeless** under the honest serving-structure bound.
Making the served model cheaper just makes it diverge more, which is exactly what
the seed-conditioned audit catches. This is the token-channel statement of why
Token-DiFR is hard to game: there is no free lunch in the cheap/divergence tradeoff.

**Figures:** results/figures/fig_entropy_divergence_1b_{roc,attack}.png.
