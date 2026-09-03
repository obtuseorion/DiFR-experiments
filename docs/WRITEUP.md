# Randomized-depth multi-checkpoint Activation-DiFR

A write-up of the idea, the attacks it must survive, the protocol
modifications, and the experiments run to date.

*Reconstruction note (2026-08-20): the original project working copy was lost
with an ephemeral GPU instance; this document and the accompanying code were
rebuilt from the project summary and the upstream DiFR codebase
(github.com/adamkarvonen/difr). Numbers below come from the result JSONs in
`../results/`; figures are in `../results/figures/`.*

---

## 1. Background: what DiFR does

As LLM inference is increasingly outsourced, a customer wants to verify that a
provider actually ran the model it claims, at the precision it claims, without
substituting a cheaper computation. The obstacle is that re-running inference
is not bit-exact — different kernels, batch sizes, and dtypes produce benign
numerical noise — so a naive "recompute and compare" cannot separate legitimate
variation from cheating.

Upstream **DiFR** (Divergence-From-Reference, Karvonen et al.) solves this in
two flavours:

- **Token-DiFR.** Provider and verifier share the sampling seed. The verifier
  recomputes the reference model's logits and checks that each emitted token is
  the one the seeded Gumbel-Max sampler would have chosen. Correct tokens
  become auditable evidence at zero extra cost to the provider.
- **Activation-DiFR.** For sample-efficient forward-pass verification, the
  provider ships a compact *fingerprint* of the hidden state feeding `lm_head`:
  the activation projected through a shared random (Johnson–Lindenstrauss)
  matrix. The verifier recomputes the forward pass and compares projected
  vectors under a calibrated noise threshold. It detects 4-bit quantization
  with AUC > 0.999 in ~2 tokens.

Activation-DiFR's cost is that the verifier pays a **full-depth** forward pass
per audited token. This project asks: can we keep the detection power while
paying less?

---

## 2. The idea: fingerprint at many depths, audit at a random one

Instead of one fingerprint at the final layer, the provider fingerprints the
residual stream at several **checkpoint layers** `C = {c_1 < c_2 < … < c_m = L}`
(here, the quarter-depths of the model). During its normal forward pass it
projects the hidden state at each checkpoint through a per-checkpoint JL matrix
(public seed) and ships all `m` fingerprints. Overhead is `m·k` floats per
token — negligible.

At audit time the verifier draws a depth `d ~ π` over `C`, recomputes layers
`1..d` **only**, and checks every fingerprint at checkpoints `≤ d`. Expected
verifier compute is `E[d]/L` of a full pass instead of 1.

### Why randomization is load-bearing: the coverage gap

A *fixed* shallow audit at depth `d < L` is trivially defeated: a cheater runs
the honest model through layer `d`, then substitutes anything cheaper for
layers `d+1..L`. Every checkpoint the verifier ever inspects is honest, so the
cheat is **never** detected, and the cheater banks exactly the compute the
verifier saved.

With depth drawn from `π`, cheating anywhere in segment `(c_{i-1}, c_i]`
corrupts every checkpoint `≥ c_i`, so it is caught whenever `d ≥ c_i` — with
per-token probability `P_π(d ≥ c_i)`. Over `T` audited tokens the escape
probability decays geometrically, `(1 − P_π(d ≥ c_i))^T`. The depth
distribution `π` is the knob trading expected compute against time-to-detection:
audit shallow often, deep occasionally.

### Threat model

- The provider claims a model `M`, a sampling seed, and a temperature. It may
  substitute quantized weights, a cheaper model, or skip work for some
  layers/tokens.
- Fingerprints are JL projections of the *true* hidden states; producing a
  passing fingerprint at checkpoint `c` should require knowing `h_c` to within
  the calibrated noise floor — i.e. actually doing the honest work up to `c`.
- Honest runs are not bit-exact, so per-checkpoint accept thresholds are
  calibrated on honest cross-implementation runs.
- Out of scope for the *activation* channel: a provider that runs the honest
  model **and** a cheap model and serves the cheap tokens. That is Token-DiFR
  territory (studied separately in this repo under `token_difr/repair_loop.py` /
  `token_difr/sweep_threshold.py`).

---

## 3. The attacks the scheme must survive

Following the forge-by-forward-pass analysis (`forgery-attack-analysis.md`),
an attacker who serves a cheap model `M_cheap` but must commit fingerprints the
verifier can reproduce from `M_honest` has a small menu:

| Strategy | Fingerprints match? | Compute vs honest | Detection |
|---|---|---|---|
| Serve cheap, forge via **full honest pass** | yes | **higher** (pays twice) | none — but no savings |
| Serve cheap, **partial pass to depth d**, fake the tail | fail above `d` | `d/L` saved | `P(audit depth > d)` per token |
| Serve cheap, **learned surrogate** for fingerprints | only if surrogate ≈ honest | savings iff surrogate ≪ honest | rises with every missed checkpoint |
| **Be honest** | yes | baseline | none |

Two facts drive the analysis:

1. **The consistency trap.** The verifier recomputes the *honest* model's
   activations on the *served* (cheap) tokens. So any forgery must reproduce
   `M_honest`'s internal states on a token sequence `M_honest` did not itself
   generate — it cannot reuse the cheap model's own activations, because those
   are exactly what the fingerprint is designed to detect.
2. **The projection can be reseeded.** If the JL matrix is refreshed per audit
   epoch, a surrogate cannot overfit a fixed `k`-dimensional shadow of the
   activation; it must approximate the full `D`-dimensional state.

The faithful full-honest-pass forgery is dominated by simply being honest (it
pays the honest cost *and* the cheap cost). The two savings-positive routes —
partial-pass-and-guess, and the learned surrogate — are what the experiments
below stress.

---

## 4. Modifications to the upstream scheme

Relative to stock Activation-DiFR, this project adds:

- **Multiple checkpoints** at quarter-depths, each with its own seeded JL
  matrix (`difr_mid/fingerprint.py`, `make_projections`). Seeds are drawn on
  CPU regardless of device, because CUDA and CPU RNGs disagree for the same
  seed and prover/verifier may run on different hardware.
- **A `epoch` parameter on the projections** so a fixed-projection audit and a
  reseeded-per-audit audit can be compared head-to-head (the `§5i` defense).
- **A sequence-level audit statistic.** Per-token fingerprint distances are
  heavy-tailed (bf16 noise occasionally spikes on a single token), so the audit
  unit is a *sequence* and the statistic is the mean distance over its tokens.
  This separates honest from cheating cleanly where per-token maxima do not.
- **Per-layer whitening before projection.** Real residual streams carry a few
  enormous-norm "massive activation" dimensions; whitening removes that
  shortcut so a forger can't match only those and let the projection average
  out the rest.
- **A vLLM activation tap** (`difr_mid/tap_vllm.py`) that hooks intermediate
  decoder layers. The canonical post-layer hidden state in vLLM's Llama/Qwen
  layers is the *sum* of the returned `(hidden_states, residual)`; the final
  checkpoint must instead tap the post-final-RMSNorm state to match what a HF
  verifier reports (pre- vs post-norm differ by ~1000× in distance — a bug
  caught during the port).

---

## 5. Experiments

Five experiments, from a toy sanity check to a real-model security probe.
Code under `experiments/`; figures under `../results/figures/`.

### 5.1 Toy-model validation of the coverage gap
`experiments/run_demo.py`

![Toy model: honest and suffix-cheat fingerprint distances overlap until layer
6, then the suffix cheat jumps above the accept threshold at layer 8;
quantized-at-all-layers sits far above throughout. Right panel: a fixed d=4
audit never detects the suffix cheat while shallow-heavy at the same compute
reaches 99% detection in ~44 tokens.](../results/figures/fig_demo.png)

An 8-layer, 64-dim toy GPT, checkpoints at layers 2/4/6/8, fingerprint width
16, 24 prompts. Three providers:

- **honest** — the declared model (verifier recomputes in fp64, so honest
  distances are the numerical noise floor);
- **quant** — 16-level weight quantization at *every* layer;
- **suffix** — honest through layer 6, quantized after (the coverage-gap
  cheater).

**Result.** Honest never flags. Quant flags at every audit depth. The suffix
cheat is invisible at depths 2/4/6 and flags only at depth 8 — exactly the
coverage gap. Across audit policies at matched compute:

| policy | E[compute] | detects suffix? | tokens to 99% |
|---|---|---|---|
| fixed d=4 | 0.50 | never | ∞ |
| shallow-heavy | 0.50 | yes | ~44 |
| uniform random | 0.625 | yes | ~16 |
| fixed d=8 (full) | 1.00 | yes | 1 |

`fixed d=4` and `shallow-heavy` pay the same compute, but only the randomized
policy ever sees the suffix cheat.

### 5.2 Detection on Qwen3-1.7B
`experiments/run_detection.py`

![Detection on Qwen3-1.7B: honest vLLM and HF points sit below the accept
threshold at every checkpoint; fp8-KV rides above it at all depths; the int4
suffix cheat tracks honest until layer 28 then jumps to ~8. Right panel: the
audit-policy tradeoff.](../results/figures/fig_detection_qwen3-1.7b.png)

28 layers, D=2048, checkpoints 7/14/21/28, k=32, 8 prompts × 64 tokens.
vLLM prover (V0 engine, eager) vs HF teacher-forced verifier. Providers:
honest (vLLM and HF), fp8 KV cache, and an int4 suffix cheat honest through
layer 21. Thresholds = 2× the max honest per-sequence mean, per prover family.

**Result** (per-sequence flag rate by audit depth):

| depth | honest (vLLM) | honest (HF) | fp8 KV | int4 suffix |
|---|---|---|---|---|
| 7  | 0 | 0 | 1.0 | 0 |
| 14 | 0 | 0 | 1.0 | 0 |
| 21 | 0 | 0 | 1.0 | 0 |
| 28 | 0 | 0 | 1.0 | 1.0 |

fp8-KV corruption touches attention at every layer, so it is caught even by a
depth-7 audit (25% of full compute). The suffix cheat is invisible until the
final checkpoint. Policy tradeoff mirrors the toy model: `fixed d=14` never
catches the suffix cheat; `shallow-heavy` at the same 0.5× compute reaches 99%
in ~44 sequences.

### 5.3 Detection on Llama-3.1-8B (scale confirmation)
`experiments/run_detection.py` (with `DIFR_MODEL`)

![Detection on Llama-3.1-8B: the same pattern as 1.7B with crisper separation —
honest below threshold everywhere, fp8-KV above it at all depths, the int4
suffix cheat invisible through layer 24 then ~8× over threshold at layer
32.](../results/figures/fig_detection_llama3.1-8b.png)

32 layers, D=4096, checkpoints 8/16/24/32, suffix cheat honest through layer
24. Llama's `lm_head` is untied (unlike Qwen's), so the suffix cheat quantizes
it too.

**Result.** Identical pattern, with *crisper* separation than at 1.7B. Honest:
0 flags everywhere. fp8-KV: caught 100% at every depth. Suffix cheat: honest
distances 0.012–0.06 through layer 24, then **7.9–9.0 at layer 32 against a
threshold of 1.06 (~8×)**. The larger model does not erode detection; if
anything the fingerprints are cleaner (layer-8 honest distances 0.012–0.015,
threshold 0.027). The randomized-depth defense holds at production scale.

### 5.4 Surrogate-forgery sweep
`experiments/run_forgery_sweep.py`, `difr_mid/collect_activations.py`,
`difr_mid/forgers.py`

![Forgery results. Left: the projection-aware forger's training AND held-out
error both exceed the accept threshold past layer 7 — missing signal, not
overfitting. Right: the full forger ladder, every forger above threshold at
depth, with the distilled 3-block suffix climbing from the layer-7 floor it
computed honestly.](../results/figures/fig_forgery.png)

The core security test. A cheap (int8-quantized) model serves tokens; the
honest model recomputes on those served tokens (the consistency trap). A ladder
of forgers tries to reconstruct the honest fingerprint at each checkpoint:

- **F0-mean / F0-copy** — corpus-mean fingerprint / commit the cheap model's
  own activations (zero honest compute);
- **F1-ridge** — linear map cheap→honest activation;
- **F2-mlp{64,512}** — small MLPs;
- **F3-shortcut / F3-distill{1,3}b** — honest to layer 7, then a linear jump or
  a distilled 1-/3-block transformer suffix to the deeper checkpoints (pays
  `≈ 0.25–0.36` honest compute);
- **F4-projaware / F4-oracle** — the projection-aware forgers that target the
  exact public projection (from cheap features / from honest layer-7 features).

Run at scale: 600 sequences (~20,000 training tokens, 10× D=2048), gentle int8
cheat, k=32, whitening on, regularization chosen on a validation split.

**Result** (held-out median distance vs threshold; forger passes only if below
threshold at *every* audited checkpoint):

| checkpoint | threshold | F0-copy | F1-ridge | F2-mlp512 | F3-distill3b | F4-projaware |
|---|---|---|---|---|---|---|
| 7  | 0.20 | 0.46 | 0.40 | 5.15 | 0.00 | 7.23 |
| 14 | 0.35 | 0.49 | 0.51 | 5.26 | 7.43 | 24.84 |
| 21 | 0.24 | 0.47 | 0.44 | 5.31 | 7.67 | 144.38 |
| 28 | 0.28 | 0.71 | 0.54 | 5.67 | 7.91 | 17.24 |

No forger cheaper than the honest pass passes the full-depth audit. Every
forger that computed layer 7 honestly passes a *layer-7-only* audit but fails
the moment a deeper checkpoint is in scope.

### 5.5 Why forgery fails: the signal ceiling
`results/forgery_signal.json` (the left panel of the figure above)

To distinguish "not enough data" (overfitting) from "no signal", the
projection-aware forger's error was measured on both its **training** set and a
held-out set, at the least-regularized setting (the one that fits training
hardest):

| checkpoint | threshold | training error | held-out error |
|---|---|---|---|
| 7  | 0.20 | 0.31 | 0.42 |
| 14 | 0.35 | 1.04 | 1.45 |
| 21 | 0.24 | 5.69 | 8.23 |
| 28 | 0.28 | 1.06 | 1.53 |

Past layer 7 the **training** error itself exceeds the threshold. High training
error is the signature of missing signal, not overfitting — more data would not
help. The honest layer-21 activation is not a linear function of the cheap
activation, nor even of the honest layer-7 activation (in a development probe
the oracle forger, reading honest layer-7 features, had a training error an
order of magnitude above threshold at layer 21). Reproducing the deep state to
fingerprint tolerance requires running the intervening transformer layers.

**Conclusion.** On this model, forgery fails because the deep activation
trajectory is *incompressible* — not recoverable from cheaper or shallower
state — which is a stronger property than projection secrecy. The reseeding
defense is sound in principle: in a synthetic control where the honest target
*was* a linear function of the features, a projection-aware forger passed under
a fixed projection and failed by roughly an order of magnitude under a fresh
one. But on real activations forgery is already blocked before projection
secrecy comes into play, so this run does not isolate reseeding's contribution.
(The synthetic control was a development sanity check, not a saved experiment.)

---

## 6. What is and isn't established

**Established.**
- Randomized-depth multi-checkpoint audit detects fp8-KV misconfig at any
  depth and a suffix cheat at full depth, on toy, 1.7B, and 8B models, with
  zero false flags on honest traffic.
- A fixed shallow audit is structurally blind to the suffix cheat; randomized
  depth restores coverage at equal expected compute.
- Against a ladder of linear and small-distilled forgers, no forger cheaper
  than the honest pass survives the full-depth audit, because the deep
  activation fingerprint is not learnable from cheaper/shallower state.

**Not yet established / caveats.**
- Forgers tested are *linear* (ridge, projection-aware) plus a *small*
  distilled transformer. A much larger distilled surrogate on far more data is
  the open frontier — F3-distill3b already fails (7.9 at layer 28), but this is
  the honest boundary of the claim.
- The benign-noise floor uses a bf16-vs-fp32 honest recompute as a stand-in;
  the deployment-real floor is the vLLM-vs-HF cross-implementation gap (what
  §5.2/5.3 measure for detection) and is still to be wired into the forgery
  sweep.
- The reseeding defense is demonstrated only on synthetic data; isolating it on
  a real model needs a forger that gets close enough under fixed-P for a fresh
  projection to matter, which this model's incompressibility prevents.
- Thresholds are calibrated in-sample; a deployed system would calibrate on
  held-out honest traffic with more sequences.

---

## 7. Repository map

```
difr_mid/fingerprint.py         JL projections (seeded, reseedable), distances
difr_mid/tap_vllm.py            vLLM multi-checkpoint activation tap
difr_mid/collect_activations.py honest+cheap activation collection (consistency trap)
difr_mid/forgers.py             the forger ladder F0–F4
experiments/run_demo.py         toy coverage-gap validation
experiments/run_detection.py    detection on real models (DIFR_MODEL env override)
experiments/run_forgery_sweep.py surrogate-forgery sweep
experiments/make_figures.py     regenerate all figures from results/*.json
experiments/plot_style.py       plotting setup
results/*.json                  experiment outputs
results/figures/*.png           figures
difr_org/                       upstream DiFR reference files (unmodified)
token_difr/                     Token-DiFR side track (repair loop, threshold sweep)
tools/                          GPU diagnostics (seed/fp8 checks, env capture, ssh config)
docs/forgery-attack-analysis.md the forge-by-forward-pass analysis
docs/EXPERIMENT_entropy_divergence.md  Token-DiFR entropy-divergence experiment design/log
```
