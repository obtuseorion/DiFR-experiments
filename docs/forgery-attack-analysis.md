# The Forge-by-Forward-Pass Attack

**Scenario.** A provider advertises an expensive model $M_{\text{hon}}$ but wants
to serve a cheaper one $M_{\text{cheap}}$ (a quantized, distilled, or
layer-skipped variant) to save compute. Under Activation-DiFR the provider must
also commit activation fingerprints that a verifier can reproduce by running
$M_{\text{hon}}$. The attack under study: **serve the cheap model's tokens to the
user, but forge passable fingerprints by separately running a forward pass.**

The question is whether any version of "run a forward pass to forge" leaves the
attacker ahead. The short answer is no: forging fingerprints faithfully requires
paying the honest model's cost, so the cheat is dominated by simply being honest.
The rest of this note makes that precise, isolates the *only* forgery with
positive expected savings (a learned surrogate), and shows why our
randomized-depth multi-checkpoint scheme makes that surrogate about as expensive
as the honest model it is trying to avoid.

---

## 1. Setup and notation

For a sequence of $n$ served tokens on an $L$-layer model:

- $a_\ell(t_{1:i})$ — the residual-stream activation at layer $\ell$, position
  $i$, of model on context $t_{1:i}$.
- $P_\ell \in \mathbb{R}^{k\times D}$ — the seeded random orthogonal projection
  for checkpoint $\ell$, shared by provider and verifier.
- Fingerprint: $f_\ell^{(i)} = P_\ell\, a_\ell(t_{1:i})$.
- The provider **commits** $\{f_\ell^{(i)}\}$ append-only at generation time,
  for every checkpoint layer $\ell \in \mathcal{C}$ and logged position $i$.
- At audit time the verifier samples a cutoff depth $\ell^\star$, recomputes
  $M_{\text{hon}}$ **to depth $\ell^\star$ on the served tokens**, and for each
  $\ell \le \ell^\star$ checks
  $\lVert P_\ell\,\hat a_\ell - f_\ell \rVert_2 \le \tau_\ell$
  against the per-layer threshold $\tau_\ell$.

Two facts about the verifier's check drive everything below:

1. **The verifier recomputes on the served tokens.** The reference activation is
   $a_\ell$ of the *honest* model evaluated on exactly the sequence the user
   received. Whatever produced those tokens, the fingerprints must match
   $M_{\text{hon}}$'s activations *on that specific sequence*.
2. **The projection is distance-preserving (Johnson–Lindenstrauss).** Being
   within $\tau_\ell$ in the $k$-dimensional projected space implies being close
   in the full $D$-dimensional activation space (up to JL distortion). The
   attacker cannot "match the projection" without approximately matching the
   underlying activation direction — and because $P_\ell$ is reseedable, they
   cannot overfit to a fixed low-dimensional target.

Let $C_{\text{hon}}$ and $C_{\text{cheap}}$ denote the FLOPs of one forward pass
(per token) of each model, with $C_{\text{cheap}} < C_{\text{hon}}$.

---

## 2. The consistency trap

The attacker serves tokens from $M_{\text{cheap}}$. Call them $t_{1:n}$. The
committed fingerprint at $(\ell, i)$ must equal $P_\ell\, a_\ell^{\text{hon}}(t_{1:i})$
— the honest model's layer-$\ell$ activation **on the cheap model's tokens** —
because that is precisely what the verifier recomputes.

This is a trap, not a convenience. The attacker cannot reuse any activation the
cheap model computed while generating, because $a_\ell^{\text{cheap}} \ne
a_\ell^{\text{hon}}$ wherever the models differ — that difference is exactly what
the fingerprint is designed to detect. So the fingerprints have to come from a
*separate* computation that reproduces $M_{\text{hon}}$'s internal states on a
token sequence $M_{\text{hon}}$ did not itself produce.

(Note: on the activation channel alone, serving cheap-model tokens is not
directly penalized — $M_{\text{hon}}$'s activations are well-defined on any
sequence. What is penalized is failing to reproduce those activations. Token-DiFR
closes the orthogonal gap of whether the tokens were honestly sampled; here we
analyze the activation channel in isolation.)

---

## 3. Core accounting: honest forgery is strictly dominated

Consider the faithful version of the attack — the attacker forges *correct*
fingerprints by actually running $M_{\text{hon}}$'s forward pass over the served
tokens to obtain the true activations.

$$
\text{Cost}_{\text{cheat}}
= \underbrace{n\,C_{\text{cheap}}}_{\text{serve tokens}}
+ \underbrace{n\,C_{\text{hon}}}_{\text{full honest pass for fingerprints}}
\;>\; n\,C_{\text{hon}}
= \text{Cost}_{\text{honest}}.
$$

The attacker pays for the honest forward pass *anyway* — that is the only way to
get activations that match at every depth — and then pays *extra* to run the
cheap model on top. The result is negative savings.

The reason is structural, and it is the crux of the whole scheme:

> **Honesty amortizes generation and fingerprinting into a single forward pass.**
> The honest provider runs $M_{\text{hon}}$ once; that same pass emits the served
> tokens *and* the activations the fingerprints project. Any attacker who runs a
> full honest pass purely to forge fingerprints has already paid the honest cost
> and gained nothing from the cheap model — so they are strictly worse off than
> an honest provider, who at least delivers the promised product for that price.

So "forge by running a forward pass" fails the moment the forward pass is a
*faithful full-depth* pass. To have any hope of positive savings, the attacker
must avoid running $M_{\text{hon}}$ to full depth — which forces one of the two
approximations below.

---

## 4. The only forgeries with positive savings

### 4a. Partial pass + guess the tail

Run $M_{\text{hon}}$ honestly only to some depth $d < L$, then fabricate
fingerprints for layers $> d$ without computing them. Cost
$\approx n\,C_{\text{cheap}} + (d/L)\,n\,C_{\text{hon}}$.

This is exactly the honest-prefix attacker, and it is what **randomized depth**
was built to defeat. The fabricated deep fingerprints cannot match
$M_{\text{hon}}$'s true deep activations (the attacker never computed them, and
deep activations are not predictable from layer-$d$ state without doing the work
of the intervening layers). Whenever the verifier draws $\ell^\star > d$, every
checkpoint in $(d, \ell^\star]$ fails. Detection probability per audit is
$P(\ell^\star > d)$; the attacker's savings are $(L-d)/L$ of the pass. Savings
and detection risk rise together, and only $d = L$ (i.e. honesty) is risk-free.

### 4b. Learned surrogate for the activations

Train a cheap function $g_\ell$ that predicts $P_\ell\, a_\ell^{\text{hon}}$
directly — a distilled/low-rank/quantized network that emits *approximate honest
fingerprints* at cost $C_{\text{surr}} \ll C_{\text{hon}}$. Serve $M_{\text{cheap}}$
tokens, run $g$ for fingerprints. This is the genuine forgery threat: the
attacker never runs $M_{\text{hon}}$ at all, yet tries to produce fingerprints
that pass. Positive savings require

$$
C_{\text{cheap}} + C_{\text{surr}} \;<\; C_{\text{hon}}.
$$

The rest of the note is about why building such a $g$ is nearly as hard as
possessing $M_{\text{hon}}$.

---

## 5. Why the surrogate is hard

**(i) The projection is incompressible.** Because $P_\ell$ is a
distance-preserving random projection and can be reseeded per audit epoch, $g$
cannot target a fixed $k$-dimensional shadow of the activation. To keep
$\lVert P_\ell g_\ell - P_\ell a_\ell^{\text{hon}}\rVert$ small under a fresh
projection, $g_\ell$ must approximate the *full* activation
$a_\ell^{\text{hon}}$ in direction and magnitude, not some compressed statistic
of it. Matching a $D$-dimensional internal state to fingerprint tolerance is a
far tighter constraint than matching output tokens — ordinary distillation
optimizes the latter and is free to reach it via very different internals.

**(ii) Multi-checkpoint forces trajectory matching.** A single checkpoint
constrains one point on the depth trajectory; there may be cheap functions that
land near one point. Independent checkpoints at
$\ell \in \{2,4,8,16,24,32\}$, each with its own seeded projection, force $g$ to
track $M_{\text{hon}}$'s activations at every sampled depth simultaneously. A
surrogate that reproduces the layer-16 state but diverges by layer-24 is caught
whenever $\ell^\star \ge 24$. Reproducing the whole trajectory to tolerance is
close to reproducing the computation that generates it.

**(iii) Randomized depth forces accuracy everywhere, not on average.** The
verifier's cost scales with $\mathbb{E}[\ell^\star]$, but the surrogate's
exposure scales with the *support* of the depth distribution. Because any
checkpoint may be the one audited, $g$ must be within-threshold at *all* of them
— a worst-case constraint, not an average-case one. There is no shallow region
the attacker can compute honestly and no deep region they can safely neglect.

**(iv) Shallow thresholds are tight.** Benign numerical noise accumulates with
depth, so $\tau_\ell$ is *smallest* at shallow layers (in our toy run,
$\tau_2 \approx 0.006$ vs $\tau_{32} \approx 0.033$). The surrogate must be
*most* accurate exactly where activations are least forgiving. A cheap
approximation that is "close enough" by layer 32's loose tolerance can still
blow the layer-2 or layer-4 budget.

**(v) The massive-activation loophole, and its patch.** Real mid-layer residual
streams carry a few enormous-norm dimensions (attention-sink features) that
dominate the raw $\ell_2$ norm. A surrogate could try to match only those and
let the projection "average out" the rest. Per-layer normalization (or
whitening) before projection removes this shortcut and should be applied — this
is the one place where an otherwise-hard surrogate could get cheap traction, so
it is worth closing explicitly.

Taken together: any $g$ cheap enough to yield savings (4b) is, by construction,
a model that reproduces $M_{\text{hon}}$'s full-depth activation trajectory to
tight tolerance on arbitrary token sequences. That capability is
computationally close to $M_{\text{hon}}$ itself — so the inequality
$C_{\text{cheap}} + C_{\text{surr}} < C_{\text{hon}}$ is very hard to satisfy in
practice, and the attack collapses back toward "just run the honest model."

---

## 6. Putting it together: the cheater's menu

Every route the attacker has:

| Strategy | Fingerprints match? | Compute vs honest | Detection |
|---|---|---|---|
| Serve cheap, forge via **full honest pass** | yes | **higher** (pays twice) | none — but no savings |
| Serve cheap, **partial pass to $d$**, fake tail | fail above $d$ | $\frac{d}{L}$ saved | $P(\ell^\star > d)$ per audit |
| Serve cheap, **surrogate** $g$ for fingerprints | only if $g\approx M_{\text{hon}}$ | savings iff $C_{\text{surr}}\ll C_{\text{hon}}$ | rises with every checkpoint $g$ misses |
| **Be honest** | yes | baseline | none |

The dominant diagonal is honesty: it is the unique strategy that both passes
every audit and delivers the advertised product for the advertised compute. The
faithful forgery is dominated by honesty (it costs more for the same detection
profile). The two savings-positive forgeries each pay for their savings in
detection probability, and randomized-depth multi-checkpoint is what converts
"savings" into "proportional detection risk" across the entire network.

The deterrence condition inherited from the protocol note: cheating above depth
$d$ is negative-EV when
$$
q \cdot P(\ell^\star > d)\cdot V \;>\; \frac{L-d}{L},
$$
for audit rate $q$ and penalty $V$. The surrogate attack does not escape this —
it merely changes *which* depths it fails at, and every failed depth contributes
its mass to $P(\ell^\star > d)$.

---

## 7. Experiment to validate the surrogate bound

The one route above that is not settled by pure accounting is 4b — it is an
empirical claim that a cheap $g$ cannot match the trajectory. Direct test:

1. Fix $M_{\text{hon}}$ (e.g. Llama-3.1-8B). Choose $M_{\text{cheap}}$ = 4-bit
   quant.
2. Train surrogates $g$ of increasing capacity/cost to predict the *projected*
   activations $P_\ell a_\ell^{\text{hon}}$ at the checkpoint layers, from the
   token sequence (or from $M_{\text{cheap}}$'s cheap activations as input
   features — the most favorable case for the attacker).
3. Sweep: pass rate at $\tau_\ell$ (1% FPR) **vs** surrogate cost
   $C_{\text{surr}}/C_{\text{hon}}$, and **vs** number of checkpoints the
   surrogate must satisfy simultaneously (1 vs 2 vs 6).
4. Headline plot: pass rate against $C_{\text{surr}}$, one line per
   checkpoint-count. The multi-checkpoint hypothesis predicts the curves drop
   sharply as checkpoints are added, and that reaching a high pass rate requires
   $C_{\text{surr}} \to C_{\text{hon}}$.

If the six-checkpoint curve only reaches a passing rate as $C_{\text{surr}}$
approaches $C_{\text{hon}}$, that is the empirical statement of "forging is as
expensive as honesty," and it is the quantitative core of the paper's security
argument.
