"""The forger ladder for the surrogate attack (forgery-attack-analysis.md
section 4b/7). Each forger maps cheap-model activations at a checkpoint to a
GUESS of the honest activation at that checkpoint, at some online FLOP cost.

All forgers predict RAW D-dim activations. The audit then projects both the
guess and the true honest activation through the audit epoch's P and measures
distance. Predicting raw activations (not a fixed projection of them) is what
the reseeded-P protocol forces: a forger that targeted one fixed P would be
defeated by a fresh P, so the honest-difficulty question is about matching the
full activation (section 5i).

COST MODEL (the C_surr/C_hon axis). The attacker ALWAYS pays for the cheap pass
that serves the user -- that is the product they sell, not part of the forgery
budget. The quantity the defense forces up, and the only thing we chart, is the
HONEST forward-pass FLOPs the forger must consume to forge, as a fraction of a
full honest pass C_hon:

    forgers that read only cheap activations   -> 0  (no honest work)
    a forger that runs honestly to layer j     -> j / L
    an exact/honest forger                     -> 1.0

The forger's own head eval (a linear or small-MLP map on D dims) is negligible
next to a transformer layer and is not counted. The claim under test is that
pass rate at the full-depth audit stays ~0 until this honest fraction -> 1.
"""

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Forgers. Each exposes: .fit(feat_train, target_train), .predict(feat) -> guess
# and .cost_fraction(checkpoint, n_layers) -> C_surr/C_hon for a token whose
# fingerprint at `checkpoint` this forger produces.
# ----------------------------------------------------------------------------

class CopyForger:
    """F0a: guess the honest activation = the cheap activation itself.
    The attacker serves the cheap model and simply commits its own activations,
    hoping they pass. Cost: only the cheap pass to depth c that it already ran.
    """
    name = "F0-copy"

    def fit(self, feat, target):
        return self

    def predict(self, feat):
        return feat

    def cost_fraction(self, c, L):
        # Reads only the cheap activation it already computed while serving:
        # zero honest forward-pass work.
        return 0.0


class MeanForger:
    """F0b: guess = corpus-mean honest activation at this checkpoint. Zero
    online cost (a stored constant); the floor any real forger must beat."""
    name = "F0-mean"

    def fit(self, feat, target):
        self.mu = target.mean(dim=0, keepdim=True)
        return self

    def predict(self, feat):
        return self.mu.expand(feat.shape[0], -1)

    def cost_fraction(self, c, L):
        return 0.0


class RidgeForger:
    """F1: affine map cheap_act -> honest_act, closed-form ridge regression.
    Online cost = one D×D matmul on top of the cheap activation the attacker
    already has. Negligible vs a transformer layer, so cost ~= reading the
    cheap activation at c (c/L)."""
    name = "F1-ridge"

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, feat, target):
        X = feat.double()
        Y = target.double()
        n, d = X.shape
        Xa = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)
        A = Xa.T @ Xa + self.lam * torch.eye(d + 1, dtype=X.dtype)
        self.W = torch.linalg.solve(A, Xa.T @ Y).float()  # [d+1, d]
        return self

    def predict(self, feat):
        n = feat.shape[0]
        Xa = torch.cat([feat, torch.ones(n, 1)], dim=1)
        return Xa @ self.W

    def cost_fraction(self, c, L):
        return 0.0  # reads only cheap activations; no honest work


class MLPForger:
    """F2: small MLP cheap_act -> honest_act, one hidden layer of width `hidden`.
    Cost knob is `hidden`; still tiny vs a transformer layer, so the dominant
    cost is reading the cheap activation at c (c/L). Trained with Adam."""

    def __init__(self, hidden=256, epochs=40, lr=1e-3, device="cpu"):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.name = f"F2-mlp{hidden}"

    def fit(self, feat, target):
        d = feat.shape[1]
        self.net = nn.Sequential(
            nn.Linear(d, self.hidden), nn.GELU(), nn.Linear(self.hidden, d)
        ).to(self.device)
        X = feat.to(self.device)
        Y = target.to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        bs = 4096
        for _ in range(self.epochs):
            perm = torch.randperm(X.shape[0], device=self.device)
            for i in range(0, X.shape[0], bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(self.net(X[b]), Y[b])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, feat):
        return self.net(feat.to(self.device)).cpu()

    def cost_fraction(self, c, L):
        return 0.0  # reads only cheap activations; no honest work


class ShortcutForger:
    """F3: the attacker truncates its honest work at depth j < c and predicts
    the checkpoint-c honest activation from the honest layer-j activation via a
    ridge map. This is the 'partial pass + learned tail' hybrid: it pays only
    j/L of the honest pass, then a linear jump to depth c.

    Note the feature here is the HONEST layer-j activation (the attacker did run
    honestly to j), passed in separately from the cheap features. Cost = j/L.
    """
    name = "F3-shortcut"

    def __init__(self, j, lam=1.0):
        self.j = j
        self.lam = lam
        self.name = f"F3-shortcut@{j}"

    def fit(self, feat_j, target_c):
        X = feat_j.double()
        Y = target_c.double()
        n, d = X.shape
        Xa = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)
        A = Xa.T @ Xa + self.lam * torch.eye(d + 1, dtype=X.dtype)
        self.W = torch.linalg.solve(A, Xa.T @ Y).float()
        return self

    def predict(self, feat_j):
        n = feat_j.shape[0]
        Xa = torch.cat([feat_j, torch.ones(n, 1)], dim=1)
        return Xa @ self.W

    def cost_fraction(self, c, L):
        return self.j / L


class ProjectionAwareForger:
    """The forger the reseeding defense (section 5i) specifically targets. It
    trains a ridge map from cheap activations DIRECTLY to the FINGERPRINT P.h at
    a given projection P -- overfitting the fixed k-dim shadow instead of the
    full D-dim activation. Cheap (k << D outputs) and, against a FIXED P, as
    accurate as the k-dim signal allows.

    KNOWN LIMITATION (measured on Qwen3-1.7B, 2026-08-20): with ~1-4k training
    tokens and D=2048 features the ridge is underdetermined (fewer samples than
    dims), so this forger CANNOT be trained to threshold -- its distances stay
    far above the pass line at all lambda, worst at mid-depth checkpoints. Its
    absolute pass rates here therefore reflect data starvation, NOT a proven
    hardness of forgery. Testing the reseeding claim properly needs >>D training
    tokens (tens of thousands) so the forger can actually approach threshold
    under fixed-P; only then is the fixed->reseeded gap meaningful.

    Interface differs from the other forgers: it predicts fingerprints, and it
    is bound to a specific P at fit time. The sweep uses it two ways:
      fixed-P    : fit and evaluate on the SAME P               (attacker wins if any forger can)
      reseeded-P : fit on one P, evaluate on a FRESH P          (defense: the fit is now useless)
    Cost: reads only cheap activations -> 0 honest work.
    """

    def __init__(self, lam=1e-2):
        self.lam = lam
        self.name = "F4-projaware"

    def fit(self, feat, target, proj):
        """feat: [N, D] cheap acts. target: [N, D] honest acts. proj: [D, k].
        Learns cheap_act -> (honest_act @ proj).

        Features are z-scored before the solve so ridge regularization acts
        uniformly across dimensions -- without this, deep-layer activations have
        such large dynamic range that X^T X is ill-conditioned and the solve
        explodes. lam scales with the sample count (Tikhonov on standardized
        features)."""
        X = feat.double()
        self.mu = X.mean(0, keepdim=True)
        sd = X.std(0, keepdim=True)
        self.sd = torch.clamp(sd, min=0.01 * sd.mean())
        Xz = (X - self.mu) / self.sd
        Yfp = (target.float() @ proj).double()
        n, d = Xz.shape
        Xa = torch.cat([Xz, torch.ones(n, 1, dtype=Xz.dtype)], dim=1)

        # Solve the ridge problem via SVD, which is stable no matter how
        # collinear the (whitened) features are -- the normal-equations solve
        # blows up when X^T X is rank-deficient, which happens at some layers.
        # Ridge in SVD space: W = V diag(s/(s^2+lam)) U^T Y.
        U, s, Vh = torch.linalg.svd(Xa, full_matrices=False)
        lam = self.lam * n
        d_inv = s / (s**2 + lam)
        self.W = (Vh.T * d_inv) @ (U.T @ Yfp)
        self.W = self.W.float()
        return self

    def predict_fp(self, feat):
        """Returns forged FINGERPRINTS [N, k] (not activations)."""
        Xz = (feat.double() - self.mu) / self.sd
        n = Xz.shape[0]
        Xa = torch.cat([Xz.float(), torch.ones(n, 1)], dim=1)
        return Xa @ self.W

    def cost_fraction(self, c, L):
        return 0.0


class _TinySuffix(nn.Module):
    """A few causal transformer blocks: honest layer-j state -> honest deep
    states at each target checkpoint. Shares a trunk, one linear read-out head
    per target checkpoint. Operates on [B, T, D] sequences (attention across
    positions, like the real suffix layers it distills)."""

    def __init__(self, d_model, n_blocks, n_head, targets, d_ff_mult=2):
        super().__init__()
        self.in_ln = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_head, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, d_ff_mult * d_model), nn.GELU(),
                    nn.Linear(d_ff_mult * d_model, d_model),
                ),
            }))
        self.targets = list(targets)
        self.heads = nn.ModuleDict({str(c): nn.Linear(d_model, d_model) for c in targets})

    def forward(self, x_BTD):
        T = x_BTD.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x_BTD.device, dtype=torch.bool), diagonal=1)
        h = self.in_ln(x_BTD)
        for blk in self.blocks:
            a = blk["ln1"](h)
            a, _ = blk["attn"](a, a, a, attn_mask=mask, need_weights=False)
            h = h + a
            h = h + blk["mlp"](blk["ln2"](h))
        return {c: self.heads[str(c)](h) for c in self.targets}


class DistilledSuffixForger:
    """F3-strong: the serious surrogate. Runs honestly to depth j, then a small
    distilled transformer predicts the honest activations at every deeper
    checkpoint from the honest layer-j state. Cost = j/L + (tiny suffix), which
    we account as j/L plus the suffix's block fraction so heavier suffixes cost
    more. Trained on raw activations (projection-free), so evaluating it under a
    fresh projection is a fair test of section 5i."""

    def __init__(self, j, targets, n_blocks=2, n_head=8, epochs=120, lr=1e-3,
                 device="cpu", n_layers=None):
        self.j = j
        self.targets = [c for c in targets if c > j]
        self.n_blocks = n_blocks
        self.n_head = n_head
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.n_layers = n_layers
        self.name = f"F3-distill{n_blocks}b@{j}"

    def fit(self, feat_j_seqs, target_seqs):
        """feat_j_seqs: list of [T, D] honest layer-j activations per sequence.
        target_seqs: {c: list of [T, D]} honest activations at target ckpts."""
        d_model = feat_j_seqs[0].shape[1]
        self.net = _TinySuffix(d_model, self.n_blocks, self.n_head, self.targets).to(self.device)
        X = torch.stack(feat_j_seqs).to(self.device)  # [S, T, D]
        Y = {c: torch.stack(target_seqs[c]).to(self.device) for c in self.targets}
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # normalize per-target so all checkpoints contribute comparably to loss
        scale = {c: Y[c].std().item() + 1e-6 for c in self.targets}
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.net(X)
            loss = sum(((pred[c] - Y[c]) / scale[c]).pow(2).mean() for c in self.targets)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict_seq(self, feat_j_seq):
        """feat_j_seq: [T, D] -> {c: [T, D] guess} for target checkpoints."""
        x = feat_j_seq.unsqueeze(0).to(self.device)
        out = self.net(x)
        return {c: out[c][0].cpu() for c in self.targets}

    def cost_fraction(self, c, L):
        # honest work to depth j, plus the distilled suffix's own transformer
        # blocks as a fraction of the network's layers (each block ~ one layer).
        return self.j / L + self.n_blocks / L
