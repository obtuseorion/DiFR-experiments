"""Shared fingerprint machinery for multi-checkpoint Activation-DiFR.

Both prover and verifier must build identical projection matrices from the
public seed. Matches upstream difr conventions (create_down_proj_matrices in
difr_org/vllm_verification.py): N(0,1) entries scaled by 1/sqrt(k), fp32
projection math. One matrix per checkpoint layer, seeded by (base_seed +
checkpoint) so checkpoints are independent.
"""

import torch

PUBLIC_SEED = 42


def make_projections(
    d_model: int,
    checkpoints: list[int],
    k: int,
    seed: int = PUBLIC_SEED,
    device: str | torch.device = "cpu",
    epoch: int = 0,
) -> dict[int, torch.Tensor]:
    """dict: checkpoint layer -> (d_model, k) JL projection matrix, fp32.

    `epoch` reseeds the whole family: fixed-P audits pass epoch=0 always;
    reseeded-P audits pass a fresh epoch per audit so the attacker cannot
    target a fixed k-dim shadow of the activation (docs/forgery-attack-analysis.md
    section 5i). The epoch mixes into the seed with a large stride so distinct
    (checkpoint, epoch) pairs never collide.
    """
    mats = {}
    for c in checkpoints:
        # Always sample on CPU: CUDA and CPU generators disagree for the same
        # seed, and prover/verifier may live on different devices.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + c + epoch * 100_003)
        m = torch.randn(d_model, k, generator=gen, dtype=torch.float32)
        mats[c] = (m / torch.sqrt(torch.tensor(float(k)))).to(device)
    return mats


def fingerprint(h_D: torch.Tensor, proj_Dk: torch.Tensor) -> torch.Tensor:
    """Project one hidden state to a k-dim fingerprint, fp32."""
    return h_D.detach().to(torch.float32) @ proj_Dk


def fp_distance(a_k: torch.Tensor, b_k: torch.Tensor) -> float:
    """L2 distance between fingerprints, normalized by sqrt(k) so the scale
    is comparable across fingerprint widths."""
    d = torch.linalg.vector_norm(a_k.float() - b_k.float()) / (a_k.numel() ** 0.5)
    return float(d)
