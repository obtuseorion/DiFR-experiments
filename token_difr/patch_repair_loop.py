"""Patch repair_loop.py: the repo's apply_top_k_top_p wants per-row tensors."""

import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "repair_loop.py"
src = open(path, encoding="utf-8").read()

HELPER = '''
def _filter(logits_BV: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """apply_top_k_top_p expects k and p as per-row tensors of shape [B]."""
    B = logits_BV.shape[0]
    device = logits_BV.device
    k = torch.full((B,), top_k, dtype=torch.long, device=device)
    p = torch.full((B,), top_p, dtype=torch.float32, device=device)
    return apply_top_k_top_p(logits_BV, k, p)


'''

# Insert the helper just before gumbel_pick.
src = src.replace("def gumbel_pick(", HELPER.lstrip("\n") + "def gumbel_pick(", 1)

# Call site 1: single row inside gumbel_pick.
src = src.replace(
    "    x = apply_top_k_top_p(x[None, :], top_k, top_p).squeeze()",
    "    x = _filter(x[None, :], top_k, top_p).squeeze()",
)

# Call site 2: J rows inside audit.
src = src.replace(
    "    filtered = apply_top_k_top_p(x.clone(), top_k, top_p)",
    "    filtered = _filter(x.clone(), top_k, top_p)",
)

open(path, "w", encoding="utf-8").write(src)

n = len(re.findall(r"_filter\(", src))
print(f"patched {path}: {n} references to _filter (expect 3: def + 2 call sites)")
