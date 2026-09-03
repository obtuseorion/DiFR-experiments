"""Collect the activation tensors the surrogate-forgery sweep trains and
tests on, honoring the consistency trap (docs/forgery-attack-analysis.md section 2).

The verifier recomputes the HONEST model's activations on the SERVED (cheap)
tokens. So for each served sequence we need, at every checkpoint layer c:

    target[c]   P-free honest activation  a_c_hon(cheap_tokens)   [T, D]
    feat[c]     cheap-model activation     a_c_cheap(cheap_tokens) [T, D]
                (the attacker's forger inputs; they compute these anyway)

Both come from a single teacher-forced pass of each model over prompt+served,
sliced to the generated positions. We store raw D-dim activations (not
fingerprints) so the sweep can apply any projection / epoch / k / normalization
afterwards without re-running the GPU.

Token generation: the CHEAP model greedily decodes the served tokens (it is
the thing actually serving the user). The honest model never generates here —
it only ever recomputes on the cheap tokens, exactly as the verifier does.
"""

import gc

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def quantize_per_channel_(module: torch.nn.Module, n_levels: int = 16) -> None:
    """Snap every 2D weight to an `n_levels`-level per-output-channel grid, 1D
    params to a per-tensor grid. Stand-in for a cheaper served model M_cheap.
    n_levels=16 ~ int4 (harsh); n_levels=256 ~ int8/fp8-ish (gentle, activations
    stay close to honest so forgery has a real chance)."""
    lv = n_levels - 1
    with torch.no_grad():
        for p in module.parameters():
            w = p.float()
            if w.dim() >= 2:
                flat = w.reshape(w.shape[0], -1)
                lo = flat.min(dim=1, keepdim=True).values
                hi = flat.max(dim=1, keepdim=True).values
                scale = torch.clamp((hi - lo) / lv, min=1e-12)
                q = torch.round((flat - lo) / scale) * scale + lo
                p.copy_(q.reshape_as(w).to(p.dtype))
            else:
                lo, hi = w.min(), w.max()
                scale = torch.clamp((hi - lo) / lv, min=1e-12)
                p.copy_((torch.round((w - lo) / scale) * scale + lo).to(p.dtype))


# Back-compat alias.
def int4_per_channel_quantize_(module):
    quantize_per_channel_(module, n_levels=16)


@torch.inference_mode()
def _hidden_at(model, input_ids, checkpoints):
    """output_hidden_states[c] for each c, one teacher-forced pass. Returns
    dict c -> [T_full, D] on CPU fp32 (index 0 of hidden_states is embeddings,
    so hidden_states[c] is the post-layer-c residual stream — matches the
    verifier convention in run_detection.py)."""
    out = model(input_ids, output_hidden_states=True, use_cache=False)
    return {c: out.hidden_states[c][0].float().cpu() for c in checkpoints}


@torch.inference_mode()
def collect(
    model_name: str,
    prompt_token_ids: list[list[int]],
    checkpoints: list[int],
    max_tokens: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cheat_levels: int = 16,
) -> dict:
    """Returns:
      {
        "d_model": D, "n_layers": L, "checkpoints": [...],
        "sequences": [ {
            "prompt_len": int, "served_tokens": [...],
            "target": {c: Tensor[T, D]},   # honest acts on served tokens
            "feat":   {c: Tensor[T, D]},   # cheap acts on served tokens
        } ... ]
      }
    T = number of generated positions (== max_tokens).

    Each sequence also carries "target_fp32": the SAME honest model recomputed
    in fp32 on the served tokens. (target[c], target_fp32[c]) is a genuine
    honest-vs-honest pair whose difference is the benign cross-precision noise
    the verifier must tolerate -- the real H0 floor, replacing synthetic jitter.
    """
    cfg = AutoConfig.from_pretrained(model_name)
    honest = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()

    # Cheap served model: deep-copy then int4-quantize the WHOLE network. This
    # is a genuinely different served model (not a suffix cheat) so honest and
    # cheap activations diverge from shallow layers on — the hard case for a
    # forger, since it can't just copy shallow honest states.
    import copy

    cheap = copy.deepcopy(honest)
    quantize_per_channel_(cheap.model.layers, cheat_levels)
    quantize_per_channel_(cheap.model.norm, cheat_levels)

    # Second honest copy in fp32 for the genuine benign-noise floor. Qwen3-1.7B
    # in fp32 is ~7GB, comfortable on the 48GB card alongside the bf16 pair.
    honest_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device
    ).eval()

    sequences = []
    for ids in prompt_token_ids:
        inp = torch.tensor([ids], device=device)
        # Cheap model serves the tokens (greedy).
        gen = cheap.generate(
            inp, attention_mask=torch.ones_like(inp),
            max_new_tokens=max_tokens, min_new_tokens=max_tokens,
            do_sample=False, use_cache=True,
        )
        served = gen[0, len(ids):].tolist()
        assert len(served) == max_tokens
        full = torch.tensor([ids + served], device=device)

        honest_h = _hidden_at(honest, full, checkpoints)
        cheap_h = _hidden_at(cheap, full, checkpoints)
        honest32_h = _hidden_at(honest_fp32, full, checkpoints)

        plen = len(ids)
        # generated position j is predicted by residual at index plen + j - 1
        idx = list(range(plen - 1, plen - 1 + max_tokens))
        sequences.append({
            "prompt_len": plen,
            "served_tokens": served,
            "target": {c: honest_h[c][idx].clone() for c in checkpoints},
            "target_fp32": {c: honest32_h[c][idx].clone() for c in checkpoints},
            "feat": {c: cheap_h[c][idx].clone() for c in checkpoints},
        })

    d_model = cfg.hidden_size
    del honest, cheap, honest_fp32
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "d_model": d_model,
        "n_layers": cfg.num_hidden_layers,
        "checkpoints": checkpoints,
        "sequences": sequences,
    }
