"""vLLM prover with multi-checkpoint activation taps.

Ports the upstream single-hook trick (difr_org/vllm_verification.py hooks the
logits processor) to intermediate decoder layers. Needs the V0 engine so the
model lives in-process where hooks can reach it, and enforce_eager so hooks
actually fire (cuda graphs would capture around them).

Alignment strategy: one request at a time, ignore_eos, fixed max_tokens.
Then the forward passes are exactly [prefill(prompt_len rows)] +
(max_tokens-1) * [decode(1 row)], and the hidden state that predicted
generated token j is the last prefill row for j=0, else the single row of
decode step j. Everything is asserted; if vLLM schedules differently
(chunked prefill etc.) we fail loudly rather than mis-align.

vLLM decoder layers (Llama/Qwen family) return (hidden_states, residual)
with the residual stream carried separately; the canonical post-layer hidden
state — the thing HF's output_hidden_states[c] reports — is their sum.
"""

import os

os.environ.setdefault("VLLM_USE_V1", "0")

import gc

import torch
from vllm import LLM, SamplingParams

from difr_mid.fingerprint import fingerprint, make_projections


def _canonical_hidden(output) -> torch.Tensor:
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], torch.Tensor):
        return output[0] + output[1]
    if isinstance(output, torch.Tensor):
        return output
    raise ValueError(f"unexpected layer output type: {type(output)}")


def prover_generate_vllm(
    model_name: str,
    prompt_token_ids: list[list[int]],
    checkpoints: list[int],
    fp_k: int,
    max_tokens: int,
    vllm_args: dict | None = None,
    dtype: str = "bfloat16",
    max_model_len: int = 4096,
) -> list[dict]:
    """Generate greedily with vLLM, fingerprinting every checkpoint at every
    generated position from the prover's own forward passes.

    Returns one dict per request:
      {"prompt_token_ids": [...], "token_ids": [...],
       "fps": [ {checkpoint: [fp_k floats], ...} per generated token ]}
    """
    vllm_args = dict(vllm_args or {})
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=0.7,
        max_model_len=max_model_len,
        **vllm_args,
    )

    hf_config = llm.llm_engine.model_config.hf_config
    d_model = hf_config.hidden_size
    n_layers = hf_config.num_hidden_layers
    assert max(checkpoints) <= n_layers, (checkpoints, n_layers)

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layers = model.model.layers

    projs = make_projections(d_model, checkpoints, fp_k, device="cuda")

    taps: dict[int, list[torch.Tensor]] = {c: [] for c in checkpoints}

    def make_hook(c):
        def hook(module, inputs, output):
            h = _canonical_hidden(output)
            # flatten [num_tokens, D] (V0 flattens the batch) and keep fp32
            taps[c].append(h.reshape(-1, d_model).float().detach())
        return hook

    def make_final_norm_hook(c):
        def hook(module, inputs, output):
            # final RMSNorm returns (normed, residual) when called with a
            # residual; the normed tensor is what HF reports as its last
            # hidden_states entry, so checkpoint L must match that.
            h = output[0] if isinstance(output, tuple) else output
            taps[c].append(h.reshape(-1, d_model).float().detach())
        return hook

    handles = []
    for c in checkpoints:
        if c == n_layers:
            handles.append(model.model.norm.register_forward_hook(make_final_norm_hook(c)))
        else:
            handles.append(layers[c - 1].register_forward_hook(make_hook(c)))

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)

    results = []
    try:
        for ids in prompt_token_ids:
            for c in checkpoints:
                taps[c].clear()

            out = llm.generate(prompt_token_ids=[ids], sampling_params=sp, use_tqdm=False)
            gen_ids = list(out[0].outputs[0].token_ids)
            assert len(gen_ids) == max_tokens, (len(gen_ids), max_tokens)

            fps = []
            for c in checkpoints:
                passes = taps[c]
                assert len(passes) == max_tokens, (
                    f"ckpt {c}: expected {max_tokens} forward passes "
                    f"(1 prefill + {max_tokens - 1} decodes), got {len(passes)} "
                    f"with row counts {[p.shape[0] for p in passes[:5]]}..."
                )
                assert passes[0].shape[0] == len(ids), (passes[0].shape, len(ids))
                assert all(p.shape[0] == 1 for p in passes[1:])

            for j in range(max_tokens):
                tok_fp = {}
                for c in checkpoints:
                    h = taps[c][0][-1] if j == 0 else taps[c][j][0]
                    tok_fp[c] = fingerprint(h, projs[c]).cpu().tolist()
                fps.append(tok_fp)

            results.append(
                {"prompt_token_ids": list(ids), "token_ids": gen_ids, "fps": fps}
            )
    finally:
        for h in handles:
            h.remove()
        del model, layers, llm
        gc.collect()
        torch.cuda.empty_cache()

    return results
