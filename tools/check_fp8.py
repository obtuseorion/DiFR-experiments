"""
fp8 KV-cache perturbation check — V0 engine only.

vLLM 0.10.1's V1 engine rejects kv_cache_dtype, so this runs under V0 and does
NOT touch prompt_logprobs. It generates under bf16 and fp8 KV cache, compares
the token streams, and writes both out in the repo's external-responses format
so they can be scored later by the unmodified verifier.

Run:  python check_fp8.py
"""

import os

os.environ["VLLM_USE_V1"] = "0"          # must be set before importing vllm
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42
MAX_TOKENS = 128
OUT_DIR = Path("/network/artifacts")

PROMPTS = [
    "Write a short paragraph about why cities grow near rivers.",
    "Explain the difference between precision and recall.",
    "Describe an unusual hobby someone might enjoy.",
    "What makes a good scientific hypothesis?",
    "Tell me about an animal that lives in the deep sea.",
    "Give me a plausible reason someone would learn Latin today.",
    "How would you explain compound interest to a teenager?",
    "Invent a name for a coffee shop and justify it.",
]

tok = AutoTokenizer.from_pretrained(MODEL)

# tokenize through the chat template so this matches how the repo builds prompts
PROMPT_IDS = [
    tok.encode(
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        ),
        add_special_tokens=False,
    )
    for p in PROMPTS
]


def run(label, **vllm_args):
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=True,
        enable_prefix_caching=False,
        **vllm_args,
    )
    sp = SamplingParams(
        temperature=1.0, top_p=0.95, top_k=50,
        max_tokens=MAX_TOKENS, seed=SEED,
    )
    prompts: list[TokensPrompt] = [{"prompt_token_ids": ids} for ids in PROMPT_IDS]
    outs = llm.generate(prompts, sp)
    result = [list(o.outputs[0].token_ids) for o in outs]
    del llm

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"gen_{label}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "samples": [
                    {"prompt_token_ids": p, "output_token_ids": o}
                    for p, o in zip(PROMPT_IDS, result)
                ]
            },
            f,
        )
    print(f"  wrote {path}")
    return result


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


print("generating: bf16 KV ...")
ref = run("bf16")

print("generating: fp8 KV ...")
fp8 = run("fp8kv", kv_cache_dtype="fp8", calculate_kv_scales=True)

print("\n" + "=" * 70)
print("bf16 KV vs fp8 KV   (want: most prompts DIVERGE)")
print("=" * 70)
n_identical = 0
for i, (r, a) in enumerate(zip(ref, fp8)):
    d = first_divergence(r, a)
    if d is None:
        n_identical += 1
        print(f"[prompt {i}] identical ({len(r)} tokens)")
        continue
    print(f"[prompt {i}] diverges at token {d}/{len(r)}")
    print(f"    shared tail : ...{tok.decode(r[max(0, d - 10):d])!r}")
    print(f"    bf16 -> {r[d]:>6} {tok.decode([r[d]])!r}   then {tok.decode(r[d:d + 15])!r}")
    print(f"    fp8  -> {a[d]:>6} {tok.decode([a[d]])!r}   then {tok.decode(a[d:d + 15])!r}")

n = len(PROMPTS)
print(f"\n  => {n_identical}/{n} identical")
print(f"  fp8 perturbs output : {'PASS' if n_identical < n else 'INCONCLUSIVE'}")
print(f"\nToken files in {OUT_DIR} are in the repo's external-responses format —")
print("score them with load_external_tokens() + verify_outputs() under V1.")
