"""
Diagnostic: does fp8 KV cache actually perturb generation, and is the seed reproducible?

Run:  python compare_kv.py
"""

import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42
MAX_TOKENS = 128

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


def run(kv_cache_dtype, seed=SEED):
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=True,          # avoid cudagraph nondeterminism while diagnosing
        enable_prefix_caching=False, # prefix cache can mask config differences
    )
    sp = SamplingParams(
        temperature=1.0, top_p=0.95, top_k=50,
        max_tokens=MAX_TOKENS, seed=seed,
    )
    outs = llm.generate(PROMPTS, sp)
    result = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    return result


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def report(label, ref, alt):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    identical = 0
    for i, (r, a) in enumerate(zip(ref, alt)):
        d = first_divergence(r, a)
        if d is None:
            identical += 1
            print(f"\n[prompt {i}] IDENTICAL ({len(r)} tokens)")
            continue
        print(f"\n[prompt {i}] diverges at token {d} of {len(r)}")
        print(f"  shared : ...{tok.decode(r[max(0, d - 12):d])!r}")
        print(f"  ref  -> {r[d]:>6}  {tok.decode([r[d]])!r}")
        print(f"  alt  -> {a[d]:>6}  {tok.decode([a[d]])!r}")
        print(f"  ref  cont: {tok.decode(r[d:d + 20])!r}")
        print(f"  alt  cont: {tok.decode(a[d:d + 20])!r}")
    print(f"\n  -> {identical}/{len(ref)} prompts identical")
    return identical


print("running bf16 KV (seed 42) ...")
ref = run("auto")

print("running bf16 KV again (seed 42) ...")
ref2 = run("auto")

print("running bf16 KV (seed 43) ...")
ref3 = run("auto", seed=43)

print("running fp8 KV (seed 42) ...")
fp8 = run("fp8")

same_seed = report("SEED CHECK: bf16 seed42 vs bf16 seed42  (want: ALL IDENTICAL)", ref, ref2)
diff_seed = report("SEED CHECK: bf16 seed42 vs bf16 seed43  (want: ALL DIVERGE)", ref, ref3)
kv_diff = report("KV CHECK:   bf16 seed42 vs fp8 seed42    (want: SOME DIVERGE)", ref, fp8)

n = len(PROMPTS)
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  seed reproducible      : {'PASS' if same_seed == n else 'FAIL'}  ({same_seed}/{n} identical)")
print(f"  different seed differs : {'PASS' if diff_seed == 0 else 'FAIL'}  ({diff_seed}/{n} identical)")
print(f"  fp8 perturbs output    : {'PASS' if kv_diff < n else 'INCONCLUSIVE'}  ({kv_diff}/{n} identical)")