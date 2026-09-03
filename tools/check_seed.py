"""
Seed reproducibility check — V1 engine only.

Validates that vLLM's seeded sampling is deterministic on this build, which is
the assumption the whole DiFR threat model rests on.

Run:  python check_seed.py
"""

import os

os.environ["VLLM_USE_V1"] = "1"
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


def run(seed):
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=True,
        enable_prefix_caching=False,
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


def report(label, want, ref, alt):
    print("\n" + "=" * 70)
    print(f"{label}   (want: {want})")
    print("=" * 70)
    n_identical = 0
    for i, (r, a) in enumerate(zip(ref, alt)):
        d = first_divergence(r, a)
        if d is None:
            n_identical += 1
            print(f"[prompt {i}] identical ({len(r)} tokens)")
            continue
        print(f"[prompt {i}] diverges at token {d}/{len(r)}")
        print(f"    shared tail : ...{tok.decode(r[max(0, d - 10):d])!r}")
        print(f"    A -> {r[d]:>6} {tok.decode([r[d]])!r}   then {tok.decode(r[d:d + 15])!r}")
        print(f"    B -> {a[d]:>6} {tok.decode([a[d]])!r}   then {tok.decode(a[d:d + 15])!r}")
    print(f"  => {n_identical}/{len(ref)} identical")
    return n_identical


print("run 1: seed 42 ...")
a = run(SEED)
print("run 2: seed 42 ...")
b = run(SEED)
print("run 3: seed 43 ...")
c = run(43)

n = len(PROMPTS)
same = report("same seed (42 vs 42)", "8/8 identical", a, b)
diff = report("different seed (42 vs 43)", "0/8 identical", a, c)

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  seed is reproducible   : {'PASS' if same == n else 'FAIL'}")
print(f"  seed actually matters  : {'PASS' if diff == 0 else 'FAIL'}")
