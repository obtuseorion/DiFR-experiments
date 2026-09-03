from vllm import LLM, SamplingParams

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
prompts = ["The capital of France is", "Explain gravity in one sentence."]
sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=50, max_tokens=32, seed=42)

for kv in ["auto", "fp8"]:
    llm = LLM(model=MODEL, dtype="bfloat16", kv_cache_dtype=kv,
              gpu_memory_utilization=0.85, max_model_len=2048)
    out = llm.generate(prompts, sp)
    print(kv, [o.outputs[0].token_ids[:8] for o in out])
    del llm