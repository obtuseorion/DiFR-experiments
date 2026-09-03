#!/bin/bash
# Executed ON the GPU box: detection experiment on Llama-3.1-8B.
# Requires a HuggingFace token in ~/.huggingface_token (or HF_TOKEN env) with
# access to the gated meta-llama repo.
set -e
cd ~/difr-mid

export DIFR_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Sanity: token present?
if [ -z "$HF_TOKEN" ] && [ ! -f ~/.huggingface_token ] && [ ! -f ~/.cache/huggingface/token ]; then
  echo "NO-TOKEN: set ~/.huggingface_token or HF_TOKEN first" >&2
  exit 3
fi

~/difr-env/bin/python -m py_compile experiments/run_detection.py difr_mid/tap_vllm.py difr_mid/fingerprint.py
echo "COMPILE-OK"

# Prefetch weights (idempotent; uses the token via _load_hf_token in-process)
~/difr-env/bin/python - <<'PYEOF'
from experiments.run_detection import HF_TOKEN
from huggingface_hub import snapshot_download
print("token present:", bool(HF_TOKEN))
snapshot_download("meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)
print("DOWNLOAD-OK")
PYEOF

exec ~/difr-env/bin/python experiments/run_detection.py 2>&1
