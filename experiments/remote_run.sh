#!/bin/bash
# Executed ON the GPU box. Downloads weights (idempotent), then runs the
# detection experiment. Launched via nohup from the local machine.
set -e
cd ~/difr-mid

~/difr-env/bin/python -m py_compile experiments/run_detection.py difr_mid/tap_vllm.py difr_mid/fingerprint.py
echo "COMPILE-OK"

~/difr-env/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-1.7B")
print("DOWNLOAD-OK")
EOF

exec ~/difr-env/bin/python experiments/run_detection.py 2>&1
