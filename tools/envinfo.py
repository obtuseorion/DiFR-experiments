"""
Environment provenance capture.

Two uses:

1. Standalone:  python envinfo.py            -> writes /network/env_<timestamp>.json
2. Imported:    from envinfo import env_info  -> dict to embed in every artifact you dump

Because your experiment measures small numerical differences between inference
configurations, "which build produced this" is data, not bookkeeping.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _sh(cmd):
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def env_info():
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # --- GPU / driver -------------------------------------------------
    gpus = _sh(
        "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap "
        "--format=csv,noheader"
    )
    if gpus:
        info["gpus"] = [
            dict(zip(["name", "driver_version", "memory_total", "compute_cap"],
                     [f.strip() for f in line.split(",")]))
            for line in gpus.splitlines()
        ]

    # --- torch / vllm -------------------------------------------------
    try:
        import torch
        info["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))
            if torch.cuda.is_available() else None,
        }
    except Exception as e:
        info["torch"] = {"error": str(e)}

    for mod in ("vllm", "transformers", "numpy", "scipy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None

    # --- code state ---------------------------------------------------
    info["git"] = {
        "commit": _sh("git rev-parse HEAD"),
        "branch": _sh("git rev-parse --abbrev-ref HEAD"),
        "dirty": bool(_sh("git status --porcelain")),
    }

    # --- env vars that change numerics --------------------------------
    info["env_vars"] = {
        k: os.environ.get(k)
        for k in (
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "VLLM_ATTENTION_BACKEND",
            "VLLM_USE_V1",
            "PYTHONHASHSEED",
            "CUBLAS_WORKSPACE_CONFIG",
        )
        if os.environ.get(k) is not None
    }

    return info


def stamp(artifact: dict, **run_config) -> dict:
    """Wrap a result dict with environment provenance and run config."""
    return {"env": env_info(), "run_config": run_config, "data": artifact}


if __name__ == "__main__":
    info = env_info()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = f"/network/env_{ts}.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print(f"\nwritten to {out}")

    # full dependency freeze alongside it
    freeze = _sh(f"{sys.executable} -m pip freeze")
    if freeze:
        lock = f"/network/requirements_{ts}.lock.txt"
        with open(lock, "w") as f:
            f.write(freeze + "\n")
        print(f"written to {lock}")