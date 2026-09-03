#!/bin/bash
# Executed ON the GPU box: compile-check then run the forgery sweep.
set -e
cd ~/difr-mid
~/difr-env/bin/python -m py_compile experiments/run_forgery_sweep.py \
    difr_mid/collect_activations.py difr_mid/forgers.py difr_mid/fingerprint.py
echo "COMPILE-OK"
exec ~/difr-env/bin/python experiments/run_forgery_sweep.py 2>&1
