# difr-mid — project context for Claude Code

Research project: randomized-depth multi-checkpoint Activation-DiFR. Verifier
collects activation fingerprints at intermediate layers to cut verification
compute; randomized audit depth closes the coverage gap (README.md has the
results summary; docs/WRITEUP.md the full protocol and threat model;
experiments/run_demo.py validates it on a toy model).

The port to real models (Qwen3-1.7B, Llama-3.1-8B) via vLLM is done and
validated; remaining open items are in docs/WRITEUP.md §6. Layout:
difr_mid/ (library), experiments/ (runnable + remote_*.sh launchers),
results/, docs/, token_difr/ (Token-DiFR side track), tools/ (GPU
diagnostics, ssh config templates).

## GPU instance workflow

You run on the LOCAL machine; there is no local GPU. All GPU work happens on
a remote instance via plain ssh (alias `gpu`, multiplexed so calls are fast).
Jobs are short (minutes), so just run them blocking and read the output:

    ssh gpu "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
    rsync -az --exclude .git --exclude results --exclude __pycache__ ./ gpu:difr-mid/
    ssh gpu "cd difr-mid && python experiments/run_detection.py 2>&1"
    rsync -az gpu:difr-mid/results/ ./results/

Typical loop: edit locally -> rsync up -> ssh run (blocking, output comes
straight back) -> rsync results down -> analyze locally.

Notes:
- Always `2>&1` on experiment commands so tracebacks come back too.
- If a job unexpectedly needs to outlive the connection, the escape hatch is
  nohup (no tmux needed):
      ssh gpu "cd difr-mid && nohup python exp.py > run.log 2>&1 & echo started"
      ssh gpu "tail -n 40 difr-mid/run.log"
- The instance is EPHEMERAL: rsync results back promptly; never keep the
  only copy of anything remotely.
- If ssh fails, the box is down or its IP changed. Do not retry in a loop -
  ask the user to update HostName in ~/.ssh/config.
- One experiment at a time; check `ssh gpu nvidia-smi` if unsure.
