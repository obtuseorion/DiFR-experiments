"""Generate plain matplotlib figures for the difr-mid experiments.

  fig_detection_<model>.png   detection: per-sequence distance by checkpoint
                              (log) + audit-policy tradeoff
  fig_demo.png                toy-model coverage-gap validation
  fig_forgery.png             forgery: train/test signal ceiling + forger ladder

Run locally:  python experiments/make_figures.py   (matplotlib + numpy, no GPU)
Outputs to results/figures/.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style as S

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

S.setup()


def threshold_line(ax, x, thr):
    """Plain dashed black threshold line with a small label at the right end."""
    ax.plot(x, thr, "k--", lw=1.0, label="accept threshold")


# ---------------------------------------------------------------------------
# Detection: distance-by-checkpoint (log) + policy tradeoff
# ---------------------------------------------------------------------------
def fig_detection(path, model_label, out_name):
    d = json.load(open(path))
    cks = d["config"]["checkpoints"]
    fam = {"honest_vllm": "vllm", "fp8kv_vllm": "vllm",
           "honest_hf": "hf", "suffix_hf": "hf"}
    pretty = {"honest_vllm": "honest (vLLM)", "fp8kv_vllm": "fp8 KV cache",
              "honest_hf": "honest (HF)", "suffix_hf": "int4 suffix cheat"}
    marker = {"honest_vllm": "o", "honest_hf": "s",
              "fp8kv_vllm": "^", "suffix_hf": "x"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(cks))
    thr = [d["thresholds"]["vllm"][str(c)] for c in cks]

    names = ["honest_vllm", "honest_hf", "fp8kv_vllm", "suffix_hf"]
    for ci, name in enumerate(names):
        n_seq = len(d["distances"][name])
        for si in range(n_seq):
            seq = d["distances"][name][si]
            ys = [np.mean([t[str(c)] for t in seq]) for c in cks]
            jitter = (si / max(1, n_seq - 1) - 0.5) * 0.22
            ax1.plot(x + jitter, ys, marker[name], color=S.COLORS[ci],
                     ms=4, mew=1.0, alpha=0.8, ls="",
                     label=pretty[name] if si == 0 else None)
    threshold_line(ax1, x, thr)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in cks])
    ax1.set_xlabel("checkpoint layer (audit depth)")
    ax1.set_ylabel("mean fingerprint distance per sequence")
    ax1.set_title(f"Detection: {model_label}")
    ax1.legend(loc="upper left")

    # audit-policy tradeoff
    pol = d["policies"]
    for name, p in pol.items():
        kind = "fixed" if name.startswith("fixed") else "random"
        c = "0.4" if kind == "fixed" else S.COLORS[0]
        m = "s" if kind == "fixed" else "o"
        comp = p["e_compute"]
        t99 = p["seqs_to_99_suffix"]
        never = isinstance(t99, str) or (isinstance(t99, float) and math.isinf(t99))
        y = 200 if never else t99
        ax2.plot([comp], [y], m, color=c, ms=6)
        ax2.annotate(name, (comp, y), textcoords="offset points",
                     xytext=(7, 0), va="center", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylim(0.8, 320)
    ax2.set_yticks([1, 4, 16, 64, 200])
    ax2.set_yticklabels(["1", "4", "16", "64", "never"])
    ax2.set_xlim(0.4, 1.15)
    ax2.set_xlabel("expected verifier compute (fraction of full pass)")
    ax2.set_ylabel("sequences to 99% detection (suffix cheat)")
    ax2.set_title("Audit policy tradeoff")

    fig.tight_layout()
    fig.savefig(FIG / out_name)
    plt.close(fig)
    print(f"wrote {FIG / out_name}")


# ---------------------------------------------------------------------------
# Toy demo
# ---------------------------------------------------------------------------
def fig_demo():
    d = json.load(open(RES / "demo_results.json"))
    cks = d["config"]["checkpoints"]
    prov = ["honest", "quant", "suffix"]
    pretty = {"honest": "honest", "quant": "quantized (all layers)",
              "suffix": "suffix cheat"}
    marker = {"honest": "o", "quant": "^", "suffix": "x"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(cks))
    thr = [d["thresholds"][str(c)] for c in cks]
    floor = 1e-7
    for ci, name in enumerate(prov):
        ys = [max(d["mean_distance"][name][str(c)], floor) for c in cks]
        ax1.plot(x, ys, marker[name] + "-", color=S.COLORS[ci], label=pretty[name])
    threshold_line(ax1, x, thr)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in cks])
    ax1.set_xlabel("checkpoint layer (audit depth)")
    ax1.set_ylabel("mean fingerprint distance")
    ax1.set_title("Toy model: distance by checkpoint")
    ax1.legend(loc="center left")

    pol = d["policies"]
    for name, p in pol.items():
        kind = "fixed" if name.startswith("fixed") else "random"
        c = "0.4" if kind == "fixed" else S.COLORS[0]
        m = "s" if kind == "fixed" else "o"
        comp = p["e_compute"]
        t99 = p["tokens_to_99_suffix"]
        never = isinstance(t99, str) or (isinstance(t99, float) and math.isinf(t99))
        y = 200 if never else t99
        ax2.plot([comp], [y], m, color=c, ms=6)
        ax2.annotate(name, (comp, y), textcoords="offset points",
                     xytext=(7, 0), va="center", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylim(0.8, 320)
    ax2.set_yticks([1, 4, 16, 64, 200])
    ax2.set_yticklabels(["1", "4", "16", "64", "never"])
    ax2.set_xlim(0.4, 1.1)
    ax2.set_xlabel("expected verifier compute (fraction of full pass)")
    ax2.set_ylabel("tokens to 99% detection (suffix cheat)")
    ax2.set_title("Audit policy tradeoff")

    fig.tight_layout()
    fig.savefig(FIG / "fig_demo.png")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_demo.png'}")


# ---------------------------------------------------------------------------
# Forgery: signal ceiling + ladder
# ---------------------------------------------------------------------------
def fig_forgery():
    sig = json.load(open(RES / "forgery_signal.json"))
    cks = sig["checkpoints"]
    x = np.arange(len(cks))
    train = [sig["signal"][str(c)]["train"] for c in cks]
    test = [sig["signal"][str(c)]["test"] for c in cks]
    thr = [sig["thr"][str(c)] for c in cks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(x, train, "o-", color=S.COLORS[0], label="training error")
    ax1.plot(x, test, "s--", color=S.COLORS[1], label="held-out error")
    threshold_line(ax1, x, thr)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in cks])
    ax1.set_xlabel("checkpoint layer (audit depth)")
    ax1.set_ylabel("projected fingerprint distance")
    ax1.set_title("Projection-aware forger: train vs held-out")
    ax1.legend(loc="upper left")

    run = list(json.load(open(RES / "forgery_sweep_results.json"))["runs"].values())[0]
    diag = run["diagnostic"]
    ladder = [
        ("F0-copy", "commit cheap acts"),
        ("F1-ridge", "ridge (cheap->honest)"),
        ("F2-mlp512", "MLP width 512"),
        ("F3-distill3b@7", "distilled 3-block suffix"),
        ("F4-projaware", "projection-aware"),
    ]
    thr2 = [diag[str(c)]["thr"] for c in cks]
    for ci, (key, label) in enumerate(ladder):
        ys = [max(diag[str(ck)][key], 1e-2) for ck in cks]
        ax2.plot(x, ys, "o-", color=S.COLORS[ci], ms=3.5, label=label)
    threshold_line(ax2, x, thr2)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in cks])
    ax2.set_xlabel("checkpoint layer (audit depth)")
    ax2.set_ylabel("held-out fingerprint distance")
    ax2.set_title("Forger ladder")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "fig_forgery.png")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_forgery.png'}")


if __name__ == "__main__":
    fig_detection(RES / "detection_results.json", "Qwen3-1.7B", "fig_detection_qwen3-1.7b.png")
    fig_detection(RES / "detection_results_Llama-3-1-8B-Instruct.json",
                  "Llama-3.1-8B", "fig_detection_llama3.1-8b.png")
    fig_demo()
    fig_forgery()
    print("\nall figures written to", FIG)
