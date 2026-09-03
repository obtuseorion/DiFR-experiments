"""Minimal matplotlib setup for difr-mid figures. Plain and utilitarian:
white background, black axes, default-ish line colours, light grid. Nothing
decorative."""

import matplotlib as mpl

# A small, plain colour cycle (matplotlib-ish defaults, muted). Assigned in
# order per plot; no design-system palette.
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def setup():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "legend.frameon": True,
        "legend.fontsize": 9,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.7",
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
