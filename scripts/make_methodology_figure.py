#!/usr/bin/env python3
"""Regenerate Figure 1 (methodology flowchart) alone; identical to the figure block of
recompute_manuscript_results.py (round-5 version)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "derived"
# 1. Corrected methodology flowchart.
fig, ax = plt.subplots(figsize=(13.6, 6.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", linewidth=1.1)
ax.text(0.5, 0.93, "Shared construction: canonical capture -> cleaning -> label taxonomy -> stratified split", ha="center", va="center", fontsize=12, bbox=box)
ax.annotate("", xy=(0.27, 0.78), xytext=(0.46, 0.89), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.73, 0.78), xytext=(0.54, 0.89), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(0.25, 0.74, "Axis 1: tabular detector measurement", ha="center", va="center", fontsize=12, fontweight="bold", bbox=box)
ax.text(0.75, 0.74, "Axis 2: few-shot language-model control", ha="center", va="center", fontsize=12, fontweight="bold", bbox=box)
ax.text(0.25, 0.57, "Strict core panel (8 configurations)\nShared capped rows within each cell and seed\nSix classical arms + TabPFN 2.5 + TabICL", ha="center", va="center", fontsize=10.5, bbox=box)
ax.text(0.75, 0.57, "Disjoint development, ICL and evaluation rows\n12 development-selected features\n5 shots per class; RF anchor on identical inputs", ha="center", va="center", fontsize=10.5, bbox=box)
ax.annotate("", xy=(0.25, 0.46), xytext=(0.25, 0.52), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.75, 0.46), xytext=(0.75, 0.52), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(0.25, 0.36, "Primary outcome: rare-class recall\nAlso accuracy, macro-F1, balanced accuracy, MCC, macro-AP, alerts/TP\n10 paired seeds; bootstrap CI; sign flip; effect sizes with MDE", ha="center", va="center", fontsize=10.5, bbox=box)
ax.text(0.75, 0.36, "Closed-label accuracy and macro-F1\nParse failures retained as errors; parse rate separate\n10 paired seeds; Bonferroni across six models", ha="center", va="center", fontsize=10.5, bbox=box)
ax.annotate("", xy=(0.25, 0.20), xytext=(0.25, 0.30), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(0.75, 0.20), xytext=(0.75, 0.30), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.text(0.25, 0.12, "Separate extensions\nTabSTAR (different CIC partition); 2x2 balance-size, cap, names, ManyClass\nRe-execution on a second platform under 5 held-out designs\n(3 row-level, chronological, host-pair) + leakage audit\nCost crossovers and operating points", ha="center", va="center", fontsize=9.8, bbox=box)
ax.text(0.75, 0.12, "Protocol checks\nBlind configuration negotiation\nPooled vs development-only vs ICL-only feature selection\nEffect sizes with MDE; cost and throughput extrapolation", ha="center", va="center", fontsize=9.8, bbox=box)
fig.tight_layout()
fig.savefig(OUT / "methodology_flowchart.pdf", bbox_inches="tight")
fig.savefig(OUT / "methodology_flowchart.png", dpi=220, bbox_inches="tight")
plt.close(fig)

