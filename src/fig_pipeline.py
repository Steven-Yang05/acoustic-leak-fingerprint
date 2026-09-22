"""
fig_pipeline.py

Full-width pipeline schematic for the ICASSP paper, drawn in a restrained
academic style: square-corner boxes, thin uniform strokes, serif type,
orthogonal arrows, a single muted accent color. No gradients, no icons.

Output: figures/fig_pipeline.png (relative to the repository root)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "fig_pipeline.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

INK = "#111111"        # box borders / arrows
ZONE_FC = "#f4f4f4"    # zone panel fill
ZONE_EC = "#9a9a9a"    # zone panel edge
ZONE_TC = "#444444"    # zone label text
ACCENT = "#1f4e79"     # frozen-model accent
ACCENT_FC = "#e9eff5"  # accent fill

fig, ax = plt.subplots(figsize=(7.16, 2.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis("off")


def zone(x0, y0, x1, y1, label):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=ZONE_FC,
                           ec=ZONE_EC, lw=0.6, linestyle=(0, (3, 2)),
                           zorder=0))
    ax.text(x0 + 1.2, y1 - 1.6, label, fontsize=5.8, color=ZONE_TC,
            ha="left", va="top", zorder=1)


def box(x0, y0, x1, y1, text, fs=6.2, accent=False):
    fc = ACCENT_FC if accent else "white"
    ec = ACCENT if accent else INK
    lw = 1.1 if accent else 0.8
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec,
                           lw=lw, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, fontsize=fs,
            ha="center", va="center", zorder=3, linespacing=1.35,
            color=INK)


def arrow(p0, p1, style="arc3,rad=0", lw=0.9, color=INK):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 connectionstyle=style, zorder=4))


def line(p0, p1, lw=0.9, color=INK):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=lw, color=color, zorder=4,
            solid_capstyle="butt")


# ------------------------------------------------------------- zone panels
zone(1.0, 2.0, 22.5, 40.0, "DATA & GROUPING")
zone(24.0, 2.0, 75.5, 40.0, "TRAINING-SIDE DEVELOPMENT  (800 CLIPS ONLY)")
zone(77.0, 2.0, 99.0, 40.0, "FROZEN EVALUATION")

# ---------------------------------------------------------------- zone 1
box(2.5, 29.8, 21.0, 36.8,
    "public dataset [12]\n1,000 clips · 8 kHz · 1 s\n"
    "leak 500 · no-leak 386\nnoise 114", fs=5.8)
box(2.5, 22.0, 21.0, 28.5,
    "group id from filenames\n276 / 71 / 114 groups", fs=6.0)
box(2.5, 14.0, 21.0, 20.0,
    "stratified group split\n(seed 42, fixed)", fs=6.0)
box(2.5, 5.0, 21.0, 12.0,
    "train  800 clips · 370 groups\nverif.  200 clips · 91 groups", fs=6.0)

arrow((11.75, 29.8), (11.75, 28.5))
arrow((11.75, 22.0), (11.75, 20.0))
arrow((11.75, 14.0), (11.75, 12.0))

# ---------------------------------------------------------------- zone 2
# branch A (handcrafted)
box(25.5, 29.5, 38.5, 37.0,
    r"212-d handcrafted" + "\n" + r"acoustic statistics" + "\n"
    r"(log-Mel, MFCC, $\Delta$)", fs=6.0)
box(40.0, 29.5, 52.0, 37.0, "RBF-SVM\n(C = 10, RBF)")
box(53.5, 29.5, 74.0, 37.0,
    "nested group-aware\nPlatt calibration\n$p=\\sigma(af(x)+b)$")
arrow((38.5, 33.25), (40.0, 33.25))
arrow((52.0, 33.25), (53.5, 33.25))

# branch B (waveform)
box(25.5, 18.5, 38.5, 26.0,
    "waveform 1 × 8000\n(peak-normalized)")
box(40.0, 18.5, 68.0, 26.0,
    "1D CNN · 4 conv blocks\n105,569 parameters")
arrow((38.5, 22.25), (40.0, 22.25))

# OOF + stability + freeze bar
box(25.5, 9.5, 74.0, 15.5,
    "5-fold group-aware OOF × 5 partitions  →  "
    "stability selection (rank-1 in 3/5)\n"
    "frozen fusion:  $p = 0.55\\,p_{SVM} + 0.45\\,p_{CNN}$,  "
    "$\\tau = 0.520$", fs=6.0)

# branches -> bar
line((70.0, 29.5), (70.0, 16.2))
arrow((70.0, 16.2), (70.0, 15.5))
arrow((57.0, 18.5), (57.0, 15.5))

# train arrow: zone1 -> zone2
arrow((21.0, 10.0), (25.5, 12.5), style="arc3,rad=-0.15")

# ---------------------------------------------------------------- zone 3
box(78.5, 24.0, 97.5, 36.0,
    "single frozen evaluation\n\nAcc 94.5%  ·  Rec 96.0%\n"
    "F1 0.9458  ·  AUC 0.9851", accent=True)
box(81.5, 7.0, 97.5, 19.5,
    "post-freeze audits\n\nleave-one-condition-out\n"
    "additive noise +20…−5 dB\nwav2vec 2.0 baseline", fs=6.0)

# frozen model -> evaluation
arrow((74.0, 13.0), (78.5, 28.0), style="arc3,rad=-0.25",
      color=ACCENT, lw=1.2)
# evaluation -> audits
arrow((89.5, 24.0), (89.5, 19.5))

# verification passthrough lane (untouched during development)
line((11.75, 5.0), (11.75, 3.3))
line((11.75, 3.3), (80.0, 3.3))
arrow((80.0, 3.3), (80.0, 24.0))
ax.text(44.0, 4.5, "verification clips kept untouched during development",
        fontsize=5.4, color=ZONE_TC, ha="center", va="bottom",
        style="italic", zorder=5)

fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"saved -> {OUT}")
