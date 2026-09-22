"""
fig_robustness.py

Single-column robustness summary for the ICASSP paper:
  (a) frozen-model F1 under additive white noise (+20 ... -5 dB)
  (b) same under pink noise
  (c) LOCO source-OOF vs unseen-target F1 of the frozen fusion

Data: results_snr_degradation/snr_degradation_summary.csv
      results_loco/loco_results.csv
Output: figures/fig_robustness.png (relative to the repository root)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "figures" / "fig_robustness.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

SNR_CSV = PROJECT / "results_snr_degradation" / "snr_degradation_summary.csv"
LOCO_CSV = PROJECT / "results_loco" / "loco_results.csv"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 6.8,
    "axes.linewidth": 0.6,
})

MODELS = [("svm_calibrated", "cal. RBF-SVM", "#1f77b4", "o"),
          ("cnn1d", "1D CNN", "#ff7f0e", "s"),
          ("fusion_fixed", "frozen fusion", "#2ca02c", "D")]
SNRS = [20, 15, 10, 5, 0, -5]
CLEAN_FUSION_F1 = 0.9458

snr_df = pd.read_csv(SNR_CSV)
loco_df = pd.read_csv(LOCO_CSV)

fig = plt.figure(figsize=(3.5, 2.62))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                      hspace=0.88, wspace=0.30)

# ------------------------------------------------ (a)(b) noise degradation
for col, (ntype, tag) in enumerate((("white", "(a) white noise"),
                                    ("pink", "(b) pink noise"))):
    ax = fig.add_subplot(gs[0, col])
    for model, label, color, marker in MODELS:
        sub = snr_df[(snr_df.noise_type == ntype) & (snr_df.model == model)]
        sub = sub.set_index("snr_db").reindex(SNRS)
        ax.errorbar(SNRS, sub["f1_mean"], yerr=sub["f1_std"],
                    color=color, marker=marker, ms=2.6, lw=1.0,
                    elinewidth=0.6, capsize=1.4, label=label)
    ax.axhline(CLEAN_FUSION_F1, color="grey", ls="--", lw=0.6)
    ax.set_xlim(21, -6)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([20, 10, 0])
    ax.set_title(tag, fontsize=7.2)
    ax.grid(alpha=0.3, lw=0.4)
    if col == 0:
        ax.set_ylabel("F1")
    ax.set_xlabel("SNR (dB)")
    if col == 1:
        ax.legend(frameon=False, fontsize=5.6, loc="lower left",
                  handlelength=1.4)

# ---------------------------------------------------- (c) LOCO, full width
ax = fig.add_subplot(gs[1, :])
tasks = ["device_hydro_to_logger", "device_logger_to_hydro",
         "material_di_to_pe", "material_pe_to_di"]
labels = ["hydro.\n$\\to$logger", "logger\n$\\to$hydro.",
          "DI$\\to$PE", "PE$\\to$DI"]
src, tgt = [], []
for t in tasks:
    sub = loco_df[(loco_df.task == t) & (loco_df.model == "fusion")]
    src.append(float(sub[sub.split == "source_oof"].f1.iloc[0]))
    tgt.append(float(sub[sub.split == "target_loco"].f1.iloc[0]))
x = np.arange(len(tasks))
w = 0.34
b1 = ax.bar(x - w / 2, src, w, color="#1f4e79", alpha=0.9,
            label="source OOF")
b2 = ax.bar(x + w / 2, tgt, w, facecolor="#9db8d2", edgecolor="#1f4e79",
            hatch="///", lw=0.7, label="unseen target")
for xi, v in zip(x - w / 2, src):
    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
            fontsize=5.6)
for xi, v in zip(x + w / 2, tgt):
    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
            fontsize=5.6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6.2)
ax.set_ylim(0, 1.50)
ax.set_ylabel("F1")
ax.set_title("(c) leave-one-condition-out (frozen fusion)", fontsize=7.2)
ax.legend(frameon=False, fontsize=5.6, loc="upper center", ncol=2,
          handlelength=1.4, handletextpad=0.6, columnspacing=1.2)
ax.grid(alpha=0.3, lw=0.4, axis="y")

fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"saved -> {OUT}")
