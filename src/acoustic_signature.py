"""
acoustic_signature.py

Acoustic signature ("leak voiceprint") and spectral-physics analysis for the
ICASSP version of the leak-detection paper.

Computes, from the raw WAV files in data/:
  1. Per-class mean power spectra (leak / no_leak / noise), dB re 1.0,
     with between-GROUP standard deviation (honest variability).
  2. Spectral contrast leak - no_leak with 95% group-cluster bootstrap CI.
  3. Band-energy fractions per class (five bands, 20-4000 Hz).
  4. Spectral statistics per class (centroid, bandwidth, rolloff85,
     flatness, ZCR, RMS) + Cohen's d (leak vs no_leak).
  5. Dominant spectral peak frequency per clip, summarized per class.
  6. Fingerprint consistency: mean pairwise correlation of group-mean
     spectra within vs between classes.
  7. Physical scaling: Spearman correlation of leak acoustic level
     (RMS, low-band level) with pressure / flow velocity parsed from
     leak filenames.

STFT parameters match extract_features.py exactly:
  n_fft=256, win=256 (hann), hop=80, center=False, sr=8000.

Outputs: results_signature/
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
OUT_DIR = PROJECT / "results_signature"

SR = 8000
N_FFT = 256
HOP = 80
F_MIN, F_MAX = 20.0, 4000.0
EPS = 1e-12

BANDS = [(20, 150), (150, 400), (400, 800), (800, 1600), (1600, 4000)]
BAND_LABELS = ["20-150", "150-400", "400-800", "800-1600", "1600-4000"]

BOOT_REPS = 5000
BOOT_SEED = 20260922

CLASS_DIRS = {"leak": 1, "no_leak": 0, "noise": 0}
CLASS_COLORS = {"leak": "#d62728", "no_leak": "#1f77b4", "noise": "#7f7f7f"}


def derive_group_id(source_class: str, filename: str) -> str:
    stem = Path(filename).stem
    base = re.sub(r"(?:_\d+)+$", "", stem)
    return f"{source_class}::{base}"


def clip_power_spectrum(y: np.ndarray) -> np.ndarray:
    """Mean STFT power spectrum, matching extract_features.py settings."""
    _, _, zxx = stft(
        y, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
        nfft=N_FFT, boundary=None, padded=False,
    )
    power = np.abs(zxx) ** 2
    return power.mean(axis=1)  # (129,)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    # ------------------------------------------------------------------ load
    records = []
    for cls in ("leak", "no_leak", "noise"):
        for wav in sorted((DATA_DIR / cls).glob("*.wav")):
            y, sr = sf.read(wav, dtype="float64", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            assert sr == SR, f"{wav}: unexpected sample rate {sr}"
            y = y - np.mean(y)  # DC removal, same as extract_features.py
            records.append({
                "source_class": cls,
                "label": CLASS_DIRS[cls],
                "file": wav.name,
                "group_id": derive_group_id(cls, wav.name),
                "y": y,
            })
    print(f"loaded {len(records)} clips")

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    fmask = (freqs >= F_MIN) & (freqs <= F_MAX)
    f_sel = freqs[fmask]

    waveforms = np.stack([rec["y"] for rec in records])  # (N, 8000)
    specs = np.stack([clip_power_spectrum(w) for w in waveforms])  # (N,129)
    specs_db = 10.0 * np.log10(specs + EPS)
    rms = np.sqrt(np.mean(waveforms ** 2, axis=1))
    zcr = np.mean(
        np.abs(np.diff(np.signbit(waveforms), axis=1)).astype(float),
        axis=1) / 2.0

    df = pd.DataFrame({
        "source_class": [r["source_class"] for r in records],
        "label": [r["label"] for r in records],
        "file": [r["file"] for r in records],
        "group_id": [r["group_id"] for r in records],
        "rms": rms,
        "zcr": zcr,
    })
    for cls in CLASS_DIRS:
        g = df[df.source_class == cls].group_id.nunique()
        print(f"  {cls}: {(df.source_class == cls).sum()} clips, {g} groups")

    # ------------------------------------------- 1. per-class mean spectra
    rows = {"freq_hz": f_sel}
    class_group_mean_db = {}
    for cls in CLASS_DIRS:
        idx = (df.source_class == cls).to_numpy()
        cls_db = specs_db[idx][:, fmask]
        rows[f"{cls}_mean_db"] = cls_db.mean(axis=0)
        # group-level means -> honest between-group SD
        gdf = pd.DataFrame(cls_db, index=df.group_id[idx]).groupby(level=0).mean()
        class_group_mean_db[cls] = gdf
        rows[f"{cls}_group_sd_db"] = gdf.to_numpy().std(axis=0)
    spec_df = pd.DataFrame(rows)
    spec_df.to_csv(OUT_DIR / "class_mean_spectra.csv", index=False)

    # --------------------------- 2. contrast leak - no_leak with bootstrap CI
    leak_groups = class_group_mean_db["leak"].to_numpy()      # (Gl, B)
    nl_groups = class_group_mean_db["no_leak"].to_numpy()     # (Gn, B)
    contrast = leak_groups.mean(axis=0) - nl_groups.mean(axis=0)
    boot = np.empty((BOOT_REPS, leak_groups.shape[1]))
    for b in range(BOOT_REPS):
        li = rng.integers(0, len(leak_groups), len(leak_groups))
        ni = rng.integers(0, len(nl_groups), len(nl_groups))
        boot[b] = leak_groups[li].mean(axis=0) - nl_groups[ni].mean(axis=0)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    contrast_df = pd.DataFrame({
        "freq_hz": f_sel, "contrast_db": contrast,
        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
    })
    contrast_df.to_csv(OUT_DIR / "spectral_contrast_leak_minus_noleak.csv",
                       index=False)
    sig_bins = (ci_lo > 0) | (ci_hi < 0)
    print(f"contrast: significant bins {sig_bins.sum()}/{len(f_sel)}")

    # ------------------------------------------------- 3. band energy shares
    band_rows = []
    for cls in CLASS_DIRS:
        idx = (df.source_class == cls).to_numpy()
        p = specs[idx][:, fmask]
        total = p.sum(axis=1, keepdims=True)
        fracs = []
        for lo, hi in BANDS:
            bm = (f_sel >= lo) & (f_sel < hi)
            fracs.append(p[:, bm].sum(axis=1, keepdims=True) / total)
        fracs = np.concatenate(fracs, axis=1)  # (n, 5)
        for bi, lab in enumerate(BAND_LABELS):
            band_rows.append({
                "source_class": cls, "band_hz": lab,
                "mean_fraction": fracs[:, bi].mean(),
                "sd_fraction": fracs[:, bi].std(),
            })
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(OUT_DIR / "band_energy_fractions.csv", index=False)

    # ------------------------------------------------- 4. spectral statistics
    def spectral_stats(p_lin: np.ndarray) -> dict:
        """p_lin: (n, len(f_sel)) linear power restricted to 20-4000 Hz."""
        psum = p_lin.sum(axis=1) + EPS
        centroid = (p_lin * f_sel).sum(axis=1) / psum
        bandwidth = np.sqrt(
            (p_lin * (f_sel[None, :] - centroid[:, None]) ** 2).sum(axis=1) / psum)
        cum = np.cumsum(p_lin, axis=1) / psum[:, None]
        rolloff = f_sel[np.argmax(cum >= 0.85, axis=1)]
        flatness = np.exp(np.mean(np.log(p_lin + EPS), axis=1)) / (psum / p_lin.shape[1])
        return {"centroid_hz": centroid, "bandwidth_hz": bandwidth,
                "rolloff85_hz": rolloff, "flatness": flatness}

    stat_rows = []
    per_class_stats = {}
    for cls in CLASS_DIRS:
        idx = (df.source_class == cls).to_numpy()
        s = spectral_stats(specs[idx][:, fmask])
        s["rms"] = rms[idx]
        s["zcr"] = zcr[idx]
        per_class_stats[cls] = s
        for name, v in s.items():
            stat_rows.append({"source_class": cls, "statistic": name,
                              "mean": v.mean(), "sd": v.std(),
                              "median": np.median(v)})
    # Cohen's d, leak vs no_leak
    for name in per_class_stats["leak"]:
        a, b = per_class_stats["leak"][name], per_class_stats["no_leak"][name]
        pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        stat_rows.append({"source_class": "cohens_d_leak_vs_noleak",
                          "statistic": name,
                          "mean": (a.mean() - b.mean()) / (pooled + EPS),
                          "sd": np.nan, "median": np.nan})
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(OUT_DIR / "spectral_statistics.csv", index=False)

    # --------------------------------------------- 5. dominant peak frequency
    kernel = np.ones(5) / 5.0
    peak_rows = []
    for cls in CLASS_DIRS:
        idx = (df.source_class == cls).to_numpy()
        p = specs[idx][:, fmask]
        smooth = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="same"), 1, p)
        peaks = f_sel[np.argmax(smooth, axis=1)]
        peak_rows.append({
            "source_class": cls,
            "peak_median_hz": np.median(peaks),
            "peak_q25_hz": np.percentile(peaks, 25),
            "peak_q75_hz": np.percentile(peaks, 75),
            "frac_peaks_below_400hz": float(np.mean(peaks < 400)),
        })
    peak_df = pd.DataFrame(peak_rows)
    peak_df.to_csv(OUT_DIR / "peak_frequency_summary.csv", index=False)

    # --------------------------------- 6. fingerprint consistency (group means)
    def mean_pairwise_corr(mat: np.ndarray) -> float:
        c = np.corrcoef(mat)
        iu = np.triu_indices_from(c, k=1)
        return float(np.mean(c[iu]))

    consist = {
        "leak_within": mean_pairwise_corr(leak_groups),
        "noleak_within": mean_pairwise_corr(nl_groups),
        "noise_within": mean_pairwise_corr(class_group_mean_db["noise"].to_numpy()),
    }
    # between-class: mean corr of each leak group mean with each no_leak group mean
    ln = np.corrcoef(np.vstack([leak_groups, nl_groups]))
    gl = len(leak_groups)
    consist["leak_vs_noleak_between"] = float(np.mean(ln[:gl, gl:]))
    print("fingerprint consistency:", consist)

    # ------------------------------------- 7. pressure / velocity scaling
    pl_rows = []
    for _, row in df[df.source_class == "leak"].iterrows():
        m = re.match(
            r"^(?P<mat>[^-]+)-(?P<area>[^-]+)-(?P<press>[^-]+)-(?P<vel>[^-]+)-",
            row["file"])
        if not m:
            continue
        press = re.match(r"([\d.]+)\s*MPa", m.group("press"))
        vel = re.match(r"([\d.]+)\s*ms", m.group("vel"))
        if press:
            pl_rows.append({"file": row["file"],
                            "pressure_mpa": float(press.group(1)),
                            "velocity_ms": float(vel.group(1)) if vel else np.nan,
                            "rms": row["rms"]})
    press_df = pd.DataFrame(pl_rows)
    corr_rows = []
    if len(press_df) > 10:
        # low-band absolute level (20-400 Hz) per clip
        low_mask = (f_sel >= 20) & (f_sel < 400)
        low_level = 10 * np.log10(
            specs[:, fmask][:, low_mask].sum(axis=1) + EPS)
        df["low_band_db"] = low_level
        merged = press_df.merge(
            df[["file", "low_band_db"]], on="file", how="left")
        for var in ("pressure_mpa", "velocity_ms"):
            for meas in ("rms", "low_band_db"):
                sub = merged[[var, meas]].dropna()
                rho, p = spearmanr(sub[var], sub[meas])
                corr_rows.append({"condition": var, "measure": meas,
                                  "n": len(sub),
                                  "spearman_rho": rho, "p_value": p})
                print(f"{var} vs {meas}: rho={rho:.3f} (n={len(sub)}, p={p:.2e})")
        merged.to_csv(OUT_DIR / "leak_pressure_velocity_levels.csv", index=False)
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "pressure_velocity_correlations.csv", index=False)

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "figure.dpi": 300, "savefig.dpi": 300})

    # Fig 1: class mean spectra + contrast
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
    ax = axes[0]
    for cls in CLASS_DIRS:
        m = spec_df[f"{cls}_mean_db"].to_numpy()
        s = spec_df[f"{cls}_group_sd_db"].to_numpy()
        ax.plot(f_sel, m, color=CLASS_COLORS[cls], lw=1.2,
                label=f"{cls.replace('_', ' ')}")
        ax.fill_between(f_sel, m - s, m + s, color=CLASS_COLORS[cls],
                        alpha=0.15, lw=0)
    ax.set_ylabel("Power (dB re 1)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("(a) Class-mean power spectra",
                 fontsize=8)
    ax.grid(alpha=0.3, lw=0.4)
    ax.set_xlim(20, 3800)
    ax = axes[1]
    ax.plot(f_sel, contrast, color="#2ca02c", lw=1.2)
    ax.fill_between(f_sel, ci_lo, ci_hi, color="#2ca02c", alpha=0.2, lw=0)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Contrast (dB)")
    ax.set_xlim(20, 3800)
    ax.set_title("(b) Leak $-$ no-leak spectral contrast",
                 fontsize=8)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_signature_spectra.png")
    plt.close(fig)

    # Fig 2: band energy fractions
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.arange(len(BAND_LABELS))
    width = 0.26
    for i, cls in enumerate(CLASS_DIRS):
        sub = band_df[band_df.source_class == cls]
        ax.bar(x + (i - 1) * width, sub["mean_fraction"], width,
               yerr=sub["sd_fraction"], capsize=2,
               color=CLASS_COLORS[cls], alpha=0.85,
               label=cls.replace("_", " "), error_kw={"lw": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, fontsize=7)
    ax.set_xlabel("Frequency band (Hz)")
    ax.set_ylabel("Fraction of total power")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3, lw=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_band_energy.png")
    plt.close(fig)

    # Fig 3: pressure scaling (only if data parsed)
    if len(press_df) > 10:
        merged = pd.read_csv(OUT_DIR / "leak_pressure_velocity_levels.csv")
        merged["rms_db"] = 20.0 * np.log10(merged["rms"] + EPS)
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        ax.scatter(merged["pressure_mpa"], merged["rms_db"], s=8, alpha=0.45,
                   color="#d62728", edgecolors="none")
        ax.set_xlabel("Pressure (MPa)")
        ax.set_ylabel("Leak clip RMS (dB)")
        ax.grid(alpha=0.3, lw=0.4)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_pressure_scaling.png")
        plt.close(fig)

    # ------------------------------------------------------------ summary
    summary = {
        "seed": BOOT_SEED, "bootstrap_reps": BOOT_REPS,
        "stft": {"sr": SR, "n_fft": N_FFT, "hop": HOP,
                 "f_range_hz": [F_MIN, F_MAX]},
        "significant_contrast_bins": int(sig_bins.sum()),
        "total_bins": int(len(f_sel)),
        "fingerprint_consistency": consist,
    }
    with open(OUT_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
