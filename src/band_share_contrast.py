from __future__ import annotations

"""
band_share_contrast.py

Companion audit for acoustic_signature.py.

What it does
------------
1) Recomputes per-clip spectral band shares with the exact STFT settings used
   by acoustic_signature.py.
2) Saves per-clip and per-class band-share summaries.
3) Quantifies environmental-noise vs on-site no-leak differences with a
   group-cluster bootstrap, preserving the source-recording grouping rule.

Produces the band-share contrasts of Section II-C (the Delta column of
Table 1 and its 95% group-bootstrap intervals).

Run from anywhere:
    python src/band_share_contrast.py

Outputs:
    results_signature/band_share_per_clip.csv
    results_signature/band_share_summary_extended.csv
    results_signature/noise_vs_noleak_band_contrast.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
OUT_DIR = PROJECT / "results_signature"

SR = 8000
N_FFT = 256
HOP = 80
F_MIN, F_MAX = 20.0, 4000.0
EPS = 1e-12

# Keep the five paper bands exactly as in acoustic_signature.py.
BANDS = [
    ("20-150", 20.0, 150.0),
    ("150-400", 150.0, 400.0),
    ("400-800", 400.0, 800.0),
    ("800-1600", 800.0, 1600.0),
    ("1600-4000", 1600.0, 4000.0),
]
# Two aggregate rows already used in Table 1.
AGG_BANDS = [
    ("150-800", 150.0, 800.0),
    (">1600", 1600.0, 4000.0),
]

BOOT_REPS = 5000
BOOT_SEED = 20260922
CLASSES = ("leak", "no_leak", "noise")


def derive_group_id(source_class: str, filename: str) -> str:
    stem = Path(filename).stem
    base = re.sub(r"(?:_\d+)+$", "", stem)
    return f"{source_class}::{base}"


def clip_power_spectrum(y: np.ndarray) -> np.ndarray:
    _, _, zxx = stft(
        y,
        fs=SR,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    return (np.abs(zxx) ** 2).mean(axis=1)


def cluster_bootstrap_difference(
    df_band: pd.DataFrame,
    class_a: str,
    class_b: str,
    rng: np.random.Generator,
    reps: int = BOOT_REPS,
) -> tuple[float, float, float]:
    """
    Difference in per-clip mean share: class_a - class_b.

    The point estimate is the ordinary per-clip mean difference so that it
    matches the Table-1 percentages. The uncertainty is cluster-aware:
    source-recording groups are resampled with replacement, and all clips in
    each sampled group are carried together.
    """
    a = df_band[df_band["source_class"] == class_a]
    b = df_band[df_band["source_class"] == class_b]

    if a.empty or b.empty:
        raise RuntimeError(f"Missing data for comparison {class_a} vs {class_b}")

    point = float(a["fraction"].mean() - b["fraction"].mean())

    a_groups = list(a["group_id"].drop_duplicates())
    b_groups = list(b["group_id"].drop_duplicates())
    a_by_group = {
        g: a.loc[a["group_id"] == g, "fraction"].to_numpy(dtype=float)
        for g in a_groups
    }
    b_by_group = {
        g: b.loc[b["group_id"] == g, "fraction"].to_numpy(dtype=float)
        for g in b_groups
    }

    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        draw_a = rng.choice(a_groups, size=len(a_groups), replace=True)
        draw_b = rng.choice(b_groups, size=len(b_groups), replace=True)

        vals_a = np.concatenate([a_by_group[g] for g in draw_a])
        vals_b = np.concatenate([b_by_group[g] for g in draw_b])
        boot[i] = float(vals_a.mean() - vals_b.mean())

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def group_level_cohens_d(
    df_band: pd.DataFrame,
    class_a: str,
    class_b: str,
) -> float:
    """
    Standardized effect size on equal-weight group means.
    This is secondary to the cluster-bootstrap CI and is useful for judging
    practical separation when class sample counts differ.
    """
    a = (
        df_band[df_band["source_class"] == class_a]
        .groupby("group_id")["fraction"]
        .mean()
        .to_numpy(dtype=float)
    )
    b = (
        df_band[df_band["source_class"] == class_b]
        .groupby("group_id")["fraction"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / (pooled + EPS))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    fmask = (freqs >= F_MIN) & (freqs <= F_MAX)
    f_sel = freqs[fmask]

    rows: list[dict[str, object]] = []

    for cls in CLASSES:
        wavs = sorted((DATA_DIR / cls).glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(
                f"No WAV files found in {DATA_DIR / cls}. "
                "Download/extract the Zenodo dataset first."
            )

        for wav in wavs:
            y, sr = sf.read(wav, dtype="float64", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                raise RuntimeError(f"{wav}: expected {SR} Hz, got {sr}")
            y = y - np.mean(y)

            p = clip_power_spectrum(y)[fmask]
            total = float(p.sum() + EPS)

            for label, lo, hi in BANDS + AGG_BANDS:
                bm = (f_sel >= lo) & (f_sel < hi)
                share = float(p[bm].sum() / total)
                rows.append(
                    {
                        "source_class": cls,
                        "file": wav.name,
                        "group_id": derive_group_id(cls, wav.name),
                        "band_hz": label,
                        "fraction": share,
                    }
                )

    per_clip = pd.DataFrame(rows)
    per_clip.to_csv(
        OUT_DIR / "band_share_per_clip.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        per_clip.groupby(["source_class", "band_hz"], sort=False)["fraction"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_fraction", "std": "sd_fraction"})
    )
    summary["mean_percent"] = 100.0 * summary["mean_fraction"]
    summary["sd_percent"] = 100.0 * summary["sd_fraction"]
    summary.to_csv(
        OUT_DIR / "band_share_summary_extended.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rng = np.random.default_rng(BOOT_SEED)
    contrast_rows: list[dict[str, object]] = []
    ordered_bands = [x[0] for x in BANDS + AGG_BANDS]

    for band in ordered_bands:
        sub = per_clip[per_clip["band_hz"] == band].copy()
        point, lo, hi = cluster_bootstrap_difference(
            sub, "noise", "no_leak", rng=rng
        )

        noise_mean = float(
            sub.loc[sub["source_class"] == "noise", "fraction"].mean()
        )
        noleak_mean = float(
            sub.loc[sub["source_class"] == "no_leak", "fraction"].mean()
        )

        contrast_rows.append(
            {
                "band_hz": band,
                "noise_mean_percent": 100.0 * noise_mean,
                "noleak_mean_percent": 100.0 * noleak_mean,
                "noise_minus_noleak_pp": 100.0 * point,
                "ci95_lo_pp": 100.0 * lo,
                "ci95_hi_pp": 100.0 * hi,
                "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
                "noise_to_noleak_ratio": (
                    noise_mean / noleak_mean if noleak_mean > 0 else np.nan
                ),
                "group_cohens_d": group_level_cohens_d(
                    sub, "noise", "no_leak"
                ),
                "noise_groups": int(
                    sub.loc[sub["source_class"] == "noise", "group_id"].nunique()
                ),
                "noleak_groups": int(
                    sub.loc[sub["source_class"] == "no_leak", "group_id"].nunique()
                ),
                "bootstrap_reps": BOOT_REPS,
                "bootstrap_seed": BOOT_SEED,
            }
        )

    contrast = pd.DataFrame(contrast_rows)
    contrast.to_csv(
        OUT_DIR / "noise_vs_noleak_band_contrast.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nNoise vs no-leak band-share contrast")
    show = contrast[
        [
            "band_hz",
            "noise_mean_percent",
            "noleak_mean_percent",
            "noise_minus_noleak_pp",
            "ci95_lo_pp",
            "ci95_hi_pp",
            "noise_to_noleak_ratio",
            "group_cohens_d",
            "ci_excludes_zero",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
