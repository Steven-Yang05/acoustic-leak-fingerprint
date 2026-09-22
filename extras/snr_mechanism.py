from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "snr-mechanism-svm-cnn1d-v1"

# ---------------------------------------------------------------------
# TRAINING-ONLY mechanism audit.
#
# It reads only:
#   features/metadata.csv
#   train_test_data/train_indices.npy
#   results_stability/
#       repeated_training_oof_probabilities.csv
#   the 800 training WAVs referenced by metadata
#
# It DOES NOT read:
#   X_test.npy / y_test.npy / test_indices.npy
#   frozen_verification.py fixed-test predictions/metrics
#
# The goal is NOT to tune a dynamic gate yet.
# The goal is only to test whether acoustic quality (SNR proxy) has a
# reproducible relationship with the RELATIVE OOF advantage of:
#       calibrated RBF-SVM vs 1D CNN
# ---------------------------------------------------------------------

MAIN_QUANTILE = 0.20
SENSITIVITY_QUANTILES = [0.10, 0.20, 0.30]

FRAME_LENGTH = 256
HOP_LENGTH = 80
EPS = 1e-12

# Frozen fixed-fusion anchor selected BEFORE the frozen_verification.py fixed-test evaluation.
ANCHOR_SVM_WEIGHT = 0.55
ANCHOR_CNN1D_WEIGHT = 0.45
ANCHOR_THRESHOLD = 0.520

WEIGHT_GRID = np.round(np.arange(0.00, 1.0001, 0.05), 10)

# Screening heuristic only; NOT a significance test.
MIN_DIRECTIONAL_RHO = 0.05


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation implemented via average ranks + Pearson correlation.
    Avoids adding a scipy dependency.
    """
    sx = pd.Series(np.asarray(x, dtype=np.float64))
    sy = pd.Series(np.asarray(y, dtype=np.float64))

    mask = np.isfinite(sx.to_numpy()) & np.isfinite(sy.to_numpy())
    sx = sx[mask]
    sy = sy[mask]

    if len(sx) < 3:
        return float("nan")

    rx = sx.rank(method="average").to_numpy(dtype=np.float64)
    ry = sy.rank(method="average").to_numpy(dtype=np.float64)

    if np.std(rx) <= 0 or np.std(ry) <= 0:
        return 0.0

    return float(np.corrcoef(rx, ry)[0, 1])


def frame_energy(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64).reshape(-1)

    if len(wave) < FRAME_LENGTH:
        wave = np.pad(
            wave,
            (0, FRAME_LENGTH - len(wave)),
            mode="constant",
        )

    starts = range(
        0,
        len(wave) - FRAME_LENGTH + 1,
        HOP_LENGTH,
    )

    energies = [
        float(
            np.mean(
                wave[start:start + FRAME_LENGTH] ** 2
            )
        )
        for start in starts
    ]

    if not energies:
        energies = [float(np.mean(wave ** 2))]

    return np.asarray(energies, dtype=np.float64)


def snr_proxy_from_energy(
    energies: np.ndarray,
    low_quantile: float,
) -> float:
    """
    Relative acoustic SNR proxy.

    1) Estimate the internal noise floor as the MEAN energy of the lowest
       q fraction of short-time frames.
    2) Estimate effective signal energy as:
           max(mean(all frame energy) - noise_floor, eps)
    3) Return:
           10 * log10(effective_signal_energy / noise_floor)

    This is a relative acoustic-quality proxy, not physical ground-truth SNR.
    """
    e = np.asarray(energies, dtype=np.float64)
    e = e[np.isfinite(e)]

    if len(e) == 0:
        return float("nan")

    n_low = max(1, int(math.ceil(len(e) * low_quantile)))
    low = np.sort(e)[:n_low]

    noise_floor = max(float(np.mean(low)), EPS)
    total_energy = max(float(np.mean(e)), EPS)
    effective_signal = max(total_energy - noise_floor, EPS)

    return float(
        10.0 * np.log10(
            effective_signal / noise_floor
        )
    )


def robust_z(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=np.float64)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad

    if scale <= 0:
        scale = float(np.std(x))
    if scale <= 0:
        scale = 1.0

    z = (x - median) / scale
    return z, median, scale


def binary_log_loss_per_sample(
    y: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(
        np.asarray(p, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )
    return -(
        y * np.log(p)
        + (1.0 - y) * np.log(1.0 - p)
    )


def compute_binary_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(np.int64)

    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))

    accuracy = (tn + tp) / max(len(y), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def rank_weight_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            by=[
                "f1",
                "recall",
                "specificity",
                "accuracy",
            ],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TRAINING-ONLY SNR mechanism audit for calibrated RBF-SVM + 1D CNN."
        )
    )
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_root()

    exp11_path = (
        root / "src" / "model_selection_oof.py"
    )

    metadata_path = root / "features" / "metadata.csv"
    train_indices_path = (
        root / "train_test_data" / "train_indices.npy"
    )
    repeated_oof_path = (
        root
        / "results_stability"
        / "repeated_training_oof_probabilities.csv"
    )

    required = [
        exp11_path,
        metadata_path,
        train_indices_path,
        repeated_oof_path,
    ]

    forbidden = [
        root / "train_test_data" / "X_test.npy",
        root / "train_test_data" / "y_test.npy",
        root / "train_test_data" / "test_indices.npy",
        root / "results_frozen_verification",
    ]

    print("=" * 96)
    print("RUNNING: extras/snr_mechanism.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("PURPOSE         : TRAINING-ONLY SNR mechanism validation")
    print("EXPERTS         : calibrated RBF-SVM vs 1D CNN")
    print("DYNAMIC GATE    : NOT tuned in this script")
    print("FIXED TEST DATA : NOT READ")
    print("=" * 96)

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required training-side file missing: {path}"
            )

    print("Files this audit WILL read:")
    for path in required:
        print(f"  {path}")

    print()
    print("Test-side files/results this audit is explicitly designed NOT to read:")
    for path in forbidden:
        print(f"  {path}")

    exp11 = load_module(
        exp11_path,
        "exp11_for_snr_mechanism",
    )

    metadata = pd.read_csv(
        metadata_path,
        encoding="utf-8-sig",
    )
    train_indices = np.load(
        train_indices_path,
        allow_pickle=False,
    ).astype(np.int64)
    repeated = pd.read_csv(
        repeated_oof_path,
        encoding="utf-8-sig",
    )

    if len(train_indices) != 800:
        raise RuntimeError(
            f"Expected 800 training samples; got {len(train_indices)}"
        )

    needed_cols = {
        "repeat_seed",
        "source_index",
        "group_id",
        "true_label",
        "oof_fold",
        "rbf_svm_calibrated_probability",
        "cnn1d_probability",
    }
    missing = needed_cols.difference(repeated.columns)
    if missing:
        raise RuntimeError(
            "Repeated OOF file is missing columns: "
            + ", ".join(sorted(missing))
        )

    repeat_seeds = sorted(
        repeated["repeat_seed"].unique().tolist()
    )

    expected_rows = len(train_indices) * len(repeat_seeds)
    if len(repeated) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} repeated OOF rows; got {len(repeated)}"
        )

    # Ensure each repeat covers exactly the same frozen training indices.
    train_index_set = set(train_indices.tolist())
    for seed, sub in repeated.groupby("repeat_seed"):
        source_set = set(
            sub["source_index"].astype(int).tolist()
        )
        if source_set != train_index_set:
            raise RuntimeError(
                f"repeat_seed={seed}: source_index set does not match "
                "the frozen training partition."
            )

    print()
    print(f"Training samples : {len(train_indices)}")
    print(f"Training repeats : {repeat_seeds}")
    print(f"Repeated OOF rows: {len(repeated)}")
    print(f"Main SNR proxy   : lowest {MAIN_QUANTILE:.0%} frame energies")
    print(
        f"Frozen anchor    : SVM={ANCHOR_SVM_WEIGHT:.2f}, "
        f"CNN1D={ANCHOR_CNN1D_WEIGHT:.2f}, "
        f"threshold={ANCHOR_THRESHOLD:.3f}"
    )

    if args.check_only:
        print()
        print("CHECK PASSED.")
        print("No SNR computation or dynamic-gate tuning was run.")
        print("No fixed-test data/results were read.")
        return

    out_dir = (
        root / "results_snr_mechanism"
    )
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: compute SNR proxies from the 800 TRAINING WAVs only
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STEP 1/5: compute relative acoustic SNR proxies on 800 training WAVs")
    print("=" * 96)

    snr_rows: list[dict[str, Any]] = []

    # Get canonical group/label values from any repeat.
    first_repeat = (
        repeated[repeated["repeat_seed"] == repeat_seeds[0]]
        .sort_values("source_index")
        .copy()
    )
    by_index = (
        first_repeat
        .set_index("source_index")
    )

    for j, source_index in enumerate(train_indices, start=1):
        raw = metadata.iloc[int(source_index)]["file_path"]
        wav_path = exp11.resolve_wav_path(root, raw)
        wave = (
            exp11.load_one_waveform(wav_path)
            .squeeze(0)
            .numpy()
            .astype(np.float64)
        )

        energies = frame_energy(wave)

        row = {
            "source_index": int(source_index),
            "group_id": str(
                by_index.loc[int(source_index), "group_id"]
            ),
            "true_label": int(
                by_index.loc[int(source_index), "true_label"]
            ),
        }

        for q in SENSITIVITY_QUANTILES:
            key = f"snr_proxy_q{int(round(q * 100)):02d}_db"
            row[key] = snr_proxy_from_energy(
                energies,
                q,
            )

        snr_rows.append(row)

        if j % 100 == 0 or j == len(train_indices):
            print(f"  processed {j}/{len(train_indices)}")

    snr_df = pd.DataFrame(snr_rows)

    main_col = "snr_proxy_q20_db"
    z, snr_median, snr_scale = robust_z(
        snr_df[main_col].to_numpy(dtype=np.float64)
    )
    snr_df["snr_robust_z_q20"] = z

    snr_df.to_csv(
        out_dir / "training_snr_proxy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # STEP 2: sensitivity of the proxy itself
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STEP 2/5: SNR proxy sensitivity")
    print("=" * 96)

    proxy_cols = [
        "snr_proxy_q10_db",
        "snr_proxy_q20_db",
        "snr_proxy_q30_db",
    ]

    sensitivity_rows: list[dict[str, Any]] = []
    for i in range(len(proxy_cols)):
        for j in range(i + 1, len(proxy_cols)):
            a = proxy_cols[i]
            b = proxy_cols[j]
            rho = spearman_rho(
                snr_df[a].to_numpy(),
                snr_df[b].to_numpy(),
            )
            sensitivity_rows.append({
                "proxy_a": a,
                "proxy_b": b,
                "spearman_rho": float(rho),
            })

    sensitivity_df = pd.DataFrame(
        sensitivity_rows
    )
    sensitivity_df.to_csv(
        out_dir / "snr_proxy_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        sensitivity_df.to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    # ------------------------------------------------------------------
    # STEP 3: relative OOF log-loss advantage
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STEP 3/5: SNR vs SVM relative OOF log-loss advantage")
    print("=" * 96)

    work = repeated.merge(
        snr_df,
        on=[
            "source_index",
            "group_id",
            "true_label",
        ],
        how="left",
        validate="many_to_one",
    )

    if work[proxy_cols].isna().any().any():
        raise RuntimeError(
            "SNR proxy could not be aligned to all repeated OOF rows."
        )

    y_rep = work["true_label"].to_numpy(dtype=np.int64)
    p_svm = work[
        "rbf_svm_calibrated_probability"
    ].to_numpy(dtype=np.float64)
    p_cnn = work[
        "cnn1d_probability"
    ].to_numpy(dtype=np.float64)

    loss_svm = binary_log_loss_per_sample(
        y_rep,
        p_svm,
    )
    loss_cnn = binary_log_loss_per_sample(
        y_rep,
        p_cnn,
    )

    # Positive value means SVM has LOWER loss and therefore an advantage.
    work["svm_logloss"] = loss_svm
    work["cnn1d_logloss"] = loss_cnn
    work["svm_relative_logloss_advantage"] = (
        loss_cnn - loss_svm
    )

    work.to_csv(
        out_dir / "repeated_oof_snr_loss_advantage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corr_rows: list[dict[str, Any]] = []

    # Aggregate over all 5 repeated OOF predictions.
    for proxy_col in proxy_cols:
        rho = spearman_rho(
            work[proxy_col].to_numpy(),
            work[
                "svm_relative_logloss_advantage"
            ].to_numpy(),
        )
        corr_rows.append({
            "scope": "all_repeated_oof_rows",
            "repeat_seed": "all",
            "level": "sample-repeat",
            "snr_proxy": proxy_col,
            "spearman_rho_snr_vs_svm_advantage": float(rho),
        })

    # Per-repeat sample-level correlations.
    for seed, sub in work.groupby("repeat_seed"):
        for proxy_col in proxy_cols:
            rho = spearman_rho(
                sub[proxy_col].to_numpy(),
                sub[
                    "svm_relative_logloss_advantage"
                ].to_numpy(),
            )
            corr_rows.append({
                "scope": "single_repeat",
                "repeat_seed": int(seed),
                "level": "sample",
                "snr_proxy": proxy_col,
                "spearman_rho_snr_vs_svm_advantage": float(rho),
            })

    # Average repeated prediction/loss evidence per physical sample.
    per_sample = (
        work.groupby(
            [
                "source_index",
                "group_id",
                "true_label",
            ],
            as_index=False,
        )
        .agg({
            "rbf_svm_calibrated_probability": "mean",
            "cnn1d_probability": "mean",
            "svm_logloss": "mean",
            "cnn1d_logloss": "mean",
            "svm_relative_logloss_advantage": "mean",
            "snr_proxy_q10_db": "first",
            "snr_proxy_q20_db": "first",
            "snr_proxy_q30_db": "first",
            "snr_robust_z_q20": "first",
        })
    )

    per_sample.to_csv(
        out_dir / "mean_oof_evidence_per_training_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for proxy_col in proxy_cols:
        rho = spearman_rho(
            per_sample[proxy_col].to_numpy(),
            per_sample[
                "svm_relative_logloss_advantage"
            ].to_numpy(),
        )
        corr_rows.append({
            "scope": "mean_over_repeats",
            "repeat_seed": "mean",
            "level": "sample",
            "snr_proxy": proxy_col,
            "spearman_rho_snr_vs_svm_advantage": float(rho),
        })

    # Group-macro mechanism check: every recording group gets equal weight.
    per_group = (
        per_sample.groupby(
            "group_id",
            as_index=False,
        )
        .agg({
            "snr_proxy_q10_db": "mean",
            "snr_proxy_q20_db": "mean",
            "snr_proxy_q30_db": "mean",
            "svm_relative_logloss_advantage": "mean",
        })
    )

    per_group.to_csv(
        out_dir / "mean_mechanism_evidence_per_group.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for proxy_col in proxy_cols:
        rho = spearman_rho(
            per_group[proxy_col].to_numpy(),
            per_group[
                "svm_relative_logloss_advantage"
            ].to_numpy(),
        )
        corr_rows.append({
            "scope": "group_macro",
            "repeat_seed": "group",
            "level": "group",
            "snr_proxy": proxy_col,
            "spearman_rho_snr_vs_svm_advantage": float(rho),
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(
        out_dir / "snr_vs_svm_advantage_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        corr_df[
            corr_df["snr_proxy"] == main_col
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    # ------------------------------------------------------------------
    # STEP 4: exploratory low/mid/high SNR-bin weight evidence
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STEP 4/5: exploratory SNR-bin weight evidence")
    print("=" * 96)
    print(
        "Weight search uses the already frozen threshold 0.520; "
        "no threshold is re-tuned here."
    )

    # Equal-sized SNR bins based on TRAINING samples only.
    per_sample["snr_bin"] = pd.qcut(
        per_sample[main_col],
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    ).astype(str)

    bin_search_rows: list[dict[str, Any]] = []
    bin_best_rows: list[dict[str, Any]] = []
    bin_model_rows: list[dict[str, Any]] = []

    for bin_name in ["low", "mid", "high"]:
        sub = per_sample[
            per_sample["snr_bin"] == bin_name
        ].copy()

        y_bin = sub[
            "true_label"
        ].to_numpy(dtype=np.int64)
        svm_bin = sub[
            "rbf_svm_calibrated_probability"
        ].to_numpy(dtype=np.float64)
        cnn_bin = sub[
            "cnn1d_probability"
        ].to_numpy(dtype=np.float64)

        # Component + frozen-anchor summaries.
        for method_name, prob, threshold in [
            ("calibrated_svm", svm_bin, 0.5),
            ("cnn1d", cnn_bin, 0.5),
            (
                "frozen_anchor_055_045",
                ANCHOR_SVM_WEIGHT * svm_bin
                + ANCHOR_CNN1D_WEIGHT * cnn_bin,
                ANCHOR_THRESHOLD,
            ),
        ]:
            m = compute_binary_metrics(
                y_bin,
                prob,
                threshold,
            )
            bin_model_rows.append({
                "snr_bin": bin_name,
                "method": method_name,
                "sample_count": int(len(sub)),
                "mean_snr_q20_db": float(
                    sub[main_col].mean()
                ),
                **m,
            })

        for svm_weight in WEIGHT_GRID:
            cnn_weight = 1.0 - svm_weight
            p = (
                svm_weight * svm_bin
                + cnn_weight * cnn_bin
            )
            m = compute_binary_metrics(
                y_bin,
                p,
                ANCHOR_THRESHOLD,
            )
            bin_search_rows.append({
                "snr_bin": bin_name,
                "sample_count": int(len(sub)),
                "svm_weight": float(svm_weight),
                "cnn1d_weight": float(cnn_weight),
                "fixed_threshold": ANCHOR_THRESHOLD,
                **m,
            })

        search_sub = pd.DataFrame(
            [
                r
                for r in bin_search_rows
                if r["snr_bin"] == bin_name
            ]
        )
        best = rank_weight_rows(
            search_sub
        ).iloc[0].to_dict()
        bin_best_rows.append(best)

    bin_search_df = pd.DataFrame(
        bin_search_rows
    )
    bin_best_df = pd.DataFrame(
        bin_best_rows
    )
    bin_models_df = pd.DataFrame(
        bin_model_rows
    )

    bin_search_df.to_csv(
        out_dir / "snr_bin_fixed_threshold_weight_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bin_best_df.to_csv(
        out_dir / "snr_bin_best_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bin_models_df.to_csv(
        out_dir / "snr_bin_component_and_anchor_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        bin_best_df[
            [
                "snr_bin",
                "sample_count",
                "svm_weight",
                "cnn1d_weight",
                "f1",
                "recall",
                "specificity",
                "fp",
                "fn",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    # ------------------------------------------------------------------
    # STEP 5: predeclared mechanism screen
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STEP 5/5: mechanism-screen decision")
    print("=" * 96)

    min_proxy_rho = float(
        sensitivity_df["spearman_rho"].min()
    )
    proxy_stable = bool(min_proxy_rho >= 0.95)

    mean_repeat_row = corr_df[
        (corr_df["scope"] == "mean_over_repeats")
        & (corr_df["snr_proxy"] == main_col)
    ]
    if len(mean_repeat_row) != 1:
        raise RuntimeError(
            "Could not uniquely locate mean-over-repeats main correlation."
        )

    main_rho = float(
        mean_repeat_row.iloc[0][
            "spearman_rho_snr_vs_svm_advantage"
        ]
    )

    group_row = corr_df[
        (corr_df["scope"] == "group_macro")
        & (corr_df["snr_proxy"] == main_col)
    ]
    group_rho = float(
        group_row.iloc[0][
            "spearman_rho_snr_vs_svm_advantage"
        ]
    )

    sensitivity_mean_rows = corr_df[
        (corr_df["scope"] == "mean_over_repeats")
        & (corr_df["snr_proxy"].isin(proxy_cols))
    ]
    sensitivity_rhos = sensitivity_mean_rows[
        "spearman_rho_snr_vs_svm_advantage"
    ].to_numpy(dtype=np.float64)

    if main_rho > 0:
        mechanism_direction = (
            "higher_snr_increases_relative_svm_contribution"
        )
        sign = 1
    elif main_rho < 0:
        mechanism_direction = (
            "higher_snr_increases_relative_cnn1d_contribution"
        )
        sign = -1
    else:
        mechanism_direction = "no_monotonic_direction"
        sign = 0

    sensitivity_sign_consistent = bool(
        sign != 0
        and np.all(
            np.sign(sensitivity_rhos) == sign
        )
    )

    per_repeat_main = corr_df[
        (corr_df["scope"] == "single_repeat")
        & (corr_df["snr_proxy"] == main_col)
    ].copy()

    repeat_same_sign = int(
        np.sum(
            np.sign(
                per_repeat_main[
                    "spearman_rho_snr_vs_svm_advantage"
                ].to_numpy(dtype=np.float64)
            )
            == sign
        )
    ) if sign != 0 else 0

    group_same_sign = bool(
        sign != 0
        and np.sign(group_rho) == sign
    )

    best_by_bin = (
        bin_best_df
        .set_index("snr_bin")
    )
    low_w = float(
        best_by_bin.loc["low", "svm_weight"]
    )
    high_w = float(
        best_by_bin.loc["high", "svm_weight"]
    )
    bin_weight_change = high_w - low_w

    if sign > 0:
        bin_direction_consistent = bool(
            bin_weight_change >= 0
        )
    elif sign < 0:
        bin_direction_consistent = bool(
            bin_weight_change <= 0
        )
    else:
        bin_direction_consistent = False

    eligible = bool(
        proxy_stable
        and sensitivity_sign_consistent
        and repeat_same_sign >= 4
        and group_same_sign
        and abs(main_rho) >= MIN_DIRECTIONAL_RHO
        and bin_direction_consistent
    )

    decision = {
        "script_version": SCRIPT_VERSION,
        "fixed_test_data_read": False,
        "dynamic_gate_tuned": False,
        "training_samples": int(len(train_indices)),
        "training_groups": int(
            per_sample["group_id"].nunique()
        ),
        "repeat_seeds": [
            int(x) for x in repeat_seeds
        ],
        "snr_proxy": {
            "main_low_energy_fraction": MAIN_QUANTILE,
            "sensitivity_fractions": SENSITIVITY_QUANTILES,
            "frame_length": FRAME_LENGTH,
            "hop_length": HOP_LENGTH,
            "definition": (
                "10*log10((mean_frame_energy - low_energy_noise_floor) "
                "/ low_energy_noise_floor), with eps protection"
            ),
            "median_q20": snr_median,
            "robust_scale_q20": snr_scale,
            "minimum_q10_q20_q30_pairwise_spearman": min_proxy_rho,
            "proxy_stable_ge_0_95": proxy_stable,
        },
        "relative_advantage_definition": (
            "CNN1D per-sample log-loss minus calibrated-SVM per-sample "
            "log-loss; positive values mean SVM has lower loss"
        ),
        "main_sample_level_mean_repeat_spearman": main_rho,
        "group_macro_spearman": group_rho,
        "mechanism_direction": mechanism_direction,
        "sensitivity_sign_consistent_q10_q20_q30": (
            sensitivity_sign_consistent
        ),
        "q20_repeat_correlations_same_direction_count": (
            repeat_same_sign
        ),
        "q20_repeat_count": int(
            len(per_repeat_main)
        ),
        "group_macro_same_direction": group_same_sign,
        "exploratory_best_svm_weight_low_snr": low_w,
        "exploratory_best_svm_weight_high_snr": high_w,
        "exploratory_high_minus_low_svm_weight": (
            bin_weight_change
        ),
        "bin_weight_direction_consistent": (
            bin_direction_consistent
        ),
        "screening_min_abs_rho": MIN_DIRECTIONAL_RHO,
        "eligible_for_stage2_monotonic_gate_exploration": eligible,
        "screening_note": (
            "This eligibility flag is a predeclared engineering/mechanism "
            "screen, not a statistical significance test."
        ),
        "frozen_anchor_from_stability_selection": {
            "svm_weight": ANCHOR_SVM_WEIGHT,
            "cnn1d_weight": ANCHOR_CNN1D_WEIGHT,
            "threshold": ANCHOR_THRESHOLD,
        },
    }

    with open(
        out_dir / "snr_mechanism_decision.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            decision,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Mechanism summary:")
    print(
        f"  Main sample-level rho (q20) : {main_rho:+.5f}"
    )
    print(
        f"  Group-macro rho (q20)       : {group_rho:+.5f}"
    )
    print(
        f"  Repeat same-sign count       : "
        f"{repeat_same_sign}/{len(per_repeat_main)}"
    )
    print(
        f"  Proxy sensitivity min rho    : {min_proxy_rho:.5f}"
    )
    print(
        f"  Low -> high best SVM weight  : "
        f"{low_w:.2f} -> {high_w:.2f}"
    )
    print(
        f"  Suggested monotonic direction: "
        f"{mechanism_direction}"
    )
    print(
        f"  Eligible for Stage 2 gate    : {eligible}"
    )

    print()
    print("=" * 96)
    print("TRAINING-ONLY SNR MECHANISM AUDIT FINISHED")
    print("=" * 96)
    print(f"Outputs: {out_dir}")
    print("Most important files:")
    print(f"  {out_dir / 'snr_mechanism_decision.json'}")
    print(f"  {out_dir / 'snr_vs_svm_advantage_correlations.csv'}")
    print(f"  {out_dir / 'snr_bin_best_weights.csv'}")
    print(f"  {out_dir / 'snr_proxy_sensitivity.csv'}")
    print()
    print("No fixed-test-set metrics were computed.")
    print("No dynamic-gate parameter was tuned.")


if __name__ == "__main__":
    main()
