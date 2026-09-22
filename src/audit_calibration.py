from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

SCRIPT_VERSION = "calibration-reliability-audit-v1"

N_BINS_PRIMARY = 10
N_BINS_SENSITIVITY = 15
EPS = 1e-7
SVM_WEIGHT = 0.55
CNN_WEIGHT = 0.45


def first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for root in [here.parents[1], Path.cwd()]:
        if (root / "features").exists() and (root / "results_frozen_verification").exists():
            return root
    raise FileNotFoundError(
        "Could not identify project root. Put this script in <repo>/src "
        "or run it with the working directory set to the repository root."
    )


def detect_true_label(df: pd.DataFrame) -> np.ndarray:
    col = first_existing(df.columns, [
        "true_label", "true_label_id", "label_id", "y_true", "target"
    ])
    if col is None:
        raise RuntimeError(f"Could not detect true label. Columns: {list(df.columns)}")
    numeric = pd.to_numeric(df[col], errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=int)
    txt = df[col].astype(str).str.strip().str.lower()
    mapping = {"leak": 1, "no_leak": 0, "noise": 0}
    y = txt.map(mapping)
    if y.isna().any():
        raise RuntimeError(f"Could not convert textual labels in column {col}.")
    return y.to_numpy(dtype=int)


def detect_probability(df: pd.DataFrame, kind: str) -> tuple[np.ndarray | None, str | None]:
    candidates = {
        "svm": [
            "rbf_svm_calibrated_probability", "rbf_svm_probability",
            "calibrated_svm_probability", "svm_probability", "p_svm",
            "probability_svm", "svm_prob", "rbf_svm_calibrated_oof_probability"
        ],
        "cnn": [
            "cnn1d_probability", "cnn_1d_probability", "p_cnn1d",
            "probability_cnn1d", "cnn1d_prob", "cnn_probability",
            "cnn1d_oof_probability"
        ],
        "fusion": [
            "fusion_probability", "frozen_fusion_probability",
            "frozen_probability", "p_final", "final_probability",
            "fused_probability", "probability_fusion"
        ],
    }[kind]
    col = first_existing(df.columns, candidates)
    if col is None:
        return None, None
    p = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(p).any():
        raise ValueError(f"Probability column {col} contains NaN/non-numeric values.")
    return p, col


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Include p=1 in the last bin.
    ids = np.digitize(np.clip(p, 0, 1), edges[1:-1], right=False)
    rows = []
    n = len(y)
    for b in range(n_bins):
        mask = ids == b
        count = int(mask.sum())
        if count:
            mean_p = float(np.mean(p[mask]))
            frac_pos = float(np.mean(y[mask]))
            abs_gap = abs(mean_p - frac_pos)
        else:
            mean_p = np.nan
            frac_pos = np.nan
            abs_gap = np.nan
        rows.append({
            "bin": b + 1,
            "lower": edges[b],
            "upper": edges[b + 1],
            "count": count,
            "mean_predicted_probability": mean_p,
            "observed_positive_fraction": frac_pos,
            "absolute_gap": abs_gap,
            "weight": count / n,
        })
    return pd.DataFrame(rows)


def ece_mce(bins: pd.DataFrame) -> tuple[float, float]:
    used = bins[bins["count"] > 0]
    ece = float(np.sum(used["weight"] * used["absolute_gap"]))
    mce = float(used["absolute_gap"].max()) if len(used) else np.nan
    return ece, mce


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    # Calibration model: y ~ intercept + slope * logit(p)
    pp = np.clip(p, EPS, 1 - EPS)
    logit = np.log(pp / (1 - pp)).reshape(-1, 1)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
    )
    try:
        model.fit(logit, y)
    except Exception:
        # Compatibility with sklearn variants using penalty="none".
        model = LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=5000,
        )
        model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_metrics(y: np.ndarray, p: np.ndarray, model_name: str, dataset: str) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    bins10 = reliability_bins(y, p, N_BINS_PRIMARY)
    ece10, mce10 = ece_mce(bins10)
    bins15 = reliability_bins(y, p, N_BINS_SENSITIVITY)
    ece15, mce15 = ece_mce(bins15)
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "dataset": dataset,
        "model": model_name,
        "n": int(len(y)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10_equal_width_bins": ece10,
        "mce_10_equal_width_bins": mce10,
        "ece_15_equal_width_bins": ece15,
        "mce_15_equal_width_bins": mce15,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def scan_training_oof(root: Path) -> tuple[pd.DataFrame | None, Path | None]:
    # Search only known training-side result directories; never fixed-test directories.
    roots = [
        root / "results_fusion_pairs",
        root / "results_stability",
        root / "results_model_selection",
    ]
    for d in roots:
        if not d.exists():
            continue
        for csv_path in d.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            svm_p, _ = detect_probability(df, "svm")
            cnn_p, _ = detect_probability(df, "cnn")
            label_col = first_existing(df.columns, [
                "true_label", "true_label_id", "label_id", "y_true", "target"
            ])
            if svm_p is not None and cnn_p is not None and label_col is not None and len(df) >= 700:
                return df, csv_path
    return None, None


def make_reliability_figure(
    out_path: Path,
    y: np.ndarray,
    models: dict[str, np.ndarray],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfect calibration")
    for name, p in models.items():
        bins = reliability_bins(y, p, N_BINS_PRIMARY)
        used = bins[bins["count"] > 0]
        ax.plot(
            used["mean_predicted_probability"],
            used["observed_positive_fraction"],
            marker="o",
            linewidth=1.6,
            label=name,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed leak fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = find_project_root()
    outdir = root / "results_audit_calibration"
    outdir.mkdir(parents=True, exist_ok=True)

    fixed_path = root / "results_frozen_verification" / "fixed_test_predictions.csv"
    if not fixed_path.exists():
        raise FileNotFoundError(f"Missing: {fixed_path}")
    fixed = pd.read_csv(fixed_path, encoding="utf-8-sig")
    y_fixed = detect_true_label(fixed)

    svm_fixed, svm_col = detect_probability(fixed, "svm")
    cnn_fixed, cnn_col = detect_probability(fixed, "cnn")
    fusion_fixed, fusion_col = detect_probability(fixed, "fusion")

    if svm_fixed is None or cnn_fixed is None:
        raise RuntimeError(
            "Could not detect calibrated SVM and CNN1D probability columns in fixed_test_predictions.csv."
        )
    if fusion_fixed is None:
        fusion_fixed = SVM_WEIGHT * svm_fixed + CNN_WEIGHT * cnn_fixed
        fusion_col = "reconstructed_0.55_svm_plus_0.45_cnn1d"

    fixed_models = {
        "Calibrated RBF-SVM": svm_fixed,
        "1D CNN": cnn_fixed,
        "Frozen fusion": fusion_fixed,
    }

    metric_rows = [
        calibration_metrics(y_fixed, p, name, "fixed_group_isolated_verification")
        for name, p in fixed_models.items()
    ]

    # Save per-model reliability bins.
    bin_rows = []
    for name, p in fixed_models.items():
        bins = reliability_bins(y_fixed, p, N_BINS_PRIMARY)
        bins.insert(0, "model", name)
        bins.insert(0, "dataset", "fixed_group_isolated_verification")
        bin_rows.append(bins)

    make_reliability_figure(
        outdir / "reliability_diagram_fixed_verification.png",
        y_fixed,
        fixed_models,
        "Probability calibration on fixed group-isolated verification",
    )

    # Optional: use stored training-only OOF probabilities if they are available.
    oof_df, oof_path = scan_training_oof(root)
    oof_used = False
    if oof_df is not None:
        y_oof = detect_true_label(oof_df)
        svm_oof, svm_oof_col = detect_probability(oof_df, "svm")
        cnn_oof, cnn_oof_col = detect_probability(oof_df, "cnn")
        if svm_oof is not None and cnn_oof is not None:
            fusion_oof = SVM_WEIGHT * svm_oof + CNN_WEIGHT * cnn_oof
            oof_models = {
                "Calibrated RBF-SVM": svm_oof,
                "1D CNN": cnn_oof,
                "Frozen-weight fusion": fusion_oof,
            }
            for name, p in oof_models.items():
                metric_rows.append(
                    calibration_metrics(y_oof, p, name, "training_only_group_aware_oof")
                )
                bins = reliability_bins(y_oof, p, N_BINS_PRIMARY)
                bins.insert(0, "model", name)
                bins.insert(0, "dataset", "training_only_group_aware_oof")
                bin_rows.append(bins)

            make_reliability_figure(
                outdir / "reliability_diagram_training_oof.png",
                y_oof,
                oof_models,
                "Probability calibration on training-only group-aware OOF",
            )
            oof_used = True

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        outdir / "calibration_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(bin_rows, ignore_index=True).to_csv(
        outdir / "reliability_bins_10_equal_width.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "script_version": SCRIPT_VERSION,
        "fixed_prediction_file": str(fixed_path),
        "fixed_probability_columns": {
            "svm": svm_col,
            "cnn1d": cnn_col,
            "fusion": fusion_col,
        },
        "primary_ECE_definition": {
            "bins": N_BINS_PRIMARY,
            "binning": "equal-width over [0,1]",
            "formula": "sum_b (n_b/N)*abs(mean_probability_b - observed_fraction_b)",
        },
        "ECE_sensitivity_bins": N_BINS_SENSITIVITY,
        "training_oof_probability_file_found": str(oof_path) if oof_path else None,
        "training_oof_analysis_used": oof_used,
        "important_note": (
            "Raw RBF-SVM decision_function margins are not probabilities. "
            "This audit therefore does NOT compute Brier/ECE on raw margins or apply "
            "an arbitrary sigmoid to them. It evaluates the nested-Platt-calibrated SVM "
            "probabilities directly, together with CNN1D and fused probabilities."
        ),
    }
    (outdir / "calibration_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("APPLIED ACOUSTICS SUPPLEMENTARY ANALYSIS 2")
    print("Probability calibration / reliability audit")
    print("=" * 88)
    print(metrics_df.to_string(index=False))
    if oof_path:
        print(f"\nTraining OOF source detected: {oof_path}")
    else:
        print(
            "\nNo compatible per-sample calibrated training-OOF CSV was auto-detected. "
            "Fixed-verification calibration analysis was completed normally."
        )
    print("\nImportant: raw SVM margins were NOT treated as probabilities.")
    print(f"Outputs: {outdir}")


if __name__ == "__main__":
    main()
