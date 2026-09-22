from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_VERSION = "negative-source-sensitivity-v1"

FUSION_THRESHOLD = 0.520
SVM_THRESHOLD = 0.500
CNN_THRESHOLD = 0.500
SVM_WEIGHT = 0.55
CNN_WEIGHT = 0.45

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260818


def norm_text(x: object) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def find_project_root() -> Path:
    # Intended location: <repo>/src/audit_negative_sources.py
    here = Path(__file__).resolve()
    candidates = [here.parents[1], Path.cwd()]
    for root in candidates:
        if (root / "features").exists() and (root / "train_test_data").exists():
            return root
    raise FileNotFoundError(
        "Could not identify project root. Put this script in <repo>/src "
        "or run it with the working directory set to the repository root."
    )


def load_fixed_predictions(root: Path) -> pd.DataFrame:
    path = root / "results_frozen_verification" / "fixed_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing final frozen prediction file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def attach_metadata(root: Path, pred: pd.DataFrame) -> pd.DataFrame:
    metadata_path = root / "features" / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    meta = pd.read_csv(metadata_path, encoding="utf-8-sig").reset_index(drop=True)

    out = pred.copy().reset_index(drop=True)

    # Prefer explicit source index.
    src_col = first_existing(out.columns, [
        "source_index", "sample_index", "original_index", "metadata_index"
    ])
    if src_col is not None:
        idx = out[src_col].astype(int).to_numpy()
        if idx.min() < 0 or idx.max() >= len(meta):
            raise ValueError(f"{src_col} contains indices outside metadata range.")
        aligned = meta.iloc[idx].reset_index(drop=True)
    else:
        test_idx_path = root / "train_test_data" / "test_indices.npy"
        if not test_idx_path.exists():
            raise FileNotFoundError(
                "No source_index in predictions and test_indices.npy is missing."
            )
        test_idx = np.load(test_idx_path, allow_pickle=False).astype(int)
        if len(test_idx) != len(out):
            raise ValueError(
                f"Prediction rows ({len(out)}) != test_indices rows ({len(test_idx)})."
            )
        aligned = meta.iloc[test_idx].reset_index(drop=True)
        out["source_index"] = test_idx

    # Attach source metadata without overwriting prediction columns.
    for col in aligned.columns:
        if col not in out.columns:
            out[col] = aligned[col].to_numpy()

    # Attach frozen group IDs if available.
    if "group_id" not in out.columns and "groupid" not in out.columns:
        manifest_path = root / "train_test_data" / "split_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
            group_col = first_existing(manifest.columns, ["group_id", "groupid", "group"])
            manifest_idx_col = first_existing(
                manifest.columns, ["source_index", "sample_index", "original_index", "metadata_index"]
            )
            if group_col is not None and manifest_idx_col is not None:
                mp = manifest.set_index(manifest_idx_col)[group_col].to_dict()
                out["group_id"] = [mp.get(int(i), np.nan) for i in out["source_index"]]

    return out


def detect_true_label(df: pd.DataFrame) -> np.ndarray:
    col = first_existing(df.columns, [
        "true_label", "true_label_id", "label_id", "y_true", "target"
    ])
    if col is None:
        raise RuntimeError(f"Could not detect true-label column. Columns: {list(df.columns)}")
    y = pd.to_numeric(df[col], errors="coerce").to_numpy()
    if np.isnan(y).any():
        # Maybe textual labels.
        vals = df[col].map(norm_text)
        y = vals.map({"leak": 1, "no_leak": 0, "noise": 0}).to_numpy()
    return y.astype(int)


def detect_probability(df: pd.DataFrame, kind: str) -> np.ndarray | None:
    candidates = {
        "svm": [
            "rbf_svm_calibrated_probability", "rbf_svm_probability",
            "calibrated_svm_probability", "svm_probability", "p_svm",
            "probability_svm", "svm_prob"
        ],
        "cnn": [
            "cnn1d_probability", "cnn_1d_probability", "p_cnn1d",
            "probability_cnn1d", "cnn1d_prob", "cnn_probability"
        ],
        "fusion": [
            "fusion_probability", "frozen_fusion_probability",
            "frozen_probability", "p_final", "final_probability",
            "fused_probability", "probability_fusion"
        ],
    }[kind]
    col = first_existing(df.columns, candidates)
    if col is None:
        return None
    p = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(p).any():
        raise ValueError(f"Probability column {col} contains non-numeric/NaN values.")
    return p


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    else:
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan
    return result


def group_bootstrap_fusion(
    df: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    group_col: str,
) -> pd.DataFrame:
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    for _ in range(BOOTSTRAP_REPS):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx_parts = [np.flatnonzero(groups == g) for g in sampled]
        idx = np.concatenate(idx_parts)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        m = metrics(yy, pp, threshold)
        rows.append({
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "recall": m["recall"],
            "specificity": m["specificity"],
        })

    boot = pd.DataFrame(rows)
    summary = []
    for metric_name in ["accuracy", "f1", "recall", "specificity"]:
        vals = boot[metric_name].dropna().to_numpy()
        summary.append({
            "metric": metric_name,
            "estimate": metrics(y, p, threshold)[metric_name],
            "ci95_low": float(np.percentile(vals, 2.5)),
            "ci95_high": float(np.percentile(vals, 97.5)),
            "bootstrap_reps_used": int(len(vals)),
            "resampling_unit": "group_id",
            "ci_method": "percentile",
        })
    return pd.DataFrame(summary)


def main() -> None:
    root = find_project_root()
    outdir = root / "results_audit_negative_sources"
    outdir.mkdir(parents=True, exist_ok=True)

    pred = attach_metadata(root, load_fixed_predictions(root))
    y = detect_true_label(pred)

    source_col = first_existing(pred.columns, ["source_class", "original_class", "class_name"])
    if source_col is None:
        raise RuntimeError(
            "metadata.csv does not expose source_class. "
            "This analysis requires leak/no_leak/noise provenance."
        )
    source = pred[source_col].map(norm_text)
    source = source.replace({
        "noleak": "no_leak",
        "no__leak": "no_leak",
        "environmental_noise": "noise",
        "environment_noise": "noise",
    })
    pred["source_class_normalized"] = source

    svm_p = detect_probability(pred, "svm")
    cnn_p = detect_probability(pred, "cnn")
    fusion_p = detect_probability(pred, "fusion")

    if svm_p is None or cnn_p is None:
        raise RuntimeError(
            "Could not detect calibrated SVM and CNN1D probabilities in "
            "fixed_test_predictions.csv. Please inspect the column names."
        )
    if fusion_p is None:
        fusion_p = SVM_WEIGHT * svm_p + CNN_WEIGHT * cnn_p
        pred["fusion_probability_reconstructed"] = fusion_p

    pred["svm_probability_audit"] = svm_p
    pred["cnn1d_probability_audit"] = cnn_p
    pred["fusion_probability_audit"] = fusion_p

    # Sanity: label provenance.
    label_by_source = {"leak": 1, "no_leak": 0, "noise": 0}
    known = source.isin(label_by_source)
    if known.any():
        expected = source[known].map(label_by_source).astype(int).to_numpy()
        observed = y[known.to_numpy()]
        mismatch = int(np.sum(expected != observed))
        if mismatch:
            raise RuntimeError(
                f"Detected {mismatch} source_class/true_label mismatches. Stop and inspect metadata."
            )

    models = {
        "Calibrated RBF-SVM": (svm_p, SVM_THRESHOLD),
        "1D CNN": (cnn_p, CNN_THRESHOLD),
        "Frozen fusion": (fusion_p, FUSION_THRESHOLD),
    }

    masks = {
        "full_fixed_verification": np.ones(len(pred), dtype=bool),
        "onsite_leak_vs_no_leak": source.isin(["leak", "no_leak"]).to_numpy(),
    }

    metric_rows = []
    for subset_name, mask in masks.items():
        yy = y[mask]
        for model_name, (pp, th) in models.items():
            row = {"subset": subset_name, "model": model_name}
            row.update(metrics(yy, pp[mask], th))
            metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        outdir / "all_vs_onsite_metrics.csv", index=False, encoding="utf-8-sig"
    )

    # Negative source breakdown: no_leak and environmental noise separately.
    breakdown = []
    for subtype in ["no_leak", "noise"]:
        mask = (source == subtype).to_numpy()
        if mask.sum() == 0:
            continue
        if np.any(y[mask] != 0):
            raise RuntimeError(f"Subtype {subtype} contains positive labels unexpectedly.")
        for model_name, (pp, th) in models.items():
            pred_bin = (pp[mask] >= th).astype(int)
            fp = int(pred_bin.sum())
            n = int(mask.sum())
            breakdown.append({
                "negative_subtype": subtype,
                "model": model_name,
                "n": n,
                "fp": fp,
                "tn": n - fp,
                "false_positive_rate": fp / n,
                "specificity": 1 - fp / n,
                "mean_predicted_leak_probability": float(np.mean(pp[mask])),
                "median_predicted_leak_probability": float(np.median(pp[mask])),
            })

    breakdown_df = pd.DataFrame(breakdown)
    breakdown_df.to_csv(
        outdir / "negative_source_breakdown.csv", index=False, encoding="utf-8-sig"
    )

    # Group bootstrap on the on-site sensitivity subset for the final fusion only.
    group_col = first_existing(pred.columns, ["group_id", "groupid", "group"])
    onsite_mask = masks["onsite_leak_vs_no_leak"]
    if group_col is not None and pred.loc[onsite_mask, group_col].notna().all():
        boot_df = group_bootstrap_fusion(
            pred.loc[onsite_mask].reset_index(drop=True),
            y[onsite_mask],
            fusion_p[onsite_mask],
            FUSION_THRESHOLD,
            group_col,
        )
        boot_df.to_csv(
            outdir / "onsite_fusion_group_bootstrap_ci95.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        (outdir / "onsite_fusion_group_bootstrap_SKIPPED.txt").write_text(
            "group_id could not be recovered for every on-site verification sample.\n",
            encoding="utf-8",
        )

    # Save augmented prediction audit.
    pred.to_csv(
        outdir / "fixed_predictions_with_source_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Figure 1: FPR by negative subtype for the final fusion.
    fusion_break = breakdown_df[breakdown_df["model"] == "Frozen fusion"].copy()
    if not fusion_break.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(
            fusion_break["negative_subtype"].replace({
                "no_leak": "On-site no-leak",
                "noise": "Environmental noise",
            }),
            fusion_break["false_positive_rate"],
        )
        ax.set_ylabel("False-positive rate")
        ax.set_title("Frozen fusion: negative-source error breakdown")
        ax.set_ylim(bottom=0)
        for i, (_, r) in enumerate(fusion_break.iterrows()):
            ax.text(
                i,
                r["false_positive_rate"],
                f'{int(r["fp"])}/{int(r["n"])}',
                ha="center",
                va="bottom",
            )
        fig.tight_layout()
        fig.savefig(outdir / "negative_source_fpr.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Figure 2: full vs on-site-only key metrics for the final fusion.
    fm = metrics_df[metrics_df["model"] == "Frozen fusion"].copy()
    if len(fm) == 2:
        key = ["accuracy", "f1", "recall", "specificity"]
        labels = ["Accuracy", "F1", "Recall", "Specificity"]
        x = np.arange(len(key))
        width = 0.36
        full = fm.loc[fm["subset"] == "full_fixed_verification", key].iloc[0].to_numpy(float)
        onsite = fm.loc[fm["subset"] == "onsite_leak_vs_no_leak", key].iloc[0].to_numpy(float)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(x - width/2, full, width, label="Full fixed verification")
        ax.bar(x + width/2, onsite, width, label="Leak vs on-site no-leak")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1.02)
        ax.set_title("Sensitivity to exclusion of environmental-noise negatives")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "full_vs_onsite_key_metrics.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    counts = source.value_counts(dropna=False).to_dict()
    summary = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(root),
        "analysis_type": "frozen-model sensitivity; no retraining and no retuning",
        "source_counts_in_fixed_verification": {str(k): int(v) for k, v in counts.items()},
        "frozen_fusion": {
            "svm_weight": SVM_WEIGHT,
            "cnn1d_weight": CNN_WEIGHT,
            "threshold": FUSION_THRESHOLD,
        },
        "interpretation_rule": (
            "If performance remains similar after removing environmental-noise negatives, "
            "the headline result is less likely to be driven primarily by easy source-domain cues. "
            "This does not replace cross-site external validation."
        ),
    }
    (outdir / "analysis_protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 88)
    print("APPLIED ACOUSTICS SUPPLEMENTARY ANALYSIS 1")
    print("Environmental-noise / on-site no-leak sensitivity audit")
    print("NO RETRAINING / NO RETUNING")
    print("=" * 88)
    print(metrics_df.to_string(index=False))
    print("\nNegative-source breakdown:")
    print(breakdown_df.to_string(index=False))
    print(f"\nOutputs: {outdir}")


if __name__ == "__main__":
    main()
