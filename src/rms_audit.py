from __future__ import annotations

"""
rms_audit.py

RMS-only ablation on the frozen 800/200 group-isolated verification split
(Section IV-D, Table 4).

How much of the handcrafted SVM's discrimination can be explained by
absolute level alone? This script trains and calibrates the same RBF-SVM
(C=10, gamma=scale, class_weight=balanced) with training-only 5-fold
group-aware OOF Platt calibration on three reduced feature sets:

  1) RMS mean            (1-D)
  2) RMS mean + RMS std  (2-D)
  3) all except RMS      (210-D)
  4) full handcrafted    (212-D, control)

and evaluates each once on the 200-clip group-isolated verification set
with a fixed 0.5 probability threshold.

Note: the paper's full-212-D row (0.935/0.936, 8 FP / 5 FN) is quoted from
frozen_verification.py, whose nested group-aware Platt calibration differs
slightly from the plain OOF Platt used here; the RMS-only rows (the actual
ablation) are produced by this script and match Table 4 exactly.

Run from anywhere:
    python src/rms_audit.py

Outputs:
    results_rms_audit/rms_ablation_results.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "results_rms_audit"

SEED = 42
FOLDS = 5


def derive_group_ids(metadata: pd.DataFrame) -> np.ndarray:
    """Reproduce the recording-group IDs used by the project."""
    groups = []
    for _, row in metadata.iterrows():
        stem = Path(str(row["file_name"])).stem
        base = re.sub(r"_\d+$", "", stem)
        groups.append(f'{row["source_class"]}::{base}')
    return np.asarray(groups, dtype=object)


def build_svm() -> Pipeline:
    """Use the same RBF-SVM configuration for all feature sets."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=False,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_calibrated_svm(X_train, y_train, groups_train):
    """
    Fit training-only, 5-fold group-aware OOF Platt calibration.

    The held-out verification set is never used for fitting either the
    classifier or the probability calibrator.
    """
    cv = StratifiedGroupKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    oof_score = np.full(len(y_train), np.nan, dtype=np.float64)

    for fit_idx, held_idx in cv.split(
        np.zeros((len(y_train), 1)),
        y_train,
        groups_train,
    ):
        model = build_svm()
        model.fit(X_train[fit_idx], y_train[fit_idx])
        oof_score[held_idx] = model.decision_function(
            X_train[held_idx]
        )

    if not np.isfinite(oof_score).all():
        raise RuntimeError("Incomplete OOF decision scores.")

    # Platt scaling fitted only on training OOF scores.
    calibrator = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=3000,
        random_state=SEED,
    )
    calibrator.fit(
        oof_score.reshape(-1, 1),
        y_train,
    )

    # Refit the SVM on all training samples for final verification.
    final_model = build_svm()
    final_model.fit(X_train, y_train)

    return final_model, calibrator


def evaluate(
    name,
    columns,
    X,
    y,
    train_idx,
    test_idx,
    groups,
):
    """Train/calibrate on the frozen training set and evaluate once."""
    X_train = X[train_idx][:, columns].astype(np.float32)
    X_test = X[test_idx][:, columns].astype(np.float32)

    y_train = y[train_idx].astype(np.int64)
    y_test = y[test_idx].astype(np.int64)

    model, calibrator = fit_calibrated_svm(
        X_train,
        y_train,
        groups[train_idx],
    )

    decision = model.decision_function(X_test)
    prob = calibrator.predict_proba(
        decision.reshape(-1, 1)
    )[:, 1]

    # Same fixed probability threshold for all four experiments.
    pred = (prob >= 0.5).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "feature_set": name,
        "dim": len(columns),
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "fp": int(fp),
        "fn": int(fn),
        "FP/FN": f"{fp}/{fn}",
        "brier": brier_score_loss(y_test, prob),
    }


def main():
    root = PROJECT
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-extracted features and frozen split.
    X = np.load(
        root / "features" / "X.npy",
        allow_pickle=False,
    )
    y = np.load(
        root / "features" / "y.npy",
        allow_pickle=False,
    )

    metadata = pd.read_csv(
        root / "features" / "metadata.csv",
        encoding="utf-8-sig",
    )

    feature_names = [
        line.strip()
        for line in (
            root / "features" / "feature_names.txt"
        )
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    ]

    train_idx = np.load(
        root / "train_test_data" / "train_indices.npy",
        allow_pickle=False,
    ).astype(np.int64)

    test_idx = np.load(
        root / "train_test_data" / "test_indices.npy",
        allow_pickle=False,
    ).astype(np.int64)

    # Sanity checks.
    if X.shape[1] != len(feature_names):
        raise RuntimeError(
            f"X contains {X.shape[1]} columns, but "
            f"{len(feature_names)} feature names were found."
        )

    if (len(train_idx), len(test_idx)) != (800, 200):
        raise RuntimeError(
            "Expected the frozen 800/200 split, but found "
            f"{len(train_idx)}/{len(test_idx)}."
        )

    groups = derive_group_ids(metadata)

    overlap = set(groups[train_idx]).intersection(
        set(groups[test_idx])
    )
    if overlap:
        raise RuntimeError(
            f"Train/test group overlap detected: "
            f"{len(overlap)} overlapping groups."
        )

    # Identify RMS mean and RMS std.
    try:
        rms_mean_idx = feature_names.index("rms_mean")
        rms_std_idx = feature_names.index("rms_std")
    except ValueError as exc:
        rms_like = [
            name
            for name in feature_names
            if "rms" in name.lower()
        ]
        raise RuntimeError(
            "Expected feature names 'rms_mean' and 'rms_std'. "
            f"RMS-like features found: {rms_like}"
        ) from exc

    rms_indices = [
        rms_mean_idx,
        rms_std_idx,
    ]

    non_rms_indices = [
        i
        for i in range(X.shape[1])
        if i not in rms_indices
    ]

    full_indices = list(range(X.shape[1]))

    # Four experiments.
    experiments = [
        ("RMS mean", [rms_mean_idx]),
        ("RMS mean + std", rms_indices),
        ("All except RMS", non_rms_indices),
        ("Full handcrafted", full_indices),
    ]

    rows = []

    for name, columns in experiments:
        print(
            f"Running: {name} "
            f"({len(columns)} feature{'s' if len(columns) != 1 else ''})"
        )

        result = evaluate(
            name=name,
            columns=columns,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            groups=groups,
        )

        rows.append(result)

    results = pd.DataFrame(rows)

    display_columns = [
        "feature_set",
        "dim",
        "accuracy",
        "f1",
        "roc_auc",
        "FP/FN",
        "brier",
    ]

    print("\n=== RMS feature ablation ===")
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    output_path = OUT_DIR / "rms_ablation_results.csv"
    results.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved CSV to: {output_path}")


if __name__ == "__main__":
    main()
