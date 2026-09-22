from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_VERSION = "heterogeneous-fusion-pair-search-v1"

SEED = 42
INNER_CALIBRATION_FOLDS = 5

WEIGHTS = np.round(np.arange(0.00, 1.0001, 0.05), 10)
THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.005), 10)

# Primary structural question:
# Which handcrafted/statistical expert is the best partner for which deep-acoustic expert?
#
# For PAIR IDENTITY we require both experts to contribute, because w=0 or w=1
# makes the partner's identity meaningless. Therefore:
#   primary ranking  : 0.05 <= handcrafted_weight <= 0.95
#   unrestricted audit: 0.00 <= handcrafted_weight <= 1.00
#
# The unrestricted result is still saved, because it tells us whether a pair
# collapses to a single model.
GENUINE_MIN_WEIGHT = 0.05

HANDCRAFTED_MODELS = [
    "logistic_regression",
    "random_forest",
    "rbf_svm_calibrated",
    "mlp",
]

DEEP_MODELS = [
    "cnn1d",
    "cnn2d",
    "crnn",
]

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "rbf_svm_calibrated": "RBF-SVM (nested Platt calibrated)",
    "mlp": "MLP",
    "cnn1d": "1D CNN",
    "cnn2d": "2D CNN",
    "crnn": "CRNN",
}

# Keep exactly the same model-selection order as the later fusion/gating work.
SELECTION_COLUMNS = [
    "f1",
    "recall",
    "specificity",
    "roc_auc",
    "average_precision",
]


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_rbf_svm() -> Pipeline:
    # Match model_selection_oof.py exactly.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=False,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])


def positive_decision_score(model: Any, x: np.ndarray) -> np.ndarray:
    score = model.decision_function(x)
    return np.asarray(score, dtype=np.float64).reshape(-1)


def stable_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return np.clip(p, 1e-7, 1.0 - 1e-7)


def compute_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = stable_prob(prob)
    pred = (prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            SELECTION_COLUMNS,
            ascending=[False] * len(SELECTION_COLUMNS),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def search_best_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    # ROC-AUC/AP/Brier/log-loss are threshold independent, so compute once.
    p = stable_prob(prob)
    auc = float(roc_auc_score(y_true, p))
    ap = float(average_precision_score(y_true, p))
    brier = float(brier_score_loss(y_true, p))
    ll = float(log_loss(y_true, p, labels=[0, 1]))

    for threshold in THRESHOLDS:
        pred = (p >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            pred,
            labels=[0, 1],
        ).ravel()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = (tp + tn) / len(y_true)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "roc_auc": auc,
            "average_precision": ap,
            "brier": brier,
            "log_loss": ll,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    all_df = pd.DataFrame(rows)
    best = rank_rows(all_df).iloc[0].to_dict()
    return best, all_df


def nested_group_platt_calibration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    outer_fold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Produce group-safe calibrated RBF-SVM OOF probabilities.

    For each OUTER OOF fold:
      1) Use only the outer-fit samples.
      2) Inside outer-fit, run INNER StratifiedGroupKFold.
      3) Generate inner OOF SVM decision scores.
      4) Fit a one-dimensional Platt logistic calibrator on those inner OOF scores.
      5) Fit the RBF-SVM on all outer-fit samples.
      6) Score the outer-heldout samples.
      7) Apply the calibrator to outer-heldout decision scores.

    Therefore neither the SVM nor the calibrator sees the current outer-heldout
    sample/group while being fitted.
    """
    y_train = np.asarray(y_train, dtype=np.int64)
    groups = np.asarray(groups, dtype=object)
    outer_fold = np.asarray(outer_fold, dtype=np.int64)

    calibrated_prob = np.full(len(y_train), np.nan, dtype=np.float64)
    outer_decision_score = np.full(len(y_train), np.nan, dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []

    unique_outer_folds = sorted(np.unique(outer_fold).tolist())

    for outer in unique_outer_folds:
        outer_fit = np.where(outer_fold != outer)[0]
        outer_heldout = np.where(outer_fold == outer)[0]

        fit_groups = groups[outer_fit]
        heldout_groups = groups[outer_heldout]
        overlap = len(set(fit_groups).intersection(set(heldout_groups)))
        if overlap != 0:
            raise RuntimeError(
                f"Outer fold {outer}: group overlap={overlap}, expected 0."
            )

        inner_cv = StratifiedGroupKFold(
            n_splits=INNER_CALIBRATION_FOLDS,
            shuffle=True,
            random_state=SEED + int(outer),
        )

        inner_oof_score = np.full(len(outer_fit), np.nan, dtype=np.float64)

        inner_splits = list(
            inner_cv.split(
                X=np.zeros((len(outer_fit), 1), dtype=np.float32),
                y=y_train[outer_fit],
                groups=groups[outer_fit],
            )
        )

        for inner_idx, (inner_fit_local, inner_hold_local) in enumerate(
            inner_splits,
            start=1,
        ):
            inner_fit_global = outer_fit[inner_fit_local]
            inner_hold_global = outer_fit[inner_hold_local]

            inner_group_overlap = len(
                set(groups[inner_fit_global]).intersection(
                    set(groups[inner_hold_global])
                )
            )
            if inner_group_overlap != 0:
                raise RuntimeError(
                    f"Outer fold {outer}, inner fold {inner_idx}: "
                    f"group overlap={inner_group_overlap}, expected 0."
                )

            svm_inner = build_rbf_svm()
            svm_inner.fit(
                X_train[inner_fit_global],
                y_train[inner_fit_global],
            )
            inner_oof_score[inner_hold_local] = positive_decision_score(
                svm_inner,
                X_train[inner_hold_global],
            )

        if not np.isfinite(inner_oof_score).all():
            raise RuntimeError(
                f"Outer fold {outer}: incomplete inner OOF decision scores."
            )

        # Near-unregularized 1D logistic Platt mapping.
        calibrator = LogisticRegression(
            solver="lbfgs",
            C=1e6,
            max_iter=3000,
            random_state=SEED,
        )
        calibrator.fit(
            inner_oof_score.reshape(-1, 1),
            y_train[outer_fit],
        )

        inner_prob = stable_prob(
            calibrator.predict_proba(
                inner_oof_score.reshape(-1, 1)
            )[:, 1]
        )

        # Refit the actual SVM on the complete outer-fit partition.
        svm_outer = build_rbf_svm()
        svm_outer.fit(
            X_train[outer_fit],
            y_train[outer_fit],
        )
        outer_score = positive_decision_score(
            svm_outer,
            X_train[outer_heldout],
        )
        outer_prob = stable_prob(
            calibrator.predict_proba(
                outer_score.reshape(-1, 1)
            )[:, 1]
        )

        outer_decision_score[outer_heldout] = outer_score
        calibrated_prob[outer_heldout] = outer_prob

        audit_rows.append({
            "outer_fold": int(outer),
            "outer_fit_samples": int(len(outer_fit)),
            "outer_heldout_samples": int(len(outer_heldout)),
            "outer_fit_groups": int(len(np.unique(groups[outer_fit]))),
            "outer_heldout_groups": int(len(np.unique(groups[outer_heldout]))),
            "outer_group_overlap": int(overlap),
            "inner_folds": int(INNER_CALIBRATION_FOLDS),
            "platt_coef": float(calibrator.coef_.reshape(-1)[0]),
            "platt_intercept": float(calibrator.intercept_.reshape(-1)[0]),
            "inner_brier": float(
                brier_score_loss(y_train[outer_fit], inner_prob)
            ),
            "inner_log_loss": float(
                log_loss(
                    y_train[outer_fit],
                    inner_prob,
                    labels=[0, 1],
                )
            ),
        })

    if not np.isfinite(calibrated_prob).all():
        raise RuntimeError("Nested SVM calibrated OOF probabilities are incomplete.")

    return (
        calibrated_prob,
        outer_decision_score,
        pd.DataFrame(audit_rows),
    )


def pair_diagnostics(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    ca = pred_a == y_true
    cb = pred_b == y_true

    both_correct = int(np.sum(ca & cb))
    a_only = int(np.sum(ca & ~cb))
    b_only = int(np.sum(~ca & cb))
    both_wrong = int(np.sum(~ca & ~cb))
    oracle_correct = int(np.sum(ca | cb))

    return {
        "both_correct": both_correct,
        "handcrafted_only_correct": a_only,
        "deep_only_correct": b_only,
        "both_wrong": both_wrong,
        "discordant_errors": a_only + b_only,
        "oracle_pair_accuracy": float(oracle_correct / len(y_true)),
    }


def search_pair(
    pair_id: str,
    handcrafted_key: str,
    deep_key: str,
    y_true: np.ndarray,
    p_hand: np.ndarray,
    p_deep: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    p_hand = stable_prob(p_hand)
    p_deep = stable_prob(p_deep)

    for w_hand in WEIGHTS:
        w_deep = 1.0 - w_hand
        p = stable_prob(w_hand * p_hand + w_deep * p_deep)

        auc = float(roc_auc_score(y_true, p))
        ap = float(average_precision_score(y_true, p))
        brier = float(brier_score_loss(y_true, p))
        ll = float(log_loss(y_true, p, labels=[0, 1]))

        # Vectorized threshold sweep.
        pred_matrix = p[None, :] >= THRESHOLDS[:, None]
        y1 = (y_true == 1)[None, :]
        y0 = ~y1

        tp = np.sum(pred_matrix & y1, axis=1)
        fp = np.sum(pred_matrix & y0, axis=1)
        fn = np.sum((~pred_matrix) & y1, axis=1)
        tn = np.sum((~pred_matrix) & y0, axis=1)

        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(tp + fp) > 0,
        )
        recall = np.divide(
            tp,
            tp + fn,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(tp + fn) > 0,
        )
        specificity = np.divide(
            tn,
            tn + fp,
            out=np.zeros_like(tn, dtype=np.float64),
            where=(tn + fp) > 0,
        )
        accuracy = (tp + tn) / len(y_true)
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision, dtype=np.float64),
            where=(precision + recall) > 0,
        )

        for i, threshold in enumerate(THRESHOLDS):
            rows.append({
                "pair_id": pair_id,
                "handcrafted_model": handcrafted_key,
                "deep_model": deep_key,
                "handcrafted_weight": float(w_hand),
                "deep_weight": float(w_deep),
                "threshold": float(threshold),
                "accuracy": float(accuracy[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "specificity": float(specificity[i]),
                "roc_auc": auc,
                "average_precision": ap,
                "brier": brier,
                "log_loss": ll,
                "tn": int(tn[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tp": int(tp[i]),
                "is_genuine_fusion": bool(
                    (w_hand >= GENUINE_MIN_WEIGHT)
                    and (w_hand <= 1.0 - GENUINE_MIN_WEIGHT)
                ),
            })

    search_df = pd.DataFrame(rows)

    unrestricted_best = rank_rows(search_df).iloc[0].to_dict()

    genuine_df = search_df[search_df["is_genuine_fusion"]].copy()
    genuine_best = rank_rows(genuine_df).iloc[0].to_dict()

    return search_df, unrestricted_best, genuine_best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TRAINING-ONLY heterogeneous fusion-pair audit using model_selection_oof.py "
            "OOF predictions and nested group-safe Platt calibration for RBF-SVM."
        )
    )
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = resolve_project_root()
    feature_dir = root / "features"
    split_dir = root / "train_test_data"
    exp11_dir = root / "results_model_selection"
    out_dir = root / "results_fusion_pairs"

    x_path = feature_dir / "X.npy"
    y_path = feature_dir / "y.npy"
    metadata_path = feature_dir / "metadata.csv"
    train_indices_path = split_dir / "train_indices.npy"
    oof_path = exp11_dir / "training_only_oof_predictions.csv"

    forbidden_test_inputs = [
        split_dir / "X_test.npy",
        split_dir / "y_test.npy",
        split_dir / "test_indices.npy",
    ]

    print("=" * 92)
    print("RUNNING: src/fusion_pair_search.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("PURPOSE         : select heterogeneous fusion pairs using TRAINING OOF only")
    print("SVM             : nested group-aware Platt calibration")
    print("FIXED TEST DATA : NOT READ")
    print("=" * 92)

    required = [
        x_path,
        y_path,
        metadata_path,
        train_indices_path,
        oof_path,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required training-side file not found: {path}")

    print("Files this audit WILL read:")
    for path in required:
        print(f"  {path}")

    print()
    print("Test-side locations this audit is explicitly designed NOT to read:")
    for path in forbidden_test_inputs:
        print(f"  {path}")

    X_all = np.load(x_path, allow_pickle=False)
    y_all = np.load(y_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    train_indices = np.load(
        train_indices_path,
        allow_pickle=False,
    ).astype(np.int64)
    oof = pd.read_csv(oof_path, encoding="utf-8-sig")

    if X_all.shape[0] != len(y_all) or len(metadata) != len(y_all):
        raise RuntimeError("X.npy / y.npy / metadata.csv row counts do not match.")

    if X_all.ndim != 2 or X_all.shape[1] != 212:
        raise RuntimeError(f"Expected X with 212 features; got {X_all.shape}")

    if len(train_indices) != 800:
        raise RuntimeError(
            f"Expected frozen 800-sample training partition; got {len(train_indices)}"
        )

    if len(oof) != 800:
        raise RuntimeError(
            f"model_selection_oof.py OOF CSV should contain 800 rows; got {len(oof)}"
        )

    required_oof_cols = [
        "source_index",
        "group_id",
        "true_label",
        "oof_fold",
        "logistic_regression_score",
        "rbf_svm_score",
        "random_forest_score",
        "mlp_score",
        "cnn1d_score",
        "cnn2d_score",
        "crnn_score",
    ]
    missing = [c for c in required_oof_cols if c not in oof.columns]
    if missing:
        raise RuntimeError(
            "model_selection_oof.py OOF CSV is missing required columns: "
            + ", ".join(missing)
        )

    oof_source_index = oof["source_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(oof_source_index, train_indices):
        raise RuntimeError(
            "model_selection_oof.py OOF row order/source_index does not exactly match "
            "train_indices.npy. Stop and inspect alignment."
        )

    X_train = X_all[train_indices].astype(np.float32)
    y_train = y_all[train_indices].astype(np.int64)

    if not np.array_equal(
        oof["true_label"].to_numpy(dtype=np.int64),
        y_train,
    ):
        raise RuntimeError("model_selection_oof.py OOF true_label does not match y_train.")

    groups = oof["group_id"].astype(str).to_numpy(dtype=object)
    outer_fold = oof["oof_fold"].to_numpy(dtype=np.int64)

    if len(np.unique(groups)) != 370:
        raise RuntimeError(
            f"Expected 370 training groups; got {len(np.unique(groups))}"
        )

    if sorted(np.unique(outer_fold).tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError(
            f"Expected model_selection_oof.py outer folds 1..5; got "
            f"{sorted(np.unique(outer_fold).tolist())}"
        )

    print()
    print(f"Training samples : {len(y_train)}")
    print(f"Training groups  : {len(np.unique(groups))}")
    print(f"Outer OOF folds  : {sorted(np.unique(outer_fold).tolist())}")
    print(f"Class counts     : class0={(y_train == 0).sum()}, class1={(y_train == 1).sum()}")

    if args.check_only:
        print()
        print("CHECK PASSED.")
        print("No fusion search was run.")
        print("No SVM calibration was fitted.")
        print("No fixed-test data or prior fixed-test results were read.")
        return

    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Nested group-safe calibration for RBF-SVM
    # ------------------------------------------------------------------
    print()
    print("=" * 92)
    print("STEP 1/5: nested group-aware Platt calibration for RBF-SVM")
    print("=" * 92)

    svm_prob, svm_outer_score, svm_cal_audit = nested_group_platt_calibration(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        outer_fold=outer_fold,
    )

    svm_cal_audit.to_csv(
        out_dir / "rbf_svm_nested_calibration_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    svm_cal_df = pd.DataFrame({
        "source_index": train_indices,
        "group_id": groups,
        "true_label": y_train,
        "oof_fold": outer_fold,
        "original_rbf_svm_decision_score_exp11": oof["rbf_svm_score"].to_numpy(
            dtype=np.float64
        ),
        "nested_outer_rbf_svm_decision_score": svm_outer_score,
        "rbf_svm_calibrated_oof_probability": svm_prob,
    })
    svm_cal_df.to_csv(
        out_dir / "rbf_svm_nested_calibrated_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 2) Assemble probability outputs for all seven models
    # ------------------------------------------------------------------
    print()
    print("=" * 92)
    print("STEP 2/5: base-model OOF probability quality + training-only thresholds")
    print("=" * 92)

    probs: dict[str, np.ndarray] = {
        "logistic_regression": stable_prob(
            oof["logistic_regression_score"].to_numpy(dtype=np.float64)
        ),
        "random_forest": stable_prob(
            oof["random_forest_score"].to_numpy(dtype=np.float64)
        ),
        "rbf_svm_calibrated": stable_prob(svm_prob),
        "mlp": stable_prob(
            oof["mlp_score"].to_numpy(dtype=np.float64)
        ),
        "cnn1d": stable_prob(
            oof["cnn1d_score"].to_numpy(dtype=np.float64)
        ),
        "cnn2d": stable_prob(
            oof["cnn2d_score"].to_numpy(dtype=np.float64)
        ),
        "crnn": stable_prob(
            oof["crnn_score"].to_numpy(dtype=np.float64)
        ),
    }

    base_quality_rows: list[dict[str, Any]] = []
    base_threshold_rows: list[dict[str, Any]] = []
    base_best: dict[str, dict[str, Any]] = {}

    all_base_threshold_search: list[pd.DataFrame] = []

    for key, p in probs.items():
        default_metrics = compute_metrics(
            y_train,
            p,
            threshold=0.5,
        )
        best, search_df = search_best_threshold(
            y_train,
            p,
        )
        search_df.insert(0, "model_key", key)
        search_df.insert(1, "model_name", DISPLAY_NAMES[key])
        all_base_threshold_search.append(search_df)

        base_best[key] = best

        base_quality_rows.append({
            "model_key": key,
            "model_name": DISPLAY_NAMES[key],
            "default_threshold": 0.5,
            "default_accuracy": default_metrics["accuracy"],
            "default_precision": default_metrics["precision"],
            "default_recall": default_metrics["recall"],
            "default_f1": default_metrics["f1"],
            "default_specificity": default_metrics["specificity"],
            "roc_auc": default_metrics["roc_auc"],
            "average_precision": default_metrics["average_precision"],
            "brier": default_metrics["brier"],
            "log_loss": default_metrics["log_loss"],
            "default_fp": default_metrics["fp"],
            "default_fn": default_metrics["fn"],
        })

        base_threshold_rows.append({
            "model_key": key,
            "model_name": DISPLAY_NAMES[key],
            **best,
        })

    pd.DataFrame(base_quality_rows).to_csv(
        out_dir / "base_model_oof_probability_quality.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base_threshold_df = rank_rows(
        pd.DataFrame(base_threshold_rows)
    )
    base_threshold_df.insert(
        0,
        "selection_rank",
        np.arange(1, len(base_threshold_df) + 1),
    )
    base_threshold_df.to_csv(
        out_dir / "base_model_oof_threshold_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        all_base_threshold_search,
        ignore_index=True,
    ).to_csv(
        out_dir / "base_model_threshold_search_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save a unified probability table for downstream reproducibility.
    prob_df = pd.DataFrame({
        "source_index": train_indices,
        "group_id": groups,
        "true_label": y_train,
        "oof_fold": outer_fold,
    })
    for key, p in probs.items():
        prob_df[f"{key}_probability"] = p

    prob_df.to_csv(
        out_dir / "all_training_oof_probabilities_for_fusion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 3) Search all 4 x 3 = 12 heterogeneous pairs
    # ------------------------------------------------------------------
    print()
    print("=" * 92)
    print("STEP 3/5: exhaustive training-only heterogeneous fusion search")
    print("=" * 92)

    full_search_frames: list[pd.DataFrame] = []
    unrestricted_best_rows: list[dict[str, Any]] = []
    genuine_best_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    total_pairs = len(HANDCRAFTED_MODELS) * len(DEEP_MODELS)
    pair_counter = 0

    for hand_key in HANDCRAFTED_MODELS:
        for deep_key in DEEP_MODELS:
            pair_counter += 1
            pair_id = f"{hand_key}__plus__{deep_key}"

            print(
                f"  [{pair_counter:02d}/{total_pairs:02d}] "
                f"{DISPLAY_NAMES[hand_key]} + {DISPLAY_NAMES[deep_key]}"
            )

            search_df, unrestricted_best, genuine_best = search_pair(
                pair_id=pair_id,
                handcrafted_key=hand_key,
                deep_key=deep_key,
                y_true=y_train,
                p_hand=probs[hand_key],
                p_deep=probs[deep_key],
            )

            full_search_frames.append(search_df)

            unrestricted_best_rows.append({
                "pair_id": pair_id,
                "handcrafted_model": hand_key,
                "handcrafted_name": DISPLAY_NAMES[hand_key],
                "deep_model": deep_key,
                "deep_name": DISPLAY_NAMES[deep_key],
                **{
                    k: v for k, v in unrestricted_best.items()
                    if k not in {
                        "pair_id",
                        "handcrafted_model",
                        "deep_model",
                    }
                },
            })

            hand_best_f1 = float(base_best[hand_key]["f1"])
            deep_best_f1 = float(base_best[deep_key]["f1"])
            stronger_individual_f1 = max(hand_best_f1, deep_best_f1)

            genuine_best_rows.append({
                "pair_id": pair_id,
                "handcrafted_model": hand_key,
                "handcrafted_name": DISPLAY_NAMES[hand_key],
                "deep_model": deep_key,
                "deep_name": DISPLAY_NAMES[deep_key],
                **{
                    k: v for k, v in genuine_best.items()
                    if k not in {
                        "pair_id",
                        "handcrafted_model",
                        "deep_model",
                    }
                },
                "best_individual_thresholded_f1": stronger_individual_f1,
                "genuine_fusion_f1_minus_best_individual": (
                    float(genuine_best["f1"]) - stronger_individual_f1
                ),
            })

            # Complementarity based on each member's own TRAINING-ONLY selected threshold.
            hand_thr = float(base_best[hand_key]["threshold"])
            deep_thr = float(base_best[deep_key]["threshold"])
            pred_hand = (probs[hand_key] >= hand_thr).astype(np.int64)
            pred_deep = (probs[deep_key] >= deep_thr).astype(np.int64)

            diagnostics = pair_diagnostics(
                y_true=y_train,
                pred_a=pred_hand,
                pred_b=pred_deep,
            )

            diagnostic_rows.append({
                "pair_id": pair_id,
                "handcrafted_model": hand_key,
                "deep_model": deep_key,
                "handcrafted_selected_threshold": hand_thr,
                "deep_selected_threshold": deep_thr,
                "probability_pearson_r": float(
                    np.corrcoef(probs[hand_key], probs[deep_key])[0, 1]
                ),
                **diagnostics,
            })

    full_search_df = pd.concat(
        full_search_frames,
        ignore_index=True,
    )
    full_search_df.to_csv(
        out_dir / "heterogeneous_pair_fusion_search_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 4) Rank pairs
    # ------------------------------------------------------------------
    print()
    print("=" * 92)
    print("STEP 4/5: pair ranking and structural decision")
    print("=" * 92)

    unrestricted_df = rank_rows(
        pd.DataFrame(unrestricted_best_rows)
    )
    unrestricted_df.insert(
        0,
        "selection_rank",
        np.arange(1, len(unrestricted_df) + 1),
    )
    unrestricted_df["collapsed_to_single_model"] = (
        (unrestricted_df["handcrafted_weight"] <= 1e-12)
        | (unrestricted_df["deep_weight"] <= 1e-12)
    )
    unrestricted_df.to_csv(
        out_dir / "pair_best_unrestricted.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # PRIMARY structural ranking: both experts must contribute.
    genuine_df = rank_rows(
        pd.DataFrame(genuine_best_rows)
    )
    genuine_df.insert(
        0,
        "selection_rank",
        np.arange(1, len(genuine_df) + 1),
    )
    genuine_df.to_csv(
        out_dir / "pair_best_genuine_fusion_primary_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics_df = pd.DataFrame(diagnostic_rows)
    diagnostics_df.to_csv(
        out_dir / "pair_training_oof_complementarity_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary = genuine_df.iloc[0].to_dict()
    unrestricted_primary = unrestricted_df.iloc[0].to_dict()

    historical_mask = (
        (genuine_df["handcrafted_model"] == "mlp")
        & (genuine_df["deep_model"] == "cnn2d")
    )
    if historical_mask.sum() != 1:
        raise RuntimeError("Could not uniquely locate historical MLP+CNN2D pair.")
    historical_row = genuine_df[historical_mask].iloc[0].to_dict()

    rf_cnn2d_row = genuine_df[
        (genuine_df["handcrafted_model"] == "random_forest")
        & (genuine_df["deep_model"] == "cnn2d")
    ].iloc[0].to_dict()

    svm_cnn2d_row = genuine_df[
        (genuine_df["handcrafted_model"] == "rbf_svm_calibrated")
        & (genuine_df["deep_model"] == "cnn2d")
    ].iloc[0].to_dict()

    decision = {
        "script_version": SCRIPT_VERSION,
        "fixed_test_data_read": False,
        "training_samples": int(len(y_train)),
        "training_groups": int(len(np.unique(groups))),
        "outer_oof_folds_reused_from_model_selection_oof": 5,
        "svm_probability_method": (
            "nested group-aware Platt calibration: inner group OOF scores "
            "inside each outer-fit partition"
        ),
        "pair_search": {
            "handcrafted_models": HANDCRAFTED_MODELS,
            "deep_models": DEEP_MODELS,
            "pair_count": int(total_pairs),
            "handcrafted_weight_grid": {
                "min": float(WEIGHTS.min()),
                "max": float(WEIGHTS.max()),
                "step": 0.05,
            },
            "threshold_grid": {
                "min": float(THRESHOLDS.min()),
                "max": float(THRESHOLDS.max()),
                "step": 0.005,
            },
        },
        "selection_rule": (
            "lexicographic descending: OOF F1 -> Recall -> Specificity "
            "-> ROC-AUC -> Average Precision"
        ),
        "primary_structural_ranking_rule": (
            "best genuine heterogeneous fusion per pair, requiring "
            "0.05 <= handcrafted_weight <= 0.95"
        ),
        "secondary_unrestricted_rule": (
            "weights 0.00..1.00 retained to detect collapse to a single model"
        ),
        "primary_training_only_pair": {
            "pair_id": primary["pair_id"],
            "handcrafted_model": primary["handcrafted_model"],
            "deep_model": primary["deep_model"],
            "handcrafted_weight": float(primary["handcrafted_weight"]),
            "deep_weight": float(primary["deep_weight"]),
            "threshold": float(primary["threshold"]),
            "f1": float(primary["f1"]),
            "recall": float(primary["recall"]),
            "specificity": float(primary["specificity"]),
            "accuracy": float(primary["accuracy"]),
            "roc_auc": float(primary["roc_auc"]),
            "average_precision": float(primary["average_precision"]),
            "f1_minus_best_individual": float(
                primary["genuine_fusion_f1_minus_best_individual"]
            ),
        },
        "unrestricted_training_only_best": {
            "pair_id": unrestricted_primary["pair_id"],
            "handcrafted_weight": float(
                unrestricted_primary["handcrafted_weight"]
            ),
            "deep_weight": float(unrestricted_primary["deep_weight"]),
            "threshold": float(unrestricted_primary["threshold"]),
            "f1": float(unrestricted_primary["f1"]),
            "collapsed_to_single_model": bool(
                unrestricted_primary["collapsed_to_single_model"]
            ),
        },
        "historical_mlp_cnn2d_pair": {
            "selection_rank": int(historical_row["selection_rank"]),
            "handcrafted_weight": float(
                historical_row["handcrafted_weight"]
            ),
            "deep_weight": float(historical_row["deep_weight"]),
            "threshold": float(historical_row["threshold"]),
            "f1": float(historical_row["f1"]),
            "recall": float(historical_row["recall"]),
            "specificity": float(historical_row["specificity"]),
            "f1_minus_best_individual": float(
                historical_row["genuine_fusion_f1_minus_best_individual"]
            ),
        },
        "rf_cnn2d_pair": {
            "selection_rank": int(rf_cnn2d_row["selection_rank"]),
            "handcrafted_weight": float(
                rf_cnn2d_row["handcrafted_weight"]
            ),
            "deep_weight": float(rf_cnn2d_row["deep_weight"]),
            "threshold": float(rf_cnn2d_row["threshold"]),
            "f1": float(rf_cnn2d_row["f1"]),
        },
        "calibrated_svm_cnn2d_pair": {
            "selection_rank": int(svm_cnn2d_row["selection_rank"]),
            "handcrafted_weight": float(
                svm_cnn2d_row["handcrafted_weight"]
            ),
            "deep_weight": float(svm_cnn2d_row["deep_weight"]),
            "threshold": float(svm_cnn2d_row["threshold"]),
            "f1": float(svm_cnn2d_row["f1"]),
        },
        "historical_pair_training_only_status": (
            "SUPPORTED_AS_PRIMARY_PAIR"
            if int(historical_row["selection_rank"]) == 1
            else "NOT_PRIMARY_TRAINING_ONLY_PAIR"
        ),
    }

    with open(
        out_dir / "training_only_fusion_pair_decision.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            decision,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 5) Console report
    # ------------------------------------------------------------------
    print()
    print("=" * 92)
    print("STEP 5/5: final TRAINING-ONLY report")
    print("=" * 92)

    report_cols = [
        "selection_rank",
        "handcrafted_name",
        "deep_name",
        "handcrafted_weight",
        "deep_weight",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "roc_auc",
        "average_precision",
        "genuine_fusion_f1_minus_best_individual",
    ]

    report = genuine_df[report_cols].copy()

    print()
    print("PRIMARY RANKING: genuine heterogeneous fusion only")
    print("Both experts must contribute (weights in [0.05, 0.95]).")
    print("Selection: F1 -> Recall -> Specificity -> ROC-AUC -> AP.")
    print()
    print(
        report.to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    print()
    print("Key structural audit:")
    print(
        f"  Training-only primary pair : "
        f"{DISPLAY_NAMES[primary['handcrafted_model']]} + "
        f"{DISPLAY_NAMES[primary['deep_model']]}"
    )
    print(
        f"  Primary weights            : "
        f"{float(primary['handcrafted_weight']):.2f} / "
        f"{float(primary['deep_weight']):.2f}"
    )
    print(
        f"  Primary threshold          : "
        f"{float(primary['threshold']):.3f}"
    )
    print(
        f"  Primary OOF F1             : "
        f"{float(primary['f1']):.5f}"
    )
    print(
        f"  MLP + CNN2D rank           : "
        f"{int(historical_row['selection_rank'])}"
    )
    print(
        f"  RF + CNN2D rank            : "
        f"{int(rf_cnn2d_row['selection_rank'])}"
    )
    print(
        f"  calibrated SVM + CNN2D rank: "
        f"{int(svm_cnn2d_row['selection_rank'])}"
    )
    print(
        f"  Historical pair status     : "
        f"{decision['historical_pair_training_only_status']}"
    )

    print()
    print("Unrestricted best:")
    print(
        f"  pair={unrestricted_primary['pair_id']}, "
        f"weights="
        f"{float(unrestricted_primary['handcrafted_weight']):.2f}/"
        f"{float(unrestricted_primary['deep_weight']):.2f}, "
        f"threshold={float(unrestricted_primary['threshold']):.3f}, "
        f"F1={float(unrestricted_primary['f1']):.5f}, "
        f"collapsed_to_single_model="
        f"{bool(unrestricted_primary['collapsed_to_single_model'])}"
    )

    print()
    print("=" * 92)
    print("TRAINING-ONLY HETEROGENEOUS FUSION AUDIT FINISHED")
    print("=" * 92)
    print(f"Outputs: {out_dir}")
    print("Most important files:")
    print(
        f"  {out_dir / 'pair_best_genuine_fusion_primary_ranking.csv'}"
    )
    print(
        f"  {out_dir / 'pair_best_unrestricted.csv'}"
    )
    print(
        f"  {out_dir / 'training_only_fusion_pair_decision.json'}"
    )
    print(
        f"  {out_dir / 'base_model_oof_probability_quality.csv'}"
    )
    print(
        f"  {out_dir / 'rbf_svm_nested_calibration_audit.csv'}"
    )
    print(
        f"  {out_dir / 'all_training_oof_probabilities_for_fusion.csv'}"
    )
    print()
    print("No fixed-test-set metrics were computed.")


if __name__ == "__main__":
    main()
