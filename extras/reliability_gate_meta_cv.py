from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


SCRIPT_VERSION = "reliability-dynamic-gate-meta-cv-v1"

# ---------------------------------------------------------------------
# TRAINING-ONLY dynamic-gate experiment.
#
# Base experts:
#   nested-Platt-calibrated RBF-SVM
#   1D CNN
#
# Fixed anchor selected before fixed-test verification:
#   p_fixed = 0.55 * p_svm + 0.45 * p_cnn1d
#   threshold = 0.520
#
# Reliability proxy:
#   r_i = 2*|p_svm - 0.5| - 2*|p_cnn1d - 0.5|
#
# Dynamic gate:
#   w_svm,i = sigmoid(logit(0.55) + beta * r_i)
#   w_cnn,i = 1 - w_svm,i
#
# Crucially:
#   beta = 0  =>  w_svm = 0.55 and w_cnn = 0.45 exactly.
#
# This experiment DOES NOT read the fixed 200-sample test set.
# It uses a second-level group-aware meta-CV so beta is evaluated on
# groups not used to choose beta.
#
# Threshold remains FIXED at 0.520 throughout this meta-CV audit.
# Therefore any observed improvement is attributable to dynamic weights,
# not threshold re-tuning.
# ---------------------------------------------------------------------

ANCHOR_SVM_WEIGHT = 0.55
ANCHOR_CNN1D_WEIGHT = 0.45
FROZEN_THRESHOLD = 0.520

META_CV_SEED = 142
META_FOLDS = 5

BETA_GRID = np.round(np.arange(0.0, 4.0001, 0.025), 10)

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 17017

# Same primary ordering used through the project.
SELECTION_COLUMNS = [
    "f1",
    "recall",
    "specificity",
    "roc_auc",
    "average_precision",
]

# Eligibility for fixed-test application:
# 1) nested meta-OOF dynamic result is lexicographically better than fixed;
# 2) at least 4/5 meta folds select beta > 0.
MIN_NONZERO_BETA_FOLDS = 4


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def stable_prob(p):
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, 1e-7, 1.0 - 1e-7)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    # Numerically stable enough for our small range.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logit_scalar(p: float) -> float:
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


ANCHOR_LOGIT = logit_scalar(ANCHOR_SVM_WEIGHT)


def relative_margin_reliability(
    p_svm: np.ndarray,
    p_cnn: np.ndarray,
) -> np.ndarray:
    p_svm = stable_prob(p_svm)
    p_cnn = stable_prob(p_cnn)
    c_svm = 2.0 * np.abs(p_svm - 0.5)
    c_cnn = 2.0 * np.abs(p_cnn - 0.5)
    return c_svm - c_cnn


def dynamic_svm_weight(
    reliability: np.ndarray,
    beta: float,
) -> np.ndarray:
    reliability = np.asarray(reliability, dtype=np.float64)
    return sigmoid(
        ANCHOR_LOGIT + float(beta) * reliability
    )


def dynamic_probability(
    p_svm: np.ndarray,
    p_cnn: np.ndarray,
    reliability: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    w_svm = dynamic_svm_weight(
        reliability,
        beta,
    )
    p = (
        w_svm * stable_prob(p_svm)
        + (1.0 - w_svm) * stable_prob(p_cnn)
    )
    return stable_prob(p), w_svm


def compute_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = FROZEN_THRESHOLD,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = stable_prob(prob)
    pred = (prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "roc_auc": float(
            roc_auc_score(y_true, prob)
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                prob,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def rank_beta_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            by=SELECTION_COLUMNS,
            ascending=[False] * len(SELECTION_COLUMNS),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def search_beta(
    y_true: np.ndarray,
    p_svm: np.ndarray,
    p_cnn: np.ndarray,
    reliability: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    for beta in BETA_GRID:
        p, w = dynamic_probability(
            p_svm,
            p_cnn,
            reliability,
            float(beta),
        )
        m = compute_metrics(
            y_true,
            p,
            threshold=FROZEN_THRESHOLD,
        )

        rows.append({
            "beta": float(beta),
            "threshold": FROZEN_THRESHOLD,
            "mean_svm_weight": float(np.mean(w)),
            "std_svm_weight": float(np.std(w)),
            "min_svm_weight": float(np.min(w)),
            "max_svm_weight": float(np.max(w)),
            **m,
        })

    search_df = pd.DataFrame(rows)
    ranked = rank_beta_rows(search_df)

    # Important tie-break: if the complete selection vector is identical,
    # choose the SMALLEST beta, i.e. the least complex dynamic model.
    best0 = ranked.iloc[0]
    tied = search_df.copy()
    for col in SELECTION_COLUMNS:
        tied = tied[
            np.isclose(
                tied[col].to_numpy(dtype=np.float64),
                float(best0[col]),
                rtol=0,
                atol=1e-12,
            )
        ]

    if len(tied) == 0:
        raise RuntimeError("Internal beta tie resolution failed.")

    best = (
        tied.sort_values(
            by="beta",
            ascending=True,
            kind="mergesort",
        )
        .iloc[0]
        .to_dict()
    )

    return best, search_df


def lexicographically_better(
    a: dict[str, Any],
    b: dict[str, Any],
    tol: float = 1e-12,
) -> bool:
    """
    Is metric vector A better than B according to the predeclared project order?
    """
    for col in SELECTION_COLUMNS:
        av = float(a[col])
        bv = float(b[col])

        if av > bv + tol:
            return True
        if av < bv - tol:
            return False

    return False


def exact_mcnemar(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)

    ca = pred_a == y_true
    cb = pred_b == y_true

    a_correct_b_wrong = int(
        np.sum(ca & ~cb)
    )
    a_wrong_b_correct = int(
        np.sum(~ca & cb)
    )
    n = a_correct_b_wrong + a_wrong_b_correct

    if n == 0:
        p_value = 1.0
    else:
        k = min(
            a_correct_b_wrong,
            a_wrong_b_correct,
        )
        tail = (
            sum(
                math.comb(n, i)
                for i in range(k + 1)
            )
            / (2 ** n)
        )
        p_value = min(
            1.0,
            2.0 * tail,
        )

    return {
        "fixed_correct_dynamic_wrong": (
            a_correct_b_wrong
        ),
        "fixed_wrong_dynamic_correct": (
            a_wrong_b_correct
        ),
        "discordant_total": int(n),
        "exact_two_sided_p": float(p_value),
    }


def group_cluster_bootstrap(
    frame: pd.DataFrame,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """
    Paired complete-group bootstrap of Dynamic - Fixed metric differences.
    """
    groups = frame["group_id"].astype(str).unique()
    group_to_indices = {
        g: frame.index[
            frame["group_id"].astype(str) == g
        ].to_numpy(dtype=np.int64)
        for g in groups
    }

    y = frame["true_label"].to_numpy(dtype=np.int64)
    p_fixed = frame["fixed_probability"].to_numpy(dtype=np.float64)
    p_dynamic = frame["dynamic_probability"].to_numpy(dtype=np.float64)

    fixed_obs = compute_metrics(
        y,
        p_fixed,
        FROZEN_THRESHOLD,
    )
    dynamic_obs = compute_metrics(
        y,
        p_dynamic,
        FROZEN_THRESHOLD,
    )

    metrics = [
        "accuracy",
        "f1",
        "recall",
        "specificity",
    ]

    rng = np.random.default_rng(seed)
    boot = {
        m: np.empty(reps, dtype=np.float64)
        for m in metrics
    }

    for r in range(reps):
        sampled_groups = rng.choice(
            groups,
            size=len(groups),
            replace=True,
        )

        idx_parts = [
            group_to_indices[g]
            for g in sampled_groups
        ]
        idx = np.concatenate(idx_parts)

        yy = y[idx]
        pf = p_fixed[idx]
        pdyn = p_dynamic[idx]

        mf = compute_metrics(
            yy,
            pf,
            FROZEN_THRESHOLD,
        )
        md = compute_metrics(
            yy,
            pdyn,
            FROZEN_THRESHOLD,
        )

        for metric in metrics:
            boot[metric][r] = (
                float(md[metric])
                - float(mf[metric])
            )

    rows = []
    for metric in metrics:
        arr = boot[metric]
        low, high = np.quantile(
            arr,
            [0.025, 0.975],
        )

        rows.append({
            "metric": metric,
            "difference_definition": "dynamic_minus_fixed",
            "observed_difference": (
                float(dynamic_obs[metric])
                - float(fixed_obs[metric])
            ),
            "bootstrap_repetitions": int(reps),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "ci_excludes_zero": bool(
                low > 0 or high < 0
            ),
        })

    return pd.DataFrame(rows)


def exact_group_sign_flip(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Exact sign-flip test on per-group accuracy differences.
    Zero-difference groups are retained conceptually but do not create
    distinct sign configurations. If too many nonzero groups occur,
    switch to deterministic Monte Carlo.
    """
    rows = []

    for group_id, sub in frame.groupby("group_id"):
        y = sub["true_label"].to_numpy(dtype=np.int64)
        fixed_pred = (
            sub["fixed_probability"].to_numpy(dtype=np.float64)
            >= FROZEN_THRESHOLD
        ).astype(np.int64)
        dyn_pred = (
            sub["dynamic_probability"].to_numpy(dtype=np.float64)
            >= FROZEN_THRESHOLD
        ).astype(np.int64)

        fixed_acc = float(
            np.mean(fixed_pred == y)
        )
        dynamic_acc = float(
            np.mean(dyn_pred == y)
        )

        rows.append({
            "group_id": group_id,
            "difference": dynamic_acc - fixed_acc,
        })

    diffs = np.asarray(
        [r["difference"] for r in rows],
        dtype=np.float64,
    )
    observed = float(np.mean(diffs))

    nz = diffs[
        np.abs(diffs) > 1e-15
    ]
    k = len(nz)

    if k == 0:
        return {
            "observed_group_macro_accuracy_difference": observed,
            "test_groups": int(len(diffs)),
            "groups_with_nonzero_difference": 0,
            "two_sided_sign_flip_p": 1.0,
            "method": "exact",
        }

    zero_count = len(diffs) - k

    # Mean over ALL groups. Since zero groups remain zero, the same
    # denominator is used in observed and permuted statistics.
    denom = len(diffs)

    if k <= 20:
        total = 2 ** k
        extreme = 0

        for mask in range(total):
            signed_sum = 0.0
            for j in range(k):
                sign = 1.0 if ((mask >> j) & 1) else -1.0
                signed_sum += sign * nz[j]

            stat = signed_sum / denom
            if abs(stat) >= abs(observed) - 1e-15:
                extreme += 1

        p = extreme / total
        method = "exact_enumeration"
    else:
        rng = np.random.default_rng(171717)
        reps = 200000
        signs = rng.choice(
            [-1.0, 1.0],
            size=(reps, k),
            replace=True,
        )
        stats = (
            signs @ nz
        ) / denom
        p = float(
            np.mean(
                np.abs(stats)
                >= abs(observed) - 1e-15
            )
        )
        method = "monte_carlo_200000"

    return {
        "observed_group_macro_accuracy_difference": observed,
        "test_groups": int(len(diffs)),
        "groups_with_nonzero_difference": int(k),
        "zero_difference_groups": int(zero_count),
        "two_sided_sign_flip_p": float(p),
        "method": method,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "TRAINING-ONLY nested meta-CV tuning/evaluation of a "
            "reliability-aware SVM+CNN1D dynamic gate."
        )
    )
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_root()

    repeated_path = (
        root
        / "results_stability"
        / "repeated_training_oof_probabilities.csv"
    )

    mechanism_path = (
        root
        / "results_reliability_mechanism"
        / "reliability_mechanism_decision.json"
    )

    forbidden = [
        root / "train_test_data" / "X_test.npy",
        root / "train_test_data" / "y_test.npy",
        root / "train_test_data" / "test_indices.npy",
        root / "results_frozen_verification",
    ]

    print("=" * 104)
    print("RUNNING: extras/reliability_gate_meta_cv.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("PURPOSE         : train/evaluate reliability-aware gate using TRAINING OOF only")
    print("BASE EXPERTS    : calibrated RBF-SVM + 1D CNN")
    print("ANCHOR          : 0.55 / 0.45")
    print(f"THRESHOLD       : FIXED at {FROZEN_THRESHOLD:.3f}")
    print("META EVALUATION : 5-fold StratifiedGroupKFold")
    print("FIXED TEST DATA : NOT READ")
    print("=" * 104)

    for path in [repeated_path, mechanism_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required training-only input missing: {path}"
            )

    print("Files this experiment WILL read:")
    print(f"  {repeated_path}")
    print(f"  {mechanism_path}")

    print()
    print("Test-side files/results this experiment is explicitly designed NOT to read:")
    for path in forbidden:
        print(f"  {path}")

    repeated = pd.read_csv(
        repeated_path,
        encoding="utf-8-sig",
    )

    with open(
        mechanism_path,
        "r",
        encoding="utf-8",
    ) as f:
        mechanism = json.load(f)

    required_cols = {
        "repeat_seed",
        "source_index",
        "group_id",
        "true_label",
        "rbf_svm_calibrated_probability",
        "cnn1d_probability",
    }
    missing = required_cols.difference(
        repeated.columns
    )
    if missing:
        raise RuntimeError(
            "Repeated OOF CSV missing columns: "
            + ", ".join(sorted(missing))
        )

    # Average the 5 independently group-held-out base-expert predictions
    # for each physical training sample. Every constituent probability is OOF.
    per_sample = (
        repeated.groupby(
            ["source_index", "group_id", "true_label"],
            as_index=False,
        )
        .agg({
            "rbf_svm_calibrated_probability": "mean",
            "cnn1d_probability": "mean",
        })
        .sort_values("source_index")
        .reset_index(drop=True)
    )

    if len(per_sample) != 800:
        raise RuntimeError(
            f"Expected 800 physical training samples; got {len(per_sample)}"
        )

    if per_sample["group_id"].nunique() != 370:
        raise RuntimeError(
            f"Expected 370 training groups; got "
            f"{per_sample['group_id'].nunique()}"
        )

    y = per_sample[
        "true_label"
    ].to_numpy(dtype=np.int64)
    groups = per_sample[
        "group_id"
    ].astype(str).to_numpy(dtype=object)

    p_svm = stable_prob(
        per_sample[
            "rbf_svm_calibrated_probability"
        ].to_numpy(dtype=np.float64)
    )
    p_cnn = stable_prob(
        per_sample[
            "cnn1d_probability"
        ].to_numpy(dtype=np.float64)
    )

    reliability = relative_margin_reliability(
        p_svm,
        p_cnn,
    )

    fixed_prob = stable_prob(
        ANCHOR_SVM_WEIGHT * p_svm
        + ANCHOR_CNN1D_WEIGHT * p_cnn
    )

    per_sample["relative_margin_reliability"] = reliability
    per_sample["fixed_probability"] = fixed_prob

    print()
    print(f"Training samples : {len(per_sample)}")
    print(f"Training groups  : {per_sample['group_id'].nunique()}")
    print(
        f"Mechanism rho    : "
        f"{mechanism.get('primary_mean_over_repeats_spearman')}"
    )
    print(
        f"Mechanism group rho: "
        f"{mechanism.get('group_macro_spearman')}"
    )
    print(
        "NOTE: reliability_mechanism.py's binary eligibility flag is not "
        "used here because its non-monotonic bin decision was affected by "
        "tied weight optima; this script evaluates the continuous gate "
        "directly with nested group-aware meta-CV."
    )

    if args.check_only:
        print()
        print("CHECK PASSED.")
        print("No beta search or meta-CV evaluation was run.")
        print("No fixed-test data/results were read.")
        return

    out_dir = (
        root
        / "results_reliability_gate"
    )
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # STEP 1: build second-level group-aware meta folds
    # ------------------------------------------------------------------
    print()
    print("=" * 104)
    print("STEP 1/5: build group-aware meta-CV folds")
    print("=" * 104)

    sgkf = StratifiedGroupKFold(
        n_splits=META_FOLDS,
        shuffle=True,
        random_state=META_CV_SEED,
    )

    splits = list(
        sgkf.split(
            X=np.zeros(
                (len(y), 1),
                dtype=np.float32,
            ),
            y=y,
            groups=groups,
        )
    )

    meta_fold_assignment = np.full(
        len(y),
        -1,
        dtype=np.int64,
    )
    fold_plan_rows = []

    for fold_idx, (fit_idx, heldout_idx) in enumerate(
        splits,
        start=1,
    ):
        fit_groups = set(groups[fit_idx])
        heldout_groups = set(
            groups[heldout_idx]
        )
        overlap = len(
            fit_groups.intersection(
                heldout_groups
            )
        )
        if overlap != 0:
            raise RuntimeError(
                f"Meta fold {fold_idx}: group overlap={overlap}"
            )

        meta_fold_assignment[
            heldout_idx
        ] = fold_idx

        fold_plan_rows.append({
            "meta_fold": int(fold_idx),
            "fit_samples": int(len(fit_idx)),
            "heldout_samples": int(len(heldout_idx)),
            "fit_groups": int(len(fit_groups)),
            "heldout_groups": int(len(heldout_groups)),
            "fit_class0": int(
                np.sum(y[fit_idx] == 0)
            ),
            "fit_class1": int(
                np.sum(y[fit_idx] == 1)
            ),
            "heldout_class0": int(
                np.sum(y[heldout_idx] == 0)
            ),
            "heldout_class1": int(
                np.sum(y[heldout_idx] == 1)
            ),
            "group_overlap": int(overlap),
        })

        print(
            f"  fold {fold_idx}: "
            f"fit={len(fit_idx)}/{len(fit_groups)} groups, "
            f"heldout={len(heldout_idx)}/{len(heldout_groups)} groups, "
            f"group_overlap={overlap}"
        )

    pd.DataFrame(
        fold_plan_rows
    ).to_csv(
        out_dir / "meta_fold_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # STEP 2: tune beta inside each meta-fit and evaluate meta-heldout
    # ------------------------------------------------------------------
    print()
    print("=" * 104)
    print("STEP 2/5: nested meta-CV beta selection")
    print("=" * 104)

    dynamic_meta_oof = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )
    dynamic_weight_meta_oof = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )
    selected_beta_rows = []
    full_search_rows = []

    for fold_idx, (fit_idx, heldout_idx) in enumerate(
        splits,
        start=1,
    ):
        best, search_df = search_beta(
            y[fit_idx],
            p_svm[fit_idx],
            p_cnn[fit_idx],
            reliability[fit_idx],
        )

        search_df.insert(
            0,
            "meta_fold",
            fold_idx,
        )
        full_search_rows.append(
            search_df
        )

        beta = float(
            best["beta"]
        )

        p_hold, w_hold = dynamic_probability(
            p_svm[heldout_idx],
            p_cnn[heldout_idx],
            reliability[heldout_idx],
            beta,
        )

        dynamic_meta_oof[
            heldout_idx
        ] = p_hold
        dynamic_weight_meta_oof[
            heldout_idx
        ] = w_hold

        fit_fixed_metrics = compute_metrics(
            y[fit_idx],
            fixed_prob[fit_idx],
            FROZEN_THRESHOLD,
        )
        hold_fixed_metrics = compute_metrics(
            y[heldout_idx],
            fixed_prob[heldout_idx],
            FROZEN_THRESHOLD,
        )
        hold_dynamic_metrics = compute_metrics(
            y[heldout_idx],
            p_hold,
            FROZEN_THRESHOLD,
        )

        selected_beta_rows.append({
            "meta_fold": int(fold_idx),
            "selected_beta": beta,
            "fit_dynamic_f1": float(
                best["f1"]
            ),
            "fit_fixed_f1": float(
                fit_fixed_metrics["f1"]
            ),
            "heldout_fixed_accuracy": float(
                hold_fixed_metrics["accuracy"]
            ),
            "heldout_dynamic_accuracy": float(
                hold_dynamic_metrics["accuracy"]
            ),
            "heldout_fixed_f1": float(
                hold_fixed_metrics["f1"]
            ),
            "heldout_dynamic_f1": float(
                hold_dynamic_metrics["f1"]
            ),
            "heldout_fixed_recall": float(
                hold_fixed_metrics["recall"]
            ),
            "heldout_dynamic_recall": float(
                hold_dynamic_metrics["recall"]
            ),
            "heldout_fixed_specificity": float(
                hold_fixed_metrics["specificity"]
            ),
            "heldout_dynamic_specificity": float(
                hold_dynamic_metrics["specificity"]
            ),
            "heldout_mean_svm_weight": float(
                np.mean(w_hold)
            ),
            "heldout_min_svm_weight": float(
                np.min(w_hold)
            ),
            "heldout_max_svm_weight": float(
                np.max(w_hold)
            ),
        })

        print(
            f"  fold {fold_idx}: "
            f"beta={beta:.3f}, "
            f"heldout F1 fixed={hold_fixed_metrics['f1']:.5f} "
            f"dynamic={hold_dynamic_metrics['f1']:.5f}, "
            f"Recall {hold_fixed_metrics['recall']:.5f}->"
            f"{hold_dynamic_metrics['recall']:.5f}, "
            f"Spec {hold_fixed_metrics['specificity']:.5f}->"
            f"{hold_dynamic_metrics['specificity']:.5f}"
        )

    if not np.isfinite(
        dynamic_meta_oof
    ).all():
        raise RuntimeError(
            "Incomplete dynamic meta-OOF probabilities."
        )

    pd.concat(
        full_search_rows,
        ignore_index=True,
    ).to_csv(
        out_dir / "meta_fold_beta_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_beta_df = pd.DataFrame(
        selected_beta_rows
    )
    selected_beta_df.to_csv(
        out_dir / "meta_fold_selected_beta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # STEP 3: aggregate nested meta-OOF comparison
    # ------------------------------------------------------------------
    print()
    print("=" * 104)
    print("STEP 3/5: aggregate nested meta-OOF dynamic vs fixed")
    print("=" * 104)

    fixed_metrics = compute_metrics(
        y,
        fixed_prob,
        FROZEN_THRESHOLD,
    )
    dynamic_metrics = compute_metrics(
        y,
        dynamic_meta_oof,
        FROZEN_THRESHOLD,
    )

    summary_df = pd.DataFrame([
        {
            "method": "fixed_anchor_055_045",
            "beta": 0.0,
            "threshold": FROZEN_THRESHOLD,
            **fixed_metrics,
        },
        {
            "method": "nested_meta_oof_dynamic",
            "beta": np.nan,
            "threshold": FROZEN_THRESHOLD,
            **dynamic_metrics,
        },
    ])

    summary_df.to_csv(
        out_dir / "nested_meta_oof_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_sample["meta_fold"] = meta_fold_assignment
    per_sample["dynamic_probability"] = dynamic_meta_oof
    per_sample["dynamic_svm_weight"] = dynamic_weight_meta_oof
    per_sample["fixed_prediction"] = (
        fixed_prob >= FROZEN_THRESHOLD
    ).astype(np.int64)
    per_sample["dynamic_prediction"] = (
        dynamic_meta_oof >= FROZEN_THRESHOLD
    ).astype(np.int64)

    per_sample.to_csv(
        out_dir / "nested_meta_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        summary_df[
            [
                "method",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "roc_auc",
                "average_precision",
                "fp",
                "fn",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    # ------------------------------------------------------------------
    # STEP 4: paired TRAINING-ONLY statistics
    # ------------------------------------------------------------------
    print()
    print("=" * 104)
    print("STEP 4/5: paired training-only statistics")
    print("=" * 104)

    fixed_pred = (
        fixed_prob >= FROZEN_THRESHOLD
    ).astype(np.int64)
    dynamic_pred = (
        dynamic_meta_oof >= FROZEN_THRESHOLD
    ).astype(np.int64)

    mcnemar = exact_mcnemar(
        y,
        fixed_pred,
        dynamic_pred,
    )

    with open(
        out_dir / "nested_meta_oof_mcnemar.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            mcnemar,
            f,
            ensure_ascii=False,
            indent=2,
        )

    bootstrap_df = group_cluster_bootstrap(
        per_sample[
            [
                "group_id",
                "true_label",
                "fixed_probability",
                "dynamic_probability",
            ]
        ].copy()
    )
    bootstrap_df.to_csv(
        out_dir / "nested_meta_oof_group_bootstrap_dynamic_minus_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sign_flip = exact_group_sign_flip(
        per_sample[
            [
                "group_id",
                "true_label",
                "fixed_probability",
                "dynamic_probability",
            ]
        ].copy()
    )

    with open(
        out_dir / "nested_meta_oof_group_sign_flip.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            sign_flip,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("McNemar:")
    print(json.dumps(
        mcnemar,
        ensure_ascii=False,
        indent=2,
    ))
    print()
    print("Group bootstrap:")
    print(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda v: f"{v:.6f}",
        )
    )
    print()
    print("Group sign-flip:")
    print(json.dumps(
        sign_flip,
        ensure_ascii=False,
        indent=2,
    ))

    # ------------------------------------------------------------------
    # STEP 5: freeze full-training beta if nested evidence supports it
    # ------------------------------------------------------------------
    print()
    print("=" * 104)
    print("STEP 5/5: training-only freeze decision")
    print("=" * 104)

    nonzero_beta_folds = int(
        np.sum(
            selected_beta_df[
                "selected_beta"
            ].to_numpy(dtype=np.float64)
            > 1e-12
        )
    )

    nested_better = lexicographically_better(
        dynamic_metrics,
        fixed_metrics,
    )

    eligible = bool(
        nested_better
        and nonzero_beta_folds
        >= MIN_NONZERO_BETA_FOLDS
    )

    # Final beta is fit on all 800 TRAINING-ONLY averaged OOF predictions.
    final_best, final_search_df = search_beta(
        y,
        p_svm,
        p_cnn,
        reliability,
    )

    final_search_df.to_csv(
        out_dir / "full_training_oof_beta_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_beta = float(
        final_best["beta"]
    )

    p_full_dynamic, w_full_dynamic = dynamic_probability(
        p_svm,
        p_cnn,
        reliability,
        final_beta,
    )

    final_dynamic_metrics = compute_metrics(
        y,
        p_full_dynamic,
        FROZEN_THRESHOLD,
    )

    decision = {
        "script_version": SCRIPT_VERSION,
        "fixed_test_data_read": False,
        "reliability_definition": (
            "2*abs(p_svm-0.5) - 2*abs(p_cnn1d-0.5)"
        ),
        "gate_formula": (
            "w_svm=sigmoid(logit(0.55)+beta*r); "
            "w_cnn1d=1-w_svm"
        ),
        "strict_ablation_property": (
            "beta=0 exactly reproduces fixed 0.55/0.45 fusion"
        ),
        "threshold_fixed": FROZEN_THRESHOLD,
        "beta_grid": {
            "min": float(BETA_GRID.min()),
            "max": float(BETA_GRID.max()),
            "step": 0.025,
        },
        "meta_cv": {
            "folds": META_FOLDS,
            "random_state": META_CV_SEED,
            "group_aware": True,
        },
        "nested_meta_oof_fixed_metrics": fixed_metrics,
        "nested_meta_oof_dynamic_metrics": dynamic_metrics,
        "nested_dynamic_lexicographically_better": nested_better,
        "selected_beta_by_meta_fold": [
            float(x)
            for x in selected_beta_df[
                "selected_beta"
            ].tolist()
        ],
        "nonzero_beta_fold_count": nonzero_beta_folds,
        "minimum_nonzero_beta_folds_for_eligibility": (
            MIN_NONZERO_BETA_FOLDS
        ),
        "eligible_for_one_frozen_fixed_test_evaluation": eligible,
        "final_beta_selected_using_all_training_oof": (
            final_beta if eligible else None
        ),
        "full_training_oof_selected_beta_metrics": (
            final_dynamic_metrics if eligible else None
        ),
        "full_training_oof_dynamic_weight_summary": (
            {
                "mean": float(np.mean(w_full_dynamic)),
                "std": float(np.std(w_full_dynamic)),
                "min": float(np.min(w_full_dynamic)),
                "max": float(np.max(w_full_dynamic)),
            }
            if eligible
            else None
        ),
        "sample_level_mcnemar": mcnemar,
        "group_sign_flip": sign_flip,
        "note": (
            "If eligible is true, beta and threshold are frozen before "
            "any fixed-test application. Fixed-test results must not be "
            "used to revise beta, threshold, or reliability definition."
        ),
    }

    with open(
        out_dir / "reliability_dynamic_gate_decision.json",
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
    print("Dynamic-gate summary:")
    print(
        f"  Nested fixed F1             : "
        f"{fixed_metrics['f1']:.5f}"
    )
    print(
        f"  Nested dynamic F1           : "
        f"{dynamic_metrics['f1']:.5f}"
    )
    print(
        f"  Nested fixed Recall/Spec    : "
        f"{fixed_metrics['recall']:.5f}/"
        f"{fixed_metrics['specificity']:.5f}"
    )
    print(
        f"  Nested dynamic Recall/Spec  : "
        f"{dynamic_metrics['recall']:.5f}/"
        f"{dynamic_metrics['specificity']:.5f}"
    )
    print(
        f"  Meta-fold selected betas    : "
        f"{selected_beta_df['selected_beta'].tolist()}"
    )
    print(
        f"  Nonzero beta folds          : "
        f"{nonzero_beta_folds}/{META_FOLDS}"
    )
    print(
        f"  Full-training OOF beta      : "
        f"{final_beta:.3f}"
    )
    print(
        f"  Eligible for frozen test    : "
        f"{eligible}"
    )

    print()
    print("=" * 104)
    print("TRAINING-ONLY RELIABILITY DYNAMIC-GATE EXPERIMENT FINISHED")
    print("=" * 104)
    print(f"Outputs: {out_dir}")
    print("Most important files:")
    print(
        f"  {out_dir / 'reliability_dynamic_gate_decision.json'}"
    )
    print(
        f"  {out_dir / 'nested_meta_oof_comparison.csv'}"
    )
    print(
        f"  {out_dir / 'meta_fold_selected_beta.csv'}"
    )
    print(
        f"  {out_dir / 'nested_meta_oof_group_bootstrap_dynamic_minus_fixed.csv'}"
    )
    print(
        f"  {out_dir / 'full_training_oof_beta_search.csv'}"
    )
    print()
    print("No fixed-test-set metrics were computed.")


if __name__ == "__main__":
    main()
