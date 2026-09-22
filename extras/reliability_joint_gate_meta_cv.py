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


SCRIPT_VERSION = "reliability-gate-joint-threshold-meta-cv-v1"

# ============================================================================
# TRAINING-ONLY joint gate + threshold meta-CV audit
# ============================================================================
#
# Goal:
#   Fairly compare THREE models on second-level group-aware meta-OOF:
#
#   A) Original frozen fixed fusion
#      p = 0.55*SVM + 0.45*CNN1D
#      threshold = 0.520
#
#   B) Threshold-only fixed fusion
#      weights stay 0.55/0.45
#      threshold is selected using META-FIT only
#
#   C) Reliability-aware dynamic fusion
#      w_svm,i = sigmoid(logit(0.55) + beta*r_i)
#      r_i = 2*|p_svm-0.5| - 2*|p_cnn1d-0.5|
#      beta AND threshold are selected using META-FIT only
#
# The critical scientific comparison is:
#
#       C  vs  B
#
# because B controls for gains that could come merely from changing threshold.
#
# No fixed-test files or fixed-test results are read.
# ============================================================================

ANCHOR_SVM_WEIGHT = 0.55
ANCHOR_CNN1D_WEIGHT = 0.45
ANCHOR_THRESHOLD = 0.520

META_FOLDS = 5
META_CV_SEED = 182

BETA_GRID = np.round(np.arange(0.0, 4.0001, 0.025), 10)
THRESHOLD_GRID = np.round(np.arange(0.30, 0.7001, 0.005), 10)

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 18018

SELECTION_COLUMNS = [
    "f1",
    "recall",
    "specificity",
    "roc_auc",
    "average_precision",
]

MIN_NONZERO_BETA_FOLDS = 4
MIN_HELDOUT_F1_NONWORSE_FOLDS = 3


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def stable_prob(p):
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, 1e-7, 1.0 - 1e-7)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logit_scalar(p: float) -> float:
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


ANCHOR_LOGIT = logit_scalar(ANCHOR_SVM_WEIGHT)


def relative_margin_reliability(p_svm, p_cnn):
    p_svm = stable_prob(p_svm)
    p_cnn = stable_prob(p_cnn)
    return (
        2.0 * np.abs(p_svm - 0.5)
        - 2.0 * np.abs(p_cnn - 0.5)
    )


def dynamic_svm_weight(reliability, beta):
    reliability = np.asarray(reliability, dtype=np.float64)
    return sigmoid(
        ANCHOR_LOGIT + float(beta) * reliability
    )


def dynamic_probability(p_svm, p_cnn, reliability, beta):
    w_svm = dynamic_svm_weight(reliability, beta)
    p = (
        w_svm * stable_prob(p_svm)
        + (1.0 - w_svm) * stable_prob(p_cnn)
    )
    return stable_prob(p), w_svm


def compute_metrics(y_true, prob, threshold):
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
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(
            average_precision_score(y_true, prob)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metrics_vector_better(a: dict[str, Any], b: dict[str, Any], tol=1e-12):
    for col in SELECTION_COLUMNS:
        av = float(a[col])
        bv = float(b[col])

        if av > bv + tol:
            return True
        if av < bv - tol:
            return False

    return False


def exact_mcnemar(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)

    ca = pred_a == y_true
    cb = pred_b == y_true

    a_correct_b_wrong = int(np.sum(ca & ~cb))
    a_wrong_b_correct = int(np.sum(~ca & cb))
    n = a_correct_b_wrong + a_wrong_b_correct

    if n == 0:
        p = 1.0
    else:
        k = min(a_correct_b_wrong, a_wrong_b_correct)
        tail = sum(
            math.comb(n, i)
            for i in range(k + 1)
        ) / (2 ** n)
        p = min(1.0, 2.0 * tail)

    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_total": n,
        "exact_two_sided_p": float(p),
    }


def choose_best_threshold(y_true, prob):
    rows = []

    prob = stable_prob(prob)
    auc = float(roc_auc_score(y_true, prob))
    ap = float(average_precision_score(y_true, prob))

    for threshold in THRESHOLD_GRID:
        m = compute_metrics(
            y_true,
            prob,
            float(threshold),
        )
        rows.append({
            "threshold": float(threshold),
            **m,
        })

    df = pd.DataFrame(rows)

    ranked = df.sort_values(
        by=SELECTION_COLUMNS,
        ascending=[False] * len(SELECTION_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)

    best0 = ranked.iloc[0]

    # Conservative tie break:
    # if metrics are identical, choose threshold closest to the original 0.520.
    tied = df.copy()
    for col in SELECTION_COLUMNS:
        tied = tied[
            np.isclose(
                tied[col].to_numpy(dtype=np.float64),
                float(best0[col]),
                atol=1e-12,
                rtol=0,
            )
        ]

    tied = tied.assign(
        distance_to_anchor=np.abs(
            tied["threshold"] - ANCHOR_THRESHOLD
        )
    )

    best = (
        tied.sort_values(
            by=["distance_to_anchor", "threshold"],
            ascending=[True, True],
            kind="mergesort",
        )
        .iloc[0]
        .to_dict()
    )

    return best, df


def choose_best_beta_threshold(
    y_true,
    p_svm,
    p_cnn,
    reliability,
):
    rows = []

    for beta in BETA_GRID:
        p_dynamic, w_svm = dynamic_probability(
            p_svm,
            p_cnn,
            reliability,
            float(beta),
        )

        auc = float(
            roc_auc_score(y_true, p_dynamic)
        )
        ap = float(
            average_precision_score(
                y_true,
                p_dynamic,
            )
        )

        for threshold in THRESHOLD_GRID:
            m = compute_metrics(
                y_true,
                p_dynamic,
                float(threshold),
            )

            rows.append({
                "beta": float(beta),
                "threshold": float(threshold),
                "mean_svm_weight": float(np.mean(w_svm)),
                "std_svm_weight": float(np.std(w_svm)),
                "min_svm_weight": float(np.min(w_svm)),
                "max_svm_weight": float(np.max(w_svm)),
                **m,
            })

    df = pd.DataFrame(rows)

    ranked = df.sort_values(
        by=SELECTION_COLUMNS,
        ascending=[False] * len(SELECTION_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)

    best0 = ranked.iloc[0]

    tied = df.copy()
    for col in SELECTION_COLUMNS:
        tied = tied[
            np.isclose(
                tied[col].to_numpy(dtype=np.float64),
                float(best0[col]),
                atol=1e-12,
                rtol=0,
            )
        ]

    # Conservative complexity tie break:
    # 1) smallest beta
    # 2) threshold closest to 0.520
    tied = tied.assign(
        distance_to_anchor=np.abs(
            tied["threshold"] - ANCHOR_THRESHOLD
        )
    )

    best = (
        tied.sort_values(
            by=[
                "beta",
                "distance_to_anchor",
                "threshold",
            ],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .iloc[0]
        .to_dict()
    )

    return best, df


def group_cluster_bootstrap(
    frame: pd.DataFrame,
    prob_a_col: str,
    threshold_a: float,
    prob_b_col: str,
    threshold_b: float,
    comparison_name: str,
):
    groups = frame["group_id"].astype(str).unique()

    group_to_indices = {
        g: frame.index[
            frame["group_id"].astype(str) == g
        ].to_numpy(dtype=np.int64)
        for g in groups
    }

    y = frame["true_label"].to_numpy(dtype=np.int64)
    p_a = frame[prob_a_col].to_numpy(dtype=np.float64)
    p_b = frame[prob_b_col].to_numpy(dtype=np.float64)

    obs_a = compute_metrics(y, p_a, threshold_a)
    obs_b = compute_metrics(y, p_b, threshold_b)

    metrics = [
        "accuracy",
        "f1",
        "recall",
        "specificity",
    ]

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    boot = {
        metric: np.empty(
            BOOTSTRAP_REPS,
            dtype=np.float64,
        )
        for metric in metrics
    }

    for r in range(BOOTSTRAP_REPS):
        sampled_groups = rng.choice(
            groups,
            size=len(groups),
            replace=True,
        )

        idx = np.concatenate(
            [
                group_to_indices[g]
                for g in sampled_groups
            ]
        )

        yy = y[idx]

        ma = compute_metrics(
            yy,
            p_a[idx],
            threshold_a,
        )
        mb = compute_metrics(
            yy,
            p_b[idx],
            threshold_b,
        )

        for metric in metrics:
            boot[metric][r] = (
                float(mb[metric])
                - float(ma[metric])
            )

    rows = []

    for metric in metrics:
        arr = boot[metric]
        low, high = np.quantile(
            arr,
            [0.025, 0.975],
        )

        rows.append({
            "comparison": comparison_name,
            "difference_definition": "method_b_minus_method_a",
            "metric": metric,
            "observed_difference": (
                float(obs_b[metric])
                - float(obs_a[metric])
            ),
            "bootstrap_repetitions": BOOTSTRAP_REPS,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "ci_excludes_zero": bool(
                low > 0 or high < 0
            ),
        })

    return pd.DataFrame(rows)


def exact_group_sign_flip(
    frame,
    pred_a_col,
    pred_b_col,
):
    diffs = []

    for _, sub in frame.groupby("group_id"):
        y = sub["true_label"].to_numpy(dtype=np.int64)
        pa = sub[pred_a_col].to_numpy(dtype=np.int64)
        pb = sub[pred_b_col].to_numpy(dtype=np.int64)

        acc_a = float(np.mean(pa == y))
        acc_b = float(np.mean(pb == y))

        diffs.append(
            acc_b - acc_a
        )

    diffs = np.asarray(
        diffs,
        dtype=np.float64,
    )

    observed = float(
        np.mean(diffs)
    )

    nz = diffs[
        np.abs(diffs) > 1e-15
    ]
    k = len(nz)
    denom = len(diffs)

    if k == 0:
        p = 1.0
        method = "exact"
    elif k <= 20:
        total = 2 ** k
        extreme = 0

        for mask in range(total):
            signed_sum = 0.0

            for j in range(k):
                sign = (
                    1.0
                    if ((mask >> j) & 1)
                    else -1.0
                )
                signed_sum += (
                    sign * nz[j]
                )

            stat = (
                signed_sum / denom
            )

            if (
                abs(stat)
                >= abs(observed) - 1e-15
            ):
                extreme += 1

        p = extreme / total
        method = "exact_enumeration"
    else:
        rng = np.random.default_rng(181818)
        reps = 200000
        signs = rng.choice(
            [-1.0, 1.0],
            size=(reps, k),
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
        "test_groups": int(denom),
        "groups_with_nonzero_difference": int(k),
        "two_sided_sign_flip_p": float(p),
        "method": method,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "TRAINING-ONLY joint beta+threshold nested meta-CV for "
            "reliability-aware SVM+CNN1D fusion."
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

    gate_path = (
        root
        / "results_reliability_gate"
        / "reliability_dynamic_gate_decision.json"
    )

    required = [
        repeated_path,
        mechanism_path,
        gate_path,
    ]

    forbidden = [
        root / "train_test_data" / "X_test.npy",
        root / "train_test_data" / "y_test.npy",
        root / "train_test_data" / "test_indices.npy",
        root / "results_frozen_verification",
    ]

    print("=" * 108)
    print("RUNNING: extras/reliability_joint_gate_meta_cv.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("PURPOSE         : separate threshold gain from reliability-gate gain")
    print("METHOD A        : fixed 0.55/0.45 + threshold 0.520")
    print("METHOD B        : fixed 0.55/0.45 + TRAINING-ONLY tuned threshold")
    print("METHOD C        : reliability gate + TRAINING-ONLY tuned beta+threshold")
    print("PRIMARY TEST    : C vs B")
    print("FIXED TEST DATA : NOT READ")
    print("=" * 108)

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required training-only input missing: {path}"
            )

    print("Files this experiment WILL read:")
    for path in required:
        print(f"  {path}")

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

    with open(
        gate_path,
        "r",
        encoding="utf-8",
    ) as f:
        gate_decision = json.load(f)

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

    per_sample = (
        repeated.groupby(
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
        })
        .sort_values("source_index")
        .reset_index(drop=True)
    )

    if len(per_sample) != 800:
        raise RuntimeError(
            f"Expected 800 training samples; got {len(per_sample)}"
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

    per_sample[
        "relative_margin_reliability"
    ] = reliability

    per_sample[
        "method_a_fixed_probability"
    ] = fixed_prob

    print()
    print(f"Training samples : {len(per_sample)}")
    print(f"Training groups  : {per_sample['group_id'].nunique()}")
    print(
        f"Reliability rho  : "
        f"{mechanism.get('primary_mean_over_repeats_spearman')}"
    )
    print(
        f"Gate-only meta-CV selected betas: "
        f"{gate_decision.get('selected_beta_by_meta_fold')}"
    )
    print(
        f"Gate-only meta-CV eligible: "
        f"{gate_decision.get('eligible_for_one_frozen_fixed_test_evaluation')}"
    )
    print(
        f"Beta grid        : {BETA_GRID.min():.3f}..{BETA_GRID.max():.3f} "
        f"step 0.025"
    )
    print(
        f"Threshold grid   : {THRESHOLD_GRID.min():.3f}..{THRESHOLD_GRID.max():.3f} "
        f"step 0.005"
    )

    if args.check_only:
        print()
        print("CHECK PASSED.")
        print("No parameter search or meta-CV evaluation was run.")
        print("No fixed-test data/results were read.")
        return

    out_dir = (
        root
        / "results_reliability_joint_gate"
    )

    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # STEP 1: meta folds
    # ========================================================================
    print()
    print("=" * 108)
    print("STEP 1/5: build second-level group-aware meta folds")
    print("=" * 108)

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

    meta_fold = np.full(
        len(y),
        -1,
        dtype=np.int64,
    )

    fold_plan_rows = []

    for fold_idx, (fit_idx, heldout_idx) in enumerate(
        splits,
        start=1,
    ):
        fit_groups = set(
            groups[fit_idx]
        )
        hold_groups = set(
            groups[heldout_idx]
        )
        overlap = len(
            fit_groups.intersection(
                hold_groups
            )
        )

        if overlap != 0:
            raise RuntimeError(
                f"Meta fold {fold_idx}: group overlap={overlap}"
            )

        meta_fold[
            heldout_idx
        ] = fold_idx

        fold_plan_rows.append({
            "meta_fold": fold_idx,
            "fit_samples": len(fit_idx),
            "heldout_samples": len(heldout_idx),
            "fit_groups": len(fit_groups),
            "heldout_groups": len(hold_groups),
            "heldout_class0": int(
                np.sum(y[heldout_idx] == 0)
            ),
            "heldout_class1": int(
                np.sum(y[heldout_idx] == 1)
            ),
            "group_overlap": overlap,
        })

        print(
            f"  fold {fold_idx}: "
            f"fit={len(fit_idx)}/{len(fit_groups)} groups, "
            f"heldout={len(heldout_idx)}/{len(hold_groups)} groups"
        )

    pd.DataFrame(
        fold_plan_rows
    ).to_csv(
        out_dir / "meta_fold_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================================
    # STEP 2: nested selection
    # ========================================================================
    print()
    print("=" * 108)
    print("STEP 2/5: nested threshold-only vs beta+threshold selection")
    print("=" * 108)

    prob_b_meta = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )
    prob_c_meta = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )
    weight_c_meta = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )

    fold_setting_rows = []
    threshold_search_frames = []
    joint_search_frames = []

    for fold_idx, (fit_idx, heldout_idx) in enumerate(
        splits,
        start=1,
    ):
        # ---------------------------------------------------------------
        # Method B: fixed weights, tune threshold on meta-fit.
        # ---------------------------------------------------------------
        best_b, search_b = choose_best_threshold(
            y[fit_idx],
            fixed_prob[fit_idx],
        )
        search_b.insert(
            0,
            "meta_fold",
            fold_idx,
        )
        threshold_search_frames.append(
            search_b
        )

        threshold_b = float(
            best_b["threshold"]
        )

        prob_b_meta[
            heldout_idx
        ] = fixed_prob[
            heldout_idx
        ]

        # ---------------------------------------------------------------
        # Method C: tune beta + threshold on same meta-fit.
        # ---------------------------------------------------------------
        best_c, search_c = choose_best_beta_threshold(
            y[fit_idx],
            p_svm[fit_idx],
            p_cnn[fit_idx],
            reliability[fit_idx],
        )
        search_c.insert(
            0,
            "meta_fold",
            fold_idx,
        )
        joint_search_frames.append(
            search_c
        )

        beta_c = float(
            best_c["beta"]
        )
        threshold_c = float(
            best_c["threshold"]
        )

        p_c_hold, w_c_hold = dynamic_probability(
            p_svm[heldout_idx],
            p_cnn[heldout_idx],
            reliability[heldout_idx],
            beta_c,
        )

        prob_c_meta[
            heldout_idx
        ] = p_c_hold
        weight_c_meta[
            heldout_idx
        ] = w_c_hold

        # Heldout metrics.
        m_a = compute_metrics(
            y[heldout_idx],
            fixed_prob[heldout_idx],
            ANCHOR_THRESHOLD,
        )
        m_b = compute_metrics(
            y[heldout_idx],
            fixed_prob[heldout_idx],
            threshold_b,
        )
        m_c = compute_metrics(
            y[heldout_idx],
            p_c_hold,
            threshold_c,
        )

        fold_setting_rows.append({
            "meta_fold": fold_idx,
            "method_b_selected_threshold": threshold_b,
            "method_c_selected_beta": beta_c,
            "method_c_selected_threshold": threshold_c,
            "heldout_a_f1": m_a["f1"],
            "heldout_b_f1": m_b["f1"],
            "heldout_c_f1": m_c["f1"],
            "heldout_a_recall": m_a["recall"],
            "heldout_b_recall": m_b["recall"],
            "heldout_c_recall": m_c["recall"],
            "heldout_a_specificity": m_a["specificity"],
            "heldout_b_specificity": m_b["specificity"],
            "heldout_c_specificity": m_c["specificity"],
            "heldout_c_mean_svm_weight": float(
                np.mean(w_c_hold)
            ),
            "heldout_c_min_svm_weight": float(
                np.min(w_c_hold)
            ),
            "heldout_c_max_svm_weight": float(
                np.max(w_c_hold)
            ),
        })

        print(
            f"  fold {fold_idx}: "
            f"B threshold={threshold_b:.3f}, "
            f"C beta={beta_c:.3f}, C threshold={threshold_c:.3f} | "
            f"heldout F1 A/B/C="
            f"{m_a['f1']:.5f}/"
            f"{m_b['f1']:.5f}/"
            f"{m_c['f1']:.5f}"
        )

    if not np.isfinite(
        prob_b_meta
    ).all():
        raise RuntimeError(
            "Incomplete Method-B meta-OOF probabilities."
        )

    if not np.isfinite(
        prob_c_meta
    ).all():
        raise RuntimeError(
            "Incomplete Method-C meta-OOF probabilities."
        )

    fold_settings = pd.DataFrame(
        fold_setting_rows
    )

    fold_settings.to_csv(
        out_dir / "meta_fold_selected_settings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        threshold_search_frames,
        ignore_index=True,
    ).to_csv(
        out_dir / "meta_fold_threshold_only_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        joint_search_frames,
        ignore_index=True,
    ).to_csv(
        out_dir / "meta_fold_joint_beta_threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================================
    # STEP 3: aggregate A/B/C comparison
    # ========================================================================
    print()
    print("=" * 108)
    print("STEP 3/5: aggregate nested meta-OOF comparison")
    print("=" * 108)

    # Each heldout sample used the threshold selected for its fold.
    # Therefore create predictions explicitly per fold.
    pred_a = (
        fixed_prob >= ANCHOR_THRESHOLD
    ).astype(np.int64)

    pred_b = np.full(
        len(y),
        -1,
        dtype=np.int64,
    )

    pred_c = np.full(
        len(y),
        -1,
        dtype=np.int64,
    )

    threshold_b_per_sample = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )

    threshold_c_per_sample = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )

    beta_c_per_sample = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )

    for row in fold_settings.itertuples():
        idx = np.where(
            meta_fold == int(row.meta_fold)
        )[0]

        threshold_b_per_sample[
            idx
        ] = float(
            row.method_b_selected_threshold
        )

        threshold_c_per_sample[
            idx
        ] = float(
            row.method_c_selected_threshold
        )

        beta_c_per_sample[
            idx
        ] = float(
            row.method_c_selected_beta
        )

        pred_b[
            idx
        ] = (
            prob_b_meta[idx]
            >= float(
                row.method_b_selected_threshold
            )
        ).astype(np.int64)

        pred_c[
            idx
        ] = (
            prob_c_meta[idx]
            >= float(
                row.method_c_selected_threshold
            )
        ).astype(np.int64)

    def metrics_from_prediction(y_true, pred, prob):
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            pred,
            labels=[0, 1],
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp)
            else 0.0
        )

        return {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    pred,
                )
            ),
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
            "specificity": float(
                specificity
            ),
            "roc_auc": float(
                roc_auc_score(
                    y_true,
                    stable_prob(prob),
                )
            ),
            "average_precision": float(
                average_precision_score(
                    y_true,
                    stable_prob(prob),
                )
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    m_a = metrics_from_prediction(
        y,
        pred_a,
        fixed_prob,
    )
    m_b = metrics_from_prediction(
        y,
        pred_b,
        prob_b_meta,
    )
    m_c = metrics_from_prediction(
        y,
        pred_c,
        prob_c_meta,
    )

    comparison = pd.DataFrame([
        {
            "method": "A_original_fixed",
            "description": "0.55/0.45, threshold=0.520",
            **m_a,
        },
        {
            "method": "B_fixed_weights_tuned_threshold",
            "description": "0.55/0.45, threshold selected inside each meta-fit",
            **m_b,
        },
        {
            "method": "C_reliability_dynamic_beta_threshold",
            "description": "beta+threshold selected inside each meta-fit",
            **m_c,
        },
    ])

    comparison.to_csv(
        out_dir / "nested_meta_oof_abc_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        comparison[
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

    per_sample["meta_fold"] = meta_fold
    per_sample["method_a_probability"] = fixed_prob
    per_sample["method_a_prediction"] = pred_a
    per_sample["method_b_probability"] = prob_b_meta
    per_sample["method_b_threshold"] = threshold_b_per_sample
    per_sample["method_b_prediction"] = pred_b
    per_sample["method_c_probability"] = prob_c_meta
    per_sample["method_c_threshold"] = threshold_c_per_sample
    per_sample["method_c_beta"] = beta_c_per_sample
    per_sample["method_c_svm_weight"] = weight_c_meta
    per_sample["method_c_prediction"] = pred_c

    per_sample.to_csv(
        out_dir / "nested_meta_oof_predictions_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================================
    # STEP 4: paired stats, especially C vs B
    # ========================================================================
    print()
    print("=" * 108)
    print("STEP 4/5: paired training-only statistics")
    print("=" * 108)

    mcnemar_rows = []

    for a_name, a_pred, b_name, b_pred in [
        (
            "A_original_fixed",
            pred_a,
            "B_fixed_weights_tuned_threshold",
            pred_b,
        ),
        (
            "B_fixed_weights_tuned_threshold",
            pred_b,
            "C_reliability_dynamic_beta_threshold",
            pred_c,
        ),
        (
            "A_original_fixed",
            pred_a,
            "C_reliability_dynamic_beta_threshold",
            pred_c,
        ),
    ]:
        stat = exact_mcnemar(
            y,
            a_pred,
            b_pred,
        )

        mcnemar_rows.append({
            "method_a": a_name,
            "method_b": b_name,
            **stat,
        })

    mcnemar_df = pd.DataFrame(
        mcnemar_rows
    )

    mcnemar_df.to_csv(
        out_dir / "nested_meta_oof_mcnemar_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # For bootstrap with per-sample varying thresholds, use the already-created
    # predictions and recompute Accuracy/Recall/Specificity/F1 from predictions.
    stats_frame = per_sample[
        [
            "group_id",
            "true_label",
            "method_b_prediction",
            "method_c_prediction",
        ]
    ].copy()

    # Custom group bootstrap from prediction columns.
    groups_unique = stats_frame[
        "group_id"
    ].astype(str).unique()

    group_to_idx = {
        g: stats_frame.index[
            stats_frame["group_id"].astype(str) == g
        ].to_numpy(dtype=np.int64)
        for g in groups_unique
    }

    yy_all = stats_frame[
        "true_label"
    ].to_numpy(dtype=np.int64)

    pb_all = stats_frame[
        "method_b_prediction"
    ].to_numpy(dtype=np.int64)

    pc_all = stats_frame[
        "method_c_prediction"
    ].to_numpy(dtype=np.int64)

    def metric_from_pred(y_true, pred):
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            pred,
            labels=[0, 1],
        ).ravel()

        acc = (tn + tp) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": acc,
            "f1": f1,
            "recall": recall,
            "specificity": specificity,
        }

    obs_b = metric_from_pred(
        yy_all,
        pb_all,
    )
    obs_c = metric_from_pred(
        yy_all,
        pc_all,
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    metrics = [
        "accuracy",
        "f1",
        "recall",
        "specificity",
    ]

    boot = {
        metric: np.empty(
            BOOTSTRAP_REPS,
            dtype=np.float64,
        )
        for metric in metrics
    }

    for r in range(
        BOOTSTRAP_REPS
    ):
        sampled_groups = rng.choice(
            groups_unique,
            size=len(groups_unique),
            replace=True,
        )

        idx = np.concatenate(
            [
                group_to_idx[g]
                for g in sampled_groups
            ]
        )

        mb = metric_from_pred(
            yy_all[idx],
            pb_all[idx],
        )

        mc = metric_from_pred(
            yy_all[idx],
            pc_all[idx],
        )

        for metric in metrics:
            boot[metric][r] = (
                mc[metric]
                - mb[metric]
            )

    bootstrap_rows = []

    for metric in metrics:
        low, high = np.quantile(
            boot[metric],
            [0.025, 0.975],
        )

        bootstrap_rows.append({
            "comparison": "C_dynamic_minus_B_threshold_only_fixed",
            "metric": metric,
            "observed_difference": (
                obs_c[metric]
                - obs_b[metric]
            ),
            "bootstrap_repetitions": BOOTSTRAP_REPS,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "ci_excludes_zero": bool(
                low > 0 or high < 0
            ),
        })

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    bootstrap_df.to_csv(
        out_dir / "group_bootstrap_c_minus_b.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sign_flip = exact_group_sign_flip(
        stats_frame,
        "method_b_prediction",
        "method_c_prediction",
    )

    with open(
        out_dir / "group_sign_flip_c_vs_b.json",
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
    print(
        mcnemar_df.to_string(
            index=False
        )
    )

    print()
    print("C - B group bootstrap:")
    print(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda v: f"{v:.6f}",
        )
    )

    print()
    print("C vs B group sign-flip:")
    print(
        json.dumps(
            sign_flip,
            ensure_ascii=False,
            indent=2,
        )
    )

    # ========================================================================
    # STEP 5: freeze decision and full-training parameter selection
    # ========================================================================
    print()
    print("=" * 108)
    print("STEP 5/5: training-only freeze decision")
    print("=" * 108)

    c_better_than_b = (
        metrics_vector_better(
            m_c,
            m_b,
        )
    )

    nonzero_beta_folds = int(
        np.sum(
            fold_settings[
                "method_c_selected_beta"
            ].to_numpy(dtype=np.float64)
            > 1e-12
        )
    )

    heldout_f1_nonworse_folds = int(
        np.sum(
            fold_settings[
                "heldout_c_f1"
            ].to_numpy(dtype=np.float64)
            >=
            fold_settings[
                "heldout_b_f1"
            ].to_numpy(dtype=np.float64)
            - 1e-12
        )
    )

    # Full 800 training OOF selection for Method B and C, ONLY if we later freeze.
    best_b_full, search_b_full = choose_best_threshold(
        y,
        fixed_prob,
    )

    best_c_full, search_c_full = choose_best_beta_threshold(
        y,
        p_svm,
        p_cnn,
        reliability,
    )

    search_b_full.to_csv(
        out_dir / "full_training_threshold_only_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    search_c_full.to_csv(
        out_dir / "full_training_joint_beta_threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eligible = bool(
        c_better_than_b
        and nonzero_beta_folds
        >= MIN_NONZERO_BETA_FOLDS
        and heldout_f1_nonworse_folds
        >= MIN_HELDOUT_F1_NONWORSE_FOLDS
        and float(m_c["recall"])
        >= float(m_b["recall"]) - 1e-12
    )

    decision = {
        "script_version": SCRIPT_VERSION,
        "fixed_test_data_read": False,
        "scientific_question": (
            "Does reliability-aware sample-wise weighting improve over "
            "a fixed 0.55/0.45 fusion after allowing the fixed fusion "
            "to retune its threshold using training-only meta-fit data?"
        ),
        "method_a": {
            "name": "original_fixed",
            "svm_weight": ANCHOR_SVM_WEIGHT,
            "cnn1d_weight": ANCHOR_CNN1D_WEIGHT,
            "threshold": ANCHOR_THRESHOLD,
            "nested_meta_oof_metrics": m_a,
        },
        "method_b": {
            "name": "fixed_weights_threshold_tuned",
            "svm_weight": ANCHOR_SVM_WEIGHT,
            "cnn1d_weight": ANCHOR_CNN1D_WEIGHT,
            "nested_meta_oof_metrics": m_b,
            "full_training_selected_threshold": float(
                best_b_full["threshold"]
            ),
        },
        "method_c": {
            "name": "reliability_dynamic_beta_threshold",
            "gate_formula": (
                "w_svm=sigmoid(logit(0.55)+beta*r), "
                "r=2*abs(p_svm-0.5)-2*abs(p_cnn1d-0.5)"
            ),
            "strict_ablation": (
                "beta=0 reproduces fixed 0.55/0.45 weights"
            ),
            "nested_meta_oof_metrics": m_c,
            "selected_beta_by_meta_fold": [
                float(x)
                for x in fold_settings[
                    "method_c_selected_beta"
                ].tolist()
            ],
            "selected_threshold_by_meta_fold": [
                float(x)
                for x in fold_settings[
                    "method_c_selected_threshold"
                ].tolist()
            ],
            "full_training_selected_beta": float(
                best_c_full["beta"]
            ),
            "full_training_selected_threshold": float(
                best_c_full["threshold"]
            ),
        },
        "primary_comparison": "method_c_vs_method_b",
        "dynamic_lexicographically_better_than_threshold_only_fixed": (
            c_better_than_b
        ),
        "nonzero_beta_fold_count": (
            nonzero_beta_folds
        ),
        "heldout_f1_dynamic_nonworse_fold_count": (
            heldout_f1_nonworse_folds
        ),
        "minimum_nonzero_beta_folds": (
            MIN_NONZERO_BETA_FOLDS
        ),
        "minimum_f1_nonworse_folds": (
            MIN_HELDOUT_F1_NONWORSE_FOLDS
        ),
        "eligible_for_one_frozen_fixed_test_evaluation": (
            eligible
        ),
        "if_eligible_frozen_parameters": (
            {
                "beta": float(
                    best_c_full["beta"]
                ),
                "threshold": float(
                    best_c_full["threshold"]
                ),
                "anchor_svm_weight": (
                    ANCHOR_SVM_WEIGHT
                ),
                "anchor_cnn1d_weight": (
                    ANCHOR_CNN1D_WEIGHT
                ),
            }
            if eligible
            else None
        ),
        "paired_sample_mcnemar_c_vs_b": (
            mcnemar_df[
                (
                    mcnemar_df["method_a"]
                    == "B_fixed_weights_tuned_threshold"
                )
                &
                (
                    mcnemar_df["method_b"]
                    == "C_reliability_dynamic_beta_threshold"
                )
            ]
            .iloc[0]
            .to_dict()
        ),
        "group_sign_flip_c_vs_b": (
            sign_flip
        ),
        "note": (
            "If eligible is true, the full-training beta and threshold "
            "are frozen before any fixed-test evaluation. Fixed-test "
            "results must not be used to revise beta, threshold, or "
            "the reliability definition."
        ),
    }

    with open(
        out_dir / "reliability_joint_gate_decision.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            decision,
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda x: (
                int(x)
                if isinstance(x, np.integer)
                else float(x)
                if isinstance(x, np.floating)
                else x
            ),
        )

    print()
    print("Joint-gate meta-CV decision summary:")
    print(
        f"  Method A F1/Recall/Spec : "
        f"{m_a['f1']:.5f}/"
        f"{m_a['recall']:.5f}/"
        f"{m_a['specificity']:.5f}"
    )
    print(
        f"  Method B F1/Recall/Spec : "
        f"{m_b['f1']:.5f}/"
        f"{m_b['recall']:.5f}/"
        f"{m_b['specificity']:.5f}"
    )
    print(
        f"  Method C F1/Recall/Spec : "
        f"{m_c['f1']:.5f}/"
        f"{m_c['recall']:.5f}/"
        f"{m_c['specificity']:.5f}"
    )
    print(
        f"  C > B lexicographically: "
        f"{c_better_than_b}"
    )
    print(
        f"  Nonzero-beta meta folds : "
        f"{nonzero_beta_folds}/{META_FOLDS}"
    )
    print(
        f"  C F1 >= B F1 folds      : "
        f"{heldout_f1_nonworse_folds}/{META_FOLDS}"
    )
    print(
        f"  Full-training B threshold: "
        f"{float(best_b_full['threshold']):.3f}"
    )
    print(
        f"  Full-training C beta/thr : "
        f"{float(best_c_full['beta']):.3f}/"
        f"{float(best_c_full['threshold']):.3f}"
    )
    print(
        f"  Eligible for frozen test : "
        f"{eligible}"
    )

    print()
    print("=" * 108)
    print("TRAINING-ONLY JOINT RELIABILITY-GATE EXPERIMENT FINISHED")
    print("=" * 108)
    print(f"Outputs: {out_dir}")
    print("Most important files:")
    print(
        f"  {out_dir / 'reliability_joint_gate_decision.json'}"
    )
    print(
        f"  {out_dir / 'nested_meta_oof_abc_comparison.csv'}"
    )
    print(
        f"  {out_dir / 'meta_fold_selected_settings.csv'}"
    )
    print(
        f"  {out_dir / 'group_bootstrap_c_minus_b.csv'}"
    )
    print(
        f"  {out_dir / 'nested_meta_oof_mcnemar_abc.csv'}"
    )
    print()
    print("No fixed-test-set metrics were computed.")


if __name__ == "__main__":
    main()
