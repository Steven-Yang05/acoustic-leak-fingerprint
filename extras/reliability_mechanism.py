
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_VERSION = "reliability-mechanism-svm-cnn1d-v1"

ANCHOR_SVM_WEIGHT = 0.55
ANCHOR_CNN1D_WEIGHT = 0.45
ANCHOR_THRESHOLD = 0.520
WEIGHT_GRID = np.round(np.arange(0.00, 1.0001, 0.05), 10)

MIN_PRIMARY_RHO = 0.10
MIN_DISAGREEMENT_SELECTION_ACCURACY = 0.55
MIN_HIGH_MINUS_LOW_SVM_WEIGHT = 0.10


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def stable_prob(p):
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, 1e-7, 1.0 - 1e-7)


def per_sample_log_loss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = stable_prob(p)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def margin_confidence(p):
    p = stable_prob(p)
    return 2.0 * np.abs(p - 0.5)


def entropy_certainty(p):
    p = stable_prob(p)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return 1.0 - h / np.log(2.0)


def logit_magnitude(p):
    p = stable_prob(p)
    return np.abs(np.log(p / (1.0 - p)))


def spearman_rho(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(rx) <= 0 or np.std(ry) <= 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = stable_prob(p)
    pred = (p >= threshold).astype(np.int64)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    accuracy = (tn + tp) / max(len(y), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
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


def rank_weight_search(df):
    return (
        df.sort_values(
            by=["f1", "recall", "specificity", "accuracy"],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def add_reliability_columns(df):
    out = df.copy()
    y = out["true_label"].to_numpy(dtype=np.int64)
    p_svm = stable_prob(
        out["rbf_svm_calibrated_probability"].to_numpy(dtype=np.float64)
    )
    p_cnn = stable_prob(
        out["cnn1d_probability"].to_numpy(dtype=np.float64)
    )

    out["svm_margin_confidence"] = margin_confidence(p_svm)
    out["cnn1d_margin_confidence"] = margin_confidence(p_cnn)
    out["relative_margin_reliability"] = (
        out["svm_margin_confidence"] - out["cnn1d_margin_confidence"]
    )

    out["svm_entropy_certainty"] = entropy_certainty(p_svm)
    out["cnn1d_entropy_certainty"] = entropy_certainty(p_cnn)
    out["relative_entropy_reliability"] = (
        out["svm_entropy_certainty"] - out["cnn1d_entropy_certainty"]
    )

    out["svm_logit_magnitude"] = logit_magnitude(p_svm)
    out["cnn1d_logit_magnitude"] = logit_magnitude(p_cnn)
    out["relative_logit_reliability"] = (
        out["svm_logit_magnitude"] - out["cnn1d_logit_magnitude"]
    )

    svm_loss = per_sample_log_loss(y, p_svm)
    cnn_loss = per_sample_log_loss(y, p_cnn)
    out["svm_logloss"] = svm_loss
    out["cnn1d_logloss"] = cnn_loss
    out["svm_relative_logloss_advantage"] = cnn_loss - svm_loss

    out["svm_prediction_05"] = (p_svm >= 0.5).astype(np.int64)
    out["cnn1d_prediction_05"] = (p_cnn >= 0.5).astype(np.int64)
    out["experts_disagree"] = (
        out["svm_prediction_05"] != out["cnn1d_prediction_05"]
    )

    out["more_reliable_expert"] = np.where(
        out["relative_margin_reliability"] >= 0,
        "svm",
        "cnn1d",
    )

    svm_correct = out["svm_prediction_05"].to_numpy(dtype=np.int64) == y
    cnn_correct = out["cnn1d_prediction_05"].to_numpy(dtype=np.int64) == y
    out["more_reliable_expert_correct"] = np.where(
        out["more_reliable_expert"].to_numpy() == "svm",
        svm_correct,
        cnn_correct,
    ).astype(bool)

    return out


def disagreement_summary(df, scope, repeat_seed):
    sub = df[df["experts_disagree"]].copy()
    if len(sub) == 0:
        return {
            "scope": scope,
            "repeat_seed": repeat_seed,
            "disagreement_rows": 0,
            "more_reliable_expert_accuracy": float("nan"),
            "svm_more_reliable_rows": 0,
            "cnn1d_more_reliable_rows": 0,
        }
    return {
        "scope": scope,
        "repeat_seed": repeat_seed,
        "disagreement_rows": int(len(sub)),
        "more_reliable_expert_accuracy": float(
            sub["more_reliable_expert_correct"].mean()
        ),
        "svm_more_reliable_rows": int(
            np.sum(sub["more_reliable_expert"] == "svm")
        ),
        "cnn1d_more_reliable_rows": int(
            np.sum(sub["more_reliable_expert"] == "cnn1d")
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "TRAINING-ONLY reliability mechanism audit for calibrated "
            "RBF-SVM + 1D CNN. No dynamic gate is tuned."
        )
    )
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_root()

    repeated_oof_path = (
        root
        / "results_stability"
        / "repeated_training_oof_probabilities.csv"
    )

    forbidden = [
        root / "train_test_data" / "X_test.npy",
        root / "train_test_data" / "y_test.npy",
        root / "train_test_data" / "test_indices.npy",
        root / "results_frozen_verification",
    ]

    print("=" * 100)
    print("RUNNING: extras/reliability_mechanism.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("PURPOSE         : validate expert reliability before dynamic gating")
    print("EXPERTS         : calibrated RBF-SVM vs 1D CNN")
    print("MAIN RELIABILITY: 2*|p_svm-0.5| - 2*|p_cnn-0.5|")
    print("DYNAMIC GATE    : NOT tuned in this script")
    print("FIXED TEST DATA : NOT READ")
    print("=" * 100)

    if not repeated_oof_path.exists():
        raise FileNotFoundError(
            f"Required training-only repeated OOF file missing: "
            f"{repeated_oof_path}"
        )

    print("File this audit WILL read:")
    print(f"  {repeated_oof_path}")
    print()
    print("Test-side files/results this audit is explicitly designed NOT to read:")
    for path in forbidden:
        print(f"  {path}")

    df = pd.read_csv(repeated_oof_path, encoding="utf-8-sig")

    required_cols = {
        "repeat_seed",
        "source_index",
        "group_id",
        "true_label",
        "oof_fold",
        "rbf_svm_calibrated_probability",
        "cnn1d_probability",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Repeated OOF CSV missing required columns: "
            + ", ".join(sorted(missing))
        )

    repeat_seeds = sorted(df["repeat_seed"].unique().tolist())
    source_count = df["source_index"].nunique()
    group_count = df["group_id"].nunique()

    if source_count != 800:
        raise RuntimeError(f"Expected 800 training samples; got {source_count}")
    if group_count != 370:
        raise RuntimeError(f"Expected 370 training groups; got {group_count}")

    expected_rows = source_count * len(repeat_seeds)
    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} repeated OOF rows; got {len(df)}"
        )

    canonical_sources = set(df["source_index"].unique().tolist())
    for seed, sub in df.groupby("repeat_seed"):
        if len(sub) != source_count:
            raise RuntimeError(
                f"repeat_seed={seed}: expected {source_count} rows, got {len(sub)}"
            )
        if set(sub["source_index"].tolist()) != canonical_sources:
            raise RuntimeError(
                f"repeat_seed={seed}: source_index set mismatch."
            )

    print()
    print(f"Training samples : {source_count}")
    print(f"Training groups  : {group_count}")
    print(f"Repeat seeds     : {repeat_seeds}")
    print(f"Repeated OOF rows: {len(df)}")
    print(
        f"Frozen anchor    : SVM={ANCHOR_SVM_WEIGHT:.2f}, "
        f"CNN1D={ANCHOR_CNN1D_WEIGHT:.2f}, "
        f"threshold={ANCHOR_THRESHOLD:.3f}"
    )

    if args.check_only:
        print()
        print("CHECK PASSED.")
        print("No mechanism calculation or dynamic-gate tuning was run.")
        print("No fixed-test data/results were read.")
        return

    out_dir = root / "results_reliability_mechanism"
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print("STEP 1/5: construct repeated OOF reliability evidence")
    print("=" * 100)

    work = add_reliability_columns(df)
    work.to_csv(
        out_dir / "repeated_oof_reliability_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 100)
    print("STEP 2/5: reliability vs relative expert advantage")
    print("=" * 100)

    reliability_cols = [
        "relative_margin_reliability",
        "relative_entropy_reliability",
        "relative_logit_reliability",
    ]
    corr_rows = []

    for rel_col in reliability_cols:
        corr_rows.append({
            "scope": "all_repeated_oof_rows",
            "repeat_seed": "all",
            "level": "sample-repeat",
            "reliability_proxy": rel_col,
            "spearman_rho_reliability_vs_svm_advantage": spearman_rho(
                work[rel_col].to_numpy(),
                work["svm_relative_logloss_advantage"].to_numpy(),
            ),
        })

    for seed, sub in work.groupby("repeat_seed"):
        for rel_col in reliability_cols:
            corr_rows.append({
                "scope": "single_repeat",
                "repeat_seed": int(seed),
                "level": "sample",
                "reliability_proxy": rel_col,
                "spearman_rho_reliability_vs_svm_advantage": spearman_rho(
                    sub[rel_col].to_numpy(),
                    sub["svm_relative_logloss_advantage"].to_numpy(),
                ),
            })

    per_sample_prob = (
        work.groupby(
            ["source_index", "group_id", "true_label"],
            as_index=False,
        )
        .agg({
            "rbf_svm_calibrated_probability": "mean",
            "cnn1d_probability": "mean",
        })
    )
    per_sample = add_reliability_columns(per_sample_prob)
    per_sample.to_csv(
        out_dir / "mean_reliability_evidence_per_training_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for rel_col in reliability_cols:
        corr_rows.append({
            "scope": "mean_over_repeats",
            "repeat_seed": "mean",
            "level": "sample",
            "reliability_proxy": rel_col,
            "spearman_rho_reliability_vs_svm_advantage": spearman_rho(
                per_sample[rel_col].to_numpy(),
                per_sample["svm_relative_logloss_advantage"].to_numpy(),
            ),
        })

    per_group = (
        per_sample.groupby("group_id", as_index=False)
        .agg({
            "relative_margin_reliability": "mean",
            "relative_entropy_reliability": "mean",
            "relative_logit_reliability": "mean",
            "svm_relative_logloss_advantage": "mean",
        })
    )
    per_group.to_csv(
        out_dir / "mean_reliability_evidence_per_group.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for rel_col in reliability_cols:
        corr_rows.append({
            "scope": "group_macro",
            "repeat_seed": "group",
            "level": "group",
            "reliability_proxy": rel_col,
            "spearman_rho_reliability_vs_svm_advantage": spearman_rho(
                per_group[rel_col].to_numpy(),
                per_group["svm_relative_logloss_advantage"].to_numpy(),
            ),
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(
        out_dir / "reliability_vs_relative_advantage_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        corr_df[
            corr_df["reliability_proxy"]
            == "relative_margin_reliability"
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    sensitivity_rows = []
    for i in range(len(reliability_cols)):
        for j in range(i + 1, len(reliability_cols)):
            a = reliability_cols[i]
            b = reliability_cols[j]
            sensitivity_rows.append({
                "proxy_a": a,
                "proxy_b": b,
                "spearman_rho_between_reliability_proxies": spearman_rho(
                    per_sample[a].to_numpy(),
                    per_sample[b].to_numpy(),
                ),
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(
        out_dir / "reliability_proxy_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 100)
    print("STEP 3/5: disagreement reliability test")
    print("=" * 100)

    disagreement_rows = []
    for seed, sub in work.groupby("repeat_seed"):
        disagreement_rows.append(
            disagreement_summary(
                sub,
                scope="single_repeat",
                repeat_seed=int(seed),
            )
        )

    disagreement_rows.append(
        disagreement_summary(
            work,
            scope="all_repeated_oof_rows",
            repeat_seed="all",
        )
    )
    disagreement_rows.append(
        disagreement_summary(
            per_sample,
            scope="mean_over_repeats",
            repeat_seed="mean",
        )
    )

    disagreement_df = pd.DataFrame(disagreement_rows)
    disagreement_df.to_csv(
        out_dir / "disagreement_more_reliable_expert_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        disagreement_df.to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    print()
    print("=" * 100)
    print("STEP 4/5: reliability-bin weight evidence")
    print("=" * 100)
    print(
        "Threshold remains frozen at 0.520. Only the SVM/CNN1D "
        "mixture weight is explored inside TRAINING-ONLY bins."
    )

    per_sample["reliability_bin"] = pd.qcut(
        per_sample["relative_margin_reliability"],
        q=3,
        labels=[
            "cnn_more_reliable",
            "similar_reliability",
            "svm_more_reliable",
        ],
        duplicates="drop",
    ).astype(str)

    bin_search_rows = []
    bin_best_rows = []
    bin_component_rows = []

    for bin_name in [
        "cnn_more_reliable",
        "similar_reliability",
        "svm_more_reliable",
    ]:
        sub = per_sample[
            per_sample["reliability_bin"] == bin_name
        ].copy()

        y_bin = sub["true_label"].to_numpy(dtype=np.int64)
        p_svm = sub[
            "rbf_svm_calibrated_probability"
        ].to_numpy(dtype=np.float64)
        p_cnn = sub[
            "cnn1d_probability"
        ].to_numpy(dtype=np.float64)

        for method, prob, threshold in [
            ("calibrated_svm", p_svm, 0.5),
            ("cnn1d", p_cnn, 0.5),
            (
                "fixed_anchor_055_045",
                ANCHOR_SVM_WEIGHT * p_svm
                + ANCHOR_CNN1D_WEIGHT * p_cnn,
                ANCHOR_THRESHOLD,
            ),
        ]:
            bin_component_rows.append({
                "reliability_bin": bin_name,
                "method": method,
                "sample_count": int(len(sub)),
                "mean_relative_margin_reliability": float(
                    sub["relative_margin_reliability"].mean()
                ),
                **binary_metrics(y_bin, prob, threshold),
            })

        local_rows = []
        for svm_weight in WEIGHT_GRID:
            cnn_weight = 1.0 - svm_weight
            p_mix = svm_weight * p_svm + cnn_weight * p_cnn
            row = {
                "reliability_bin": bin_name,
                "sample_count": int(len(sub)),
                "svm_weight": float(svm_weight),
                "cnn1d_weight": float(cnn_weight),
                "fixed_threshold": ANCHOR_THRESHOLD,
                **binary_metrics(
                    y_bin,
                    p_mix,
                    ANCHOR_THRESHOLD,
                ),
            }
            local_rows.append(row)
            bin_search_rows.append(row)

        best = rank_weight_search(
            pd.DataFrame(local_rows)
        ).iloc[0].to_dict()
        bin_best_rows.append(best)

    bin_search_df = pd.DataFrame(bin_search_rows)
    bin_best_df = pd.DataFrame(bin_best_rows)
    bin_component_df = pd.DataFrame(bin_component_rows)

    bin_search_df.to_csv(
        out_dir / "reliability_bin_fixed_threshold_weight_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bin_best_df.to_csv(
        out_dir / "reliability_bin_best_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bin_component_df.to_csv(
        out_dir / "reliability_bin_component_and_anchor_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        bin_best_df[
            [
                "reliability_bin",
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

    print()
    print("=" * 100)
    print("STEP 5/5: mechanism-screen decision")
    print("=" * 100)

    primary_rho = float(
        corr_df[
            (corr_df["scope"] == "mean_over_repeats")
            & (
                corr_df["reliability_proxy"]
                == "relative_margin_reliability"
            )
        ].iloc[0][
            "spearman_rho_reliability_vs_svm_advantage"
        ]
    )

    group_rho = float(
        corr_df[
            (corr_df["scope"] == "group_macro")
            & (
                corr_df["reliability_proxy"]
                == "relative_margin_reliability"
            )
        ].iloc[0][
            "spearman_rho_reliability_vs_svm_advantage"
        ]
    )

    repeat_main = corr_df[
        (corr_df["scope"] == "single_repeat")
        & (
            corr_df["reliability_proxy"]
            == "relative_margin_reliability"
        )
    ]
    repeat_positive_count = int(
        np.sum(
            repeat_main[
                "spearman_rho_reliability_vs_svm_advantage"
            ].to_numpy(dtype=np.float64) > 0
        )
    )

    mean_proxy_rows = corr_df[
        (corr_df["scope"] == "mean_over_repeats")
        & (corr_df["reliability_proxy"].isin(reliability_cols))
    ]
    proxy_rhos = mean_proxy_rows[
        "spearman_rho_reliability_vs_svm_advantage"
    ].to_numpy(dtype=np.float64)
    all_proxy_rhos_positive = bool(np.all(proxy_rhos > 0))

    disagreement_mean = disagreement_df[
        disagreement_df["scope"] == "mean_over_repeats"
    ].iloc[0]
    disagreement_accuracy = float(
        disagreement_mean["more_reliable_expert_accuracy"]
    )
    disagreement_count = int(
        disagreement_mean["disagreement_rows"]
    )

    best_by_bin = bin_best_df.set_index("reliability_bin")
    w_low = float(
        best_by_bin.loc["cnn_more_reliable", "svm_weight"]
    )
    w_mid = float(
        best_by_bin.loc["similar_reliability", "svm_weight"]
    )
    w_high = float(
        best_by_bin.loc["svm_more_reliable", "svm_weight"]
    )

    monotonic_weights = bool(w_low <= w_mid <= w_high)
    high_minus_low = w_high - w_low

    min_proxy_agreement = float(
        sensitivity_df[
            "spearman_rho_between_reliability_proxies"
        ].min()
    )

    eligible = bool(
        primary_rho >= MIN_PRIMARY_RHO
        and group_rho > 0
        and repeat_positive_count >= 4
        and all_proxy_rhos_positive
        and disagreement_count > 0
        and disagreement_accuracy
        >= MIN_DISAGREEMENT_SELECTION_ACCURACY
        and monotonic_weights
        and high_minus_low
        >= MIN_HIGH_MINUS_LOW_SVM_WEIGHT
    )

    decision = {
        "script_version": SCRIPT_VERSION,
        "fixed_test_data_read": False,
        "dynamic_gate_tuned": False,
        "experts": [
            "nested-Platt-calibrated RBF-SVM",
            "1D CNN",
        ],
        "training_samples": int(source_count),
        "training_groups": int(group_count),
        "repeat_seeds": [int(x) for x in repeat_seeds],
        "main_reliability_definition": (
            "2*abs(p_svm-0.5) - 2*abs(p_cnn1d-0.5)"
        ),
        "relative_advantage_definition": (
            "CNN1D per-sample log-loss minus calibrated-SVM "
            "per-sample log-loss; positive means SVM has lower loss"
        ),
        "primary_mean_over_repeats_spearman": primary_rho,
        "group_macro_spearman": group_rho,
        "positive_direction_repeat_count": repeat_positive_count,
        "repeat_count": int(len(repeat_main)),
        "all_three_reliability_proxies_positive_direction": (
            all_proxy_rhos_positive
        ),
        "minimum_pairwise_spearman_between_reliability_proxies": (
            min_proxy_agreement
        ),
        "disagreement_rows_mean_over_repeats": disagreement_count,
        "more_reliable_expert_accuracy_on_disagreements": (
            disagreement_accuracy
        ),
        "best_svm_weight_cnn_more_reliable_bin": w_low,
        "best_svm_weight_similar_reliability_bin": w_mid,
        "best_svm_weight_svm_more_reliable_bin": w_high,
        "reliability_bin_weights_monotonic_non_decreasing": (
            monotonic_weights
        ),
        "svm_weight_high_minus_low_reliability_bin": (
            high_minus_low
        ),
        "screening_thresholds": {
            "minimum_primary_spearman": MIN_PRIMARY_RHO,
            "minimum_positive_repeat_count": 4,
            "minimum_disagreement_selection_accuracy": (
                MIN_DISAGREEMENT_SELECTION_ACCURACY
            ),
            "minimum_high_minus_low_svm_weight": (
                MIN_HIGH_MINUS_LOW_SVM_WEIGHT
            ),
        },
        "eligible_for_stage2_reliability_aware_dynamic_gate": (
            eligible
        ),
        "frozen_fixed_fusion_anchor": {
            "svm_weight": ANCHOR_SVM_WEIGHT,
            "cnn1d_weight": ANCHOR_CNN1D_WEIGHT,
            "threshold": ANCHOR_THRESHOLD,
        },
        "screening_note": (
            "Eligibility is a predeclared mechanism/engineering screen, "
            "not a statistical-significance claim."
        ),
    }

    with open(
        out_dir / "reliability_mechanism_decision.json",
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
    print("Reliability mechanism summary:")
    print(f"  Primary mean-repeat rho     : {primary_rho:+.5f}")
    print(f"  Group-macro rho             : {group_rho:+.5f}")
    print(
        f"  Positive repeat count       : "
        f"{repeat_positive_count}/{len(repeat_main)}"
    )
    print(
        f"  Disagreement selector acc   : "
        f"{disagreement_accuracy:.5f} (n={disagreement_count})"
    )
    print(
        f"  Best SVM weights low/mid/high reliability: "
        f"{w_low:.2f} / {w_mid:.2f} / {w_high:.2f}"
    )
    print(f"  Monotonic bin weights       : {monotonic_weights}")
    print(f"  Eligible for dynamic gate   : {eligible}")

    print()
    print("=" * 100)
    print("TRAINING-ONLY RELIABILITY MECHANISM AUDIT FINISHED")
    print("=" * 100)
    print(f"Outputs: {out_dir}")
    print("Most important files:")
    print(f"  {out_dir / 'reliability_mechanism_decision.json'}")
    print(
        f"  {out_dir / 'reliability_vs_relative_advantage_correlations.csv'}"
    )
    print(
        f"  {out_dir / 'disagreement_more_reliable_expert_accuracy.csv'}"
    )
    print(f"  {out_dir / 'reliability_bin_best_weights.csv'}")
    print(f"  {out_dir / 'reliability_proxy_sensitivity.csv'}")
    print()
    print("No fixed-test-set metrics were computed.")
    print("No dynamic-gate parameter was tuned.")


if __name__ == "__main__":
    main()
