from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)


SCRIPT_VERSION = "verification-comparison-stats-v1"

# ---------------------------------------------------------------------
# FROZEN ANALYSIS INPUTS
# ---------------------------------------------------------------------
# No model is retrained here. All inputs are the recorded per-sample
# probabilities of the frozen 200-sample verification set:
#   P0 (ours): results_frozen_verification/fixed_test_predictions.csv
#              -> fusion_probability, frozen threshold 0.520
#   P1..P5   : results_fusion_pairs_verification/
#              verification_predictions.csv -> pX_fusion_probability
# All thresholds are the frozen stability_selection.py median values.
# ---------------------------------------------------------------------

SEED = 42
N_BOOTSTRAP = 5000
ECE_BINS = 15

PAIRS = {
    "P0": {"threshold": 0.520, "name": "cal. RBF-SVM + 1D CNN (ours)"},
    "P1": {"threshold": 0.505, "name": "cal. RBF-SVM + 2D CNN"},
    "P2": {"threshold": 0.490, "name": "MLP + 2D CNN"},
    "P3": {"threshold": 0.550, "name": "Random Forest + 2D CNN"},
    "P4": {"threshold": 0.475, "name": "MLP + 1D CNN"},
    "P5": {"threshold": 0.505, "name": "Random Forest + CRNN"},
}


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def exact_mcnemar(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    """
    Exact two-sided McNemar p-value using the binomial distribution.
    pred_a is the rival, pred_b is ours (P0). No scipy dependency.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    n = a_correct_b_wrong + a_wrong_b_correct

    if n == 0:
        p_value = 1.0
    else:
        k = min(a_correct_b_wrong, a_wrong_b_correct)
        tail = sum(
            math.comb(n, i)
            for i in range(0, k + 1)
        ) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)

    return {
        "rival_correct_ours_wrong": a_correct_b_wrong,
        "rival_wrong_ours_correct": a_wrong_b_correct,
        "discordant_total": n,
        "exact_two_sided_p": float(p_value),
    }


def expected_calibration_error(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_bins: int = ECE_BINS,
) -> float:
    """
    Standard ECE with equal-width bins over [0, 1]:
      ECE = sum_b (n_b / N) * |acc_b - conf_b|
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-inclusive last edge so p=1.0 falls into the final bin.
    bin_idx = np.clip(
        np.digitize(prob, edges[1:-1], right=False),
        0,
        n_bins - 1,
    )

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        acc_b = float(np.mean(y_true[mask]))
        conf_b = float(np.mean(prob[mask]))
        ece += (n_b / n) * abs(acc_b - conf_b)
    return float(ece)


def count_fp_fn(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> tuple[int, int, float, float]:
    pred = (np.asarray(prob) >= threshold).astype(np.int64)
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    precision = float(
        precision_score(y_true, pred, pos_label=1, zero_division=0)
    )
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return fp, fn, precision, float(specificity)


def group_bootstrap_delta(
    y_true: np.ndarray,
    groups: np.ndarray,
    prob_ours: np.ndarray,
    prob_rival: np.ndarray,
    threshold_ours: float,
    threshold_rival: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """
    Cluster bootstrap over whole groups (group_id resampled with
    replacement). Per resample:
      dF1  = F1(ours)  - F1(rival)   at each frozen threshold
      dAUC = AUC(ours) - AUC(rival)  from fusion probabilities
    95% percentile CI. Resamples where either AUC is undefined
    (single-class draw) are skipped for the AUC statistic only.
    """
    rng = np.random.default_rng(seed)

    unique_groups = np.unique(groups)
    group_to_rows: dict[Any, np.ndarray] = {
        g: np.flatnonzero(groups == g) for g in unique_groups
    }

    d_f1: list[float] = []
    d_auc: list[float] = []

    for _ in range(n_bootstrap):
        draw = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        rows = np.concatenate([group_to_rows[g] for g in draw])

        y_b = y_true[rows]
        p_o = prob_ours[rows]
        p_r = prob_rival[rows]

        f1_o = f1_score(
            y_b, (p_o >= threshold_ours).astype(np.int64),
            pos_label=1, zero_division=0,
        )
        f1_r = f1_score(
            y_b, (p_r >= threshold_rival).astype(np.int64),
            pos_label=1, zero_division=0,
        )
        d_f1.append(float(f1_o - f1_r))

        if len(np.unique(y_b)) == 2:
            d_auc.append(float(
                roc_auc_score(y_b, p_o) - roc_auc_score(y_b, p_r)
            ))

    d_f1_arr = np.asarray(d_f1, dtype=np.float64)
    d_auc_arr = np.asarray(d_auc, dtype=np.float64)

    return {
        "delta_f1_point": float(
            f1_score(
                y_true,
                (prob_ours >= threshold_ours).astype(np.int64),
                pos_label=1, zero_division=0,
            )
            - f1_score(
                y_true,
                (prob_rival >= threshold_rival).astype(np.int64),
                pos_label=1, zero_division=0,
            )
        ),
        "delta_f1_ci_low": float(np.percentile(d_f1_arr, 2.5)),
        "delta_f1_ci_high": float(np.percentile(d_f1_arr, 97.5)),
        "delta_auc_point": float(
            roc_auc_score(y_true, prob_ours)
            - roc_auc_score(y_true, prob_rival)
        ),
        "delta_auc_ci_low": float(np.percentile(d_auc_arr, 2.5)),
        "delta_auc_ci_high": float(np.percentile(d_auc_arr, 97.5)),
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_auc_valid_n": int(len(d_auc_arr)),
        "groups_total": int(len(unique_groups)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Statistical comparison of frozen fusion pairs on the "
            "200-sample verification set. No model retraining."
        )
    )
    p.add_argument("--check-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_root()

    p0_path = (
        root / "results_frozen_verification" / "fixed_test_predictions.csv"
    )
    p15_path = (
        root / "results_fusion_pairs_verification"
        / "verification_predictions.csv"
    )
    w2v2_dir = root / "results_wav2vec2"

    required = [p0_path, p15_path]

    print("=" * 96)
    print("RUNNING: src/verification_comparison_stats.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("INPUTS          : recorded per-sample frozen-test probabilities only")
    print("RETRAINING      : none")
    print(f"BOOTSTRAP       : {N_BOOTSTRAP} group-cluster resamples, seed {SEED}")
    print("=" * 96)

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    p0_df = pd.read_csv(p0_path, encoding="utf-8-sig")
    p15_df = pd.read_csv(p15_path, encoding="utf-8-sig")

    # Alignment audit: same samples, same order, same labels, same groups.
    if len(p0_df) != 200 or len(p15_df) != 200:
        raise RuntimeError(
            f"Expected 200 rows in each predictions file; "
            f"got {len(p0_df)} and {len(p15_df)}"
        )
    if not p0_df["source_index"].equals(p15_df["source_index"]):
        raise RuntimeError("source_index columns are not aligned.")
    if not p0_df["true_label"].equals(p15_df["true_label"]):
        raise RuntimeError("true_label columns are not aligned.")
    if not p0_df["group_id"].equals(p15_df["group_id"]):
        raise RuntimeError("group_id columns are not aligned.")
    print()
    print("Alignment audit PASSED (200 rows, identical order/labels/groups).")

    y_test = p0_df["true_label"].to_numpy(dtype=np.int64)
    groups = p0_df["group_id"].to_numpy()

    probs: dict[str, np.ndarray] = {
        "P0": p0_df["fusion_probability"].to_numpy(dtype=np.float64),
    }
    for pair_id in ["P1", "P2", "P3", "P4", "P5"]:
        col = f"{pair_id.lower()}_fusion_probability"
        if col not in p15_df.columns:
            raise RuntimeError(f"Missing column in {p15_path.name}: {col}")
        probs[pair_id] = p15_df[col].to_numpy(dtype=np.float64)

    if args.check_only:
        print()
        print("CHECK PASSED. No statistics were computed.")
        return

    out_dir = root / "results_verification_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.perf_counter()

    preds = {
        pair_id: (probs[pair_id] >= PAIRS[pair_id]["threshold"]).astype(
            np.int64
        )
        for pair_id in PAIRS
    }

    # ------------------------------------------------------------------
    # Task A: ours (P0) vs each rival P1..P5
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("TASK A: ours (P0) vs rival fusion pairs")
    print("=" * 96)

    comparison_rows: list[dict[str, Any]] = []

    for pair_id in ["P1", "P2", "P3", "P4", "P5"]:
        mcnemar = exact_mcnemar(
            y_true=y_test,
            pred_a=preds[pair_id],
            pred_b=preds["P0"],
        )
        boot = group_bootstrap_delta(
            y_true=y_test,
            groups=groups,
            prob_ours=probs["P0"],
            prob_rival=probs[pair_id],
            threshold_ours=PAIRS["P0"]["threshold"],
            threshold_rival=PAIRS[pair_id]["threshold"],
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED,
        )

        fp_ours, fn_ours, _, _ = count_fp_fn(
            y_test, probs["P0"], PAIRS["P0"]["threshold"]
        )
        fp_rival, fn_rival, _, _ = count_fp_fn(
            y_test, probs[pair_id], PAIRS[pair_id]["threshold"]
        )

        row: dict[str, Any] = {
            "rival_pair": pair_id,
            "rival_name": PAIRS[pair_id]["name"],
            "rival_threshold": PAIRS[pair_id]["threshold"],
            "ours_threshold": PAIRS["P0"]["threshold"],
            **mcnemar,
            **boot,
            "ours_fp": fp_ours,
            "ours_fn": fn_ours,
            "rival_fp": fp_rival,
            "rival_fn": fn_rival,
            "delta_f1_ci_contains_zero": bool(
                boot["delta_f1_ci_low"] <= 0.0 <= boot["delta_f1_ci_high"]
            ),
            "delta_auc_ci_contains_zero": bool(
                boot["delta_auc_ci_low"] <= 0.0 <= boot["delta_auc_ci_high"]
            ),
        }
        comparison_rows.append(row)

        print()
        print(
            f"{pair_id} ({PAIRS[pair_id]['name']}) vs P0 | "
            f"McNemar p={mcnemar['exact_two_sided_p']:.5f} "
            f"(discordant={mcnemar['discordant_total']}) | "
            f"dF1={boot['delta_f1_point']:+.5f} "
            f"[{boot['delta_f1_ci_low']:+.5f}, {boot['delta_f1_ci_high']:+.5f}] | "
            f"dAUC={boot['delta_auc_point']:+.5f} "
            f"[{boot['delta_auc_ci_low']:+.5f}, {boot['delta_auc_ci_high']:+.5f}]"
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        out_dir / "comparison_vs_ours.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Task B: calibration / error metrics for all six fusion pairs
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("TASK B: calibration and error metrics (P0..P5)")
    print("=" * 96)

    calib_rows: list[dict[str, Any]] = []

    for pair_id, info in PAIRS.items():
        prob = np.clip(probs[pair_id], 1e-7, 1 - 1e-7)
        fp, fn, precision, specificity = count_fp_fn(
            y_test, probs[pair_id], info["threshold"]
        )

        calib_rows.append({
            "pair_id": pair_id,
            "name": info["name"],
            "threshold": info["threshold"],
            "brier": float(brier_score_loss(y_test, prob)),
            "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
            "ece_15bin": float(expected_calibration_error(y_test, prob)),
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "specificity": specificity,
        })

    # Optional wav2vec2 fine-tuned model: only if per-sample probabilities
    # on the same 200-sample verification set were recorded.
    w2v2_note = (
        "no per-sample 200-sample verification probabilities recorded"
    )
    w2v2_candidates = sorted(w2v2_dir.glob("*.csv")) if w2v2_dir.exists() else []
    for cand in w2v2_candidates:
        df = pd.read_csv(cand, encoding="utf-8-sig")
        prob_cols = [
            c for c in df.columns
            if "finetun" in c.lower() and "oof" not in c.lower()
        ]
        if (
            len(df) == 200
            and "y_true" in df.columns
            and prob_cols
            and df["y_true"].to_numpy(dtype=np.int64).tolist()
                == y_test.tolist()
        ):
            prob = np.clip(
                df[prob_cols[0]].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7
            )
            calib_rows.append({
                "pair_id": "W2V2-FT",
                "name": (
                    f"wav2vec2 fine-tuned (source: {cand.name}, "
                    f"column {prob_cols[0]})"
                ),
                "threshold": 0.5,
                "brier": float(brier_score_loss(y_test, prob)),
                "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
                "ece_15bin": float(expected_calibration_error(y_test, prob)),
                "fp": count_fp_fn(y_test, prob, 0.5)[0],
                "fn": count_fp_fn(y_test, prob, 0.5)[1],
                "precision": count_fp_fn(y_test, prob, 0.5)[2],
                "specificity": count_fp_fn(y_test, prob, 0.5)[3],
            })
            w2v2_note = f"included from {cand.name}:{prob_cols[0]}"
            break

    print()
    print(f"wav2vec2 fine-tuned per-sample check: {w2v2_note}")

    calib = pd.DataFrame(calib_rows)
    calib.to_csv(
        out_dir / "calibration_error_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "script_version": SCRIPT_VERSION,
        "inputs": {
            "p0_probabilities": str(p0_path),
            "p1_to_p5_probabilities": str(p15_path),
        },
        "no_model_retraining": True,
        "frozen_thresholds": {
            k: v["threshold"] for k, v in PAIRS.items()
        },
        "mcnemar": "exact two-sided binomial test",
        "bootstrap": {
            "type": "cluster bootstrap over group_id",
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED,
            "ci": "95% percentile",
        },
        "ece": f"{ECE_BINS} equal-width bins over [0,1]",
        "wav2vec2_finetuned": w2v2_note,
    }
    with open(out_dir / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - start_all

    print()
    print("=" * 96)
    print("COMPARISON STATISTICS FINISHED")
    print("=" * 96)
    print()
    print("Task A: ours (P0) vs rivals")
    print(
        comparison[
            [
                "rival_pair",
                "exact_two_sided_p",
                "delta_f1_point",
                "delta_f1_ci_low",
                "delta_f1_ci_high",
                "delta_auc_point",
                "delta_auc_ci_low",
                "delta_auc_ci_high",
                "ours_fp",
                "ours_fn",
                "rival_fp",
                "rival_fn",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )
    print()
    print("Task B: calibration / error metrics")
    print(
        calib.to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )
    print()
    print(f"Total time : {total_elapsed:.1f} s")
    print(f"Results    : {out_dir}")


if __name__ == "__main__":
    main()
