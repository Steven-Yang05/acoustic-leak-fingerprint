from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch


SCRIPT_VERSION = "repeated-grouped-fusion-stability-v1"

DEFAULT_REPEAT_SEEDS = [42, 52, 62, 72, 82]

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


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TRAINING-ONLY repeated group-aware fusion-pair stability audit. "
            "No fixed-test files/results are read."
        )
    )
    p.add_argument(
        "--repeat-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_REPEAT_SEEDS,
        help="StratifiedGroupKFold shuffle seeds. Default: 42 52 62 72 82",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def fit_oof_for_repeat(
    exp11,
    exp12,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    waveforms: torch.Tensor,
    device: torch.device,
    cv_seed: int,
    folds: int,
    epochs: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    """
    Create one complete 800-sample OOF probability set for a new grouped fold plan.
    Model initialization/training seed remains the recorded fixed seed 42 inside
    model_selection_oof.py. Only the group-aware fold assignment changes across repeats.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=cv_seed,
    )
    splits = list(
        sgkf.split(
            X=np.zeros((len(y_train), 1), dtype=np.float32),
            y=y_train,
            groups=groups,
        )
    )

    outer_fold = np.full(len(y_train), -1, dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []

    for fold_idx, (fit_idx, heldout_idx) in enumerate(splits, start=1):
        fit_groups = set(groups[fit_idx])
        heldout_groups = set(groups[heldout_idx])
        overlap = len(fit_groups.intersection(heldout_groups))
        if overlap != 0:
            raise RuntimeError(
                f"repeat seed={cv_seed}, fold={fold_idx}: group overlap={overlap}"
            )
        outer_fold[heldout_idx] = fold_idx

        fold_rows.append({
            "repeat_seed": int(cv_seed),
            "fold": int(fold_idx),
            "fit_samples": int(len(fit_idx)),
            "heldout_samples": int(len(heldout_idx)),
            "fit_groups": int(len(fit_groups)),
            "heldout_groups": int(len(heldout_groups)),
            "heldout_class0": int(np.sum(y_train[heldout_idx] == 0)),
            "heldout_class1": int(np.sum(y_train[heldout_idx] == 1)),
            "group_overlap": int(overlap),
        })

    if np.any(outer_fold < 1):
        raise RuntimeError(f"repeat seed={cv_seed}: incomplete fold assignment")

    probs: dict[str, np.ndarray] = {
        key: np.full(len(y_train), np.nan, dtype=np.float64)
        for key in [
            "logistic_regression",
            "random_forest",
            "mlp",
            "cnn1d",
            "cnn2d",
            "crnn",
        ]
    }

    # Logistic Regression + Random Forest
    for model_key in ["logistic_regression", "random_forest"]:
        print(f"    {DISPLAY_NAMES[model_key]}")
        for fold_idx, (fit_idx, heldout_idx) in enumerate(splits, start=1):
            exp11.set_all_seeds(exp11.SEED)
            model = exp11.build_classical_model(model_key)
            model.fit(X_train[fit_idx], y_train[fit_idx])
            score = exp11.classical_score(model, X_train[heldout_idx])
            probs[model_key][heldout_idx] = np.asarray(score, dtype=np.float64)

    # Deep / neural models: MLP + 1D CNN + 2D CNN + CRNN
    for model_key in ["mlp", "cnn1d", "cnn2d", "crnn"]:
        print(f"    {DISPLAY_NAMES[model_key]}")
        for fold_idx, (fit_idx, heldout_idx) in enumerate(splits, start=1):
            _, score, _ = exp11.train_deep_fold(
                model_key=model_key,
                fit_idx=fit_idx,
                heldout_idx=heldout_idx,
                x_train_full=X_train,
                y_train=y_train,
                waveforms=waveforms,
                device=device,
                epochs=epochs,
            )
            probs[model_key][heldout_idx] = np.asarray(score, dtype=np.float64)

    # RBF-SVM is recalibrated for this repeat using nested group-safe Platt mapping.
    print("    RBF-SVM (nested Platt calibrated)")
    svm_prob, _, _ = exp12.nested_group_platt_calibration(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        outer_fold=outer_fold,
    )
    probs["rbf_svm_calibrated"] = np.asarray(svm_prob, dtype=np.float64)

    for key, p in probs.items():
        if not np.isfinite(p).all():
            raise RuntimeError(
                f"repeat seed={cv_seed}: incomplete probabilities for {key}"
            )

    return probs, outer_fold, fold_rows


def main() -> None:
    args = parse_args()
    root = resolve_root()

    exp11_path = root / "src" / "model_selection_oof.py"
    exp12_path = root / "src" / "fusion_pair_search.py"

    required = [
        root / "features" / "X.npy",
        root / "features" / "y.npy",
        root / "features" / "metadata.csv",
        root / "train_test_data" / "train_indices.npy",
        exp11_path,
        exp12_path,
    ]

    forbidden = [
        root / "train_test_data" / "X_test.npy",
        root / "train_test_data" / "y_test.npy",
        root / "train_test_data" / "test_indices.npy",
    ]

    print("=" * 96)
    print("RUNNING: src/stability_selection.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("PURPOSE         : repeated group-aware structural stability audit")
    print("FIXED TEST DATA : NOT READ")
    print("=" * 96)

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required training-side file missing: {p}")

    print("Files/modules this audit WILL use:")
    for p in required:
        print(f"  {p}")

    print()
    print("Test-side files/results this audit is explicitly designed NOT to read:")
    for p in forbidden:
        print(f"  {p}")

    exp11 = load_module(exp11_path, "model_selection_oof")
    exp12 = load_module(exp12_path, "fusion_pair_search")

    X_all = np.load(root / "features" / "X.npy", allow_pickle=False)
    y_all = np.load(root / "features" / "y.npy", allow_pickle=False)
    metadata = pd.read_csv(
        root / "features" / "metadata.csv",
        encoding="utf-8-sig",
    )
    train_indices = np.load(
        root / "train_test_data" / "train_indices.npy",
        allow_pickle=False,
    ).astype(np.int64)

    if len(train_indices) != 800:
        raise RuntimeError(
            f"Expected frozen 800-sample training partition; got {len(train_indices)}"
        )

    X_train = X_all[train_indices].astype(np.float32)
    y_train = y_all[train_indices].astype(np.int64)

    groups_all = exp11.derive_group_ids(metadata)
    groups = groups_all[train_indices]

    if len(np.unique(groups)) != 370:
        raise RuntimeError(
            f"Expected 370 training groups; got {len(np.unique(groups))}"
        )

    print()
    print(f"Training samples : {len(y_train)}")
    print(f"Training groups  : {len(np.unique(groups))}")
    print(f"Repeat seeds     : {args.repeat_seeds}")
    print(f"Folds per repeat : {args.folds}")
    print(f"Deep epochs      : {args.epochs}")
    print(
        "Pair selection   : genuine fusion, "
        "F1 -> Recall -> Specificity -> ROC-AUC -> AP"
    )

    if args.check_only:
        print()
        print("CHECK PASSED.")
        print("No model training was run.")
        print("No fixed-test data or prior fixed-test results were read.")
        return

    out_dir = root / "results_stability"
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preload only the 800 training waveforms once.
    waveforms = exp11.preload_training_waveforms(
        root,
        metadata,
        train_indices,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print(f"PyTorch device   : {device}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")

    repeat_pair_rows: list[dict[str, Any]] = []
    repeat_primary_rows: list[dict[str, Any]] = []
    fold_plan_rows: list[dict[str, Any]] = []
    probability_rows: list[pd.DataFrame] = []

    start_all = time.perf_counter()

    for repeat_number, cv_seed in enumerate(args.repeat_seeds, start=1):
        print()
        print("#" * 96)
        print(
            f"REPEAT {repeat_number}/{len(args.repeat_seeds)} "
            f"| StratifiedGroupKFold seed={cv_seed}"
        )
        print("#" * 96)

        repeat_start = time.perf_counter()

        probs, outer_fold, fold_rows = fit_oof_for_repeat(
            exp11=exp11,
            exp12=exp12,
            X_train=X_train,
            y_train=y_train,
            groups=groups,
            waveforms=waveforms,
            device=device,
            cv_seed=cv_seed,
            folds=args.folds,
            epochs=args.epochs,
        )
        fold_plan_rows.extend(fold_rows)

        prob_frame = pd.DataFrame({
            "repeat_seed": int(cv_seed),
            "source_index": train_indices,
            "group_id": groups,
            "true_label": y_train,
            "oof_fold": outer_fold,
        })
        for key, p in probs.items():
            prob_frame[f"{key}_probability"] = p
        probability_rows.append(prob_frame)

        best_rows: list[dict[str, Any]] = []

        for hand_key in HANDCRAFTED_MODELS:
            for deep_key in DEEP_MODELS:
                pair_id = f"{hand_key}__plus__{deep_key}"

                _, unrestricted_best, genuine_best = exp12.search_pair(
                    pair_id=pair_id,
                    handcrafted_key=hand_key,
                    deep_key=deep_key,
                    y_true=y_train,
                    p_hand=probs[hand_key],
                    p_deep=probs[deep_key],
                )

                row = {
                    "repeat_seed": int(cv_seed),
                    "pair_id": pair_id,
                    "handcrafted_model": hand_key,
                    "handcrafted_name": DISPLAY_NAMES[hand_key],
                    "deep_model": deep_key,
                    "deep_name": DISPLAY_NAMES[deep_key],
                    "handcrafted_weight": float(
                        genuine_best["handcrafted_weight"]
                    ),
                    "deep_weight": float(genuine_best["deep_weight"]),
                    "threshold": float(genuine_best["threshold"]),
                    "accuracy": float(genuine_best["accuracy"]),
                    "precision": float(genuine_best["precision"]),
                    "recall": float(genuine_best["recall"]),
                    "f1": float(genuine_best["f1"]),
                    "specificity": float(genuine_best["specificity"]),
                    "roc_auc": float(genuine_best["roc_auc"]),
                    "average_precision": float(
                        genuine_best["average_precision"]
                    ),
                    "brier": float(genuine_best["brier"]),
                    "log_loss": float(genuine_best["log_loss"]),
                    "fp": int(genuine_best["fp"]),
                    "fn": int(genuine_best["fn"]),
                    "unrestricted_handcrafted_weight": float(
                        unrestricted_best["handcrafted_weight"]
                    ),
                    "unrestricted_deep_weight": float(
                        unrestricted_best["deep_weight"]
                    ),
                    "unrestricted_threshold": float(
                        unrestricted_best["threshold"]
                    ),
                    "unrestricted_f1": float(unrestricted_best["f1"]),
                    "unrestricted_collapsed_to_single": bool(
                        float(unrestricted_best["handcrafted_weight"]) <= 1e-12
                        or float(unrestricted_best["deep_weight"]) <= 1e-12
                    ),
                }
                best_rows.append(row)

        repeat_df = pd.DataFrame(best_rows)
        repeat_df = exp12.rank_rows(repeat_df)
        repeat_df.insert(
            0,
            "rank_within_repeat",
            np.arange(1, len(repeat_df) + 1),
        )
        repeat_pair_rows.extend(repeat_df.to_dict(orient="records"))

        primary = repeat_df.iloc[0]
        repeat_primary_rows.append({
            "repeat_seed": int(cv_seed),
            "pair_id": primary["pair_id"],
            "handcrafted_model": primary["handcrafted_model"],
            "deep_model": primary["deep_model"],
            "handcrafted_weight": float(primary["handcrafted_weight"]),
            "deep_weight": float(primary["deep_weight"]),
            "threshold": float(primary["threshold"]),
            "accuracy": float(primary["accuracy"]),
            "recall": float(primary["recall"]),
            "f1": float(primary["f1"]),
            "specificity": float(primary["specificity"]),
            "roc_auc": float(primary["roc_auc"]),
            "average_precision": float(primary["average_precision"]),
            "fp": int(primary["fp"]),
            "fn": int(primary["fn"]),
        })

        elapsed = time.perf_counter() - repeat_start
        print()
        print(
            f"Repeat winner: {primary['handcrafted_name']} + "
            f"{primary['deep_name']}"
        )
        print(
            f"  weights={primary['handcrafted_weight']:.2f}/"
            f"{primary['deep_weight']:.2f}, "
            f"threshold={primary['threshold']:.3f}, "
            f"F1={primary['f1']:.5f}, "
            f"Recall={primary['recall']:.5f}, "
            f"Spec={primary['specificity']:.5f}"
        )
        print(f"Repeat time: {elapsed:.1f} s")

    # Save raw repeat-level data.
    repeat_pairs = pd.DataFrame(repeat_pair_rows)
    repeat_pairs.to_csv(
        out_dir / "repeated_pair_rankings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    repeat_primary = pd.DataFrame(repeat_primary_rows)
    repeat_primary.to_csv(
        out_dir / "repeat_primary_pair.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(fold_plan_rows).to_csv(
        out_dir / "repeated_fold_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        probability_rows,
        ignore_index=True,
    ).to_csv(
        out_dir / "repeated_training_oof_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Aggregate stability by pair.
    summary_rows: list[dict[str, Any]] = []
    n_repeats = len(args.repeat_seeds)

    for pair_id, sub in repeat_pairs.groupby("pair_id", sort=False):
        first = sub.iloc[0]
        ranks = sub["rank_within_repeat"].to_numpy(dtype=float)

        summary_rows.append({
            "pair_id": pair_id,
            "handcrafted_model": first["handcrafted_model"],
            "handcrafted_name": first["handcrafted_name"],
            "deep_model": first["deep_model"],
            "deep_name": first["deep_name"],
            "rank1_count": int(np.sum(ranks == 1)),
            "rank1_fraction": float(np.mean(ranks == 1)),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "best_rank": int(np.min(ranks)),
            "worst_rank": int(np.max(ranks)),
            "mean_f1": float(sub["f1"].mean()),
            "std_f1": float(sub["f1"].std(ddof=1)),
            "min_f1": float(sub["f1"].min()),
            "max_f1": float(sub["f1"].max()),
            "mean_recall": float(sub["recall"].mean()),
            "std_recall": float(sub["recall"].std(ddof=1)),
            "mean_specificity": float(sub["specificity"].mean()),
            "std_specificity": float(sub["specificity"].std(ddof=1)),
            "median_handcrafted_weight": float(
                sub["handcrafted_weight"].median()
            ),
            "min_handcrafted_weight": float(
                sub["handcrafted_weight"].min()
            ),
            "max_handcrafted_weight": float(
                sub["handcrafted_weight"].max()
            ),
            "median_threshold": float(sub["threshold"].median()),
            "min_threshold": float(sub["threshold"].min()),
            "max_threshold": float(sub["threshold"].max()),
            "unrestricted_collapse_count": int(
                sub["unrestricted_collapsed_to_single"].sum()
            ),
        })

    stability = pd.DataFrame(summary_rows)
    stability = stability.sort_values(
        by=[
            "rank1_count",
            "mean_rank",
            "mean_f1",
            "mean_recall",
            "mean_specificity",
        ],
        ascending=[False, True, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    stability.insert(
        0,
        "stability_rank",
        np.arange(1, len(stability) + 1),
    )
    stability.to_csv(
        out_dir / "pair_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top = stability.iloc[0]
    win_count = int(top["rank1_count"])

    if win_count >= max(4, n_repeats - 1):
        stability_label = "strong"
    elif win_count >= int(np.ceil(n_repeats / 2)):
        stability_label = "moderate"
    else:
        stability_label = "weak_or_unstable"

    historical = stability[
        (stability["handcrafted_model"] == "mlp")
        & (stability["deep_model"] == "cnn2d")
    ].iloc[0]

    svm2d = stability[
        (stability["handcrafted_model"] == "rbf_svm_calibrated")
        & (stability["deep_model"] == "cnn2d")
    ].iloc[0]

    rf2d = stability[
        (stability["handcrafted_model"] == "random_forest")
        & (stability["deep_model"] == "cnn2d")
    ].iloc[0]

    decision = {
        "script_version": SCRIPT_VERSION,
        "fixed_test_data_read": False,
        "repeat_seeds": [int(x) for x in args.repeat_seeds],
        "folds_per_repeat": int(args.folds),
        "deep_epochs": int(args.epochs),
        "training_samples": int(len(y_train)),
        "training_groups": int(len(np.unique(groups))),
        "stability_decision_rule": (
            "rank pairs independently within each repeated group-aware OOF run; "
            "primary stability summary prioritizes rank-1 frequency, then mean rank, "
            "then mean F1/Recall/Specificity"
        ),
        "overall_stability_winner": {
            "pair_id": top["pair_id"],
            "handcrafted_model": top["handcrafted_model"],
            "deep_model": top["deep_model"],
            "rank1_count": int(top["rank1_count"]),
            "rank1_fraction": float(top["rank1_fraction"]),
            "mean_rank": float(top["mean_rank"]),
            "mean_f1": float(top["mean_f1"]),
            "std_f1": float(top["std_f1"]),
            "median_handcrafted_weight": float(
                top["median_handcrafted_weight"]
            ),
            "median_threshold": float(top["median_threshold"]),
            "stability_strength": stability_label,
        },
        "key_pair_comparison": {
            "svm_cnn2d": {
                "rank1_count": int(svm2d["rank1_count"]),
                "mean_rank": float(svm2d["mean_rank"]),
                "mean_f1": float(svm2d["mean_f1"]),
                "std_f1": float(svm2d["std_f1"]),
                "median_handcrafted_weight": float(
                    svm2d["median_handcrafted_weight"]
                ),
                "median_threshold": float(svm2d["median_threshold"]),
            },
            "rf_cnn2d": {
                "rank1_count": int(rf2d["rank1_count"]),
                "mean_rank": float(rf2d["mean_rank"]),
                "mean_f1": float(rf2d["mean_f1"]),
                "std_f1": float(rf2d["std_f1"]),
                "median_handcrafted_weight": float(
                    rf2d["median_handcrafted_weight"]
                ),
                "median_threshold": float(rf2d["median_threshold"]),
            },
            "mlp_cnn2d_historical": {
                "rank1_count": int(historical["rank1_count"]),
                "mean_rank": float(historical["mean_rank"]),
                "mean_f1": float(historical["mean_f1"]),
                "std_f1": float(historical["std_f1"]),
                "median_handcrafted_weight": float(
                    historical["median_handcrafted_weight"]
                ),
                "median_threshold": float(
                    historical["median_threshold"]
                ),
            },
        },
    }

    with open(
        out_dir / "fusion_structure_stability_decision.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - start_all

    print()
    print("=" * 96)
    print("REPEATED TRAINING-ONLY FUSION STABILITY AUDIT FINISHED")
    print("=" * 96)

    print(
        stability[
            [
                "stability_rank",
                "handcrafted_name",
                "deep_name",
                "rank1_count",
                "mean_rank",
                "mean_f1",
                "std_f1",
                "mean_recall",
                "mean_specificity",
                "median_handcrafted_weight",
                "median_threshold",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )

    print()
    print(
        f"Stability winner : {top['handcrafted_name']} + {top['deep_name']}"
    )
    print(
        f"Rank-1 frequency : {int(top['rank1_count'])}/{n_repeats}"
    )
    print(f"Stability label  : {stability_label}")
    print(f"Total time       : {total_elapsed:.1f} s")
    print()
    print(f"Outputs: {out_dir}")
    print("Most important files:")
    print(f"  {out_dir / 'pair_stability_summary.csv'}")
    print(f"  {out_dir / 'repeat_primary_pair.csv'}")
    print(f"  {out_dir / 'repeated_pair_rankings.csv'}")
    print(f"  {out_dir / 'fusion_structure_stability_decision.json'}")
    print()
    print("No fixed-test-set metrics were computed.")


if __name__ == "__main__":
    main()
