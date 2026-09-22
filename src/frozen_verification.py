from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

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
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "frozen-training-selected-svm-cnn1d-fixed-test-v1"

# ---------------------------------------------------------------------
# FROZEN BEFORE FIXED-TEST EVALUATION
# ---------------------------------------------------------------------
SEED = 42
FOLDS = 5
EPOCHS = 40

SVM_WEIGHT = 0.55
CNN1D_WEIGHT = 0.45
FUSION_THRESHOLD = 0.520

# These values come from the stability_selection.py repeated TRAINING-ONLY audit:
#   Stability winner: calibrated RBF-SVM + 1D CNN
#   rank-1 frequency: 3/5
#   median handcrafted weight: 0.55
#   median threshold: 0.520
#
# This script MUST NOT modify them after seeing fixed-test performance.
# ---------------------------------------------------------------------


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compute_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    pred = (prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
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


def derive_test_waveforms(
    exp11,
    root: Path,
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
) -> torch.Tensor:
    waves: list[torch.Tensor] = []
    total = len(test_indices)

    print()
    print("Preloading the 200 FIXED-TEST waveforms for final verification...")
    for j, source_index in enumerate(test_indices, start=1):
        raw = metadata.iloc[int(source_index)]["file_path"]
        path = exp11.resolve_wav_path(root, raw)
        waves.append(exp11.load_one_waveform(path))

        if j % 50 == 0 or j == total:
            print(f"  loaded {j}/{total}")

    result = torch.stack(waves, dim=0)
    if result.shape != (len(test_indices), 1, exp11.NUM_SAMPLES):
        raise RuntimeError(
            f"Unexpected fixed-test waveform tensor shape: {tuple(result.shape)}"
        )
    return result


def fit_training_only_svm_calibrator(
    exp11,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
) -> tuple[Any, Any, pd.DataFrame, np.ndarray]:
    """
    Calibration is fit from TRAINING-ONLY group-aware OOF SVM scores.

    1) Generate 5-fold StratifiedGroupKFold OOF decision scores on the 800
       training samples.
    2) Fit a 1D logistic Platt mapper on those OOF scores and training labels.
    3) Refit RBF-SVM on all 800 training samples.
    4) Return full-training SVM + frozen calibrator.

    The fixed 200-sample test set is not used anywhere in calibration.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    splits = list(
        sgkf.split(
            X=np.zeros((len(y_train), 1), dtype=np.float32),
            y=y_train,
            groups=groups_train,
        )
    )

    oof_score = np.full(len(y_train), np.nan, dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []

    for fold_idx, (fit_idx, heldout_idx) in enumerate(splits, start=1):
        fit_groups = set(groups_train[fit_idx])
        heldout_groups = set(groups_train[heldout_idx])
        overlap = len(fit_groups.intersection(heldout_groups))
        if overlap != 0:
            raise RuntimeError(
                f"SVM calibration fold {fold_idx}: group overlap={overlap}"
            )

        svm = exp11.build_classical_model("rbf_svm")
        svm.fit(X_train[fit_idx], y_train[fit_idx])

        score = exp11.classical_score(
            svm,
            X_train[heldout_idx],
        )
        oof_score[heldout_idx] = np.asarray(score, dtype=np.float64)

        audit_rows.append({
            "fold": int(fold_idx),
            "fit_samples": int(len(fit_idx)),
            "heldout_samples": int(len(heldout_idx)),
            "fit_groups": int(len(fit_groups)),
            "heldout_groups": int(len(heldout_groups)),
            "group_overlap": int(overlap),
            "heldout_class0": int(np.sum(y_train[heldout_idx] == 0)),
            "heldout_class1": int(np.sum(y_train[heldout_idx] == 1)),
        })

    if not np.isfinite(oof_score).all():
        raise RuntimeError("Incomplete training-only SVM OOF decision scores.")

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

    oof_prob = calibrator.predict_proba(
        oof_score.reshape(-1, 1)
    )[:, 1]

    full_svm = exp11.build_classical_model("rbf_svm")
    full_svm.fit(X_train, y_train)

    return (
        full_svm,
        calibrator,
        pd.DataFrame(audit_rows),
        np.asarray(oof_prob, dtype=np.float64),
    )


def train_full_cnn1d(
    exp11,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_waveforms: torch.Tensor,
    device: torch.device,
) -> torch.nn.Module:
    """
    Train the recorded 1D CNN architecture on all 800 training samples
    for the frozen 40 epochs. No early stopping, no fixed-test feedback.
    """
    exp11.set_all_seeds(SEED)

    model = exp11.build_deep_model(
        "cnn1d",
        X_train.shape[1],
    ).to(device)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight_value = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=0.00001,
    )

    dataset = exp11.WaveDataset(
        train_waveforms,
        y_train,
        np.arange(len(y_train), dtype=np.int64),
        augment=True,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=exp11.BATCH_SIZE["cnn1d"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    amp_enabled = device.type == "cuda"
    grad_scaler = exp11.make_grad_scaler(amp_enabled)

    print()
    print("Training frozen full-data 1D CNN...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with exp11.autocast_context(amp_enabled):
                logits = model(xb)
                loss = criterion(logits, yb)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            grad_scaler.step(optimizer)
            grad_scaler.update()

            batch_n = int(yb.shape[0])
            total_loss += float(loss.detach().item()) * batch_n
            seen += batch_n

        scheduler.step()
        epoch_loss = total_loss / max(seen, 1)

        if epoch in {1, 10, 20, 30, EPOCHS}:
            print(
                f"  epoch {epoch:03d}/{EPOCHS:03d} "
                f"loss={epoch_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.7f}"
            )

    return model


def infer_cnn1d(
    exp11,
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    dataset = exp11.WaveDataset(
        waveforms,
        np.zeros(len(waveforms), dtype=np.float32),
        np.arange(len(waveforms), dtype=np.int64),
        augment=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=exp11.BATCH_SIZE["cnn1d"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    amp_enabled = device.type == "cuda"
    probs: list[np.ndarray] = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            with exp11.autocast_context(amp_enabled):
                logits = model(xb)
                p = torch.sigmoid(logits)
            probs.append(
                p.float().cpu().numpy()
            )

    return np.concatenate(probs).astype(np.float64)


def exact_mcnemar(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    """
    Exact two-sided McNemar p-value using the binomial distribution.
    No scipy dependency is required.
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
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant_total": n,
        "exact_two_sided_p": float(p_value),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Frozen fixed-test verification of the TRAINING-ONLY selected "
            "RBF-SVM + 1D CNN structure."
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

    required = [
        root / "features" / "X.npy",
        root / "features" / "y.npy",
        root / "features" / "metadata.csv",
        root / "train_test_data" / "train_indices.npy",
        root / "train_test_data" / "test_indices.npy",
        exp11_path,
    ]

    print("=" * 96)
    print("RUNNING: src/frozen_verification.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("STRUCTURE       : RBF-SVM + 1D CNN")
    print(f"FROZEN WEIGHTS  : SVM={SVM_WEIGHT:.2f}, CNN1D={CNN1D_WEIGHT:.2f}")
    print(f"FROZEN THRESHOLD: {FUSION_THRESHOLD:.3f}")
    print("SELECTION BASIS : stability_selection.py repeated TRAINING-ONLY group-aware OOF")
    print("TEST USE        : ONE frozen verification; no tuning")
    print("=" * 96)

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    exp11 = load_module(
        exp11_path,
        "exp11_for_final_verification",
    )

    X_all = np.load(
        root / "features" / "X.npy",
        allow_pickle=False,
    )
    y_all = np.load(
        root / "features" / "y.npy",
        allow_pickle=False,
    )
    metadata = pd.read_csv(
        root / "features" / "metadata.csv",
        encoding="utf-8-sig",
    )
    train_indices = np.load(
        root / "train_test_data" / "train_indices.npy",
        allow_pickle=False,
    ).astype(np.int64)
    test_indices = np.load(
        root / "train_test_data" / "test_indices.npy",
        allow_pickle=False,
    ).astype(np.int64)

    if len(train_indices) != 800:
        raise RuntimeError(
            f"Expected 800 training samples; got {len(train_indices)}"
        )
    if len(test_indices) != 200:
        raise RuntimeError(
            f"Expected 200 fixed-test samples; got {len(test_indices)}"
        )

    sample_overlap = len(
        set(train_indices.tolist()).intersection(
            set(test_indices.tolist())
        )
    )
    if sample_overlap != 0:
        raise RuntimeError(
            f"Train/test sample-index overlap={sample_overlap}"
        )

    groups_all = exp11.derive_group_ids(metadata)
    groups_train = groups_all[train_indices]
    groups_test = groups_all[test_indices]

    group_overlap = len(
        set(groups_train).intersection(
            set(groups_test)
        )
    )
    if group_overlap != 0:
        raise RuntimeError(
            f"Train/test group overlap={group_overlap}"
        )

    X_train = X_all[train_indices].astype(np.float32)
    y_train = y_all[train_indices].astype(np.int64)
    X_test = X_all[test_indices].astype(np.float32)
    y_test = y_all[test_indices].astype(np.int64)

    print()
    print(f"Train samples      : {len(y_train)}")
    print(f"Test samples       : {len(y_test)}")
    print(f"Train groups       : {len(np.unique(groups_train))}")
    print(f"Test groups        : {len(np.unique(groups_test))}")
    print(f"Sample overlap     : {sample_overlap}")
    print(f"Group overlap      : {group_overlap}")
    print(
        f"Train labels       : class0={(y_train == 0).sum()}, "
        f"class1={(y_train == 1).sum()}"
    )
    print(
        f"Test labels        : class0={(y_test == 0).sum()}, "
        f"class1={(y_test == 1).sum()}"
    )

    if args.check_only:
        print()
        print("CHECK PASSED.")
        print("No model training or fixed-test inference was run.")
        return

    out_dir = root / "results_frozen_verification"
    model_dir = root / "models_frozen"

    if args.clean_output:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if model_dir.exists():
            shutil.rmtree(model_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage A: TRAINING-ONLY SVM calibration + full training
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STAGE A: training-only RBF-SVM calibration and full refit")
    print("=" * 96)

    full_svm, svm_calibrator, svm_audit, svm_train_oof_prob = (
        fit_training_only_svm_calibrator(
            exp11=exp11,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
        )
    )

    svm_audit.to_csv(
        out_dir / "svm_training_only_calibration_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_cal_metrics = compute_metrics(
        y_train,
        svm_train_oof_prob,
        threshold=0.5,
    )
    with open(
        out_dir / "svm_training_only_calibration_quality.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            train_cal_metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(
        full_svm,
        model_dir / "rbf_svm_full_train.joblib",
    )
    joblib.dump(
        svm_calibrator,
        model_dir / "rbf_svm_platt_calibrator_training_oof.joblib",
    )

    # ------------------------------------------------------------------
    # Stage B: full 800-sample frozen 1D CNN training
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STAGE B: full 800-sample 1D CNN training")
    print("=" * 96)

    train_waveforms = exp11.preload_training_waveforms(
        root,
        metadata,
        train_indices,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"PyTorch device     : {device}")
    if device.type == "cuda":
        print(
            f"GPU                : "
            f"{torch.cuda.get_device_name(0)}"
        )

    cnn1d = train_full_cnn1d(
        exp11=exp11,
        X_train=X_train,
        y_train=y_train,
        train_waveforms=train_waveforms,
        device=device,
    )

    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "model_key": "cnn1d",
            "state_dict": cnn1d.state_dict(),
            "epochs": EPOCHS,
            "seed": SEED,
        },
        model_dir / "cnn1d_full_train_final.pt",
    )

    # ------------------------------------------------------------------
    # Stage C: ONE frozen fixed-test evaluation
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STAGE C: ONE frozen fixed-test verification")
    print("=" * 96)
    print("No weights or thresholds will be searched.")

    # SVM
    svm_test_score = exp11.classical_score(
        full_svm,
        X_test,
    )
    svm_test_prob = svm_calibrator.predict_proba(
        np.asarray(svm_test_score).reshape(-1, 1)
    )[:, 1]

    # CNN1D
    test_waveforms = derive_test_waveforms(
        exp11=exp11,
        root=root,
        metadata=metadata,
        test_indices=test_indices,
    )
    cnn1d_test_prob = infer_cnn1d(
        exp11=exp11,
        model=cnn1d,
        waveforms=test_waveforms,
        device=device,
    )

    # Frozen fusion
    fusion_prob = (
        SVM_WEIGHT * svm_test_prob
        + CNN1D_WEIGHT * cnn1d_test_prob
    )

    svm_metrics = compute_metrics(
        y_test,
        svm_test_prob,
        threshold=0.5,
    )
    cnn1d_metrics = compute_metrics(
        y_test,
        cnn1d_test_prob,
        threshold=0.5,
    )
    fusion_metrics = compute_metrics(
        y_test,
        fusion_prob,
        threshold=FUSION_THRESHOLD,
    )

    summary = pd.DataFrame([
        {
            "method": "rbf_svm_calibrated",
            "weight_svm": 1.0,
            "weight_cnn1d": 0.0,
            **svm_metrics,
        },
        {
            "method": "cnn1d",
            "weight_svm": 0.0,
            "weight_cnn1d": 1.0,
            **cnn1d_metrics,
        },
        {
            "method": "frozen_svm055_cnn1d045",
            "weight_svm": SVM_WEIGHT,
            "weight_cnn1d": CNN1D_WEIGHT,
            **fusion_metrics,
        },
    ])

    summary.to_csv(
        out_dir / "fixed_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions = metadata.iloc[
        test_indices
    ].copy().reset_index(drop=True)

    predictions.insert(
        0,
        "source_index",
        test_indices,
    )
    predictions["group_id"] = groups_test
    predictions["true_label"] = y_test
    predictions["svm_probability"] = svm_test_prob
    predictions["cnn1d_probability"] = cnn1d_test_prob
    predictions["fusion_probability"] = fusion_prob
    predictions["svm_prediction"] = (
        svm_test_prob >= 0.5
    ).astype(np.int64)
    predictions["cnn1d_prediction"] = (
        cnn1d_test_prob >= 0.5
    ).astype(np.int64)
    predictions["fusion_prediction"] = (
        fusion_prob >= FUSION_THRESHOLD
    ).astype(np.int64)
    predictions["fusion_correct"] = (
        predictions["fusion_prediction"].to_numpy()
        == y_test
    )

    predictions.to_csv(
        out_dir / "fixed_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Exact paired comparisons against the two components.
    fusion_pred = (
        fusion_prob >= FUSION_THRESHOLD
    ).astype(np.int64)
    svm_pred = (
        svm_test_prob >= 0.5
    ).astype(np.int64)
    cnn_pred = (
        cnn1d_test_prob >= 0.5
    ).astype(np.int64)

    mcnemar_rows = []

    for method_a, pred_a in [
        ("rbf_svm_calibrated", svm_pred),
        ("cnn1d", cnn_pred),
    ]:
        stat = exact_mcnemar(
            y_true=y_test,
            pred_a=pred_a,
            pred_b=fusion_pred,
        )
        mcnemar_rows.append({
            "method_a": method_a,
            "method_b": "frozen_svm055_cnn1d045",
            **stat,
        })

    pd.DataFrame(
        mcnemar_rows
    ).to_csv(
        out_dir / "fixed_test_mcnemar_vs_components.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Frozen protocol metadata.
    protocol = {
        "script_version": SCRIPT_VERSION,
        "structure_selected_by": (
            "stability_selection.py repeated TRAINING-ONLY group-aware OOF stability audit"
        ),
        "structure": "RBF-SVM + 1D CNN",
        "svm_probability": (
            "Platt logistic calibrator fit on 5-fold group-aware OOF "
            "decision scores from the 800 training samples; final RBF-SVM "
            "then refit on all 800 training samples"
        ),
        "cnn1d_training": {
            "epochs": EPOCHS,
            "seed": SEED,
            "architecture": "same recorded architecture as model_selection_oof.py",
        },
        "frozen_svm_weight": SVM_WEIGHT,
        "frozen_cnn1d_weight": CNN1D_WEIGHT,
        "frozen_threshold": FUSION_THRESHOLD,
        "parameter_source": (
            "median values from the stability_selection.py repeated TRAINING-ONLY audit"
        ),
        "test_samples": int(len(y_test)),
        "test_groups": int(len(np.unique(groups_test))),
        "train_test_sample_overlap": int(sample_overlap),
        "train_test_group_overlap": int(group_overlap),
        "no_test_tuning_performed": True,
    }

    with open(
        out_dir / "frozen_protocol.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            protocol,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Console result.
    print()
    print("=" * 96)
    print("FROZEN FIXED-TEST VERIFICATION FINISHED")
    print("=" * 96)
    print(
        summary[
            [
                "method",
                "threshold",
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
    print()
    print(
        f"Frozen fusion = "
        f"{SVM_WEIGHT:.2f} * calibrated SVM + "
        f"{CNN1D_WEIGHT:.2f} * CNN1D"
    )
    print(
        f"Frozen threshold = {FUSION_THRESHOLD:.3f}"
    )
    print()
    print(f"Results: {out_dir}")
    print(f"Models : {model_dir}")
    print()
    print("IMPORTANT: Do not change weights/threshold based on these results.")


if __name__ == "__main__":
    main()
