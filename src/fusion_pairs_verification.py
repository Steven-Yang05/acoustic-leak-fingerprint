from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

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
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "all-fusion-pairs-fixed-test-verification-v1"

# ---------------------------------------------------------------------
# FROZEN BEFORE FIXED-TEST EVALUATION
# ---------------------------------------------------------------------
SEED = 42
EPOCHS = 40

# The five fusion pairs evaluated here. All weights/thresholds are the
# median values from stability_selection.py's repeated TRAINING-ONLY
# group-aware OOF stability audit. They MUST NOT be modified after
# seeing fixed-test performance.
#
# fusion_prob = weight * p_handcrafted + (1 - weight) * p_deep
FUSION_PAIRS = [
    {
        "pair_id": "P1",
        "handcrafted_model": "rbf_svm_calibrated",
        "deep_model": "cnn2d",
        "weight": 0.45,
        "threshold": 0.505,
    },
    {
        "pair_id": "P2",
        "handcrafted_model": "mlp",
        "deep_model": "cnn2d",
        "weight": 0.45,
        "threshold": 0.490,
    },
    {
        "pair_id": "P3",
        "handcrafted_model": "random_forest",
        "deep_model": "cnn2d",
        "weight": 0.75,
        "threshold": 0.550,
    },
    {
        "pair_id": "P4",
        "handcrafted_model": "mlp",
        "deep_model": "cnn1d",
        "weight": 0.45,
        "threshold": 0.475,
    },
    {
        "pair_id": "P5",
        "handcrafted_model": "random_forest",
        "deep_model": "crnn",
        "weight": 0.75,
        "threshold": 0.505,
    },
]

# Branch artifacts reused from frozen_verification.py (already trained on
# the same frozen 800-sample training partition with the same recipe):
#   models_frozen/rbf_svm_full_train.joblib
#   models_frozen/rbf_svm_platt_calibrator_training_oof.joblib
#   models_frozen/cnn1d_full_train_final.pt
# Branches trained fresh in this script (shared across pairs because the
# input, recipe and seed are identical):
#   mlp            -> P2, P4
#   random_forest  -> P3, P5
#   cnn2d          -> P1, P2, P3
#   crnn           -> P5
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


def train_full_deep_model(
    exp11,
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_waveforms: torch.Tensor,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Any]:
    """
    Train one deep branch on all 800 training samples for the frozen 40
    epochs, following the exact model_selection_oof.py recipe:
      - AdamW lr=1e-3, weight_decay=1e-4
      - CosineAnnealingLR T_max=epochs, eta_min=1e-5
      - balanced BCEWithLogitsLoss
      - AMP + grad clipping 5.0
      - seed 42, no early stopping, no fixed-test feedback

    Input pipelines (as recorded in model_selection_oof.py):
      - mlp           : 212-d handcrafted features, StandardScaler fit on
                        the 800 training samples, VectorDataset
      - cnn2d / crnn  : raw waveform -> WaveDataset(augment=True) ->
                        LogMelFrontend + SpecAugment inside the loop
    Returns (model, fitted_scaler_or_None, frontend_or_None).
    """
    if model_key not in {"mlp", "cnn2d", "crnn"}:
        raise KeyError(f"train_full_deep_model does not handle: {model_key}")

    exp11.set_all_seeds(SEED)

    model = exp11.build_deep_model(
        model_key,
        X_train.shape[1],
    ).to(device)

    frontend = None
    if model_key in {"cnn2d", "crnn"}:
        frontend = exp11.LogMelFrontend().to(device)

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

    scaler = None
    if model_key == "mlp":
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X_train).astype(np.float32)
        dataset = exp11.VectorDataset(x_scaled, y_train)
    else:
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
        batch_size=exp11.BATCH_SIZE[model_key],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    amp_enabled = device.type == "cuda"
    grad_scaler = exp11.make_grad_scaler(amp_enabled)

    print()
    print(f"Training frozen full-data {model_key}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with exp11.autocast_context(amp_enabled):
                if model_key in {"cnn2d", "crnn"}:
                    assert frontend is not None
                    xb = frontend(xb)
                    xb = exp11.apply_spec_augment(xb)

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

    return model, scaler, frontend


def infer_deep_model(
    exp11,
    model_key: str,
    model: torch.nn.Module,
    device: torch.device,
    waveforms: torch.Tensor | None = None,
    X: np.ndarray | None = None,
    scaler: Any = None,
    frontend: Any = None,
) -> np.ndarray:
    """
    Fixed-test inference for one deep branch. No augmentation; the
    Log-Mel frontend is applied for cnn2d/crnn exactly as in
    model_selection_oof.py.
    """
    model.eval()

    if model_key == "mlp":
        if X is None or scaler is None:
            raise RuntimeError("MLP inference requires X and fitted scaler.")
        x_scaled = scaler.transform(X).astype(np.float32)
        dataset = exp11.VectorDataset(
            x_scaled,
            np.zeros(len(x_scaled), dtype=np.float32),
        )
    else:
        if waveforms is None:
            raise RuntimeError(f"{model_key} inference requires waveforms.")
        dataset = exp11.WaveDataset(
            waveforms,
            np.zeros(len(waveforms), dtype=np.float32),
            np.arange(len(waveforms), dtype=np.int64),
            augment=False,
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=exp11.BATCH_SIZE[model_key],
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
                if model_key in {"cnn2d", "crnn"}:
                    assert frontend is not None
                    xb = frontend(xb)
                logits = model(xb)
                p = torch.sigmoid(logits)
            probs.append(
                p.float().cpu().numpy()
            )

    return np.concatenate(probs).astype(np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Frozen fixed-test verification of five TRAINING-ONLY selected "
            "handcrafted + deep fusion pairs."
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
    exp14_model_dir = root / "models_frozen"
    exp14_pred_path = (
        root / "results_frozen_verification" / "fixed_test_predictions.csv"
    )

    required = [
        root / "features" / "X.npy",
        root / "features" / "y.npy",
        root / "features" / "metadata.csv",
        root / "train_test_data" / "train_indices.npy",
        root / "train_test_data" / "test_indices.npy",
        exp11_path,
        exp14_model_dir / "rbf_svm_full_train.joblib",
        exp14_model_dir / "rbf_svm_platt_calibrator_training_oof.joblib",
        exp14_model_dir / "cnn1d_full_train_final.pt",
        exp14_pred_path,
    ]

    print("=" * 96)
    print("RUNNING: src/fusion_pairs_verification.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("PAIRS           : 5 frozen handcrafted+deep fusion pairs (P1..P5)")
    print("SELECTION BASIS : stability_selection.py repeated TRAINING-ONLY "
          "group-aware OOF")
    print("TEST USE        : ONE frozen verification per pair; no tuning")
    print("=" * 96)

    for pair in FUSION_PAIRS:
        print(
            f"  {pair['pair_id']}: {pair['handcrafted_model']} + "
            f"{pair['deep_model']} | w={pair['weight']:.2f} "
            f"tau={pair['threshold']:.3f}"
        )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    exp11 = load_module(
        exp11_path,
        "exp11_for_all_pairs_verification",
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

    out_dir = root / "results_fusion_pairs_verification"
    model_dir = root / "models_fusion_pairs"

    if args.clean_output:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if model_dir.exists():
            shutil.rmtree(model_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print()
    print(f"PyTorch device     : {device}")
    if device.type == "cuda":
        print(
            f"GPU                : "
            f"{torch.cuda.get_device_name(0)}"
        )

    start_all = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage A: preload waveforms (800 train + 200 test, once)
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STAGE A: preload waveforms")
    print("=" * 96)

    train_waveforms = exp11.preload_training_waveforms(
        root,
        metadata,
        train_indices,
    )
    test_waveforms = derive_test_waveforms(
        exp11=exp11,
        root=root,
        metadata=metadata,
        test_indices=test_indices,
    )

    # ------------------------------------------------------------------
    # Stage B: load reused frozen_verification.py branches and verify
    # reproduction
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STAGE B: load frozen frozen_verification.py branches (SVM + CNN1D)")
    print("=" * 96)

    full_svm = joblib.load(
        exp14_model_dir / "rbf_svm_full_train.joblib"
    )
    svm_calibrator = joblib.load(
        exp14_model_dir / "rbf_svm_platt_calibrator_training_oof.joblib"
    )

    svm_test_score = exp11.classical_score(
        full_svm,
        X_test,
    )
    svm_test_prob = svm_calibrator.predict_proba(
        np.asarray(svm_test_score).reshape(-1, 1)
    )[:, 1]

    cnn1d = exp11.build_deep_model(
        "cnn1d",
        X_train.shape[1],
    ).to(device)
    checkpoint = torch.load(
        exp14_model_dir / "cnn1d_full_train_final.pt",
        map_location=device,
        weights_only=False,
    )
    cnn1d.load_state_dict(checkpoint["state_dict"])

    cnn1d_test_prob = infer_deep_model(
        exp11=exp11,
        model_key="cnn1d",
        model=cnn1d,
        device=device,
        waveforms=test_waveforms,
    )

    # Reproducibility audit: the reloaded branches must reproduce the
    # frozen_verification.py fixed-test probabilities exactly.
    exp14_pred = pd.read_csv(
        exp14_pred_path,
        encoding="utf-8-sig",
    )
    svm_diff = float(
        np.max(np.abs(
            exp14_pred["svm_probability"].to_numpy(dtype=np.float64)
            - svm_test_prob
        ))
    )
    cnn1d_diff = float(
        np.max(np.abs(
            exp14_pred["cnn1d_probability"].to_numpy(dtype=np.float64)
            - cnn1d_test_prob
        ))
    )
    print()
    print(f"Reload audit | max |SVM prob diff|   : {svm_diff:.3e}")
    print(f"Reload audit | max |CNN1D prob diff| : {cnn1d_diff:.3e}")
    if svm_diff > 1e-9 or cnn1d_diff > 1e-6:
        raise RuntimeError(
            "Reloaded frozen_verification.py branches do not reproduce the "
            "recorded fixed-test probabilities; aborting before any new "
            "evaluation."
        )
    print("Reload audit PASSED.")

    # ------------------------------------------------------------------
    # Stage C: train the remaining branches once each (shared by pairs)
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STAGE C: full 800-sample frozen branch training")
    print("=" * 96)

    # Random Forest (classical, shared by P3 and P5)
    print()
    print("Fitting frozen full-data random_forest...")
    exp11.set_all_seeds(SEED)
    random_forest = exp11.build_classical_model("random_forest")
    random_forest.fit(X_train, y_train)
    rf_test_prob = np.asarray(
        exp11.classical_score(random_forest, X_test),
        dtype=np.float64,
    )
    joblib.dump(
        random_forest,
        model_dir / "random_forest_full_train.joblib",
    )

    # MLP (torch, handcrafted-feature input, shared by P2 and P4)
    mlp, mlp_scaler, _ = train_full_deep_model(
        exp11=exp11,
        model_key="mlp",
        X_train=X_train,
        y_train=y_train,
        train_waveforms=train_waveforms,
        device=device,
    )
    mlp_test_prob = infer_deep_model(
        exp11=exp11,
        model_key="mlp",
        model=mlp,
        device=device,
        X=X_test,
        scaler=mlp_scaler,
    )
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "model_key": "mlp",
            "state_dict": mlp.state_dict(),
            "epochs": EPOCHS,
            "seed": SEED,
        },
        model_dir / "mlp_full_train_final.pt",
    )
    joblib.dump(
        mlp_scaler,
        model_dir / "mlp_standard_scaler.joblib",
    )

    # 2D CNN (waveform -> Log-Mel frontend, shared by P1, P2, P3)
    cnn2d, _, cnn2d_frontend = train_full_deep_model(
        exp11=exp11,
        model_key="cnn2d",
        X_train=X_train,
        y_train=y_train,
        train_waveforms=train_waveforms,
        device=device,
    )
    cnn2d_test_prob = infer_deep_model(
        exp11=exp11,
        model_key="cnn2d",
        model=cnn2d,
        device=device,
        waveforms=test_waveforms,
        frontend=cnn2d_frontend,
    )
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "model_key": "cnn2d",
            "state_dict": cnn2d.state_dict(),
            "epochs": EPOCHS,
            "seed": SEED,
        },
        model_dir / "cnn2d_full_train_final.pt",
    )

    # CRNN (waveform -> Log-Mel frontend, used by P5)
    crnn, _, crnn_frontend = train_full_deep_model(
        exp11=exp11,
        model_key="crnn",
        X_train=X_train,
        y_train=y_train,
        train_waveforms=train_waveforms,
        device=device,
    )
    crnn_test_prob = infer_deep_model(
        exp11=exp11,
        model_key="crnn",
        model=crnn,
        device=device,
        waveforms=test_waveforms,
        frontend=crnn_frontend,
    )
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "model_key": "crnn",
            "state_dict": crnn.state_dict(),
            "epochs": EPOCHS,
            "seed": SEED,
        },
        model_dir / "crnn_full_train_final.pt",
    )

    branch_test_probs: dict[str, np.ndarray] = {
        "rbf_svm_calibrated": svm_test_prob,
        "random_forest": rf_test_prob,
        "mlp": mlp_test_prob,
        "cnn1d": cnn1d_test_prob,
        "cnn2d": cnn2d_test_prob,
        "crnn": crnn_test_prob,
    }

    # ------------------------------------------------------------------
    # Stage D: ONE frozen fixed-test evaluation per pair
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STAGE D: ONE frozen fixed-test verification per pair")
    print("=" * 96)
    print("No weights or thresholds will be searched.")

    branch_metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "roc_auc",
        "fp",
        "fn",
    ]

    summary_rows: list[dict[str, Any]] = []

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

    for branch_key, prob in branch_test_probs.items():
        predictions[f"{branch_key}_probability"] = prob
        predictions[f"{branch_key}_prediction"] = (
            prob >= 0.5
        ).astype(np.int64)

    for pair in FUSION_PAIRS:
        hand_key = pair["handcrafted_model"]
        deep_key = pair["deep_model"]
        w = float(pair["weight"])
        tau = float(pair["threshold"])

        p_hand = branch_test_probs[hand_key]
        p_deep = branch_test_probs[deep_key]

        fusion_prob = w * p_hand + (1.0 - w) * p_deep

        hand_metrics = compute_metrics(y_test, p_hand, threshold=0.5)
        deep_metrics = compute_metrics(y_test, p_deep, threshold=0.5)
        fusion_metrics = compute_metrics(y_test, fusion_prob, threshold=tau)

        row: dict[str, Any] = {
            "pair_id": pair["pair_id"],
            "handcrafted_model": hand_key,
            "deep_model": deep_key,
            "weight": w,
            "deep_weight": 1.0 - w,
            **fusion_metrics,
        }
        for k in branch_metric_keys:
            row[f"handcrafted_branch_{k}"] = hand_metrics[k]
            row[f"deep_branch_{k}"] = deep_metrics[k]
        summary_rows.append(row)

        pair_tag = f"{pair['pair_id'].lower()}_fusion"
        predictions[f"{pair_tag}_probability"] = fusion_prob
        predictions[f"{pair_tag}_prediction"] = (
            fusion_prob >= tau
        ).astype(np.int64)
        predictions[f"{pair_tag}_correct"] = (
            predictions[f"{pair_tag}_prediction"].to_numpy()
            == y_test
        )

        print()
        print(
            f"{pair['pair_id']}: {hand_key} + {deep_key} | "
            f"w={w:.2f} tau={tau:.3f} | "
            f"F1={fusion_metrics['f1']:.5f} "
            f"Acc={fusion_metrics['accuracy']:.5f} "
            f"AUC={fusion_metrics['roc_auc']:.5f}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        out_dir / "verification_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        out_dir / "verification_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Frozen protocol metadata.
    protocol = {
        "script_version": SCRIPT_VERSION,
        "pairs_selected_by": (
            "stability_selection.py repeated TRAINING-ONLY group-aware OOF "
            "stability audit; weights and thresholds are the repeat "
            "median values, frozen before any fixed-test evaluation"
        ),
        "fusion_rule": (
            "fusion_prob = weight * p_handcrafted + "
            "(1 - weight) * p_deep; decision at the frozen threshold"
        ),
        "fusion_pairs": FUSION_PAIRS,
        "reused_branches": {
            "rbf_svm_calibrated": {
                "svm": str(exp14_model_dir / "rbf_svm_full_train.joblib"),
                "platt_calibrator": str(
                    exp14_model_dir
                    / "rbf_svm_platt_calibrator_training_oof.joblib"
                ),
                "source": "frozen_verification.py (800-sample full training "
                          "refit)",
            },
            "cnn1d": {
                "checkpoint": str(
                    exp14_model_dir / "cnn1d_full_train_final.pt"
                ),
                "source": "frozen_verification.py (40 epochs, seed 42)",
            },
        },
        "branches_trained_in_this_script": {
            key: {
                "epochs": EPOCHS,
                "seed": SEED,
                "architecture": "same recorded architecture as "
                                "model_selection_oof.py",
                "recipe": (
                    "AdamW lr=1e-3 wd=1e-4, cosine annealing, balanced "
                    "BCEWithLogits, AMP, grad clip 5.0, "
                    "model_selection_oof.py augmentation; no early stopping"
                ) if key in {"mlp", "cnn2d", "crnn"} else (
                    "model_selection_oof.py build_classical_model "
                    "configuration"
                ),
            }
            for key in ["random_forest", "mlp", "cnn2d", "crnn"]
        },
        "reload_audit": {
            "svm_max_abs_prob_diff_vs_frozen_verification": svm_diff,
            "cnn1d_max_abs_prob_diff_vs_frozen_verification": cnn1d_diff,
        },
        "test_samples": int(len(y_test)),
        "test_groups": int(len(np.unique(groups_test))),
        "train_test_sample_overlap": int(sample_overlap),
        "train_test_group_overlap": int(group_overlap),
        "no_test_tuning_performed": True,
    }

    with open(
        out_dir / "protocol.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            protocol,
            f,
            ensure_ascii=False,
            indent=2,
        )

    total_elapsed = time.perf_counter() - start_all

    # Console result.
    print()
    print("=" * 96)
    print("FROZEN FIXED-TEST VERIFICATION FINISHED")
    print("=" * 96)
    print(
        summary[
            [
                "pair_id",
                "handcrafted_model",
                "deep_model",
                "weight",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "roc_auc",
                "fp",
                "fn",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )
    print()
    print("Branch-only metrics (threshold=0.5):")
    branch_cols = ["pair_id"]
    for prefix in ["handcrafted_branch", "deep_branch"]:
        branch_cols.extend(
            f"{prefix}_{k}" for k in ["accuracy", "recall", "f1", "roc_auc"]
        )
    print(
        summary[branch_cols].to_string(
            index=False,
            float_format=lambda v: f"{v:.5f}",
        )
    )
    print()
    print(f"Total time : {total_elapsed:.1f} s")
    print(f"Results    : {out_dir}")
    print(f"Models     : {model_dir}")
    print()
    print("IMPORTANT: Do not change weights/threshold based on these results.")


if __name__ == "__main__":
    main()
