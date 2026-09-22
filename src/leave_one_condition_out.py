from __future__ import annotations

"""
leave_one_condition_out.py: leave-one-condition-out (LOCO) generalization audit.

The original fixed verification set is an internal, same-site partition.
This experiment instead holds out an entire operating condition:

  - device type : hydrophone <-> noise logger
  - pipe material: ductile iron <-> PE

For each direction, all development (group-aware OOF, nested Platt
calibration, fusion weight / threshold search) is performed on the source
condition only. The frozen pipeline is then evaluated once on the unseen
target condition. This measures cross-condition generalization of the
calibrated RBF-SVM, the 1D CNN, and their frozen fusion.

Only leak and no_leak clips are used. Environmental-noise clips are
excluded because they exist almost entirely on noise-logger devices and
carry no pipe-material metadata, which would confound the condition
split. Pressure and flow velocity cannot be used as LOCO variables
because no_leak clips have no such metadata.

Outputs: results_loco/
"""

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_VERSION = "leave-one-condition-out-v1"

SEED = 42
FOLDS = 5
EPOCHS = 40

WEIGHT_GRID = np.round(np.arange(0.05, 0.951, 0.05), 4)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 4)


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_conditions(metadata: pd.DataFrame) -> pd.DataFrame:
    """Parse pipe material and device type from filenames.

    leak (7 fields)   : material-area-pressure-flow-device-X-Y
    no_leak (6 fields): material-area-pressure-flow-device_N-Y
    noise             : excluded (no material metadata; single device type)
    """
    material: list[str] = []
    device: list[str] = []
    for _, row in metadata.iterrows():
        stem = Path(str(row["file_name"])).stem
        source_class = str(row["source_class"])
        parts = stem.split("-")

        mat = "NA"
        dev = "NA"
        if source_class == "leak" and len(parts) == 7:
            mat = parts[0]
            dev = parts[4]
        elif source_class == "no_leak" and len(parts) == 6:
            mat = parts[0]
            dev = re.sub(r"_\d+$", "", parts[4])

        material.append(mat)
        device.append(dev)

    out = metadata.copy()
    out["cond_material"] = material
    out["cond_device"] = device
    return out


def nested_platt_oof(
    exp11,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware OOF calibrated probabilities for the RBF-SVM.

    Returns (oof_prob, oof_decision_score). Within each outer fold, inner
    group-aware OOF decision scores fit a Platt logistic mapper; the SVM
    is refit on the full outer-fit subset and outer held-out scores are
    calibrated with that mapping. Same protocol as Section 3.6.
    """
    outer = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_prob = np.full(len(y), np.nan, dtype=np.float64)
    oof_score = np.full(len(y), np.nan, dtype=np.float64)

    for fit_idx, held_idx in outer.split(X, y, groups):
        # inner OOF decision scores on the outer-fit subset
        inner = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        inner_score = np.full(len(fit_idx), np.nan, dtype=np.float64)
        for in_fit, in_held in inner.split(X[fit_idx], y[fit_idx], groups[fit_idx]):
            svm_in = exp11.build_classical_model("rbf_svm")
            svm_in.fit(X[fit_idx][in_fit], y[fit_idx][in_fit])
            inner_score[in_held] = exp11.classical_score(svm_in, X[fit_idx][in_held])

        platt = LogisticRegression(solver="lbfgs", C=1e6, max_iter=3000, random_state=SEED)
        platt.fit(inner_score.reshape(-1, 1), y[fit_idx])

        svm = exp11.build_classical_model("rbf_svm")
        svm.fit(X[fit_idx], y[fit_idx])
        score = exp11.classical_score(svm, X[held_idx])
        oof_score[held_idx] = score
        oof_prob[held_idx] = platt.predict_proba(score.reshape(-1, 1))[:, 1]

    return oof_prob, oof_score


def cnn_oof(
    exp11,
    X_src: np.ndarray,
    y_src: np.ndarray,
    waves_src: torch.Tensor,
    groups_src: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    outer = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_prob = np.full(len(y_src), np.nan, dtype=np.float64)
    for fold, (fit_idx, held_idx) in enumerate(outer.split(X_src, y_src, groups_src), 1):
        print(f"    CNN OOF fold {fold}/{FOLDS} (fit={len(fit_idx)}, held={len(held_idx)})")
        _, score, _ = exp11.train_deep_fold(
            "cnn1d", fit_idx, held_idx, X_src, y_src, waves_src, device, EPOCHS,
        )
        oof_prob[held_idx] = score
    return oof_prob


def search_fusion(
    y: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> dict[str, float]:
    """Joint grid search over handcrafted weight w and threshold tau.

    Lexicographic selection: F1 -> Recall -> Specificity -> ROC-AUC -> AP.
    """
    best: dict[str, float] | None = None
    for w in WEIGHT_GRID:
        fused = w * prob_a + (1.0 - w) * prob_b
        for tau in THRESHOLD_GRID:
            pred = (fused >= tau).astype(np.int64)
            m = quick_metrics(y, pred, fused)
            key = (m["f1"], m["recall"], m["specificity"], m["roc_auc"], m["average_precision"])
            if best is None or key > best["key"]:
                best = {"key": key, "weight_a": float(w), "threshold": float(tau), **m}
    assert best is not None
    best.pop("key")
    return best


def best_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best: dict[str, float] | None = None
    best_tau = 0.5
    for tau in THRESHOLD_GRID:
        pred = (prob >= tau).astype(np.int64)
        m = quick_metrics(y, pred, prob)
        key = (m["f1"], m["recall"], m["specificity"], m["roc_auc"], m["average_precision"])
        if best is None or key > best["key"]:
            best = m
            best["key"] = key
            best_tau = float(tau)
    assert best is not None
    best.pop("key")
    return best_tau, best


def quick_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "fp": int(fp),
        "fn": int(fn),
    }


def run_task(
    exp11,
    exp14,
    task_name: str,
    variable: str,
    source_level: str,
    target_level: str,
    cond: pd.DataFrame,
    X_all: np.ndarray,
    y_all: np.ndarray,
    groups_all: np.ndarray,
    root: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    on_site = cond["source_class"].isin(["leak", "no_leak"]).to_numpy()
    src_mask = on_site & (cond[variable] == source_level).to_numpy()
    tgt_mask = on_site & (cond[variable] == target_level).to_numpy()

    src_idx = np.where(src_mask)[0]
    tgt_idx = np.where(tgt_mask)[0]

    X_src, y_src = X_all[src_idx], y_all[src_idx]
    X_tgt, y_tgt = X_all[tgt_idx], y_all[tgt_idx]
    groups_src = groups_all[src_idx]

    n_groups = len(np.unique(groups_src))
    print()
    print("=" * 96)
    print(f"TASK {task_name}: {variable}: {source_level} -> {target_level}")
    print(f"  source: {len(src_idx)} samples / {n_groups} groups "
          f"(pos={int(y_src.sum())}, neg={int((1 - y_src).sum())})")
    print(f"  target: {len(tgt_idx)} samples "
          f"(pos={int(y_tgt.sum())}, neg={int((1 - y_tgt).sum())})")

    # ---------------- source-side development ----------------
    print("  [1/4] nested Platt-calibrated SVM OOF on source ...")
    svm_oof_prob, svm_oof_score = nested_platt_oof(exp11, X_src, y_src, groups_src)

    print("  [2/4] 1D CNN OOF on source ...")
    waves_src = torch.stack([
        exp11.load_one_waveform(exp11.resolve_wav_path(root, cond.iloc[int(i)]["file_path"]))
        for i in src_idx
    ])
    cnn_oof_prob = cnn_oof(exp11, X_src, y_src, waves_src, groups_src, device)

    print("  [3/4] fusion weight / threshold search on source OOF ...")
    fusion_sel = search_fusion(y_src, svm_oof_prob, cnn_oof_prob)
    tau_svm, svm_oof_best = best_threshold(y_src, svm_oof_prob)
    tau_cnn, cnn_oof_best = best_threshold(y_src, cnn_oof_prob)
    w = fusion_sel["weight_a"]
    tau = fusion_sel["threshold"]
    print(f"    fusion: w_svm={w:.2f}, tau={tau:.3f}, OOF F1={fusion_sel['f1']:.4f}")
    print(f"    svm tau={tau_svm:.3f} (OOF F1={svm_oof_best['f1']:.4f}), "
          f"cnn tau={tau_cnn:.3f} (OOF F1={cnn_oof_best['f1']:.4f})")

    # ---------------- frozen refit on all source data ----------------
    print("  [4/4] frozen refit on source and one-shot target evaluation ...")
    platt = LogisticRegression(solver="lbfgs", C=1e6, max_iter=3000, random_state=SEED)
    platt.fit(svm_oof_score.reshape(-1, 1), y_src)
    full_svm = exp11.build_classical_model("rbf_svm")
    full_svm.fit(X_src, y_src)

    full_cnn = exp14.train_full_cnn1d(
        exp11=exp11, X_train=X_src, y_train=y_src,
        train_waveforms=waves_src, device=device,
    )

    # target predictions
    svm_tgt_score = exp11.classical_score(full_svm, X_tgt)
    svm_tgt_prob = platt.predict_proba(np.asarray(svm_tgt_score).reshape(-1, 1))[:, 1]

    waves_tgt = torch.stack([
        exp11.load_one_waveform(exp11.resolve_wav_path(root, cond.iloc[int(i)]["file_path"]))
        for i in tgt_idx
    ])
    cnn_tgt_prob = exp14.infer_cnn1d(exp11=exp11, model=full_cnn, waveforms=waves_tgt, device=device)

    fusion_tgt_prob = w * svm_tgt_prob + (1.0 - w) * cnn_tgt_prob

    rows: list[dict[str, Any]] = []

    def add_row(split: str, model: str, y_true: np.ndarray, prob: np.ndarray, thr: float) -> None:
        pred = (prob >= thr).astype(np.int64)
        m = quick_metrics(y_true, pred, prob)
        rows.append({
            "task": task_name,
            "variable": variable,
            "direction": f"{source_level}->{target_level}",
            "split": split,
            "model": model,
            "threshold": float(thr),
            "fusion_weight_svm": float(w) if model == "fusion" else np.nan,
            **m,
        })

    add_row("source_oof", "svm_calibrated", y_src, svm_oof_prob, tau_svm)
    add_row("source_oof", "cnn1d", y_src, cnn_oof_prob, tau_cnn)
    add_row("source_oof", "fusion", y_src, w * svm_oof_prob + (1 - w) * cnn_oof_prob, tau)
    add_row("target_loco", "svm_calibrated", y_tgt, svm_tgt_prob, tau_svm)
    add_row("target_loco", "cnn1d", y_tgt, cnn_tgt_prob, tau_cnn)
    add_row("target_loco", "fusion", y_tgt, fusion_tgt_prob, tau)

    # save target predictions for later paired tests
    out_dir = root / "results_loco"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({
        "index": tgt_idx,
        "y_true": y_tgt,
        "svm_prob": svm_tgt_prob,
        "cnn_prob": cnn_tgt_prob,
        "fusion_prob": fusion_tgt_prob,
    }).to_csv(out_dir / f"target_predictions_{task_name}.csv", index=False)

    return rows


def main() -> None:
    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results_loco"
    out_dir.mkdir(exist_ok=True)

    exp11 = load_module(root / "src" / "model_selection_oof.py", "exp11_loco")
    exp14 = load_module(root / "src" / "frozen_verification.py", "exp14_loco")

    X_all = np.load(root / "features" / "X.npy", allow_pickle=False)
    y_all = np.load(root / "features" / "y.npy", allow_pickle=False).astype(np.int64)
    metadata = pd.read_csv(root / "features" / "metadata.csv", encoding="utf-8-sig")
    cond = parse_conditions(metadata)
    cond.to_csv(out_dir / "parsed_conditions.csv", index=False)
    groups_all = exp11.derive_group_ids(metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tasks = [
        ("device_hydro_to_logger", "cond_device", "hydrophone", "noise logger"),
        ("device_logger_to_hydro", "cond_device", "noise logger", "hydrophone"),
        ("material_di_to_pe", "cond_material", "ductile iron", "pe"),
        ("material_pe_to_di", "cond_material", "pe", "ductile iron"),
    ]

    all_rows: list[dict[str, Any]] = []
    for task_name, variable, source_level, target_level in tasks:
        all_rows.extend(run_task(
            exp11, exp14, task_name, variable, source_level, target_level,
            cond, X_all, y_all, groups_all, root, device,
        ))

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "loco_results.csv", index=False)

    protocol = {
        "script_version": SCRIPT_VERSION,
        "seed": SEED,
        "folds": FOLDS,
        "epochs": EPOCHS,
        "classes_used": ["leak", "no_leak"],
        "noise_clips_excluded": True,
        "reason_noise_excluded": (
            "environmental-noise clips carry no pipe-material metadata and "
            "exist almost entirely on noise-logger devices, which would "
            "confound the condition split"
        ),
        "variables": {
            "cond_device": ["hydrophone", "noise logger"],
            "cond_material": ["ductile iron", "pe"],
            "pressure_flow_unusable": "no_leak clips have no pressure/flow metadata",
        },
        "selection": "source-side 5-fold StratifiedGroupKFold OOF (seed 42); "
                     "nested group-aware Platt calibration for the SVM; "
                     "fusion weight/threshold and component thresholds from "
                     "source OOF only, lexicographic F1->Recall->Spec->AUC->AP",
        "target_use": "single frozen evaluation per direction; no target feedback",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "loco_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print()
    print(df.to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {out_dir}")


if __name__ == "__main__":
    main()
