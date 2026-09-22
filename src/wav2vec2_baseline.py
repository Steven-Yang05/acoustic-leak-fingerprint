from __future__ import annotations

"""
wav2vec2_baseline.py: pretrained audio representation audit (wav2vec 2.0).

The original benchmark (model_selection_oof.py / fusion_pair_search.py /
stability_selection.py) compared seven candidate models trained from
scratch. Reviewers of small-sample acoustic work reasonably ask whether a
large pretrained audio model changes the conclusion. This experiment
audits wav2vec 2.0 Base (94M parameters, pretrained on 960 h of
LibriSpeech) under exactly the same group-aware protocol as the
from-scratch models:

  1) frozen wav2vec2 embeddings (mean-pooled final layer, 768-d)
     + logistic regression
  2) frozen wav2vec2 embeddings + RBF-SVM
  3) full fine-tuning of wav2vec2 + linear head

All three use the same 5-fold StratifiedGroupKFold (seed 42) on the 800
training samples as model_selection_oof.py, then a frozen refit on all
800 and a single evaluation on the fixed 200-sample verification
partition, so the numbers are directly comparable to the main results
table of the manuscript.

Waveforms are preprocessed exactly like the 1D-CNN branch (mono, 8 kHz,
8000 samples, DC removed, peak-scaled), then resampled to 16 kHz for
wav2vec2. Waveform augmentation for fine-tuning is the recorded protocol
(gain / shift / Gaussian noise) applied at 8 kHz before resampling.

Note: the first run downloads the torchaudio WAV2VEC2_BASE weights
(Internet access required).

Outputs: results_wav2vec2/
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_VERSION = "pretrained-wav2vec2-audit-v1"

SEED = 42
FOLDS = 5
FT_EPOCHS = 15
FT_BATCH = 16
FT_LR = 2e-5
FT_WEIGHT_DECAY = 0.01
TARGET_SR = 16000

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 4)


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class W2V2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(768, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, _ = self.backbone.extract_features(x.squeeze(1))
        z = feats[-1].mean(dim=1)
        return self.head(z).squeeze(1)


class W2V2Dataset(Dataset):
    """8-kHz waveforms; augmentation at 8 kHz; resample to 16 kHz."""

    def __init__(self, waves8k: torch.Tensor, y: np.ndarray, indices: np.ndarray, augment: bool, exp11) -> None:
        self.waves = waves8k
        self.y = np.asarray(y, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.exp11 = exp11

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int):
        idx = int(self.indices[j])
        x = self.waves[idx]  # [1, 8000]
        if self.augment:
            x = self.exp11.augment_waveform(x)
        x16 = torchaudio.functional.resample(x, 8000, TARGET_SR)
        return x16, torch.tensor(self.y[idx], dtype=torch.float32)


def extract_embeddings(backbone: nn.Module, ds: Dataset, device: torch.device) -> np.ndarray:
    backbone.eval()
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            feats, _ = backbone.extract_features(xb.squeeze(1))
            z = feats[-1].mean(dim=1)
            out.append(z.float().cpu().numpy())
    emb = np.concatenate(out, axis=0)
    return emb.astype(np.float32)


def finetune_fold(
    bundle,
    waves8k: torch.Tensor,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    device: torch.device,
    exp11,
) -> tuple[np.ndarray, nn.Module]:
    exp11.set_all_seeds(SEED)

    model = W2V2Classifier(bundle.get_model()).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [(int((y[fit_idx] == 0).sum())) / max(int((y[fit_idx] == 1).sum()), 1)],
            dtype=torch.float32, device=device,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT_EPOCHS, eta_min=1e-6)
    amp_enabled = device.type == "cuda"
    scaler = exp11.make_grad_scaler(amp_enabled)

    train_ds = W2V2Dataset(waves8k, y, fit_idx, augment=True, exp11=exp11)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(train_ds, batch_size=FT_BATCH, shuffle=True, num_workers=0,
                        pin_memory=amp_enabled, generator=generator)

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with exp11.autocast_context(amp_enabled):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().item()) * int(yb.shape[0])
            seen += int(yb.shape[0])
        scheduler.step()
        if epoch in {1, 5, 10, FT_EPOCHS}:
            print(f"      epoch {epoch:03d}/{FT_EPOCHS} loss={total / max(seen, 1):.6f}")

    probs = predict_finetuned(model, waves8k, y, eval_idx, device, exp11)
    return probs, model


def predict_finetuned(model: nn.Module, waves8k: torch.Tensor, y: np.ndarray,
                      idx: np.ndarray, device: torch.device, exp11) -> np.ndarray:
    model.eval()
    ds = W2V2Dataset(waves8k, y, idx, augment=False, exp11=exp11)
    loader = DataLoader(ds, batch_size=FT_BATCH, shuffle=False, num_workers=0)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            with exp11.autocast_context(device.type == "cuda"):
                p = torch.sigmoid(model(xb))
            out.append(p.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def quick_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    pred = (score >= threshold).astype(np.int64)
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


def best_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    best_key = None
    best_tau = 0.5
    for tau in THRESHOLD_GRID:
        m = quick_metrics(y, prob, float(tau))
        key = (m["f1"], m["recall"], m["specificity"], m["roc_auc"], m["average_precision"])
        if best_key is None or key > best_key:
            best_key = key
            best_tau = float(tau)
    return best_tau


def main() -> None:
    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results_wav2vec2"
    out_dir.mkdir(exist_ok=True)

    exp11 = load_module(root / "src" / "model_selection_oof.py", "exp11_w2v2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    metadata = pd.read_csv(root / "features" / "metadata.csv", encoding="utf-8-sig")
    y_all = np.load(root / "features" / "y.npy", allow_pickle=False).astype(np.int64)
    groups_all = exp11.derive_group_ids(metadata)
    train_indices = np.load(root / "train_test_data" / "train_indices.npy", allow_pickle=False).astype(np.int64)
    test_indices = np.load(root / "train_test_data" / "test_indices.npy", allow_pickle=False).astype(np.int64)

    print("loading all 1000 waveforms (8 kHz, peak-scaled, CNN-branch preprocessing) ...")
    waves8k = torch.stack([
        exp11.load_one_waveform(exp11.resolve_wav_path(root, metadata.iloc[int(i)]["file_path"]))
        for i in range(len(metadata))
    ])

    try:
        from torchaudio.pipelines import WAV2VEC2_BASE
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torchaudio WAV2VEC2_BASE bundle unavailable: {exc}")
    bundle = WAV2VEC2_BASE

    # ---------------- frozen-embedding models ----------------
    emb_path = root / "features" / "wav2vec2_base_embeddings.npy"
    if emb_path.exists():
        print(f"loading cached embeddings: {emb_path}")
        emb_all = np.load(emb_path)
        if emb_all.shape != (len(metadata), 768):
            raise RuntimeError(f"cached embedding shape mismatch: {emb_all.shape}")
    else:
        print("extracting wav2vec2 embeddings for all 1000 clips ...")
        backbone = bundle.get_model().to(device)
        ds_all = W2V2Dataset(waves8k, y_all, np.arange(len(metadata)), augment=False, exp11=exp11)
        emb_all = extract_embeddings(backbone, ds_all, device)
        np.save(emb_path, emb_all)
        del backbone
        torch.cuda.empty_cache()
        print(f"  saved {emb_path} shape={emb_all.shape}")

    y_train = y_all[train_indices]
    groups_train = groups_all[train_indices]
    emb_train = emb_all[train_indices]
    emb_test = emb_all[test_indices]
    y_test = y_all[test_indices]

    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(emb_train, y_train, groups_train))

    oof_rows: list[dict[str, Any]] = []
    ver_rows: list[dict[str, Any]] = []

    def run_embedding_model(model_key: str, label: str) -> None:
        oof = np.full(len(y_train), np.nan, dtype=np.float64)
        fold_f1: list[float] = []
        for fit_idx, held_idx in folds:
            clf = exp11.build_classical_model(model_key)
            clf.fit(emb_train[fit_idx], y_train[fit_idx])
            oof[held_idx] = exp11.classical_score(clf, emb_train[held_idx])
            fold_f1.append(quick_metrics(y_train[held_idx], oof[held_idx], 0.5)["f1"])

        m = quick_metrics(y_train, oof, 0.5)
        oof_rows.append({"model": label, "stage": "training_oof", "threshold": 0.5,
                         "fold_f1_sd": float(np.std(fold_f1, ddof=1)), **m})
        print(f"  {label}: OOF F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")

        tau = best_threshold(y_train, oof)
        clf_full = exp11.build_classical_model(model_key)
        clf_full.fit(emb_train, y_train)
        score_test = exp11.classical_score(clf_full, emb_test)
        for thr_label, thr in [("default_0.5", 0.5), ("oof_tuned", tau)]:
            mv = quick_metrics(y_test, score_test, thr)
            ver_rows.append({"model": label, "threshold_rule": thr_label, "threshold": thr, **mv})
            print(f"    verification ({thr_label} tau={thr:.3f}): F1={mv['f1']:.4f} acc={mv['accuracy']:.4f}")

    print()
    print("=== frozen wav2vec2 embeddings + classical heads ===")
    run_embedding_model("logistic_regression", "w2v2_embedding_lr")
    run_embedding_model("rbf_svm", "w2v2_embedding_svm")

    # ---------------- full fine-tuning ----------------
    print()
    print("=== full fine-tuning of wav2vec2 + linear head ===")
    oof_ft = np.full(len(y_train), np.nan, dtype=np.float64)
    fold_f1_ft: list[float] = []
    for k, (fit_idx, held_idx) in enumerate(folds, 1):
        print(f"  fine-tune fold {k}/{FOLDS} (fit={len(fit_idx)}, held={len(held_idx)})")
        fit_global = train_indices[fit_idx]
        held_global = train_indices[held_idx]
        probs, model = finetune_fold(bundle, waves8k, y_all, fit_global, held_global, device, exp11)
        oof_ft[held_idx] = probs
        fold_f1_ft.append(quick_metrics(y_all[held_global], probs, 0.5)["f1"])
        del model
        torch.cuda.empty_cache()

    m = quick_metrics(y_train, oof_ft, 0.5)
    oof_rows.append({"model": "w2v2_finetuned", "stage": "training_oof", "threshold": 0.5,
                     "fold_f1_sd": float(np.std(fold_f1_ft, ddof=1)), **m})
    print(f"  w2v2_finetuned: OOF F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")

    tau_ft = best_threshold(y_train, oof_ft)
    print("  final fine-tune on all 800 training samples ...")
    _, model_full = finetune_fold(bundle, waves8k, y_all, train_indices, train_indices, device, exp11)
    ft_test_prob = predict_finetuned(model_full, waves8k, y_all, test_indices, device, exp11)
    for thr_label, thr in [("default_0.5", 0.5), ("oof_tuned", tau_ft)]:
        mv = quick_metrics(y_test, ft_test_prob, thr)
        ver_rows.append({"model": "w2v2_finetuned", "threshold_rule": thr_label, "threshold": thr, **mv})
        print(f"    verification ({thr_label} tau={thr:.3f}): F1={mv['f1']:.4f} acc={mv['accuracy']:.4f}")

    pd.DataFrame(oof_rows).to_csv(out_dir / "pretrained_oof_results.csv", index=False)
    pd.DataFrame(ver_rows).to_csv(out_dir / "pretrained_verification_results.csv", index=False)
    pd.DataFrame({
        "y_true": y_train,
        "emb_lr_oof": np.nan,
        "w2v2_finetuned_oof": oof_ft,
    }).to_csv(out_dir / "pretrained_oof_predictions.csv", index=False)

    protocol = {
        "script_version": SCRIPT_VERSION,
        "backbone": "torchaudio WAV2VEC2_BASE (wav2vec2-base, 94M params, LibriSpeech 960h)",
        "input": "CNN-branch preprocessing (peak-scaled 8 kHz waveform) resampled to 16 kHz",
        "embedding": "mean-pooled final transformer layer, 768-d, frozen encoder",
        "heads": ["logistic_regression (C=1.0, liblinear, balanced)",
                  "rbf_svm (C=10, gamma=scale, balanced)"],
        "finetune": {"epochs": FT_EPOCHS, "batch": FT_BATCH, "lr": FT_LR,
                     "weight_decay": FT_WEIGHT_DECAY, "full_backbone": True,
                     "augmentation": "recorded waveform protocol at 8 kHz before resampling"},
        "evaluation": "same 5-fold StratifiedGroupKFold (seed 42) on 800 training samples; "
                      "frozen refit on all 800; single evaluation on fixed 200-sample partition",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "pretrained_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {out_dir}")


if __name__ == "__main__":
    main()
