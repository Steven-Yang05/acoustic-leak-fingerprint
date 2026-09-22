from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import soundfile as sf


SCRIPT_VERSION = "seven-model-oof-model-selection-v1"

# ---------------------------------------------------------------------
# IMPORTANT DESIGN RULE
# ---------------------------------------------------------------------
# This script is intentionally TRAINING-ONLY.
#
# It reads:
#   features/X.npy
#   features/y.npy
#   features/metadata.csv
#   train_test_data/train_indices.npy
#
# It DOES NOT read:
#   X_test.npy
#   y_test.npy
#   test_indices.npy
#   any prior test prediction/result CSV
#
# The purpose is to answer:
# "If the fixed 200-sample test set were unavailable, which model(s)
#  would the 800-sample training partition select by group-aware OOF?"
# ---------------------------------------------------------------------


# =========================
# Fixed experimental setup
# =========================
SEED = 42
N_SPLITS = 5
EPOCHS = 40

SAMPLE_RATE = 8000
NUM_SAMPLES = 8000

N_FFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 80
N_MELS = 64
F_MIN = 20.0
F_MAX = 4000.0

BATCH_SIZE = {
    "mlp": 64,
    "cnn1d": 64,
    "cnn2d": 32,
    "crnn": 32,
}

MODEL_ORDER = [
    "logistic_regression",
    "rbf_svm",
    "random_forest",
    "mlp",
    "cnn1d",
    "cnn2d",
    "crnn",
]

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "rbf_svm": "RBF-SVM",
    "random_forest": "Random Forest",
    "mlp": "MLP",
    "cnn1d": "1D CNN",
    "cnn2d": "2D CNN",
    "crnn": "CRNN",
}

HANDCRAFTED_FAMILY = [
    "logistic_regression",
    "rbf_svm",
    "random_forest",
    "mlp",
]

DEEP_ACOUSTIC_FAMILY = [
    "cnn1d",
    "cnn2d",
    "crnn",
]

# Selection order matches the later OOF fusion/dynamic-gate protocol:
# F1 -> Recall -> Specificity -> ROC-AUC -> AP
SELECTION_COLUMNS = [
    "oof_f1",
    "oof_recall",
    "oof_specificity",
    "oof_roc_auc",
    "oof_average_precision",
]


# =========================
# Utility functions
# =========================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is preferred over maximum speed for this audit.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def derive_group_ids(metadata: pd.DataFrame) -> np.ndarray:
    """
    Reproduce the established project grouping rule:
      group_id = source_class :: base_recording_name

    base_recording_name is obtained by removing a trailing _<digits>
    suffix from the WAV stem, so:
      xxx.wav, xxx_1.wav, xxx_2.wav -> the same recording group.
    """
    required = {"file_name", "source_class"}
    missing = required.difference(metadata.columns)
    if missing:
        raise RuntimeError(
            "metadata.csv is missing required columns for group reconstruction: "
            + ", ".join(sorted(missing))
        )

    groups: list[str] = []
    for _, row in metadata.iterrows():
        stem = Path(str(row["file_name"])).stem
        base = re.sub(r"_\d+$", "", stem)
        source_class = str(row["source_class"])
        groups.append(f"{source_class}::{base}")
    return np.asarray(groups, dtype=object)


def resolve_wav_path(project_root: Path, raw_value: Any) -> Path:
    raw = str(raw_value).strip()
    path = Path(raw)
    if path.is_absolute():
        return path

    # Normal path used by this project: data\leak\xxx.wav
    p1 = project_root / path
    if p1.exists():
        return p1

    # Be tolerant of Windows separators embedded in a CSV.
    p2 = project_root / Path(raw.replace("\\", "/"))
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Cannot resolve WAV path: {raw_value}")


def load_one_waveform(path: Path) -> torch.Tensor:
    """
    Final deep-model preprocessing recorded in the experiment:
      - mono
      - 8 kHz
      - exactly 8000 samples
      - remove DC offset
      - per-file peak scaling to [-1, 1]
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = np.mean(data, axis=1).astype(np.float32)

    x = torch.from_numpy(wave)

    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)

    if x.numel() < NUM_SAMPLES:
        x = torch.nn.functional.pad(x, (0, NUM_SAMPLES - x.numel()))
    elif x.numel() > NUM_SAMPLES:
        x = x[:NUM_SAMPLES]

    x = x - x.mean()
    peak = x.abs().max()
    if torch.isfinite(peak) and peak > 0:
        x = x / peak

    if not torch.isfinite(x).all():
        raise ValueError(f"Non-finite waveform after preprocessing: {path}")

    return x.unsqueeze(0).float()  # [1, 8000]


def preload_training_waveforms(
    project_root: Path,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
) -> torch.Tensor:
    if "file_path" not in metadata.columns:
        raise RuntimeError(
            "metadata.csv does not contain 'file_path'. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    waves: list[torch.Tensor] = []
    total = len(train_indices)

    print()
    print("Preloading the 800 TRAINING waveforms only...")
    for j, source_index in enumerate(train_indices, start=1):
        raw = metadata.iloc[int(source_index)]["file_path"]
        path = resolve_wav_path(project_root, raw)
        waves.append(load_one_waveform(path))

        if j % 100 == 0 or j == total:
            print(f"  loaded {j}/{total}")

    result = torch.stack(waves, dim=0)  # [N, 1, 8000]
    if result.shape != (len(train_indices), 1, NUM_SAMPLES):
        raise RuntimeError(f"Unexpected waveform tensor shape: {tuple(result.shape)}")
    return result


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# Deep model definitions
# =========================
class MLP(nn.Module):
    """
    Exact recorded architecture:
      212 -> 128 -> 64 -> 1
      BN after first linear
      Dropout 0.35 / 0.25
    Expected trainable params: 35,841
    """
    def __init__(self, input_dim: int = 212) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class CNN1D(nn.Module):
    """
    Exact recorded architecture:
      Conv1: 1->32, k=9, stride=2, BN/ReLU/MaxPool(4)
      Conv2: 32->64, k=7, stride=1, BN/ReLU/MaxPool(4)
      Conv3: 64->128, k=5, stride=1, BN/ReLU/MaxPool(4)
      Conv4: 128->128, k=3, stride=1, BN/ReLU
      AdaptiveAvgPool -> Dropout(0.30) -> Linear
    Conv biases are disabled because BN follows each convolution.
    Expected trainable params: 105,569
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.30)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


class CNN2D(nn.Module):
    """
    Exact recorded architecture:
      Conv2d 1->16->32->64->128, all k=3, BN/ReLU
      MaxPool(2,2) after first three blocks
      AdaptiveAvgPool -> Dropout(0.35) -> Linear(128,1)
    Expected trainable params: 97,521
    """
    def __init__(self) -> None:
        super().__init__()

        def block(cin: int, cout: int, pool: bool) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 16, True),
            block(16, 32, True),
            block(32, 64, True),
            block(64, 128, False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


class CRNN(nn.Module):
    """
    Exact recorded architecture:
      Conv2d 1->16 -> MaxPool(2,2)
      Conv2d 16->32 -> MaxPool(2,2)
      [32 x 16] per time step -> Linear(512,128)
      1-layer bidirectional GRU(input=128, hidden=64)
      temporal mean -> Dropout(0.35) -> Linear(128,1)
    Expected trainable params: 145,137
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.projection = nn.Linear(32 * 16, 128)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # [B,32,16,T]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,T,32,16]
        x = x.flatten(2)  # [B,T,512]
        x = self.projection(x)  # [B,T,128]
        x, _ = self.gru(x)  # [B,T,128]
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


EXPECTED_PARAM_COUNTS = {
    "mlp": 35841,
    "cnn1d": 105569,
    "cnn2d": 97521,
    "crnn": 145137,
}


class LogMelFrontend(nn.Module):
    """
    64-band Log-Mel frontend used by CNN2D/CRNN.

    Recorded settings:
      n_fft=256, win_length=256, hop_length=80
      n_mels=64, f=20..4000 Hz, power=2
      Slaney mel scale + Slaney norm
      top_db=80
      map dB values to [0,1]
    """
    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            center=True,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,8000]
        b, c, n = x.shape
        x2 = x.reshape(b * c, n)
        mel = self.mel(x2)  # [B,64,101]
        db = self.to_db(mel)
        scaled = torch.clamp((db + 80.0) / 80.0, 0.0, 1.0)
        return scaled.unsqueeze(1)  # [B,1,64,101]


# =========================
# Training datasets
# =========================
class VectorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def shift_with_zeros(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    y = torch.zeros_like(x)
    if shift > 0:
        y[..., shift:] = x[..., :-shift]
    else:
        k = -shift
        y[..., :-k] = x[..., k:]
    return y


def augment_waveform(x: torch.Tensor) -> torch.Tensor:
    """
    Training-only waveform augmentation from the recorded protocol:
      gain p=.70, -6..+6 dB
      shift p=.60, max 10%
      Gaussian noise p=.55, target SNR 20..40 dB
    """
    x = x.clone()

    if torch.rand(()) < 0.70:
        gain_db = torch.empty(()).uniform_(-6.0, 6.0)
        x = x * torch.pow(torch.tensor(10.0), gain_db / 20.0)

    if torch.rand(()) < 0.60:
        max_shift = int(0.10 * NUM_SAMPLES)
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        x = shift_with_zeros(x, shift)

    if torch.rand(()) < 0.55:
        snr_db = torch.empty(()).uniform_(20.0, 40.0)
        signal_power = x.pow(2).mean().clamp_min(1e-12)
        noise_power = signal_power / torch.pow(torch.tensor(10.0), snr_db / 10.0)
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        x = x + noise

    return x


class WaveDataset(Dataset):
    def __init__(
        self,
        waves: torch.Tensor,
        y: np.ndarray,
        indices: np.ndarray,
        augment: bool,
    ) -> None:
        self.waves = waves
        self.y = np.asarray(y, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int):
        idx = int(self.indices[j])
        x = self.waves[idx]
        if self.augment:
            x = augment_waveform(x)
        return x, torch.tensor(self.y[idx], dtype=torch.float32)


def apply_spec_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Training-only SpecAugment:
      frequency mask p=.60, <=8 Mel bands
      time mask p=.60, <=12 frames
    Mask value 0 corresponds to the floor of the [0,1] Log-Mel map.
    """
    x = x.clone()
    b, c, f, t = x.shape

    for i in range(b):
        if torch.rand((), device=x.device) < 0.60:
            width = int(torch.randint(1, min(8, f) + 1, (1,), device=x.device).item())
            start_max = f - width
            start = int(torch.randint(0, start_max + 1, (1,), device=x.device).item())
            x[i, :, start:start + width, :] = 0.0

        if torch.rand((), device=x.device) < 0.60:
            width = int(torch.randint(1, min(12, t) + 1, (1,), device=x.device).item())
            start_max = t - width
            start = int(torch.randint(0, start_max + 1, (1,), device=x.device).item())
            x[i, :, :, start:start + width] = 0.0

    return x


# =========================
# Classical models
# =========================
def build_classical_model(model_key: str) -> Any:
    if model_key == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="liblinear",
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=SEED,
            )),
        ])

    if model_key == "rbf_svm":
        # Keep the final recorded baseline configuration: probability=False.
        # ROC-AUC/AP are computed from decision_function scores.
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

    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )

    raise KeyError(model_key)


def classical_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        classes = np.asarray(model.classes_)
        pos_idx = int(np.where(classes == 1)[0][0])
        return np.asarray(p[:, pos_idx], dtype=np.float64)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)

    raise RuntimeError("Classical model has no predict_proba or decision_function.")


# =========================
# Deep training helpers
# =========================
def build_deep_model(model_key: str, input_dim: int) -> nn.Module:
    if model_key == "mlp":
        model = MLP(input_dim)
    elif model_key == "cnn1d":
        model = CNN1D()
    elif model_key == "cnn2d":
        model = CNN2D()
    elif model_key == "crnn":
        model = CRNN()
    else:
        raise KeyError(model_key)

    count = count_parameters(model)
    expected = EXPECTED_PARAM_COUNTS[model_key]
    if count != expected:
        raise RuntimeError(
            f"{model_key} parameter count mismatch: expected={expected}, actual={count}"
        )
    return model


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if enabled:
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except Exception:
            return torch.cuda.amp.autocast(dtype=torch.float16)

    # torch.autocast needs a device_type; disabled context is easiest this way.
    return torch.amp.autocast("cpu", enabled=False)


def train_deep_fold(
    model_key: str,
    fit_idx: np.ndarray,
    heldout_idx: np.ndarray,
    x_train_full: np.ndarray,
    y_train: np.ndarray,
    waveforms: torch.Tensor | None,
    device: torch.device,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    set_all_seeds(SEED)

    model = build_deep_model(model_key, x_train_full.shape[1]).to(device)
    frontend = None
    if model_key in {"cnn2d", "crnn"}:
        frontend = LogMelFrontend().to(device)

    # Per-fold positive class weight only uses fold-fit labels.
    y_fit = y_train[fit_idx]
    n_pos = int((y_fit == 1).sum())
    n_neg = int((y_fit == 0).sum())
    pos_weight_value = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.00001,
    )

    amp_enabled = device.type == "cuda"
    grad_scaler = make_grad_scaler(amp_enabled)

    if model_key == "mlp":
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_train_full[fit_idx]).astype(np.float32)
        x_heldout = scaler.transform(x_train_full[heldout_idx]).astype(np.float32)

        train_ds = VectorDataset(x_fit, y_fit)
        heldout_ds = VectorDataset(x_heldout, y_train[heldout_idx])
    else:
        if waveforms is None:
            raise RuntimeError("Waveforms were not preloaded.")
        train_ds = WaveDataset(waveforms, y_train, fit_idx, augment=True)
        heldout_ds = WaveDataset(waveforms, y_train, heldout_idx, augment=False)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE[model_key],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    heldout_loader = DataLoader(
        heldout_ds,
        batch_size=BATCH_SIZE[model_key],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    final_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled):
                if model_key in {"cnn2d", "crnn"}:
                    assert frontend is not None
                    xb = frontend(xb)
                    xb = apply_spec_augment(xb)

                logits = model(xb)
                loss = criterion(logits, yb)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            batch_n = int(yb.shape[0])
            running_loss += float(loss.detach().item()) * batch_n
            seen += batch_n

        scheduler.step()
        final_loss = running_loss / max(seen, 1)

        if epoch in {1, 10, 20, 30, epochs}:
            print(
                f"      epoch {epoch:03d}/{epochs:03d} "
                f"loss={final_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.7f}"
            )

    # OOF evaluation
    model.eval()
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for xb, _ in heldout_loader:
            xb = xb.to(device, non_blocking=True)

            with autocast_context(amp_enabled):
                if model_key in {"cnn2d", "crnn"}:
                    assert frontend is not None
                    xb = frontend(xb)
                logits = model(xb)
                probs = torch.sigmoid(logits)

            p = probs.float().cpu().numpy()
            all_probs.append(p)
            all_preds.append((p >= 0.5).astype(np.int64))

    score = np.concatenate(all_probs)
    pred = np.concatenate(all_preds)
    return pred, score, final_loss


# =========================
# Ranking / complementarity
# =========================
def ranked_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        SELECTION_COLUMNS,
        ascending=[False] * len(SELECTION_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)


def choose_family_winner(
    summary: pd.DataFrame,
    family: list[str],
) -> str:
    sub = summary[summary["model_key"].isin(family)].copy()
    sub = ranked_summary(sub)
    return str(sub.iloc[0]["model_key"])


def pairwise_complementarity(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = list(predictions.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            ca = predictions[a] == y_true
            cb = predictions[b] == y_true

            both_correct = int(np.sum(ca & cb))
            a_only = int(np.sum(ca & ~cb))
            b_only = int(np.sum(~ca & cb))
            both_wrong = int(np.sum(~ca & ~cb))

            oracle_correct = int(np.sum(ca | cb))
            rows.append({
                "model_a": a,
                "model_b": b,
                "both_correct": both_correct,
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "both_wrong": both_wrong,
                "discordant_errors": a_only + b_only,
                "oracle_pair_accuracy": oracle_correct / len(y_true),
            })

    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Training-only 7-model group-aware OOF model-selection audit."
    )
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--clean-output", action="store_true")
    p.add_argument("--folds", type=int, default=N_SPLITS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Subset to run. Default: all seven models.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(SEED)

    root = resolve_project_root()
    feature_dir = root / "features"
    split_dir = root / "train_test_data"
    out_dir = root / "results_model_selection"

    x_path = feature_dir / "X.npy"
    y_path = feature_dir / "y.npy"
    metadata_path = feature_dir / "metadata.csv"
    train_indices_path = split_dir / "train_indices.npy"

    # Explicitly define forbidden test inputs for audit transparency.
    forbidden_test_inputs = [
        split_dir / "X_test.npy",
        split_dir / "y_test.npy",
        split_dir / "test_indices.npy",
    ]

    print("=" * 84)
    print("RUNNING: src/model_selection_oof.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("PURPOSE         : 7-model model-selection audit using TRAINING OOF only")
    print("TEST DATA       : NOT READ")
    print("=" * 84)

    for path in [x_path, y_path, metadata_path, train_indices_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required training-side file not found: {path}")

    print("Files this audit WILL read:")
    print(f"  {x_path}")
    print(f"  {y_path}")
    print(f"  {metadata_path}")
    print(f"  {train_indices_path}")

    print()
    print("Files this audit is explicitly designed NOT to read:")
    for path in forbidden_test_inputs:
        print(f"  {path}")

    X_all = np.load(x_path, allow_pickle=False)
    y_all = np.load(y_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    train_indices = np.load(train_indices_path, allow_pickle=False).astype(np.int64)

    if X_all.shape[0] != len(y_all) or len(metadata) != len(y_all):
        raise RuntimeError(
            "X.npy / y.npy / metadata.csv row counts do not match."
        )

    if X_all.ndim != 2 or X_all.shape[1] != 212:
        raise RuntimeError(f"Expected X with 212 features; got {X_all.shape}")

    if len(train_indices) != 800:
        raise RuntimeError(
            f"Expected exactly 800 frozen training samples; got {len(train_indices)}"
        )

    if len(np.unique(train_indices)) != len(train_indices):
        raise RuntimeError("train_indices.npy contains duplicate indices.")

    if np.min(train_indices) < 0 or np.max(train_indices) >= len(y_all):
        raise RuntimeError("train_indices.npy contains out-of-range indices.")

    X_train = X_all[train_indices].astype(np.float32)
    y_train = y_all[train_indices].astype(np.int64)

    groups_all = derive_group_ids(metadata)
    groups_train = groups_all[train_indices]

    n_groups = len(np.unique(groups_train))
    class0 = int((y_train == 0).sum())
    class1 = int((y_train == 1).sum())

    print()
    print(f"Training samples : {len(y_train)}")
    print(f"Training groups  : {n_groups}")
    print(f"Training labels  : class0={class0}, class1={class1}")
    print(f"Feature shape    : {X_train.shape}")

    # These values are expected from the established dataset protocol.
    if n_groups != 370:
        raise RuntimeError(
            f"Expected 370 training groups from the frozen protocol; got {n_groups}. "
            "Stop here and inspect group reconstruction before training."
        )

    if not np.array_equal(np.unique(y_train), np.array([0, 1])):
        raise RuntimeError(f"Unexpected training labels: {np.unique(y_train)}")

    # Check group label purity.
    group_check = pd.DataFrame({
        "group_id": groups_train,
        "label": y_train,
    }).groupby("group_id")["label"].nunique()

    mixed_groups = group_check[group_check > 1]
    if len(mixed_groups):
        raise RuntimeError(
            f"Found {len(mixed_groups)} training groups with mixed labels."
        )

    # Build ONE fold plan shared by all seven models.
    sgkf = StratifiedGroupKFold(
        n_splits=args.folds,
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

    fold_assignment = np.full(len(y_train), -1, dtype=np.int64)
    fold_plan_rows: list[dict[str, Any]] = []

    print()
    print("Group-aware OOF fold plan:")
    for fold_idx, (fit_idx, heldout_idx) in enumerate(splits, start=1):
        fit_groups = set(groups_train[fit_idx])
        heldout_groups = set(groups_train[heldout_idx])
        overlap = len(fit_groups.intersection(heldout_groups))
        if overlap != 0:
            raise RuntimeError(f"Fold {fold_idx} group overlap is {overlap}, expected 0.")

        fold_assignment[heldout_idx] = fold_idx

        fit_y = y_train[fit_idx]
        ho_y = y_train[heldout_idx]
        row = {
            "fold": fold_idx,
            "fit_samples": len(fit_idx),
            "heldout_samples": len(heldout_idx),
            "fit_groups": len(fit_groups),
            "heldout_groups": len(heldout_groups),
            "fit_class0": int((fit_y == 0).sum()),
            "fit_class1": int((fit_y == 1).sum()),
            "heldout_class0": int((ho_y == 0).sum()),
            "heldout_class1": int((ho_y == 1).sum()),
            "group_overlap": overlap,
        }
        fold_plan_rows.append(row)
        print(
            f"  fold {fold_idx}: "
            f"fit={len(fit_idx)} samples/{len(fit_groups)} groups, "
            f"heldout={len(heldout_idx)} samples/{len(heldout_groups)} groups, "
            f"labels=({row['heldout_class0']},{row['heldout_class1']}), "
            f"group_overlap={overlap}"
        )

    if np.any(fold_assignment < 1):
        raise RuntimeError("Some training samples did not receive an OOF fold.")

    # Architecture self-check before any training.
    print()
    print("Deep architecture parameter-count check:")
    for key in ["mlp", "cnn1d", "cnn2d", "crnn"]:
        m = build_deep_model(key, X_train.shape[1])
        print(f"  {DISPLAY_NAMES[key]:20s}: {count_parameters(m):,}")
        del m

    if args.check_only:
        print()
        print("CHECK PASSED.")
        print("No model training was run.")
        print("No test data or prior test results were read.")
        return

    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fold_plan_rows).to_csv(
        out_dir / "fold_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    assignments = metadata.iloc[train_indices].copy().reset_index(drop=True)
    assignments.insert(0, "source_index", train_indices)
    assignments["group_id"] = groups_train
    assignments["oof_fold"] = fold_assignment
    assignments.to_csv(
        out_dir / "training_fold_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_models = list(args.models)

    need_audio = any(
        key in {"cnn1d", "cnn2d", "crnn"}
        for key in selected_models
    )
    waveforms = None
    if need_audio:
        waveforms = preload_training_waveforms(root, metadata, train_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print(f"PyTorch device   : {device}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability  : {torch.cuda.get_device_capability(0)}")

    oof_pred: dict[str, np.ndarray] = {
        key: np.full(len(y_train), -1, dtype=np.int64)
        for key in selected_models
    }
    oof_score: dict[str, np.ndarray] = {
        key: np.full(len(y_train), np.nan, dtype=np.float64)
        for key in selected_models
    }

    fold_metric_rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # Run all models on the SAME OOF fold plan.
    # -------------------------------------------------------------
    for model_key in selected_models:
        print()
        print("#" * 84)
        print(f"MODEL: {DISPLAY_NAMES[model_key]}")
        print("#" * 84)

        model_start = time.perf_counter()

        for fold_idx, (fit_idx, heldout_idx) in enumerate(splits, start=1):
            print()
            print(
                f"  Fold {fold_idx}/{args.folds}: "
                f"fit={len(fit_idx)}, heldout={len(heldout_idx)}"
            )
            fold_start = time.perf_counter()

            if model_key in {
                "logistic_regression",
                "rbf_svm",
                "random_forest",
            }:
                set_all_seeds(SEED)
                model = build_classical_model(model_key)
                model.fit(X_train[fit_idx], y_train[fit_idx])
                pred = np.asarray(model.predict(X_train[heldout_idx]), dtype=np.int64)
                score = classical_score(model, X_train[heldout_idx])
                final_train_loss = float("nan")
            else:
                pred, score, final_train_loss = train_deep_fold(
                    model_key=model_key,
                    fit_idx=fit_idx,
                    heldout_idx=heldout_idx,
                    x_train_full=X_train,
                    y_train=y_train,
                    waveforms=waveforms,
                    device=device,
                    epochs=args.epochs,
                )

            if len(pred) != len(heldout_idx) or len(score) != len(heldout_idx):
                raise RuntimeError(
                    f"{model_key} fold {fold_idx} output length mismatch."
                )

            oof_pred[model_key][heldout_idx] = pred
            oof_score[model_key][heldout_idx] = score

            metrics = compute_metrics(
                y_train[heldout_idx],
                pred,
                score,
            )
            elapsed = time.perf_counter() - fold_start

            fold_metric_rows.append({
                "model_key": model_key,
                "model_name": DISPLAY_NAMES[model_key],
                "fold": fold_idx,
                "fit_samples": len(fit_idx),
                "heldout_samples": len(heldout_idx),
                "fit_groups": len(np.unique(groups_train[fit_idx])),
                "heldout_groups": len(np.unique(groups_train[heldout_idx])),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "specificity": metrics["specificity"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "final_train_loss": final_train_loss,
                "seconds": elapsed,
            })

            print(
                "    "
                f"F1={metrics['f1']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"Specificity={metrics['specificity']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}"
            )

        model_elapsed = time.perf_counter() - model_start
        print(f"\n  Completed {DISPLAY_NAMES[model_key]} in {model_elapsed:.1f} s")

        if np.any(oof_pred[model_key] < 0) or not np.isfinite(oof_score[model_key]).all():
            raise RuntimeError(f"Incomplete OOF predictions for {model_key}.")

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    fold_metrics_df.to_csv(
        out_dir / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Combined 800-sample OOF metrics
    # -------------------------------------------------------------
    summary_rows: list[dict[str, Any]] = []

    for model_key in selected_models:
        metrics = compute_metrics(
            y_train,
            oof_pred[model_key],
            oof_score[model_key],
        )

        fdf = fold_metrics_df[
            fold_metrics_df["model_key"] == model_key
        ]

        summary_rows.append({
            "model_key": model_key,
            "model_name": DISPLAY_NAMES[model_key],
            "oof_accuracy": metrics["accuracy"],
            "oof_precision": metrics["precision"],
            "oof_recall": metrics["recall"],
            "oof_f1": metrics["f1"],
            "oof_specificity": metrics["specificity"],
            "oof_roc_auc": metrics["roc_auc"],
            "oof_average_precision": metrics["average_precision"],
            "oof_tn": metrics["tn"],
            "oof_fp": metrics["fp"],
            "oof_fn": metrics["fn"],
            "oof_tp": metrics["tp"],
            "fold_f1_mean": float(fdf["f1"].mean()),
            "fold_f1_std": float(fdf["f1"].std(ddof=1)),
            "fold_recall_mean": float(fdf["recall"].mean()),
            "fold_recall_std": float(fdf["recall"].std(ddof=1)),
            "fold_specificity_mean": float(fdf["specificity"].mean()),
            "fold_specificity_std": float(fdf["specificity"].std(ddof=1)),
        })

    summary = pd.DataFrame(summary_rows)
    summary_ranked = ranked_summary(summary)
    summary_ranked.insert(0, "selection_rank", np.arange(1, len(summary_ranked) + 1))
    summary_ranked.to_csv(
        out_dir / "training_only_oof_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # OOF sample-level predictions
    pred_df = metadata.iloc[train_indices].copy().reset_index(drop=True)
    pred_df.insert(0, "source_index", train_indices)
    pred_df["group_id"] = groups_train
    pred_df["true_label"] = y_train
    pred_df["oof_fold"] = fold_assignment

    for model_key in selected_models:
        pred_df[f"{model_key}_prediction"] = oof_pred[model_key]
        pred_df[f"{model_key}_score"] = oof_score[model_key]

    pred_df.to_csv(
        out_dir / "training_only_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Pairwise OOF error complementarity: useful evidence, but NOT used
    # to override the prespecified family winner rule.
    comp_df = pairwise_complementarity(
        y_train,
        {k: oof_pred[k] for k in selected_models},
    )
    comp_df.to_csv(
        out_dir / "pairwise_oof_complementarity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Prespecified training-only model-selection decision.
    decision: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "test_data_read": False,
        "selection_rule": (
            "lexicographic descending: OOF F1 -> OOF Recall -> "
            "OOF Specificity -> OOF ROC-AUC -> OOF Average Precision"
        ),
        "training_samples": int(len(y_train)),
        "training_groups": int(n_groups),
        "folds": int(args.folds),
        "epochs_deep_models": int(args.epochs),
        "models_run": selected_models,
        "overall_winner": str(summary_ranked.iloc[0]["model_key"]),
    }

    if all(k in selected_models for k in HANDCRAFTED_FAMILY):
        hand_winner = choose_family_winner(summary, HANDCRAFTED_FAMILY)
        decision["best_handcrafted_representation_model"] = hand_winner
    else:
        hand_winner = None

    if all(k in selected_models for k in DEEP_ACOUSTIC_FAMILY):
        deep_winner = choose_family_winner(summary, DEEP_ACOUSTIC_FAMILY)
        decision["best_deep_acoustic_model"] = deep_winner
    else:
        deep_winner = None

    if hand_winner is not None and deep_winner is not None:
        if hand_winner == "mlp" and deep_winner == "cnn2d":
            status = "fully_supported_by_training_only_oof"
        elif deep_winner == "cnn2d":
            status = "cnn2d_supported_but_mlp_is_not_best_handcrafted_model"
        elif hand_winner == "mlp":
            status = "mlp_supported_but_cnn2d_is_not_best_deep_model"
        else:
            status = "mlp_cnn2d_pair_not_selected_by_prespecified_training_only_rule"

        decision["historical_mlp_cnn2d_pair_audit"] = status

    with open(
        out_dir / "training_only_selection_decision.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    # Short human-readable report.
    lines = []
    lines.append("TRAINING-ONLY MODEL-SELECTION AUDIT")
    lines.append("=" * 72)
    lines.append("The fixed 200-sample test set was not read by this script.")
    lines.append(
        "Selection rule: OOF F1 -> Recall -> Specificity -> ROC-AUC -> AP."
    )
    lines.append("")
    lines.append(summary_ranked[
        [
            "selection_rank",
            "model_name",
            "oof_accuracy",
            "oof_precision",
            "oof_recall",
            "oof_f1",
            "oof_specificity",
            "oof_roc_auc",
            "oof_average_precision",
            "oof_fp",
            "oof_fn",
            "fold_f1_std",
        ]
    ].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    lines.append("")
    lines.append(f"Overall OOF winner: {decision['overall_winner']}")

    if hand_winner is not None:
        lines.append(f"Best handcrafted-family model: {hand_winner}")
    if deep_winner is not None:
        lines.append(f"Best deep-acoustic model: {deep_winner}")
    if "historical_mlp_cnn2d_pair_audit" in decision:
        lines.append(
            "MLP+CNN2D historical-pair audit: "
            + decision["historical_mlp_cnn2d_pair_audit"]
        )

    report_text = "\n".join(lines)
    with open(
        out_dir / "training_only_audit_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_text)

    print()
    print("=" * 84)
    print("TRAINING-ONLY OOF AUDIT FINISHED")
    print("=" * 84)
    print(report_text)
    print()
    print(f"Outputs: {out_dir}")
    print("Most important files:")
    print(f"  {out_dir / 'training_only_oof_model_summary.csv'}")
    print(f"  {out_dir / 'training_only_selection_decision.json'}")
    print(f"  {out_dir / 'pairwise_oof_complementarity.csv'}")
    print(f"  {out_dir / 'training_only_oof_predictions.csv'}")
    print()
    print("No fixed-test-set metrics were computed.")


if __name__ == "__main__":
    main()
