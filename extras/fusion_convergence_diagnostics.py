from __future__ import annotations

"""
fusion_convergence_diagnostics.py
=================================
Convergence diagnostics for the THREE fusion structures that actually won
at least one repeat in the repeated group-aware training-only audit:

Seed 42      : RBF-SVM + 2D CNN
Seeds 52/62/72: RBF-SVM + 1D CNN
Seed 82      : MLP + 2D CNN

This script is a DIAGNOSTIC experiment only.

It uses ONLY the fixed 800-sample training partition and reconstructs the
corresponding group-aware 5-fold splits. It DOES NOT read:
    X_test.npy
    y_test.npy
    test_indices.npy
    any fixed-test prediction/result

What "convergence" means here
-----------------------------
1) RBF-SVM:
   sklearn SVC does not have an epoch-by-epoch neural-network loss curve.
   Therefore SVM convergence is checked from the solver termination flag
   (`fit_status_ == 0`) and the reported number of iterations (`n_iter_`).

2) MLP / 1D CNN / 2D CNN:
   The network is trained for the already-fixed 40 epochs.
   After EVERY epoch, the script evaluates:
       - clean fit-fold weighted BCE loss
       - clean group-heldout OOF weighted BCE loss
   The held-out curve is NEVER used for early stopping or epoch selection.

Output
------
One publication-ready PNG:
    results_convergence/
        three_selected_fusion_convergence.png

The figure contains three panels:
A. RBF-SVM + 1D CNN  (3/5 repeat winners; seeds 52, 62, 72)
B. RBF-SVM + 2D CNN  (1/5 repeat winner; seed 42)
C. MLP + 2D CNN      (1/5 repeat winner; seed 82)

All text in the figure is English for direct use in a paper.
"""

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset


SCRIPT_VERSION = "three-selected-fusion-convergence-v1"

# ============================================================
# Established experiment settings
# ============================================================
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

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-5

BATCH_SIZE = {
    "mlp": 64,
    "cnn1d": 64,
    "cnn2d": 32,
}

EXPECTED_PARAM_COUNTS = {
    "mlp": 35_841,
    "cnn1d": 105_569,
    "cnn2d": 97_521,
}

# These are the repeat winners already established by stability_selection.py.
PAIR_SPECS = {
    "svm_cnn1d": {
        "title": "RBF-SVM + 1D CNN",
        "subtitle": "Selected in 3/5 repeats · seeds 52, 62, 72",
        "seeds": [52, 62, 72],
        "deep_models": ["cnn1d"],
        "check_svm": True,
    },
    "svm_cnn2d": {
        "title": "RBF-SVM + 2D CNN",
        "subtitle": "Selected in 1/5 repeats · seed 42",
        "seeds": [42],
        "deep_models": ["cnn2d"],
        "check_svm": True,
    },
    "mlp_cnn2d": {
        "title": "MLP + 2D CNN",
        "subtitle": "Selected in 1/5 repeats · seed 82",
        "seeds": [82],
        "deep_models": ["mlp", "cnn2d"],
        "check_svm": False,
    },
}


# ============================================================
# Reproducibility
# ============================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# Project / grouping
# ============================================================
def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def derive_group_ids(metadata: pd.DataFrame) -> np.ndarray:
    """
    Established grouping rule:
        group_id = source_class :: base_recording_name

    Trailing _<digits> is removed from WAV stem:
        xxx.wav, xxx_1.wav, xxx_2.wav -> same group
    """
    required = {"file_name", "source_class"}
    missing = required.difference(metadata.columns)

    if missing:
        raise RuntimeError(
            "metadata.csv is missing required grouping columns: "
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

    if path.is_absolute() and path.exists():
        return path

    candidate_1 = project_root / path
    if candidate_1.exists():
        return candidate_1

    candidate_2 = project_root / Path(raw.replace("\\", "/"))
    if candidate_2.exists():
        return candidate_2

    raise FileNotFoundError(f"Cannot resolve WAV path: {raw_value}")


# ============================================================
# Audio
# ============================================================
def load_one_waveform(path: Path) -> torch.Tensor:
    """
    Established preprocessing:
      mono
      8 kHz
      exactly 8000 samples
      DC removal
      per-file peak scaling
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = np.mean(data, axis=1).astype(np.float32)

    x = torch.from_numpy(wave)

    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)

    if x.numel() < NUM_SAMPLES:
        x = torch.nn.functional.pad(
            x,
            (0, NUM_SAMPLES - x.numel()),
        )
    elif x.numel() > NUM_SAMPLES:
        x = x[:NUM_SAMPLES]

    x = x - x.mean()

    peak = x.abs().max()
    if torch.isfinite(peak) and peak > 0:
        x = x / peak

    if not torch.isfinite(x).all():
        raise ValueError(
            f"Non-finite waveform after preprocessing: {path}"
        )

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
    print("Preloading the TRAINING waveforms only...")

    for j, source_index in enumerate(train_indices, start=1):
        raw = metadata.iloc[int(source_index)]["file_path"]
        wav_path = resolve_wav_path(project_root, raw)
        waves.append(load_one_waveform(wav_path))

        if j % 100 == 0 or j == total:
            print(f"  loaded {j}/{total}")

    tensor = torch.stack(waves, dim=0)

    expected_shape = (
        len(train_indices),
        1,
        NUM_SAMPLES,
    )

    if tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected waveform tensor shape: {tuple(tensor.shape)}, "
            f"expected {expected_shape}"
        )

    return tensor


# ============================================================
# Training augmentation
# ============================================================
def shift_with_zeros(
    x: torch.Tensor,
    shift: int,
) -> torch.Tensor:
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
    Same waveform augmentation used by the established deep protocol:
      gain p=.70, -6..+6 dB
      shift p=.60, max 10%
      Gaussian noise p=.55, target SNR 20..40 dB
    """
    x = x.clone()

    if torch.rand(()) < 0.70:
        gain_db = torch.empty(()).uniform_(-6.0, 6.0)
        x = x * torch.pow(
            torch.tensor(10.0),
            gain_db / 20.0,
        )

    if torch.rand(()) < 0.60:
        max_shift = int(0.10 * NUM_SAMPLES)
        shift = int(
            torch.randint(
                -max_shift,
                max_shift + 1,
                (1,),
            ).item()
        )
        x = shift_with_zeros(x, shift)

    if torch.rand(()) < 0.55:
        snr_db = torch.empty(()).uniform_(20.0, 40.0)

        signal_power = (
            x.pow(2)
            .mean()
            .clamp_min(1e-12)
        )

        noise_power = signal_power / torch.pow(
            torch.tensor(10.0),
            snr_db / 10.0,
        )

        x = x + (
            torch.randn_like(x)
            * torch.sqrt(noise_power)
        )

    return x


def apply_spec_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Same training-only SpecAugment:
      frequency mask p=.60, <=8 Mel bands
      time mask p=.60, <=12 frames
    """
    x = x.clone()
    b, _, f, t = x.shape

    for i in range(b):
        if torch.rand((), device=x.device) < 0.60:
            width = int(
                torch.randint(
                    1,
                    min(8, f) + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            start = int(
                torch.randint(
                    0,
                    f - width + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            x[i, :, start:start + width, :] = 0.0

        if torch.rand((), device=x.device) < 0.60:
            width = int(
                torch.randint(
                    1,
                    min(12, t) + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            start = int(
                torch.randint(
                    0,
                    t - width + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            x[i, :, :, start:start + width] = 0.0

    return x


# ============================================================
# Datasets
# ============================================================
class VectorDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(
            np.asarray(x, dtype=np.float32)
        )
        self.y = torch.from_numpy(
            np.asarray(y, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


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
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int):
        idx = int(self.indices[j])
        x = self.waves[idx]

        if self.augment:
            x = augment_waveform(x)

        return (
            x,
            torch.tensor(
                self.y[idx],
                dtype=torch.float32,
            ),
        )


# ============================================================
# Models
# ============================================================
class MLP(nn.Module):
    """
    Exact established MLP:
      212 -> 128 -> 64 -> 1
      BatchNorm after first linear
      Dropout 0.35 / 0.25
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
    Exact established 1D CNN.
    """
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                1, 32,
                kernel_size=9,
                stride=2,
                padding=4,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                32, 64,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                64, 128,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                128, 128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
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
    Exact established 2D CNN.
    """
    def __init__(self) -> None:
        super().__init__()

        def block(
            cin: int,
            cout: int,
            pool: bool,
        ) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
            ]

            if pool:
                layers.append(
                    nn.MaxPool2d(2)
                )

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


class LogMelFrontend(nn.Module):
    """
    Established 64-band Log-Mel frontend.
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

        mel = self.mel(x2)
        db = self.to_db(mel)

        scaled = torch.clamp(
            (db + 80.0) / 80.0,
            0.0,
            1.0,
        )

        return scaled.unsqueeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def build_deep_model(
    model_key: str,
    input_dim: int,
) -> nn.Module:
    if model_key == "mlp":
        model = MLP(input_dim)

    elif model_key == "cnn1d":
        model = CNN1D()

    elif model_key == "cnn2d":
        model = CNN2D()

    else:
        raise KeyError(model_key)

    actual = count_parameters(model)
    expected = EXPECTED_PARAM_COUNTS[model_key]

    if actual != expected:
        raise RuntimeError(
            f"{model_key} parameter count mismatch: "
            f"{actual:,} != {expected:,}"
        )

    return model


# ============================================================
# AMP
# ============================================================
def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


def autocast_context(enabled: bool):
    if enabled:
        try:
            return torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            )
        except Exception:
            return torch.cuda.amp.autocast(
                dtype=torch.float16,
            )

    return torch.amp.autocast(
        "cpu",
        enabled=False,
    )


# ============================================================
# SVM convergence
# ============================================================
@dataclass
class SVMConvergence:
    total_fits: int = 0
    converged_fits: int = 0
    iterations: list[int] | None = None

    def __post_init__(self):
        if self.iterations is None:
            self.iterations = []


def check_svm_outer_fold_convergence(
    x_train_full: np.ndarray,
    y_train: np.ndarray,
    fit_idx: np.ndarray,
    seed: int,
) -> tuple[bool, int]:
    """
    This checks solver convergence of the RBF-SVM expert itself.

    Probability calibration is intentionally not required for this specific
    diagnostic because the question here is optimizer convergence, not
    calibrated probability quality.
    """
    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=False,
                class_weight="balanced",
                random_state=seed,
                max_iter=-1,
            ),
        ),
    ])

    model.fit(
        x_train_full[fit_idx],
        y_train[fit_idx],
    )

    clf: SVC = model.named_steps["classifier"]

    fit_status = int(
        getattr(clf, "fit_status_", 1)
    )

    n_iter_raw = getattr(
        clf,
        "n_iter_",
        np.asarray([-1]),
    )

    n_iter = int(
        np.asarray(n_iter_raw)
        .reshape(-1)
        .max()
    )

    return fit_status == 0, n_iter


# ============================================================
# Loss evaluation
# ============================================================
@torch.no_grad()
def evaluate_deep_loss(
    model_key: str,
    model: nn.Module,
    frontend: LogMelFrontend | None,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """
    CLEAN evaluation:
    - no waveform augmentation
    - no SpecAugment
    """
    model.eval()

    total_loss = 0.0
    seen = 0

    for xb, yb in loader:
        xb = xb.to(
            device,
            non_blocking=True,
        )
        yb = yb.to(
            device,
            non_blocking=True,
        )

        with autocast_context(amp_enabled):
            if model_key == "cnn2d":
                assert frontend is not None
                xb = frontend(xb)

            logits = model(xb)
            loss = criterion(logits, yb)

        batch_n = int(yb.shape[0])

        total_loss += (
            float(loss.detach().item())
            * batch_n
        )

        seen += batch_n

    return total_loss / max(seen, 1)


# ============================================================
# One deep model / one fold
# ============================================================
def train_deep_fold_with_curve(
    model_key: str,
    fit_idx: np.ndarray,
    heldout_idx: np.ndarray,
    x_train_full: np.ndarray,
    y_train: np.ndarray,
    waveforms: torch.Tensor,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_all_seeds(seed)

    model = build_deep_model(
        model_key=model_key,
        input_dim=x_train_full.shape[1],
    ).to(device)

    frontend: LogMelFrontend | None = None

    if model_key == "cnn2d":
        frontend = LogMelFrontend().to(device)

    y_fit = y_train[fit_idx]

    n_pos = int(
        (y_fit == 1).sum()
    )
    n_neg = int(
        (y_fit == 0).sum()
    )

    pos_weight_value = (
        n_neg / max(n_pos, 1)
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=ETA_MIN,
        )
    )

    amp_enabled = (
        device.type == "cuda"
    )

    grad_scaler = make_grad_scaler(
        amp_enabled
    )

    # --------------------------------
    # datasets / loaders
    # --------------------------------
    if model_key == "mlp":
        scaler = StandardScaler()

        x_fit = scaler.fit_transform(
            x_train_full[fit_idx]
        ).astype(np.float32)

        x_heldout = scaler.transform(
            x_train_full[heldout_idx]
        ).astype(np.float32)

        train_opt_ds = VectorDataset(
            x_fit,
            y_fit,
        )

        train_clean_ds = VectorDataset(
            x_fit,
            y_fit,
        )

        heldout_clean_ds = VectorDataset(
            x_heldout,
            y_train[heldout_idx],
        )

    else:
        train_opt_ds = WaveDataset(
            waveforms,
            y_train,
            fit_idx,
            augment=True,
        )

        train_clean_ds = WaveDataset(
            waveforms,
            y_train,
            fit_idx,
            augment=False,
        )

        heldout_clean_ds = WaveDataset(
            waveforms,
            y_train,
            heldout_idx,
            augment=False,
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_opt_loader = DataLoader(
        train_opt_ds,
        batch_size=BATCH_SIZE[model_key],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    train_clean_loader = DataLoader(
        train_clean_ds,
        batch_size=BATCH_SIZE[model_key],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    heldout_clean_loader = DataLoader(
        heldout_clean_ds,
        batch_size=BATCH_SIZE[model_key],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    fit_curve = np.zeros(
        epochs,
        dtype=np.float64,
    )

    oof_curve = np.zeros(
        epochs,
        dtype=np.float64,
    )

    for epoch in range(1, epochs + 1):
        model.train()

        for xb, yb in train_opt_loader:
            xb = xb.to(
                device,
                non_blocking=True,
            )
            yb = yb.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast_context(amp_enabled):
                if model_key == "cnn2d":
                    assert frontend is not None

                    xb = frontend(xb)
                    xb = apply_spec_augment(xb)

                logits = model(xb)
                loss = criterion(
                    logits,
                    yb,
                )

            grad_scaler.scale(
                loss
            ).backward()

            grad_scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            grad_scaler.step(
                optimizer
            )

            grad_scaler.update()

        scheduler.step()

        fit_curve[epoch - 1] = (
            evaluate_deep_loss(
                model_key=model_key,
                model=model,
                frontend=frontend,
                loader=train_clean_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
            )
        )

        oof_curve[epoch - 1] = (
            evaluate_deep_loss(
                model_key=model_key,
                model=model,
                frontend=frontend,
                loader=heldout_clean_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
            )
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"      {model_key:<6s} "
                f"epoch {epoch:02d}/{epochs} | "
                f"fit={fit_curve[epoch - 1]:.5f} | "
                f"OOF={oof_curve[epoch - 1]:.5f}"
            )

    del model

    if frontend is not None:
        del frontend

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return fit_curve, oof_curve


# ============================================================
# Convergence heuristic
# ============================================================
@dataclass
class CurveDiagnostic:
    verdict: str
    tail_cv: float
    final_vs_best: float
    best_epoch: int


def smooth_curve(
    values: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if window <= 1:
        return values.copy()

    out = np.empty_like(values)

    for i in range(len(values)):
        left = max(
            0,
            i - window + 1,
        )
        out[i] = values[left:i + 1].mean()

    return out


def diagnose_oof_curve(
    mean_oof: np.ndarray,
) -> CurveDiagnostic:
    """
    Heuristic diagnostic only.

    Stable / converged:
        last-5-epoch coefficient of variation <= 5%
        AND final smoothed loss <= 10% above its best smoothed value

    Late overfit signal:
        final smoothed loss > 15% above best smoothed value

    Otherwise:
        near plateau
    """
    smoothed = smooth_curve(
        mean_oof,
        window=5,
    )

    tail = smoothed[-5:]

    tail_cv = float(
        np.std(tail)
        / max(
            np.mean(tail),
            1e-12,
        )
    )

    best_index = int(
        np.argmin(smoothed)
    )

    best_value = float(
        smoothed[best_index]
    )

    final_value = float(
        smoothed[-1]
    )

    final_vs_best = (
        final_value
        / max(best_value, 1e-12)
        - 1.0
    )

    if (
        tail_cv <= 0.05
        and final_vs_best <= 0.10
    ):
        verdict = "Stable / converged"

    elif final_vs_best > 0.15:
        verdict = "Late overfit signal"

    else:
        verdict = "Near plateau"

    return CurveDiagnostic(
        verdict=verdict,
        tail_cv=tail_cv,
        final_vs_best=final_vs_best,
        best_epoch=best_index + 1,
    )


# ============================================================
# Pair experiment
# ============================================================
@dataclass
class PairResult:
    pair_key: str
    model_fit_curves: dict[str, np.ndarray]
    model_oof_curves: dict[str, np.ndarray]
    svm_summary: SVMConvergence | None


def run_pair(
    pair_key: str,
    pair_spec: dict[str, Any],
    x_train_full: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    waveforms: torch.Tensor,
    device: torch.device,
    epochs: int,
    n_splits: int,
) -> PairResult:
    fit_storage: dict[
        str,
        list[np.ndarray],
    ] = {
        key: []
        for key in pair_spec["deep_models"]
    }

    oof_storage: dict[
        str,
        list[np.ndarray],
    ] = {
        key: []
        for key in pair_spec["deep_models"]
    }

    svm_summary = (
        SVMConvergence()
        if pair_spec["check_svm"]
        else None
    )

    print()
    print("=" * 78)
    print(pair_spec["title"])
    print(pair_spec["subtitle"])
    print("=" * 78)

    for repeat_seed in pair_spec["seeds"]:
        print()
        print(
            f"Repeat seed {repeat_seed}"
        )

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=repeat_seed,
        )

        splits = list(
            splitter.split(
                np.zeros(
                    len(y_train)
                ),
                y_train,
                groups_train,
            )
        )

        for fold_id, (
            fit_idx,
            heldout_idx,
        ) in enumerate(
            splits,
            start=1,
        ):
            fit_groups = set(
                groups_train[fit_idx]
                .tolist()
            )

            heldout_groups = set(
                groups_train[heldout_idx]
                .tolist()
            )

            overlap = (
                fit_groups
                .intersection(
                    heldout_groups
                )
            )

            if overlap:
                raise RuntimeError(
                    f"Group leakage in "
                    f"{pair_spec['title']} "
                    f"seed={repeat_seed}, "
                    f"fold={fold_id}: "
                    f"{len(overlap)} groups"
                )

            print(
                f"  Fold {fold_id}/{n_splits}: "
                f"fit={len(fit_idx)}, "
                f"heldout={len(heldout_idx)}, "
                f"group overlap=0"
            )

            # ---------------------------
            # RBF-SVM convergence check
            # ---------------------------
            if svm_summary is not None:
                converged, n_iter = (
                    check_svm_outer_fold_convergence(
                        x_train_full=x_train_full,
                        y_train=y_train,
                        fit_idx=fit_idx,
                        seed=repeat_seed,
                    )
                )

                svm_summary.total_fits += 1
                svm_summary.converged_fits += int(
                    converged
                )
                svm_summary.iterations.append(
                    n_iter
                )

                print(
                    f"      SVM solver: "
                    f"{'converged' if converged else 'NOT converged'} "
                    f"(iterations={n_iter})"
                )

            # ---------------------------
            # Deep branches
            # ---------------------------
            for model_key in (
                pair_spec["deep_models"]
            ):
                fit_curve, oof_curve = (
                    train_deep_fold_with_curve(
                        model_key=model_key,
                        fit_idx=fit_idx,
                        heldout_idx=heldout_idx,
                        x_train_full=x_train_full,
                        y_train=y_train,
                        waveforms=waveforms,
                        device=device,
                        epochs=epochs,
                        seed=repeat_seed,
                    )
                )

                fit_storage[
                    model_key
                ].append(
                    fit_curve
                )

                oof_storage[
                    model_key
                ].append(
                    oof_curve
                )

    fit_arrays = {
        key: np.stack(
            values,
            axis=0,
        )
        for key, values
        in fit_storage.items()
    }

    oof_arrays = {
        key: np.stack(
            values,
            axis=0,
        )
        for key, values
        in oof_storage.items()
    }

    return PairResult(
        pair_key=pair_key,
        model_fit_curves=fit_arrays,
        model_oof_curves=oof_arrays,
        svm_summary=svm_summary,
    )


# ============================================================
# Plot
# ============================================================
DISPLAY_MODEL = {
    "mlp": "MLP",
    "cnn1d": "1D CNN",
    "cnn2d": "2D CNN",
}


def add_model_curves(
    ax,
    model_key: str,
    fit_array: np.ndarray,
    oof_array: np.ndarray,
):
    epochs = np.arange(
        1,
        fit_array.shape[1] + 1,
    )

    fit_mean = fit_array.mean(
        axis=0
    )

    oof_mean = oof_array.mean(
        axis=0
    )

    oof_std = oof_array.std(
        axis=0,
        ddof=1,
    ) if oof_array.shape[0] > 1 else np.zeros_like(
        oof_mean
    )

    model_name = DISPLAY_MODEL[
        model_key
    ]

    # Use Matplotlib's default color cycle.
    fit_line, = ax.plot(
        epochs,
        fit_mean,
        linewidth=1.8,
        linestyle="--",
        alpha=0.75,
        label=f"{model_name} fit",
    )

    oof_line, = ax.plot(
        epochs,
        oof_mean,
        linewidth=2.5,
        label=f"{model_name} OOF",
    )

    # Match the OOF band to the OOF line automatically.
    ax.fill_between(
        epochs,
        np.maximum(
            oof_mean - oof_std,
            0.0,
        ),
        oof_mean + oof_std,
        alpha=0.12,
        color=oof_line.get_color(),
    )

    diagnostic = diagnose_oof_curve(
        oof_mean
    )

    return diagnostic


def plot_results(
    results: dict[str, PairResult],
    output_png: Path,
    epochs: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.2, 5.6),
        sharex=True,
    )

    ordered_keys = [
        "svm_cnn1d",
        "svm_cnn2d",
        "mlp_cnn2d",
    ]

    panel_letters = [
        "A",
        "B",
        "C",
    ]

    all_handles = []
    all_labels = []

    for ax, pair_key, panel_letter in zip(
        axes,
        ordered_keys,
        panel_letters,
    ):
        spec = PAIR_SPECS[pair_key]
        result = results[pair_key]

        diagnostics: list[
            tuple[str, CurveDiagnostic]
        ] = []

        for model_key in (
            spec["deep_models"]
        ):
            diagnostic = add_model_curves(
                ax=ax,
                model_key=model_key,
                fit_array=result.model_fit_curves[
                    model_key
                ],
                oof_array=result.model_oof_curves[
                    model_key
                ],
            )

            diagnostics.append(
                (
                    DISPLAY_MODEL[
                        model_key
                    ],
                    diagnostic,
                )
            )

        ax.set_title(
            f"{panel_letter}. {spec['title']}",
            fontsize=13,
            fontweight="semibold",
            pad=12,
        )

        ax.set_xlabel(
            "Epoch",
            fontsize=11,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.8,
            alpha=0.25,
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.set_xlim(
            1,
            epochs,
        )

        ax.set_xticks(
            np.unique(
                np.concatenate(
                    (
                        [1],
                        np.arange(
                            5,
                            epochs + 1,
                            5,
                        ),
                    )
                )
            )
        )

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        for handle, label in zip(
            handles,
            labels,
        ):
            if label not in all_labels:
                all_handles.append(
                    handle
                )
                all_labels.append(
                    label
                )

    axes[0].set_ylabel(
        "Weighted BCE loss",
        fontsize=11,
    )

    fig.suptitle(
        "Convergence diagnostics of fusion structures selected across repeated group-aware OOF",
        fontsize=15.5,
        fontweight="semibold",
        y=1.02,
    )

    fig.text(
        0.5,
        0.955,
        "Training partition only · 5-fold group-aware OOF · fixed 40 epochs · no early stopping",
        ha="center",
        va="top",
        fontsize=10.3,
    )

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=min(
            len(all_labels),
            6,
        ),
        frameon=False,
        bbox_to_anchor=(
            0.5,
            -0.01,
        ),
        fontsize=9.5,
    )

    fig.text(
        0.5,
        -0.035,
        "OOF bands indicate ±1 SD across the relevant winning-repeat folds. "
        "The convergence labels are descriptive diagnostics and were not used to retune training.",
        ha="center",
        va="bottom",
        fontsize=9.1,
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.08,
            1.0,
            0.93,
        ]
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_png,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Console summary
# ============================================================
def print_summary(
    results: dict[str, PairResult],
) -> None:
    print()
    print("=" * 78)
    print("CONVERGENCE DIAGNOSTIC SUMMARY")
    print("=" * 78)

    for pair_key in [
        "svm_cnn1d",
        "svm_cnn2d",
        "mlp_cnn2d",
    ]:
        spec = PAIR_SPECS[
            pair_key
        ]
        result = results[
            pair_key
        ]

        print()
        print(
            spec["title"]
        )

        if result.svm_summary is not None:
            svm = result.svm_summary

            median_iter = int(
                np.median(
                    svm.iterations
                )
            )

            print(
                f"  SVM solver: "
                f"{svm.converged_fits}/"
                f"{svm.total_fits} "
                f"fits converged"
            )

            print(
                f"  SVM median iterations: "
                f"{median_iter}"
            )

        for model_key in (
            spec["deep_models"]
        ):
            mean_oof = (
                result
                .model_oof_curves[
                    model_key
                ]
                .mean(axis=0)
            )

            diag = diagnose_oof_curve(
                mean_oof
            )

            print(
                f"  {DISPLAY_MODEL[model_key]}: "
                f"{diag.verdict}"
            )

            print(
                f"      best smoothed OOF epoch = "
                f"{diag.best_epoch}"
            )

            print(
                f"      tail CV = "
                f"{diag.tail_cv:.4f}"
            )

            print(
                f"      final vs best = "
                f"{diag.final_vs_best:+.2%}"
            )

    print()
    print(
        "IMPORTANT: do not use these curves to change the already-fixed "
        "40-epoch protocol. They are reported only as convergence evidence."
    )


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convergence diagnostics for "
            "SVM+1D CNN, SVM+2D CNN, and MLP+2D CNN"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root. Default: parent directory of src/"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=N_SPLITS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. "
            "Default: results_convergence/"
            "three_selected_fusion_convergence.png"
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Check paths, data, grouping, and model definitions only."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else resolve_project_root()
    )

    output_png = (
        args.output.resolve()
        if args.output is not None
        else (
            project_root
            / "results_convergence"
            / "three_selected_fusion_convergence.png"
        )
    )

    x_path = (
        project_root
        / "features"
        / "X.npy"
    )

    y_path = (
        project_root
        / "features"
        / "y.npy"
    )

    metadata_path = (
        project_root
        / "features"
        / "metadata.csv"
    )

    train_indices_path = (
        project_root
        / "train_test_data"
        / "train_indices.npy"
    )

    print("=" * 78)
    print("fusion_convergence_diagnostics.py")
    print("Three selected fusion structures - convergence diagnostic")
    print("=" * 78)
    print(
        f"Script version : {SCRIPT_VERSION}"
    )
    print(
        f"Project root   : {project_root}"
    )
    print(
        f"Epochs         : {args.epochs}"
    )
    print(
        f"Folds          : {args.folds}"
    )
    print(
        f"Output PNG     : {output_png}"
    )

    print()
    print("TRAINING-ONLY RULE")
    print("  no X_test.npy")
    print("  no y_test.npy")
    print("  no test_indices.npy")
    print("  no fixed-test predictions/results")

    for path in [
        x_path,
        y_path,
        metadata_path,
        train_indices_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    x_all = np.load(
        x_path
    )

    y_all = np.load(
        y_path
    )

    metadata = pd.read_csv(
        metadata_path
    )

    train_indices = np.load(
        train_indices_path
    ).astype(
        np.int64
    )

    if len(x_all) != len(y_all):
        raise RuntimeError(
            f"X/y length mismatch: "
            f"{len(x_all)} vs {len(y_all)}"
        )

    if len(metadata) != len(y_all):
        raise RuntimeError(
            f"metadata/y length mismatch: "
            f"{len(metadata)} vs {len(y_all)}"
        )

    if len(train_indices) != 800:
        print(
            f"WARNING: expected 800 training samples, "
            f"found {len(train_indices)}"
        )

    x_train = np.asarray(
        x_all[train_indices],
        dtype=np.float32,
    )

    y_train = np.asarray(
        y_all[train_indices],
        dtype=np.int64,
    )

    all_groups = derive_group_ids(
        metadata
    )

    groups_train = np.asarray(
        all_groups[train_indices],
        dtype=object,
    )

    if set(
        np.unique(
            y_train
        ).tolist()
    ) != {0, 1}:
        raise RuntimeError(
            f"Expected binary labels 0/1, "
            f"got {np.unique(y_train)}"
        )

    print()
    print(
        f"Training samples: {len(y_train)}"
    )
    print(
        f"Training groups : "
        f"{len(np.unique(groups_train))}"
    )
    print(
        f"Feature dim     : "
        f"{x_train.shape[1]}"
    )
    print(
        "Repeat winners : "
        "seed42=SVM+2D, "
        "seed52/62/72=SVM+1D, "
        "seed82=MLP+2D"
    )

    # Model-definition check.
    for model_key in [
        "mlp",
        "cnn1d",
        "cnn2d",
    ]:
        model = build_deep_model(
            model_key,
            x_train.shape[1],
        )

        print(
            f"{DISPLAY_MODEL[model_key]:<6s} "
            f"parameters: "
            f"{count_parameters(model):,}"
        )

        del model

    # Check group isolation for every relevant repeat.
    for pair_key, spec in (
        PAIR_SPECS.items()
    ):
        for repeat_seed in (
            spec["seeds"]
        ):
            splitter = (
                StratifiedGroupKFold(
                    n_splits=args.folds,
                    shuffle=True,
                    random_state=repeat_seed,
                )
            )

            for fold_id, (
                fit_idx,
                heldout_idx,
            ) in enumerate(
                splitter.split(
                    np.zeros(
                        len(y_train)
                    ),
                    y_train,
                    groups_train,
                ),
                start=1,
            ):
                fit_groups = set(
                    groups_train[fit_idx]
                    .tolist()
                )

                heldout_groups = set(
                    groups_train[
                        heldout_idx
                    ].tolist()
                )

                overlap = (
                    fit_groups
                    .intersection(
                        heldout_groups
                    )
                )

                if overlap:
                    raise RuntimeError(
                        f"Group leakage: "
                        f"{pair_key}, seed={repeat_seed}, "
                        f"fold={fold_id}"
                    )

    print(
        "Group-isolation check: PASSED"
    )

    if args.check_only:
        print()
        print(
            "CHECK PASSED. No training was started."
        )
        return

    # Waveforms are required by both CNN1D and CNN2D.
    waveforms = preload_training_waveforms(
        project_root=project_root,
        metadata=metadata,
        train_indices=train_indices,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device          : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"Capability      : "
            f"{torch.cuda.get_device_capability(0)}"
        )

    results: dict[
        str,
        PairResult,
    ] = {}

    for pair_key in [
        "svm_cnn1d",
        "svm_cnn2d",
        "mlp_cnn2d",
    ]:
        results[pair_key] = run_pair(
            pair_key=pair_key,
            pair_spec=PAIR_SPECS[pair_key],
            x_train_full=x_train,
            y_train=y_train,
            groups_train=groups_train,
            waveforms=waveforms,
            device=device,
            epochs=args.epochs,
            n_splits=args.folds,
        )

    plot_results(
        results=results,
        output_png=output_png,
        epochs=args.epochs,
    )

    print_summary(
        results
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(
        f"Figure saved to: {output_png}"
    )


if __name__ == "__main__":
    main()
