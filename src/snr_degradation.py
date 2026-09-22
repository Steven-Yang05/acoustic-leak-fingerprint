from __future__ import annotations

"""
snr_degradation.py: post-freeze acoustic-noise robustness audit.

The frozen final pipeline from frozen_verification.py (Platt-calibrated
RBF-SVM, 1D CNN, fixed 0.55/0.45 fusion at tau=0.520) is evaluated on the
fixed 200-sample verification partition after controlled noise injection.
No retraining or retuning is performed.

Noise conditions:
  - types : additive white Gaussian noise, pink (1/f) noise
  - SNR   : +20, +15, +10, +5, 0, -5 dB (per-clip RMS matched)
  - seeds : 3 noise realizations per condition

Two questions:
  1) Does the frozen fusion degrade more gracefully than either
     component as SNR falls?
  2) Does the reliability-aware dynamic gate (rejected on clean data in
     the extras/ mechanism and meta-CV audits) become beneficial under
     noise stress?
     Tested post hoc with a representative beta = 2.0 (the meta-CV
     selections in extras/reliability_gate_meta_cv.py were 1.925-2.225).

Outputs: results_snr_degradation/
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch

SCRIPT_VERSION = "frozen-snr-degradation-v1"

# Note: noise realizations are seeded by NOISE_SEEDS below; there is no
# single global SEED in this script.
SNR_LEVELS_DB = [20, 15, 10, 5, 0, -5]
NOISE_TYPES = ["white", "pink"]
NOISE_SEEDS = [2026, 2027, 2028]

SVM_WEIGHT = 0.55
CNN_WEIGHT = 0.45
FUSION_THRESHOLD = 0.520
COMPONENT_THRESHOLD = 0.5
GATE_BETA = 2.0


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_raw_waveform(path: Path) -> np.ndarray:
    """Mono, 8 kHz, exactly 8000 samples, DC removed, NOT peak-normalized.

    Peak normalization is a positive scalar gain and therefore leaves the
    SNR of a signal+noise mixture unchanged; it is applied per branch
    afterwards exactly as in the original pipelines.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = np.mean(data, axis=1).astype(np.float32)
    if x.size < 8000:
        x = np.pad(x, (0, 8000 - x.size))
    elif x.size > 8000:
        x = x[:8000]
    x = x - x.mean()
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite waveform: {path}")
    return x.astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Voss-style 1/f spectrum shaping of white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / 8000.0)
    freqs[0] = freqs[1]  # avoid division by zero at DC
    spec = spec / np.sqrt(freqs)
    out = np.fft.irfft(spec, n=n)
    out = out / (np.std(out) + 1e-12)
    return out.astype(np.float32)


def make_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "white":
        w = rng.standard_normal(n)
        return (w / (np.std(w) + 1e-12)).astype(np.float32)
    if kind == "pink":
        return pink_noise(n, rng)
    raise KeyError(kind)


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    rms_s = float(np.sqrt(np.mean(x**2)) + 1e-12)
    rms_n = float(np.sqrt(np.mean(noise**2)) + 1e-12)
    gain = rms_s / (rms_n * 10.0 ** (snr_db / 20.0))
    return (x + gain * noise).astype(np.float32)


def peak_normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak
    return x.astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def gated_fusion(p_svm: np.ndarray, p_cnn: np.ndarray, beta: float) -> np.ndarray:
    r = 2.0 * np.abs(p_svm - 0.5) - 2.0 * np.abs(p_cnn - 0.5)
    w = sigmoid(logit(SVM_WEIGHT) + beta * r)
    return w * p_svm + (1.0 - w) * p_cnn


def metric_row(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    pred = (prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "fp": int(fp),
        "fn": int(fn),
    }


def main() -> None:
    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results_snr_degradation"
    out_dir.mkdir(exist_ok=True)

    exp11 = load_module(root / "src" / "model_selection_oof.py", "exp11_snr")
    exp01 = load_module(root / "src" / "extract_features.py", "exp01_snr")

    # ------------- frozen artefacts from frozen_verification.py -------------
    model_dir = root / "models_frozen"
    full_svm = joblib.load(model_dir / "rbf_svm_full_train.joblib")
    calibrator = joblib.load(model_dir / "rbf_svm_platt_calibrator_training_oof.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = exp11.build_deep_model("cnn1d", 212).to(device)
    ckpt = torch.load(model_dir / "cnn1d_full_train_final.pt", map_location=device, weights_only=True)
    cnn.load_state_dict(ckpt["state_dict"])
    cnn.eval()
    print(f"frozen models loaded; device={device}")

    # ---------------- fixed verification data ----------------
    metadata = pd.read_csv(root / "features" / "metadata.csv", encoding="utf-8-sig")
    y_all = np.load(root / "features" / "y.npy", allow_pickle=False).astype(np.int64)
    test_indices = np.load(root / "train_test_data" / "test_indices.npy", allow_pickle=False).astype(np.int64)
    y_test = y_all[test_indices]
    groups_test = exp11.derive_group_ids(metadata)[test_indices]

    print("loading 200 raw verification waveforms ...")
    raw = np.stack([
        load_raw_waveform(exp11.resolve_wav_path(root, metadata.iloc[int(i)]["file_path"]))
        for i in test_indices
    ])  # [200, 8000]

    # ---------------- CNN branch: batch inference for all conditions ----------------
    conds = [(nt, snr, ns) for nt in NOISE_TYPES for snr in SNR_LEVELS_DB for ns in NOISE_SEEDS]
    print(f"conditions: {len(conds)} (plus clean)")

    def build_mixed(nt: str, snr: float, ns: int) -> np.ndarray:
        rng = np.random.default_rng(ns)
        noisy = np.empty_like(raw)
        for i in range(len(raw)):
            noisy[i] = mix_at_snr(raw[i], make_noise(nt, 8000, rng), snr)
        return noisy

    def cnn_probs(mixed: np.ndarray) -> np.ndarray:
        normed = np.stack([peak_normalize(m) for m in mixed])
        waves = torch.from_numpy(normed).unsqueeze(1)  # [N, 1, 8000]
        return exp14_infer(exp11, cnn, waves, device)

    def exp14_infer(exp11_mod, model, waves, dev) -> np.ndarray:
        probs: list[np.ndarray] = []
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(waves),
            batch_size=256, shuffle=False,
        )
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(dev)
                with exp11_mod.autocast_context(dev.type == "cuda"):
                    logits = model(xb)
                    p = torch.sigmoid(logits)
                probs.append(p.float().cpu().numpy())
        return np.concatenate(probs).astype(np.float64)

    def svm_probs(exp01_mod, mixed: np.ndarray, tmp_dir: Path) -> np.ndarray:
        tmp_wav = tmp_dir / "tmp_clip.wav"
        feats = np.empty((len(mixed), 212), dtype=np.float64)
        for i, m in enumerate(mixed):
            sf.write(tmp_wav, m, 8000, subtype="FLOAT")
            vec, _ = exp01_mod.extract_features(tmp_wav)
            feats[i] = vec
        score = exp11.classical_score(full_svm, feats)
        return calibrator.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]

    rows: list[dict[str, Any]] = []
    prob_store: dict[str, np.ndarray] = {}

    def evaluate(tag: str, p_svm: np.ndarray, p_cnn: np.ndarray, noise_type: str, snr: float, ns: int) -> None:
        p_fusion = SVM_WEIGHT * p_svm + CNN_WEIGHT * p_cnn
        p_gate = gated_fusion(p_svm, p_cnn, GATE_BETA)
        prob_store[tag] = np.stack([p_svm, p_cnn, p_fusion, p_gate], axis=1)
        for model, prob, thr in [
            ("svm_calibrated", p_svm, COMPONENT_THRESHOLD),
            ("cnn1d", p_cnn, COMPONENT_THRESHOLD),
            ("fusion_fixed", p_fusion, FUSION_THRESHOLD),
            ("fusion_reliability_gate_b2", p_gate, FUSION_THRESHOLD),
        ]:
            m = metric_row(y_test, prob, thr)
            rows.append({
                "noise_type": noise_type, "snr_db": snr, "noise_seed": ns,
                "model": model, "threshold": thr, **m,
            })

    with tempfile.TemporaryDirectory(prefix="snr21_", dir=out_dir) as tmp:
        tmp_dir = Path(tmp)

        # clean baseline
        print("evaluating clean baseline ...")
        p_svm_clean = svm_probs_from_array_features(exp11, full_svm, calibrator, root, test_indices)
        p_cnn_clean = cnn_probs(raw)
        evaluate("clean", p_svm_clean, p_cnn_clean, "clean", np.inf, 0)

        for ci, (nt, snr, ns) in enumerate(conds, 1):
            print(f"[{ci}/{len(conds)}] noise={nt} snr={snr} seed={ns}")
            mixed = build_mixed(nt, snr, ns)
            p_cnn = cnn_probs(mixed)
            p_svm = svm_probs(exp01, mixed, tmp_dir)
            evaluate(f"{nt}_{snr}_{ns}", p_svm, p_cnn, nt, snr, ns)
            if ci % 6 == 0:
                pd.DataFrame(rows).to_csv(out_dir / "snr_degradation_per_seed.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "snr_degradation_per_seed.csv", index=False)

    agg = (
        df.groupby(["noise_type", "snr_db", "model"], as_index=False)
        [["accuracy", "recall", "specificity", "f1", "roc_auc"]]
        .agg(["mean", "std"])
    )
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.to_csv(out_dir / "snr_degradation_summary.csv", index=False)

    np.savez_compressed(
        out_dir / "per_clip_probabilities.npz",
        y_test=y_test,
        groups_test=groups_test,
        **prob_store,
    )

    protocol = {
        "script_version": SCRIPT_VERSION,
        "frozen_models": "models_frozen (frozen_verification.py; no retraining)",
        "snr_levels_db": SNR_LEVELS_DB,
        "noise_types": NOISE_TYPES,
        "noise_seeds": NOISE_SEEDS,
        "snr_definition": "per-clip RMS ratio, 20*log10(rms_signal/rms_noise)",
        "fusion": {"svm_weight": SVM_WEIGHT, "cnn_weight": CNN_WEIGHT, "threshold": FUSION_THRESHOLD},
        "component_threshold": COMPONENT_THRESHOLD,
        "gate": {"type": "reliability", "beta": GATE_BETA,
                 "note": "post hoc; beta representative of the reliability_gate_meta_cv.py meta-CV selections (1.925-2.225)"},
        "cnn_branch": "peak normalization after mixing (SNR invariant to scalar gain)",
        "svm_branch": "features recomputed from noisy waveform, absolute amplitude preserved",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "snr_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print()
    print(df[df["model"] == "fusion_fixed"].to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {out_dir}")


def svm_probs_from_array_features(exp11, full_svm, calibrator, root: Path, test_indices: np.ndarray) -> np.ndarray:
    """Clean baseline: reuse the precomputed 212-d features from features/X.npy."""
    X_all = np.load(root / "features" / "X.npy", allow_pickle=False)
    score = exp11.classical_score(full_svm, X_all[test_indices])
    return calibrator.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


if __name__ == "__main__":
    main()
