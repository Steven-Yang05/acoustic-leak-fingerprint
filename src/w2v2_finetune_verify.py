from __future__ import annotations

"""
Re-run, persist and verify the wav2vec 2.0 fine-tuned benchmark, then
complete the paired statistical comparison vs ours.

Background: wav2vec2_baseline.py never persisted the fine-tuned weights,
so the per-sample verification probabilities were lost. This script
performs exactly one faithful re-run of the recorded fine-tuning recipe.

Step 1: re-run the full 800-sample fine-tune exactly as recorded in
        src/wav2vec2_baseline.py (the module's own finetune_fold /
        predict_finetuned / W2V2Dataset / W2V2Classifier are reused
        verbatim, so the recipe cannot drift), and persist the weights +
        full training config to models_w2v2_finetuned/.
Step 2: consistency gate — recompute Acc/F1/ROC-AUC/FP/FN at tau=0.5 on
        the frozen 200-sample verification partition and compare against
        the recorded 0.950 / 0.950 / 0.9676 / 5 / 5. One optional retry
        with torch deterministic algorithms is allowed only if the
        mismatch is within +/-2 misclassified samples.
Step 3: (only if the gate passes) paired statistics vs ours
        (0.55*calRBF-SVM + 0.45*CNN1D, tau=0.520):
        calibration (Brier / log-loss / 15-bin ECE), exact two-sided
        McNemar, and 5000x group-cluster bootstrap CIs for dF1 and dAUC.
"""

import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

SCRIPT_VERSION = "w2v2-finetune-save-verify-v1"

SEED = 42
N_BOOTSTRAP = 5000
ECE_BINS = 15

OURS_THRESHOLD = 0.520
W2V2_THRESHOLD = 0.5

# Recorded wav2vec2_baseline.py verification numbers (the consistency
# gate).
RECORDED = {
    "accuracy": 0.950,
    "f1": 0.950,
    "roc_auc": 0.9676,
    "fp": 5,
    "fn": 5,
}
AUC_TOL = 1e-4  # recorded AUC has 4 decimal places


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def exact_mcnemar(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    """Exact two-sided McNemar (binomial). a=rival (w2v2), b=ours."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    n = a_correct_b_wrong + a_wrong_b_correct
    if n == 0:
        p_value = 1.0
    else:
        k = min(a_correct_b_wrong, a_wrong_b_correct)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
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
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        ece += (n_b / n) * abs(
            float(np.mean(y_true[mask])) - float(np.mean(prob[mask]))
        )
    return float(ece)


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
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_to_rows: dict[Any, np.ndarray] = {
        g: np.flatnonzero(groups == g) for g in unique_groups
    }
    d_f1: list[float] = []
    d_auc: list[float] = []
    for _ in range(n_bootstrap):
        draw = rng.choice(unique_groups, size=len(unique_groups), replace=True)
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
                y_true, (prob_ours >= threshold_ours).astype(np.int64),
                pos_label=1, zero_division=0,
            )
            - f1_score(
                y_true, (prob_rival >= threshold_rival).astype(np.int64),
                pos_label=1, zero_division=0,
            )
        ),
        "delta_f1_ci_low": float(np.percentile(d_f1_arr, 2.5)),
        "delta_f1_ci_high": float(np.percentile(d_f1_arr, 97.5)),
        "delta_auc_point": float(
            roc_auc_score(y_true, prob_ours) - roc_auc_score(y_true, prob_rival)
        ),
        "delta_auc_ci_low": float(np.percentile(d_auc_arr, 2.5)),
        "delta_auc_ci_high": float(np.percentile(d_auc_arr, 97.5)),
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_auc_valid_n": int(len(d_auc_arr)),
        "groups_total": int(len(unique_groups)),
    }


def run_recorded_finetune(
    exp22,
    exp11,
    bundle,
    waves8k: torch.Tensor,
    y_all: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> tuple[Any, np.ndarray]:
    """
    One faithful call of the recorded wav2vec2_baseline.py full
    fine-tune:
    finetune_fold(bundle, waves8k, y_all, train_indices, train_indices)
    followed by predict_finetuned on the frozen test partition.
    The recipe lives entirely inside the wav2vec2_baseline.py module
    (seed 42 reset at entry, 15 epochs, batch 16, AdamW 2e-5/0.01,
    cosine, balanced BCE, AMP, grad clip 5.0, 8-kHz augmentation then
    16-kHz resample).
    """
    _, model_full = exp22.finetune_fold(
        bundle, waves8k, y_all, train_indices, train_indices, device, exp11
    )
    test_prob = exp22.predict_finetuned(
        model_full, waves8k, y_all, test_indices, device, exp11
    )
    return model_full, np.asarray(test_prob, dtype=np.float64)


def gate_check(metrics: dict[str, Any]) -> tuple[bool, str]:
    ok_counts = (
        int(metrics["fp"]) == RECORDED["fp"]
        and int(metrics["fn"]) == RECORDED["fn"]
    )
    ok_acc = abs(metrics["accuracy"] - RECORDED["accuracy"]) < 5e-4
    ok_f1 = abs(metrics["f1"] - RECORDED["f1"]) < 5e-4
    ok_auc = abs(metrics["roc_auc"] - RECORDED["roc_auc"]) < AUC_TOL
    passed = ok_counts and ok_acc and ok_f1 and ok_auc
    detail = (
        f"acc={metrics['accuracy']:.5f} (rec 0.950) | "
        f"f1={metrics['f1']:.5f} (rec 0.950) | "
        f"auc={metrics['roc_auc']:.5f} (rec 0.9676) | "
        f"fp={metrics['fp']} (rec 5) | fn={metrics['fn']} (rec 5)"
    )
    return passed, detail


def main() -> None:
    t0 = time.time()
    root = resolve_root()
    out_dir = root / "results_w2v2_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = root / "models_w2v2_finetuned"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("RUNNING: src/w2v2_finetune_verify.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("STEP 1          : faithful re-run of the recorded "
          "wav2vec2_baseline.py fine-tune")
    print("STEP 2          : consistency gate vs 0.950/0.950/0.9676/5/5")
    print("STEP 3          : paired stats vs ours (only if gate passes)")
    print("=" * 96)

    exp11 = load_module(
        root / "src" / "model_selection_oof.py",
        "exp11_for_finetune_verify",
    )
    exp22 = load_module(
        root / "src" / "wav2vec2_baseline.py",
        "w2v2_baseline_module",
    )

    metadata = pd.read_csv(
        root / "features" / "metadata.csv", encoding="utf-8-sig"
    )
    y_all = np.load(
        root / "features" / "y.npy", allow_pickle=False
    ).astype(np.int64)
    train_indices = np.load(
        root / "train_test_data" / "train_indices.npy", allow_pickle=False
    ).astype(np.int64)
    test_indices = np.load(
        root / "train_test_data" / "test_indices.npy", allow_pickle=False
    ).astype(np.int64)

    backbone_cache = (
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        / "wav2vec2_fairseq_base_ls960.pth"
    )
    if not backbone_cache.exists():
        raise FileNotFoundError(
            f"Local wav2vec2 backbone cache missing: {backbone_cache} "
            "(network download is not allowed)"
        )
    print(f"Backbone cache : {backbone_cache}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")
    if device.type == "cuda":
        print(f"GPU            : {torch.cuda.get_device_name(0)}")

    # Load waveforms exactly like wav2vec2_baseline.py: all 1000,
    # CNN-branch preprocessing.
    print()
    print("Loading all 1000 waveforms (8 kHz, peak-scaled) ...")
    waves8k = torch.stack([
        exp11.load_one_waveform(
            exp11.resolve_wav_path(root, metadata.iloc[int(i)]["file_path"])
        )
        for i in range(len(metadata))
    ])

    from torchaudio.pipelines import WAV2VEC2_BASE
    bundle = WAV2VEC2_BASE

    y_test = y_all[test_indices]

    # ------------------------------------------------------------------
    # Step 1 + 2: re-run, save, consistency gate (one optional retry)
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STEP 1: full 800-sample fine-tune (recorded "
          "wav2vec2_baseline.py recipe)")
    print("=" * 96)

    attempts: list[dict[str, Any]] = []
    model_full = None
    test_prob = None
    gate_passed = False

    for attempt in [1, 2]:
        if attempt == 2:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print()
            print("Retry attempt 2 with deterministic algorithms enabled.")

        model_full, test_prob = run_recorded_finetune(
            exp22=exp22,
            exp11=exp11,
            bundle=bundle,
            waves8k=waves8k,
            y_all=y_all,
            train_indices=train_indices,
            test_indices=test_indices,
            device=device,
        )

        metrics = exp22.quick_metrics(y_test, test_prob, W2V2_THRESHOLD)
        passed, detail = gate_check(metrics)
        attempts.append({
            "attempt": attempt,
            "deterministic_algorithms": attempt == 2,
            "metrics": metrics,
            "gate_passed": bool(passed),
        })
        print()
        print(f"Attempt {attempt} gate: {'PASSED' if passed else 'FAILED'}")
        print(f"  {detail}")

        if passed:
            gate_passed = True
            break

        # One retry is allowed only when the mismatch is within +/-2
        # misclassified samples relative to the recorded 5 FP / 5 FN.
        error_drift = (
            abs(int(metrics["fp"]) - RECORDED["fp"])
            + abs(int(metrics["fn"]) - RECORDED["fn"])
        )
        if attempt == 1 and error_drift > 2:
            print(
                f"  Mismatch too large (FP/FN drift={error_drift} > 2); "
                "no retry will be attempted."
            )
            break

    # Persist the fine-tuned model from the (last) attempt.
    weights_path = model_dir / "w2v2_finetuned_full_train.pt"
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "architecture": (
                "torchaudio WAV2VEC2_BASE backbone + nn.Linear(768, 1) "
                "head (wav2vec2_baseline.py W2V2Classifier)"
            ),
            "state_dict": model_full.state_dict(),
            "gate_passed": bool(gate_passed),
            "attempt_used": attempts[-1]["attempt"],
        },
        weights_path,
    )
    size_mb = weights_path.stat().st_size / (1024 ** 2)

    train_config = {
        "script_version": SCRIPT_VERSION,
        "source_recipe": "src/wav2vec2_baseline.py (verbatim reuse of "
                         "finetune_fold / predict_finetuned / W2V2Dataset "
                         "/ W2V2Classifier)",
        "backbone": "torchaudio WAV2VEC2_BASE (wav2vec2-base, 94,370,944 "
                    "params, LibriSpeech 960h)",
        "backbone_local_cache": str(backbone_cache),
        "head": "nn.Linear(768, 1) on mean-pooled final transformer layer",
        "preprocessing": (
            "model_selection_oof.py load_one_waveform: mono, 8 kHz, 8000 "
            "samples, DC removed, peak-scaled to [-1,1]; resampled to "
            "16 kHz inside the dataset; no tokenizer/processor is used "
            "(raw-waveform feature encoder)"
        ),
        "finetune": {
            "seed": SEED,
            "epochs": exp22.FT_EPOCHS,
            "batch_size": exp22.FT_BATCH,
            "optimizer": "AdamW",
            "lr": exp22.FT_LR,
            "weight_decay": exp22.FT_WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR(T_max=15, eta_min=1e-6)",
            "loss": "BCEWithLogitsLoss with balanced pos_weight",
            "amp": "cuda autocast fp16 + GradScaler",
            "grad_clip": 5.0,
            "augmentation": "model_selection_oof.py waveform protocol at "
                            "8 kHz (gain/shift/Gaussian noise) before "
                            "resampling",
            "train_samples": 800,
        },
        "verification": {
            "partition": "train_test_data/test_indices.npy (200 samples)",
            "threshold": W2V2_THRESHOLD,
        },
        "weights_file": str(weights_path),
        "weights_size_mb": round(size_mb, 1),
        "reload_instructions": (
            "m = w2v2_baseline.W2V2Classifier(WAV2VEC2_BASE.get_model()); "
            "m.load_state_dict(torch.load(weights_file)['state_dict'])"
        ),
        "consistency_gate": {
            "recorded": RECORDED,
            "attempts": attempts,
            "passed": bool(gate_passed),
        },
    }
    with open(model_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    print()
    print(f"Saved fine-tuned weights : {weights_path} ({size_mb:.1f} MB)")
    print(f"Saved training config    : {model_dir / 'training_config.json'}")

    # Always persist the per-sample probabilities of the re-run, flagged
    # with the gate outcome, so nothing is lost either way.
    groups_all = exp11.derive_group_ids(metadata)
    probs_df = pd.DataFrame({
        "source_index": test_indices,
        "group_id": groups_all[test_indices],
        "true_label": y_test,
        "probability": test_prob,
    })
    probs_path = out_dir / "w2v2_finetuned_verification_probs.csv"
    probs_df.to_csv(probs_path, index=False, encoding="utf-8-sig")
    print(f"Saved per-sample probs   : {probs_path}")

    if not gate_passed:
        print()
        print("=" * 96)
        print("CONSISTENCY GATE FAILED - Step 3 will NOT be run.")
        print("The re-run metrics above are the faithful record; no further")
        print("retries were performed (no number shopping).")
        print("=" * 96)
        _update_stats_json(
            out_dir,
            {
                "task1_rerun": {
                    "status": "rerun_completed_gate_failed",
                    "weights_file": str(weights_path),
                    "weights_size_mb": round(size_mb, 1),
                    "gate": train_config["consistency_gate"],
                }
            },
        )
        return

    print()
    print("Reproduction check PASSED - proceeding to Step 3.")

    # ------------------------------------------------------------------
    # Step 3: paired statistics ours vs w2v2-FT
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("STEP 3: paired statistics ours vs w2v2-FT")
    print("=" * 96)

    p0 = pd.read_csv(
        root / "results_frozen_verification" / "fixed_test_predictions.csv",
        encoding="utf-8-sig",
    )
    if not p0["source_index"].equals(probs_df["source_index"]):
        raise RuntimeError("P0 and w2v2 rows are not aligned.")
    if not p0["true_label"].equals(probs_df["true_label"]):
        raise RuntimeError("Label mismatch between P0 and w2v2 rows.")

    groups = p0["group_id"].to_numpy()
    ours_prob = p0["fusion_probability"].to_numpy(dtype=np.float64)

    ours_pred = (ours_prob >= OURS_THRESHOLD).astype(np.int64)
    w2v2_pred = (test_prob >= W2V2_THRESHOLD).astype(np.int64)

    # Calibration (raw sigmoid outputs, no recalibration - by design).
    ours_clip = np.clip(ours_prob, 1e-7, 1 - 1e-7)
    w2v2_clip = np.clip(test_prob, 1e-7, 1 - 1e-7)
    calib_rows = [
        {
            "model": "ours (cal. RBF-SVM + CNN1D, tau=0.520)",
            "brier": float(brier_score_loss(y_test, ours_clip)),
            "log_loss": float(log_loss(y_test, ours_clip, labels=[0, 1])),
            "ece_15bin": expected_calibration_error(y_test, ours_clip),
        },
        {
            "model": "w2v2_finetuned (raw sigmoid, tau=0.5)",
            "brier": float(brier_score_loss(y_test, w2v2_clip)),
            "log_loss": float(log_loss(y_test, w2v2_clip, labels=[0, 1])),
            "ece_15bin": expected_calibration_error(y_test, w2v2_clip),
        },
    ]
    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(
        out_dir / "w2v2_calibration_vs_ours.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mcnemar = exact_mcnemar(y_test, w2v2_pred, ours_pred)
    boot = group_bootstrap_delta(
        y_true=y_test,
        groups=groups,
        prob_ours=ours_prob,
        prob_rival=test_prob,
        threshold_ours=OURS_THRESHOLD,
        threshold_rival=W2V2_THRESHOLD,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    comparison = {
        "ours_threshold": OURS_THRESHOLD,
        "w2v2_threshold": W2V2_THRESHOLD,
        "mcnemar": mcnemar,
        "bootstrap": boot,
        "delta_f1_ci_contains_zero": bool(
            boot["delta_f1_ci_low"] <= 0.0 <= boot["delta_f1_ci_high"]
        ),
        "delta_auc_ci_contains_zero": bool(
            boot["delta_auc_ci_low"] <= 0.0 <= boot["delta_auc_ci_high"]
        ),
    }
    pd.DataFrame([{
        "rival": "w2v2_finetuned",
        **mcnemar,
        **boot,
        "ours_fp": int(np.sum((ours_pred == 1) & (y_test == 0))),
        "ours_fn": int(np.sum((ours_pred == 0) & (y_test == 1))),
        "rival_fp": int(np.sum((w2v2_pred == 1) & (y_test == 0))),
        "rival_fn": int(np.sum((w2v2_pred == 0) & (y_test == 1))),
    }]).to_csv(
        out_dir / "w2v2_comparison_vs_ours.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _update_stats_json(
        out_dir,
        {
            "task1_rerun": {
                "status": "rerun_completed_gate_passed",
                "weights_file": str(weights_path),
                "weights_size_mb": round(size_mb, 1),
                "per_sample_probs": str(probs_path),
                "gate": train_config["consistency_gate"],
            },
            "task2_paired_statistics_rerun": {
                "status": "completed",
                "calibration": calib_rows,
                "comparison": comparison,
            },
        },
    )

    print()
    print("Calibration (Brier / log-loss / ECE-15):")
    print(calib_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print()
    print(
        f"McNemar: discordant={mcnemar['discordant_total']} "
        f"(w2v2-correct/ours-wrong={mcnemar['rival_correct_ours_wrong']}, "
        f"ours-correct/w2v2-wrong={mcnemar['rival_wrong_ours_correct']}) "
        f"exact two-sided p={mcnemar['exact_two_sided_p']:.5f}"
    )
    print(
        f"dF1 (ours - w2v2)  = {boot['delta_f1_point']:+.5f} "
        f"[{boot['delta_f1_ci_low']:+.5f}, {boot['delta_f1_ci_high']:+.5f}]"
    )
    print(
        f"dAUC (ours - w2v2) = {boot['delta_auc_point']:+.5f} "
        f"[{boot['delta_auc_ci_low']:+.5f}, {boot['delta_auc_ci_high']:+.5f}]"
    )
    print()
    print(f"Total time : {(time.time() - t0):.1f} s")
    print(f"Results    : {out_dir}")
    print(f"Models     : {model_dir}")


def _update_stats_json(out_dir: Path, patch: dict[str, Any]) -> None:
    stats_path = out_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {}
    stats.update(patch)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
