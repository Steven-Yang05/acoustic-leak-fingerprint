from __future__ import annotations

"""
ours (frozen cal. RBF-SVM + 1D CNN fusion) vs the wav2vec 2.0 fine-tuned
benchmark — statistical and cost comparison.

Task 1 (per-sample w2v2-FT verification probabilities):
    wav2vec2_baseline.py never persisted the fine-tuned weights (no
    torch.save / save_pretrained / joblib.dump anywhere in the script;
    verified by source inspection and a full project + cache search).
    This script does NOT re-run the fine-tune. To obtain the per-sample
    probabilities, run src/w2v2_finetune_verify.py first (it re-runs the
    recorded recipe once, persists the weights to models_w2v2_finetuned/,
    verifies reproduction against the recorded aggregate metrics, and
    writes the per-sample probabilities into this script's output
    directory).

Task 2 (paired statistics ours vs w2v2-FT):
    McNemar and paired group-bootstrap require per-sample w2v2-FT
    probabilities. If they are unavailable, the aggregate recorded
    metrics are still tabulated for context.

Task 3 (parameter count + cost-sensitive comparison):
    Pure computation from recorded artifacts; fully executed below.
"""

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_VERSION = "w2v2-verification-stats-v1"

COST_RATIOS = [1, 2, 3, 5, 10]

# Recorded frozen-test confusion counts (from the recorded CSVs).
RECORDED = {
    "ours": {"fp": 7, "fn": 4},
    "w2v2_finetuned": {"fp": 5, "fn": 5},
    "w2v2_embedding_lr": {"fp": 14, "fn": 11},
}


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def search_finetuned_weights(root: Path) -> dict[str, Any]:
    """
    Document the search for any persisted w2v2 fine-tuned checkpoint.
    Returns the search record and whether anything was found.
    """
    searched: list[str] = []
    found: list[str] = []

    patterns = ["*.pt", "*.pth", "*.safetensors", "*.joblib", "*.bin"]
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if "__pycache__" in str(path):
                continue
            searched.append(str(path))
            name = path.name.lower()
            if "w2v2" in name or "wav2vec" in name or "finetun" in name:
                found.append(str(path))

    hf_cache = Path.home() / ".cache" / "huggingface"
    torch_hub = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cache_note: list[str] = []
    for cache_dir in [hf_cache, torch_hub]:
        if cache_dir.exists():
            for path in sorted(cache_dir.rglob("*")):
                if path.is_file() and (
                    "wav2vec" in path.name.lower()
                    or "w2v2" in path.name.lower()
                ):
                    cache_note.append(str(path))

    return {
        "project_checkpoint_files_scanned": searched,
        "w2v2_finetuned_candidates_found": found,
        "pretrained_backbone_cache_files": cache_note,
        "finetuned_weights_available": len(found) > 0,
    }


def count_w2v2_params() -> dict[str, Any]:
    """
    Exact parameter count of the torchaudio WAV2VEC2_BASE backbone plus
    the linear head used by wav2vec2_baseline.py. Loads the locally
    cached pretrained backbone (~/.cache/torch/hub/checkpoints/
    wav2vec2_fairseq_base_ls960.pth); no download, no training.
    """
    try:
        from torchaudio.pipelines import WAV2VEC2_BASE

        backbone = WAV2VEC2_BASE.get_model()
        backbone_params = sum(p.numel() for p in backbone.parameters())
        head_params = 768 + 1  # nn.Linear(768, 1) in W2V2Classifier
        del backbone
        return {
            "backbone_params": int(backbone_params),
            "head_params": int(head_params),
            "total_params": int(backbone_params + head_params),
            "count_method": "instantiated from locally cached torchaudio "
                            "WAV2VEC2_BASE bundle",
        }
    except Exception as exc:
        return {
            "backbone_params": 94370944,
            "head_params": 769,
            "total_params": 94371713,
            "count_method": f"documented fallback value (instantiation "
                            f"failed: {exc})",
        }


def svm_parameter_accounting(svm_path: Path) -> dict[str, Any]:
    """
    SVM 'parameter' accounting convention (stated explicitly):
      stored model = support vectors (n_sv x 212) + dual coefficients
      (n_sv) + intercept (1); plus the preprocessing StandardScaler
      (212 means + 212 scales) and the Platt calibrator (coef + intercept).
    """
    pipe = joblib.load(svm_path)
    svc = pipe.named_steps["classifier"]
    scaler = pipe.named_steps["scaler"]

    n_sv = int(svc.support_vectors_.shape[0])
    dim = int(svc.support_vectors_.shape[1])
    sv_params = n_sv * dim
    dual_params = int(svc.dual_coef_.size)
    intercept_params = int(svc.intercept_.size)
    scaler_params = int(scaler.mean_.size + scaler.scale_.size)
    platt_params = 2

    return {
        "n_support_vectors": n_sv,
        "n_support_per_class": [int(x) for x in svc.n_support_],
        "feature_dim": dim,
        "support_vector_params": sv_params,
        "dual_coefficient_params": dual_params,
        "intercept_params": intercept_params,
        "standard_scaler_params": scaler_params,
        "platt_calibrator_params": platt_params,
        "total_params": int(
            sv_params + dual_params + intercept_params
            + scaler_params + platt_params
        ),
        "convention": (
            "support vectors (n_sv x dim) + dual coefficients (n_sv) + "
            "intercept (1) + StandardScaler (2 x dim) + Platt calibrator (2)"
        ),
    }


def main() -> None:
    t0 = time.time()
    root = resolve_root()
    out_dir = root / "results_w2v2_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("RUNNING: src/w2v2_verification_stats.py")
    print(f"VERSION         : {SCRIPT_VERSION}")
    print("RETRAINING      : none (w2v2 fine-tune is NOT re-run here; see "
          "src/w2v2_finetune_verify.py)")
    print("=" * 96)

    svm_path = root / "models_frozen" / "rbf_svm_full_train.joblib"
    ver22_path = (
        root / "results_wav2vec2"
        / "pretrained_verification_results.csv"
    )
    p0_path = (
        root / "results_frozen_verification" / "fixed_test_predictions.csv"
    )
    for path in [svm_path, ver22_path, p0_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # ------------------------------------------------------------------
    # Task 1: fine-tuned weight search (documented; no retraining)
    # ------------------------------------------------------------------
    print()
    print("TASK 1: searching for persisted w2v2 fine-tuned weights ...")
    search = search_finetuned_weights(root)
    print(f"  project checkpoints scanned : "
          f"{len(search['project_checkpoint_files_scanned'])}")
    print(f"  fine-tuned candidates found : "
          f"{len(search['w2v2_finetuned_candidates_found'])}")
    for c in search["pretrained_backbone_cache_files"]:
        print(f"  pretrained backbone cache   : {c}")

    if search["finetuned_weights_available"]:
        task1 = {
            "status": "weights_found",
            "candidates": search["w2v2_finetuned_candidates_found"],
            "note": (
                "Per-sample probabilities are produced by "
                "src/w2v2_finetune_verify.py, not by this script."
            ),
            "search_record": search,
        }
        print("  -> fine-tuned weights found "
              "(produced by w2v2_finetune_verify.py).")
    else:
        task1 = {
            "status": "infeasible",
            "reason": (
                "wav2vec2_baseline.py never persisted the fine-tuned "
                "wav2vec2 weights (source inspection shows no torch.save / "
                "save_pretrained / joblib.dump; project-wide checkpoint "
                "scan found no w2v2 fine-tuned file). Only the PRETRAINED "
                "backbone is cached locally, which is the fine-tuning "
                "STARTING POINT, not the fine-tuned model. Run "
                "src/w2v2_finetune_verify.py to re-run the recorded recipe "
                "once and persist the weights."
            ),
            "search_record": search,
        }
        print("  -> no persisted fine-tuned weights; run "
              "src/w2v2_finetune_verify.py first.")

    # ------------------------------------------------------------------
    # Task 2: paired statistics (blocked) + aggregate context
    # ------------------------------------------------------------------
    ver22 = pd.read_csv(ver22_path, encoding="utf-8-sig")
    w2v2_ft_row = ver22[
        (ver22["model"] == "w2v2_finetuned")
        & (ver22["threshold_rule"] == "default_0.5")
    ].iloc[0]

    p0 = pd.read_csv(p0_path, encoding="utf-8-sig")
    y_test = p0["true_label"].to_numpy(dtype=np.int64)
    ours_prob = p0["fusion_probability"].to_numpy(dtype=np.float64)
    ours_pred = (ours_prob >= 0.520).astype(np.int64)
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
    )
    ours_agg = {
        "accuracy": float(accuracy_score(y_test, ours_pred)),
        "f1": float(f1_score(y_test, ours_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_test, ours_prob)),
        "fp": int(np.sum((ours_pred == 1) & (y_test == 0))),
        "fn": int(np.sum((ours_pred == 0) & (y_test == 1))),
    }

    probs_path = out_dir / "w2v2_finetuned_verification_probs.csv"
    task2: dict[str, Any]
    if probs_path.exists():
        task2 = {
            "status": "probabilities_available",
            "per_sample_probs": str(probs_path),
            "note": (
                "Paired McNemar / bootstrap / calibration statistics are "
                "computed by src/w2v2_finetune_verify.py; see its outputs "
                "in this directory."
            ),
        }
    else:
        task2 = {
            "status": "blocked",
            "reason": (
                "Exact McNemar and paired group-cluster bootstrap require "
                "per-sample w2v2 fine-tuned probabilities on the same 200 "
                "samples; run src/w2v2_finetune_verify.py to produce them. "
                "Aggregate metrics alone do not determine the paired "
                "discordant counts."
            ),
        }
    task2["aggregate_context"] = {
        "ours": ours_agg,
        "w2v2_finetuned": {
            "accuracy": float(w2v2_ft_row["accuracy"]),
            "f1": float(w2v2_ft_row["f1"]),
            "roc_auc": float(w2v2_ft_row["roc_auc"]),
            "fp": int(w2v2_ft_row["fp"]),
            "fn": int(w2v2_ft_row["fn"]),
        },
        "note": (
            "Aggregate-only context, not a paired test: w2v2-FT has "
            "+0.5 pp accuracy/F1 but -1.75 pp ROC-AUC vs ours."
        ),
    }
    print()
    print(f"TASK 2: {task2['status']}")
    print(f"  ours      : acc={ours_agg['accuracy']:.3f} "
          f"f1={ours_agg['f1']:.5f} auc={ours_agg['roc_auc']:.4f} "
          f"fp={ours_agg['fp']} fn={ours_agg['fn']}")
    print(f"  w2v2-FT   : acc={float(w2v2_ft_row['accuracy']):.3f} "
          f"f1={float(w2v2_ft_row['f1']):.5f} "
          f"auc={float(w2v2_ft_row['roc_auc']):.4f} "
          f"fp={int(w2v2_ft_row['fp'])} fn={int(w2v2_ft_row['fn'])}")

    # ------------------------------------------------------------------
    # Task 3: parameter counts + cost-sensitive comparison
    # ------------------------------------------------------------------
    print()
    print("TASK 3: parameter counts and cost-sensitive comparison")

    w2v2_params = count_w2v2_params()
    svm_params = svm_parameter_accounting(svm_path)
    cnn1d_params = 105569  # model_selection_oof.py EXPECTED_PARAM_COUNTS

    ours_total = svm_params["total_params"] + cnn1d_params

    param_rows = [
        {
            "model": "ours_fusion_total",
            "params": ours_total,
            "detail": (
                f"cal. RBF-SVM {svm_params['total_params']} "
                f"({svm_params['convention']}) + CNN1D {cnn1d_params}"
            ),
        },
        {
            "model": "ours_cnn1d_branch",
            "params": cnn1d_params,
            "detail": "model_selection_oof.py CNN1D, EXPECTED_PARAM_COUNTS "
                      "verified",
        },
        {
            "model": "ours_rbf_svm_branch",
            "params": svm_params["total_params"],
            "detail": svm_params["convention"],
        },
        {
            "model": "w2v2_finetuned",
            "params": w2v2_params["total_params"],
            "detail": (
                f"WAV2VEC2_BASE backbone {w2v2_params['backbone_params']} "
                f"+ linear head {w2v2_params['head_params']}; "
                f"{w2v2_params['count_method']}"
            ),
        },
    ]
    param_df = pd.DataFrame(param_rows)
    param_df["vs_w2v2_ratio"] = (
        w2v2_params["total_params"] / param_df["params"]
    )
    param_df.to_csv(
        out_dir / "parameter_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print()
    print(param_df.to_string(index=False))

    # Expected cost: cost(k) = FP + k * FN
    cost_rows: list[dict[str, Any]] = []
    for model_key, counts in RECORDED.items():
        row: dict[str, Any] = {
            "model": model_key,
            "fp": counts["fp"],
            "fn": counts["fn"],
        }
        for k in COST_RATIOS:
            row[f"cost_fn_x{k}"] = (
                counts["fp"] + k * counts["fn"]
            )
        cost_rows.append(row)
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(
        out_dir / "expected_cost_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print()
    print(cost_df.to_string(index=False))

    # Critical FN-cost ratio where ours overtakes w2v2-FT:
    #   7 + 4k < 5 + 5k  <=>  k > 2
    fp_o, fn_o = RECORDED["ours"]["fp"], RECORDED["ours"]["fn"]
    fp_w, fn_w = (
        RECORDED["w2v2_finetuned"]["fp"],
        RECORDED["w2v2_finetuned"]["fn"],
    )
    # Solve fp_o + k*fn_o = fp_w + k*fn_w
    if fn_o != fn_w:
        k_star = (fp_w - fp_o) / (fn_o - fn_w)
    else:
        k_star = None

    critical = {
        "ours_vs_w2v2_finetuned": {
            "equation": "cost_ours(k)=7+4k vs cost_w2v2(k)=5+5k",
            "break_even_k": float(k_star) if k_star is not None else None,
            "interpretation": (
                "tie at k=2 (both cost 15); ours is cheaper whenever the "
                "FN cost exceeds 2x the FP cost; w2v2-FT is cheaper only "
                "for k < 2"
            ),
        },
        "ours_vs_w2v2_embedding_lr": {
            "equation": "cost_ours(k)=7+4k vs cost_embLR(k)=14+11k",
            "break_even_k": None,
            "interpretation": (
                "ours is cheaper at every k >= 0 (lower FP AND lower FN); "
                "no cross-over exists"
            ),
        },
    }
    print()
    print(f"Critical ratio ours vs w2v2-FT : k* = {k_star:.2f} "
          f"(ours cheaper for FN-cost ratio > {k_star:.2f})")

    stats = {
        "script_version": SCRIPT_VERSION,
        "task1_w2v2_per_sample_probabilities": task1,
        "task2_paired_statistics": task2,
        "task3": {
            "parameter_counts": {
                "ours_fusion_total": int(ours_total),
                "ours_cnn1d_branch": int(cnn1d_params),
                "ours_rbf_svm_branch": svm_params,
                "w2v2_finetuned": w2v2_params,
                "w2v2_over_ours_fusion_ratio": float(
                    w2v2_params["total_params"] / ours_total
                ),
                "w2v2_over_cnn1d_ratio": float(
                    w2v2_params["total_params"] / cnn1d_params
                ),
            },
            "expected_cost": {
                "definition": "cost(k) = FP + k * FN on the 200-sample "
                              "verification set",
                "ratios": COST_RATIOS,
                "table": cost_rows,
            },
            "critical_fn_cost_ratio": critical,
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print()
    print(f"Results: {out_dir}")
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
