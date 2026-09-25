from __future__ import annotations

"""
leave_one_condition_out_extra.py

Adds two stronger baselines to the leave_one_condition_out.py LOCO audit
(Section IV-B):

  1) MLP + 2D CNN heterogeneous fusion
  2) wav2vec 2.0 BASE full fine-tuning (94M backbone)

It reuses the recorded helpers of the main chain:
  - model_selection_oof.py        (deep-model training recipe, waveforms)
  - leave_one_condition_out.py    (conditions, fusion/threshold search)
  - fusion_pairs_verification.py  (full-source deep refit + inference)
  - wav2vec2_baseline.py          (fine-tuning recipe)

Protocol
--------
For each transfer direction:
  - use leak + on-site no_leak only, matching leave_one_condition_out.py;
  - do all model/threshold/fusion selection on the SOURCE condition only;
  - use 5-fold StratifiedGroupKFold OOF on source groups;
  - MLP+2D CNN: select fusion weight + threshold on source OOF only;
  - w2v2: select decision threshold on source OOF only;
  - refit once on all source-condition samples;
  - evaluate once on the unseen target condition with no target feedback.

Run:
    python src/leave_one_condition_out_extra.py

Quick first pass without wav2vec 2.0:
    python src/leave_one_condition_out_extra.py --skip-w2v2

Outputs are written next to the base LOCO results:
    results_loco/loco_extra_results.csv
    results_loco/target_predictions_extra_<task>.csv
    results_loco/loco_extra_protocol.json
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_repo_module(
    src: Path,
    filename: str,
    module_name: str,
):
    path = src / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find required helper module: {path}"
        )
    print(f"loading helper: {path.name}")
    return load_module(path, module_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extra LOCO baselines: MLP+2D CNN and fine-tuned wav2vec 2.0."
    )
    p.add_argument(
        "--skip-w2v2",
        action="store_true",
        help="Run only MLP+2D CNN (useful for a quicker first pass).",
    )
    return p.parse_args()


def deep_oof(
    exp11,
    model_key: str,
    X_src: np.ndarray,
    y_src: np.ndarray,
    waves_src: torch.Tensor,
    groups_src: np.ndarray,
    device: torch.device,
    folds: int,
    seed: int,
    epochs: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    oof = np.full(len(y_src), np.nan, dtype=np.float64)

    for fold, (fit_idx, held_idx) in enumerate(
        splitter.split(X_src, y_src, groups_src), 1
    ):
        print(
            f"      {model_key} OOF fold {fold}/{folds} "
            f"(fit={len(fit_idx)}, held={len(held_idx)})"
        )
        _, score, _ = exp11.train_deep_fold(
            model_key,
            fit_idx,
            held_idx,
            X_src,
            y_src,
            waves_src,
            device,
            epochs,
        )
        oof[held_idx] = score

    if np.isnan(oof).any():
        raise RuntimeError(f"{model_key}: incomplete OOF predictions")
    return oof


def add_metric_row(
    rows: list[dict[str, Any]],
    base_loco,
    *,
    task_name: str,
    variable: str,
    direction: str,
    split: str,
    model: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    fusion_weight_mlp: float | None = None,
) -> None:
    pred = (prob >= threshold).astype(np.int64)
    m = base_loco.quick_metrics(y_true, pred, prob)
    rows.append(
        {
            "task": task_name,
            "variable": variable,
            "direction": direction,
            "split": split,
            "model": model,
            "threshold": float(threshold),
            "fusion_weight_mlp": (
                float(fusion_weight_mlp)
                if fusion_weight_mlp is not None
                else np.nan
            ),
            **m,
        }
    )


def main() -> None:
    args = parse_args()
    t0 = time.time()

    root = Path(__file__).resolve().parents[1]
    src = root / "src"

    # Match the base LOCO script's output directory.
    out_dir = root / "results_loco"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp11 = load_repo_module(
        src,
        "model_selection_oof.py",
        "exp11_loco_extra",
    )
    base_loco = load_repo_module(
        src,
        "leave_one_condition_out.py",
        "base_loco_extra",
    )
    exp_pairs = load_repo_module(
        src,
        "fusion_pairs_verification.py",
        "exp_pairs_loco_extra",
    )

    w2v2 = None
    if not args.skip_w2v2:
        w2v2 = load_repo_module(
            src,
            "wav2vec2_baseline.py",
            "w2v2_loco_extra",
        )

    seed = int(base_loco.SEED)
    folds = int(base_loco.FOLDS)
    epochs = int(base_loco.EPOCHS)

    X_all = np.load(
        root / "features" / "X.npy",
        allow_pickle=False,
    ).astype(np.float32)
    y_all = np.load(
        root / "features" / "y.npy",
        allow_pickle=False,
    ).astype(np.int64)
    metadata = pd.read_csv(
        root / "features" / "metadata.csv",
        encoding="utf-8-sig",
    )

    cond = base_loco.parse_conditions(metadata)
    groups_all = exp11.derive_group_ids(metadata)

    if not (
        len(X_all) == len(y_all) == len(metadata) == len(groups_all)
    ):
        raise RuntimeError(
            "Feature/label/metadata/group lengths do not match: "
            f"X={len(X_all)}, y={len(y_all)}, "
            f"metadata={len(metadata)}, groups={len(groups_all)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Preload all 8-kHz, DC-removed, peak-scaled waveforms once using the
    # exact model_selection_oof.py helper. The w2v2 helper resamples to
    # 16 kHz inside W2V2Dataset, matching wav2vec2_baseline.py.
    print("preloading all waveforms once ...")
    waves8k = torch.stack(
        [
            exp11.load_one_waveform(
                exp11.resolve_wav_path(
                    root,
                    metadata.iloc[int(i)]["file_path"],
                )
            )
            for i in range(len(metadata))
        ]
    )

    bundle = None
    if not args.skip_w2v2:
        from torchaudio.pipelines import WAV2VEC2_BASE

        bundle = WAV2VEC2_BASE

    tasks = [
        ("device_hydro_to_logger", "cond_device", "hydrophone", "noise logger"),
        ("device_logger_to_hydro", "cond_device", "noise logger", "hydrophone"),
        ("material_di_to_pe", "cond_material", "ductile iron", "pe"),
        ("material_pe_to_di", "cond_material", "pe", "ductile iron"),
    ]

    all_rows: list[dict[str, Any]] = []

    for task_name, variable, source_level, target_level in tasks:
        print("\n" + "=" * 100)
        print(f"{task_name}: {source_level} -> {target_level}")
        print("=" * 100)

        on_site = cond["source_class"].isin(["leak", "no_leak"]).to_numpy()
        src_mask = on_site & (cond[variable] == source_level).to_numpy()
        tgt_mask = on_site & (cond[variable] == target_level).to_numpy()

        src_idx = np.where(src_mask)[0]
        tgt_idx = np.where(tgt_mask)[0]

        if len(src_idx) == 0 or len(tgt_idx) == 0:
            raise RuntimeError(
                f"{task_name}: empty source or target split. "
                f"source={len(src_idx)}, target={len(tgt_idx)}"
            )

        X_src = X_all[src_idx]
        y_src = y_all[src_idx]
        groups_src = groups_all[src_idx]
        waves_src = waves8k[src_idx]

        X_tgt = X_all[tgt_idx]
        y_tgt = y_all[tgt_idx]
        waves_tgt = waves8k[tgt_idx]

        print(
            f"source: {len(src_idx)} samples / "
            f"{len(np.unique(groups_src))} groups "
            f"(pos={int(y_src.sum())}, neg={int((1-y_src).sum())})"
        )
        print(
            f"target: {len(tgt_idx)} samples "
            f"(pos={int(y_tgt.sum())}, neg={int((1-y_tgt).sum())})"
        )

        # -------------------------------------------------------------
        # A) MLP + 2D CNN
        # -------------------------------------------------------------
        print("\n  [MLP+2D CNN] source OOF ...")
        mlp_oof = deep_oof(
            exp11,
            "mlp",
            X_src,
            y_src,
            waves_src,
            groups_src,
            device,
            folds,
            seed,
            epochs,
        )
        cnn2d_oof = deep_oof(
            exp11,
            "cnn2d",
            X_src,
            y_src,
            waves_src,
            groups_src,
            device,
            folds,
            seed,
            epochs,
        )

        pair_sel = base_loco.search_fusion(
            y_src,
            mlp_oof,
            cnn2d_oof,
        )
        w_mlp = float(pair_sel["weight_a"])
        tau_pair = float(pair_sel["threshold"])
        pair_oof = w_mlp * mlp_oof + (1.0 - w_mlp) * cnn2d_oof

        print(
            f"    source OOF selection: w_mlp={w_mlp:.2f}, "
            f"tau={tau_pair:.3f}, F1={pair_sel['f1']:.4f}, "
            f"AUC={pair_sel['roc_auc']:.4f}"
        )

        print("  [MLP+2D CNN] full source refit + unseen target ...")
        mlp_full, mlp_scaler, _ = exp_pairs.train_full_deep_model(
            exp11=exp11,
            model_key="mlp",
            X_train=X_src,
            y_train=y_src,
            train_waveforms=waves_src,
            device=device,
        )
        cnn2d_full, _, cnn2d_frontend = exp_pairs.train_full_deep_model(
            exp11=exp11,
            model_key="cnn2d",
            X_train=X_src,
            y_train=y_src,
            train_waveforms=waves_src,
            device=device,
        )

        mlp_tgt = exp_pairs.infer_deep_model(
            exp11=exp11,
            model_key="mlp",
            model=mlp_full,
            device=device,
            X=X_tgt,
            scaler=mlp_scaler,
        )
        cnn2d_tgt = exp_pairs.infer_deep_model(
            exp11=exp11,
            model_key="cnn2d",
            model=cnn2d_full,
            device=device,
            waveforms=waves_tgt,
            frontend=cnn2d_frontend,
        )
        pair_tgt = w_mlp * mlp_tgt + (1.0 - w_mlp) * cnn2d_tgt

        direction = f"{source_level}->{target_level}"

        add_metric_row(
            all_rows,
            base_loco,
            task_name=task_name,
            variable=variable,
            direction=direction,
            split="source_oof",
            model="mlp_cnn2d",
            y_true=y_src,
            prob=pair_oof,
            threshold=tau_pair,
            fusion_weight_mlp=w_mlp,
        )
        add_metric_row(
            all_rows,
            base_loco,
            task_name=task_name,
            variable=variable,
            direction=direction,
            split="target_loco",
            model="mlp_cnn2d",
            y_true=y_tgt,
            prob=pair_tgt,
            threshold=tau_pair,
            fusion_weight_mlp=w_mlp,
        )

        del mlp_full, cnn2d_full, mlp_scaler, cnn2d_frontend
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # B) fine-tuned wav2vec 2.0
        # -------------------------------------------------------------
        w2v2_tgt = None

        if not args.skip_w2v2:
            assert w2v2 is not None
            assert bundle is not None

            print("\n  [w2v2 fine-tuned] source OOF ...")
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            )
            w2v2_oof = np.full(
                len(y_src),
                np.nan,
                dtype=np.float64,
            )

            for fold, (fit_local, held_local) in enumerate(
                splitter.split(X_src, y_src, groups_src), 1
            ):
                # The wav2vec2_baseline.py helper indexes the GLOBAL
                # waveform/label arrays, so convert source-local fold
                # indices back to global indices.
                fit_global = src_idx[fit_local]
                held_global = src_idx[held_local]

                print(
                    f"      w2v2 OOF fold {fold}/{folds} "
                    f"(fit={len(fit_local)}, held={len(held_local)})"
                )
                probs, model = w2v2.finetune_fold(
                    bundle,
                    waves8k,
                    y_all,
                    fit_global,
                    held_global,
                    device,
                    exp11,
                )
                w2v2_oof[held_local] = probs

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if np.isnan(w2v2_oof).any():
                raise RuntimeError("w2v2: incomplete OOF predictions")

            tau_w2v2, w2v2_oof_metrics = base_loco.best_threshold(
                y_src,
                w2v2_oof,
            )
            print(
                f"    source OOF selection: tau={tau_w2v2:.3f}, "
                f"F1={w2v2_oof_metrics['f1']:.4f}, "
                f"AUC={w2v2_oof_metrics['roc_auc']:.4f}"
            )

            print("  [w2v2 fine-tuned] full source refit + unseen target ...")

            # finetune_fold is the repository's recorded fine-tuning recipe.
            # Passing src_idx as eval_idx produces source predictions that
            # are intentionally discarded; the returned model is then used
            # once on the unseen target.
            _, w2v2_full = w2v2.finetune_fold(
                bundle,
                waves8k,
                y_all,
                src_idx,
                src_idx,
                device,
                exp11,
            )

            w2v2_tgt = w2v2.predict_finetuned(
                w2v2_full,
                waves8k,
                y_all,
                tgt_idx,
                device,
                exp11,
            )

            add_metric_row(
                all_rows,
                base_loco,
                task_name=task_name,
                variable=variable,
                direction=direction,
                split="source_oof",
                model="w2v2_finetuned",
                y_true=y_src,
                prob=w2v2_oof,
                threshold=tau_w2v2,
            )
            add_metric_row(
                all_rows,
                base_loco,
                task_name=task_name,
                variable=variable,
                direction=direction,
                split="target_loco",
                model="w2v2_finetuned",
                y_true=y_tgt,
                prob=w2v2_tgt,
                threshold=tau_w2v2,
            )

            del w2v2_full
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Per-target probabilities for later paired comparisons.
        pred_frame = pd.DataFrame(
            {
                "index": tgt_idx,
                "y_true": y_tgt,
                "mlp_cnn2d_prob": pair_tgt,
            }
        )
        if w2v2_tgt is not None:
            pred_frame["w2v2_finetuned_prob"] = w2v2_tgt

        pred_frame.to_csv(
            out_dir / f"target_predictions_extra_{task_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Save incremental results after every transfer direction so a long
        # w2v2 run is not lost if a later task is interrupted.
        pd.DataFrame(all_rows).to_csv(
            out_dir / "loco_extra_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    results = pd.DataFrame(all_rows)
    results.to_csv(
        out_dir / "loco_extra_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "script": "leave_one_condition_out_extra.py",
        "matched_repo_helpers": {
            "model_selection": "model_selection_oof.py",
            "base_loco": "leave_one_condition_out.py",
            "wav2vec2": (
                None
                if args.skip_w2v2
                else "wav2vec2_baseline.py"
            ),
            "full_deep_refit": "fusion_pairs_verification.py",
        },
        "seed": seed,
        "folds": folds,
        "mlp_cnn2d_epochs": epochs,
        "w2v2_ft_epochs": (
            None
            if args.skip_w2v2
            else int(w2v2.FT_EPOCHS)
        ),
        "classes_used": ["leak", "no_leak"],
        "environmental_noise_excluded": True,
        "models": {
            "mlp_cnn2d": (
                "source-only group-aware OOF for each branch; "
                "fusion weight and threshold selected on source OOF; "
                "full-source refit; one-shot target evaluation"
            ),
            "w2v2_finetuned": (
                "source-only group-aware OOF fine-tuning; threshold selected "
                "on source OOF; full-source fine-tune; one-shot target evaluation"
                if not args.skip_w2v2
                else "skipped by CLI"
            ),
        },
        "target_feedback": False,
        "runtime_seconds": round(time.time() - t0, 1),
    }

    with open(
        out_dir / "loco_extra_protocol.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(results.to_string(index=False))
    print(
        f"\nDone in {(time.time() - t0) / 60.0:.1f} min. "
        f"Results: {out_dir / 'loco_extra_results.csv'}"
    )


if __name__ == "__main__":
    main()
