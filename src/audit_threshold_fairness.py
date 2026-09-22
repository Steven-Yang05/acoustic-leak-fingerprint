from __future__ import annotations

"""
audit_threshold_fairness.py — Threshold-fairness audit for the final calibrated RBF-SVM + 1D CNN fusion.

Purpose
-------
A reviewer may reasonably ask whether the frozen fusion's observed advantage
over its component experts partly reflects unequal threshold treatment:
the fusion uses a training-selected threshold (0.520), while the component
experts are commonly reported at the default 0.500 threshold.

This script:
1) NEVER trains a new classifier.
2) NEVER changes the frozen fusion.
3) Selects component thresholds using TRAINING-ONLY OOF probabilities.
4) Applies those frozen component thresholds to the existing fixed
   group-isolated verification probabilities.
5) Compares the frozen fusion against threshold-tuned SVM and 1D CNN
   using the same fixed 200 samples.
6) If group_id is available, performs exact McNemar and paired
   group-cluster bootstrap comparisons.

Preferred threshold derivation
------------------------------
If repeated-OOF probability files from multiple seeds can be found, the script
selects a threshold independently in each repeat and uses the median threshold.
If only the single training-OOF probability file from fusion_pair_search.py is found,
it performs a clearly labeled single-OOF sensitivity audit instead.

This is a fairness / robustness analysis. It must NOT be used to retune the
already frozen final fusion model.
"""

from pathlib import Path
import json
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_VERSION = "threshold-fairness-audit-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results_audit_threshold_fairness"

THRESHOLDS = np.round(np.arange(0.05, 0.9500001, 0.005), 3)
FUSION_THRESHOLD = 0.520
FUSION_SVM_WEIGHT = 0.55
FUSION_CNN_WEIGHT = 0.45
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260818


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def detect_col(df: pd.DataFrame, exact: list[str], contains_all: list[list[str]]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in exact:
        if name.lower() in lower:
            return lower[name.lower()]
    for parts in contains_all:
        candidates = []
        for c in df.columns:
            n = norm(c)
            if all(part in n for part in parts):
                candidates.append(c)
        if len(candidates) == 1:
            return candidates[0]
    raise RuntimeError(
        "Could not safely detect a required column.\n"
        f"Exact candidates: {exact}\n"
        f"Contains rules: {contains_all}\n"
        f"Available columns: {list(df.columns)}"
    )


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def selection_key(m: dict) -> tuple:
    # Same conceptual order used in the paper.
    return (
        m["f1"],
        m["recall"],
        m["specificity"],
        m["roc_auc"],
        m["average_precision"],
    )


def select_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = [metrics(y, p, float(t)) for t in THRESHOLDS]
    table = pd.DataFrame(rows)
    best = max(rows, key=selection_key)
    return float(best["threshold"]), table


def exact_mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    a_correct = pred_a == y
    b_correct = pred_b == y
    b = int(np.sum(a_correct & ~b_correct))  # A correct, B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong, B correct
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        # Exact two-sided binomial p under p=0.5
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "discordant": n, "p_exact": float(p)}


def cluster_bootstrap_delta(
    df: pd.DataFrame,
    y_col: str,
    pred_a_col: str,
    pred_b_col: str,
    group_col: str,
    reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = df[group_col].astype(str).unique()
    group_frames = {g: df[df[group_col].astype(str) == g] for g in groups}

    deltas = {"accuracy": [], "f1": [], "recall": [], "specificity": []}

    def _m(y, pred):
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        }

    for _ in range(reps):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        bdf = pd.concat([group_frames[g] for g in sampled], ignore_index=True)
        y = bdf[y_col].to_numpy(int)
        a = bdf[pred_a_col].to_numpy(int)
        b = bdf[pred_b_col].to_numpy(int)
        ma, mb = _m(y, a), _m(y, b)
        for k in deltas:
            if np.isfinite(ma[k]) and np.isfinite(mb[k]):
                deltas[k].append(ma[k] - mb[k])

    rows = []
    for k, vals in deltas.items():
        arr = np.asarray(vals, float)
        rows.append({
            "metric": k,
            "bootstrap_reps_used": int(len(arr)),
            "delta_mean_bootstrap": float(np.mean(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
            "ci_method": "percentile",
            "resampling_unit": "group_id",
        })
    return pd.DataFrame(rows)


def find_oof_files() -> list[Path]:
    preferred = [
        PROJECT_ROOT / "results_fusion_pairs" / "all_training_oof_probabilities_for_fusion.csv",
    ]
    found = [p for p in preferred if p.exists()]

    # Look for repeated probability outputs if stability_selection.py saved them.
    stab = PROJECT_ROOT / "results_stability"
    if stab.exists():
        for p in stab.rglob("*.csv"):
            name = p.name.lower()
            if ("probab" in name or "oof" in name) and p not in found:
                try:
                    cols = pd.read_csv(p, nrows=2).columns
                    cn = [norm(c) for c in cols]
                    if any("cnn1d" in x and "prob" in x for x in cn) and any("svm" in x and "prob" in x for x in cn):
                        found.append(p)
                except Exception:
                    pass

    # Last-resort project-wide exact basename search.
    if not found:
        for p in PROJECT_ROOT.rglob("all_training_oof_probabilities_for_fusion.csv"):
            found.append(p)

    return found


def load_oof(path: Path) -> tuple[pd.DataFrame, str, str, str, str | None]:
    df = pd.read_csv(path)
    y_col = detect_col(df,
        ["true_label", "label_id", "y_true"],
        [["true","label"], ["label","id"]]
    )
    svm_col = detect_col(df,
        ["svm_probability", "rbf_svm_calibrated_probability", "rbf_svm_calibrated_oof_probability"],
        [["svm","prob"]]
    )
    cnn_col = detect_col(df,
        ["cnn1d_probability", "cnn1d_oof_probability"],
        [["cnn1d","prob"]]
    )
    seed_col = None
    for c in df.columns:
        if norm(c) in {"seed", "repeatseed", "randomseed"}:
            seed_col = c
            break
    return df, y_col, svm_col, cnn_col, seed_col


def derive_thresholds(oof_files: list[Path]) -> tuple[dict, pd.DataFrame]:
    all_rows = []

    # Prefer a file with an explicit seed/repeat column.
    for p in oof_files:
        df, y_col, svm_col, cnn_col, seed_col = load_oof(p)
        if seed_col is not None and df[seed_col].nunique() > 1:
            for seed, sub in df.groupby(seed_col):
                y = sub[y_col].to_numpy(int)
                for model, col in [("Calibrated RBF-SVM", svm_col), ("1D CNN", cnn_col)]:
                    th, _ = select_threshold(y, sub[col].to_numpy(float))
                    m = metrics(y, sub[col].to_numpy(float), th)
                    all_rows.append({"source_file": str(p), "repeat": str(seed), "model": model, **m})
            mode = "repeated_oof_median_threshold"
            break
    else:
        # If multiple separate probability files exist, treat each as a repeat.
        valid = []
        for p in oof_files:
            try:
                df, y_col, svm_col, cnn_col, seed_col = load_oof(p)
                if len(df) >= 500:
                    valid.append((p, df, y_col, svm_col, cnn_col))
            except Exception:
                continue
        if len(valid) > 1:
            for idx, (p, df, y_col, svm_col, cnn_col) in enumerate(valid, start=1):
                y = df[y_col].to_numpy(int)
                for model, col in [("Calibrated RBF-SVM", svm_col), ("1D CNN", cnn_col)]:
                    th, _ = select_threshold(y, df[col].to_numpy(float))
                    m = metrics(y, df[col].to_numpy(float), th)
                    all_rows.append({"source_file": str(p), "repeat": f"file_{idx}", "model": model, **m})
            mode = "multiple_oof_files_median_threshold"
        elif valid:
            p, df, y_col, svm_col, cnn_col = valid[0]
            y = df[y_col].to_numpy(int)
            for model, col in [("Calibrated RBF-SVM", svm_col), ("1D CNN", cnn_col)]:
                th, grid = select_threshold(y, df[col].to_numpy(float))
                m = metrics(y, df[col].to_numpy(float), th)
                all_rows.append({"source_file": str(p), "repeat": "single_oof", "model": model, **m})
                grid.to_csv(OUT_DIR / f"{'svm' if model.startswith('Calibrated') else 'cnn1d'}_threshold_grid_single_oof.csv", index=False)
            mode = "single_oof_sensitivity"
        else:
            raise RuntimeError("No usable training-only OOF probability file could be located.")

    rep = pd.DataFrame(all_rows)
    thresholds = {}
    for model, sub in rep.groupby("model"):
        thresholds[model] = float(np.median(sub["threshold"].to_numpy(float)))
    return {"mode": mode, "thresholds": thresholds}, rep


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 86)
    print("Threshold fairness audit")
    print("Version :", SCRIPT_VERSION)
    print("Purpose : give component experts training-only threshold selection before fixed-set comparison")
    print("NO MODEL TRAINING / NO FUSION RETUNING")
    print("=" * 86)

    oof_files = find_oof_files()
    if not oof_files:
        raise RuntimeError(
            "Could not find training-only OOF probabilities. Expected at least:\n"
            + str(PROJECT_ROOT / "results_fusion_pairs" / "all_training_oof_probabilities_for_fusion.csv")
        )

    derivation, repeat_table = derive_thresholds(oof_files)
    repeat_table.to_csv(OUT_DIR / "component_thresholds_by_oof_repeat.csv", index=False)

    svm_th = derivation["thresholds"]["Calibrated RBF-SVM"]
    cnn_th = derivation["thresholds"]["1D CNN"]

    fixed_path = find_first_existing([
        PROJECT_ROOT / "results_frozen_verification" / "fixed_test_predictions.csv",
        PROJECT_ROOT / "results_audit_negative_sources" / "fixed_predictions_with_source_audit.csv",
    ])
    if fixed_path is None:
        # project-wide fallback
        candidates = list(PROJECT_ROOT.rglob("fixed_test_predictions.csv"))
        if candidates:
            fixed_path = candidates[0]
        else:
            raise RuntimeError("Could not locate the frozen fixed-verification prediction file.")

    fixed = pd.read_csv(fixed_path)
    y_col = detect_col(fixed, ["true_label", "label_id", "y_true"], [["true","label"]])
    svm_col = detect_col(fixed, ["svm_probability"], [["svm","prob"]])
    cnn_col = detect_col(fixed, ["cnn1d_probability"], [["cnn1d","prob"]])
    fusion_col = detect_col(fixed, ["fusion_probability"], [["fusion","prob"]])

    y = fixed[y_col].to_numpy(int)
    p_svm = fixed[svm_col].to_numpy(float)
    p_cnn = fixed[cnn_col].to_numpy(float)
    p_fusion = fixed[fusion_col].to_numpy(float)

    methods = [
        ("Calibrated RBF-SVM default 0.500", p_svm, 0.500),
        ("Calibrated RBF-SVM OOF-tuned", p_svm, svm_th),
        ("1D CNN default 0.500", p_cnn, 0.500),
        ("1D CNN OOF-tuned", p_cnn, cnn_th),
        ("Frozen fusion 0.55/0.45, tau=0.520", p_fusion, FUSION_THRESHOLD),
    ]

    rows = []
    pred_cols = {}
    audit = fixed.copy()
    audit[y_col] = y

    for name, p, th in methods:
        m = metrics(y, p, th)
        rows.append({"method": name, **m})
        pred = (p >= th).astype(int)
        key = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        audit[key + "_prediction"] = pred
        pred_cols[name] = key + "_prediction"

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "fixed_verification_threshold_fairness_summary.csv", index=False)

    comparisons = []
    for baseline in ["Calibrated RBF-SVM OOF-tuned", "1D CNN OOF-tuned"]:
        pa = audit[pred_cols["Frozen fusion 0.55/0.45, tau=0.520"]].to_numpy(int)
        pb = audit[pred_cols[baseline]].to_numpy(int)
        mc = exact_mcnemar(y, pa, pb)
        comparisons.append({"comparison": f"Frozen fusion vs {baseline}", **mc})
    pd.DataFrame(comparisons).to_csv(OUT_DIR / "mcnemar_fusion_vs_tuned_components.csv", index=False)

    group_col = None
    for c in fixed.columns:
        if norm(c) in {"groupid", "group"}:
            group_col = c
            break
    if group_col:
        boot_all = []
        fusion_pred_col = pred_cols["Frozen fusion 0.55/0.45, tau=0.520"]
        for baseline in ["Calibrated RBF-SVM OOF-tuned", "1D CNN OOF-tuned"]:
            bcol = pred_cols[baseline]
            b = cluster_bootstrap_delta(
                audit, y_col, fusion_pred_col, bcol, group_col, BOOTSTRAP_REPS
            )
            b.insert(0, "comparison", f"Frozen fusion minus {baseline}")
            boot_all.append(b)
        pd.concat(boot_all, ignore_index=True).to_csv(
            OUT_DIR / "group_bootstrap_fusion_minus_tuned_components.csv", index=False
        )

    audit.to_csv(OUT_DIR / "fixed_predictions_threshold_fairness_audit.csv", index=False)

    protocol = {
        "script_version": SCRIPT_VERSION,
        "analysis_type": "post-hoc threshold-fairness sensitivity; no model training; no fusion retuning",
        "oof_threshold_derivation": derivation,
        "fusion_remains_frozen": {
            "svm_weight": FUSION_SVM_WEIGHT,
            "cnn1d_weight": FUSION_CNN_WEIGHT,
            "threshold": FUSION_THRESHOLD,
        },
        "threshold_grid": {"min": 0.05, "max": 0.95, "step": 0.005},
        "selection_rule": "F1 -> Recall -> Specificity -> ROC-AUC -> AP",
        "fixed_prediction_file": str(fixed_path),
        "important_interpretation": (
            "This audit asks whether the frozen fusion remains competitive after its component experts are "
            "given training-only threshold optimization. It does not change the final fusion or claim that "
            "the fixed verification partition is a never-seen blind test set."
        ),
    }
    (OUT_DIR / "analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nThreshold derivation mode:", derivation["mode"])
    print(f"Calibrated RBF-SVM threshold: {svm_th:.3f}")
    print(f"1D CNN threshold            : {cnn_th:.3f}")
    print("\nFixed-verification comparison:")
    print(summary[["method","threshold","accuracy","precision","recall","f1","specificity","fp","fn"]]
          .to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nOutputs:", OUT_DIR)


if __name__ == "__main__":
    main()
