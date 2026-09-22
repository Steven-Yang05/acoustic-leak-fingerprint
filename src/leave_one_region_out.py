from __future__ import annotations

"""
leave_one_region_out.py: leave-one-region-out (LORO) generalization audit.

A stronger external-validity probe than leave-one-condition-out (LOCO):
an entire acquisition zone is held out. Zone membership bundles several
shift factors at once (recording location/session, background conditions,
and---for zone 1---pvc/steel pipe materials that never appear in zone 2),
so transfer across zones is a closer proxy for deployment on unseen sites
than any single-condition split.

  - zone 2 -> zone 1 : develop on 265 clips (74 groups), test on 66 clips (29 groups)
  - zone 1 -> zone 2 : develop on 66 clips (29 groups), test on 265 clips (74 groups)

Protocol is identical to leave_one_condition_out.py: all development
(group-aware OOF, nested Platt calibration, fusion weight / threshold
search) on the source zone only, then one frozen evaluation on the unseen
target zone.

Only leak and no_leak clips with a zone label are used; NA-region clips
and environmental-noise clips are excluded.

Outputs: results_loro/
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_VERSION = "leave-one-region-out-v1"


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_regions(metadata: pd.DataFrame) -> pd.DataFrame:
    """Parse acquisition zone from filenames (field 2 of the label rule).

    leak (7 fields)   : material-area-pressure-flow-device-X-Y
    no_leak (6 fields): material-area-pressure-flow-device_N-Y
    noise             : excluded
    """
    regions: list[str] = []
    for _, row in metadata.iterrows():
        stem = Path(str(row["file_name"])).stem
        source_class = str(row["source_class"])
        parts = stem.split("-")
        region = "NA"
        if source_class in ("leak", "no_leak") and len(parts) >= 2:
            region = parts[1]
        regions.append(region)

    out = metadata.copy()
    out["cond_region"] = regions
    return out


def main() -> None:
    t0 = time.time()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results_loro"
    out_dir.mkdir(exist_ok=True)

    exp11 = load_module(root / "src" / "model_selection_oof.py", "exp11_loro")
    exp14 = load_module(root / "src" / "frozen_verification.py", "exp14_loro")
    exp20 = load_module(root / "src" / "leave_one_condition_out.py", "exp20_loro")

    X_all = np.load(root / "features" / "X.npy", allow_pickle=False)
    y_all = np.load(root / "features" / "y.npy", allow_pickle=False).astype(np.int64)
    metadata = pd.read_csv(root / "features" / "metadata.csv", encoding="utf-8-sig")
    cond = parse_regions(metadata)
    cond.to_csv(out_dir / "parsed_regions.csv", index=False)
    groups_all = exp11.derive_group_ids(metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tasks = [
        ("region_zone2_to_zone1", "zone 2", "zone 1"),
        ("region_zone1_to_zone2", "zone 1", "zone 2"),
    ]

    all_rows: list[dict[str, Any]] = []
    for task_name, source_level, target_level in tasks:
        all_rows.extend(exp20.run_task(
            exp11, exp14, task_name, "cond_region", source_level, target_level,
            cond, X_all, y_all, groups_all, root, device,
        ))

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "loro_results.csv", index=False)

    # move the per-task prediction CSVs that run_task wrote into the LOCO dir
    pred_dir_loco = root / "results_loco"
    for task_name, _, _ in tasks:
        src = pred_dir_loco / f"target_predictions_{task_name}.csv"
        if src.exists():
            src.replace(out_dir / src.name)

    protocol = {
        "script_version": SCRIPT_VERSION,
        "seed": exp20.SEED,
        "folds": exp20.FOLDS,
        "epochs": exp20.EPOCHS,
        "classes_used": ["leak", "no_leak"],
        "na_region_clips_excluded": True,
        "noise_clips_excluded": True,
        "variable": "cond_region (filename field 2: acquisition zone)",
        "tasks": [t[0] for t in tasks],
        "selection": "same as leave_one_condition_out.py: source-zone 5-fold StratifiedGroupKFold OOF "
                     "(seed 42); nested group-aware Platt calibration for the SVM; "
                     "fusion weight/threshold and component thresholds from source "
                     "OOF only, lexicographic F1->Recall->Spec->AUC->AP",
        "target_use": "single frozen evaluation per direction; no target feedback",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "loro_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print()
    print(df.to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {out_dir}")


if __name__ == "__main__":
    main()
