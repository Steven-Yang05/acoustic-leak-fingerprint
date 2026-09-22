from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 只划分训练集和测试集
# ============================================================

SCRIPT_VERSION = "1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "features"

# 使用全新的目录名，避免旧文件残留干扰
OUTPUT_DIR = PROJECT_ROOT / "train_test_data"

X_PATH = FEATURE_DIR / "X.npy"
Y_PATH = FEATURE_DIR / "y.npy"
METADATA_PATH = FEATURE_DIR / "metadata.csv"

RANDOM_STATE = 42
N_SPLITS = 5

# 去掉文件名末尾的切片编号：
# xxx.wav、xxx_1.wav、xxx_2.wav 会归入同一组
SEGMENT_SUFFIX_PATTERN = re.compile(r"(?:_\d+)+$")


def get_base_recording_name(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    return SEGMENT_SUFFIX_PATTERN.sub("", stem)


def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    for path in [X_PATH, Y_PATH, METADATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"找不到必要文件：{path}")

    X = np.load(X_PATH, allow_pickle=False)
    y = np.load(Y_PATH, allow_pickle=False)

    metadata = pd.read_csv(
        METADATA_PATH,
        encoding="utf-8-sig",
    )

    return X, y, metadata


def check_data(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    if X.ndim != 2:
        raise ValueError(f"X 必须是二维矩阵，实际形状：{X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y 必须是一维数组，实际形状：{y.shape}")

    if X.shape[0] != len(y) or len(y) != len(metadata):
        raise ValueError(
            "X、y、metadata 数量不一致："
            f"X={X.shape[0]}，y={len(y)}，metadata={len(metadata)}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("X 中存在 NaN 或 Inf")

    required_columns = {
        "file_path",
        "file_name",
        "source_class",
        "label",
        "label_id",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"metadata.csv 缺少字段：{sorted(missing_columns)}"
        )

    metadata_y = metadata["label_id"].to_numpy(dtype=np.int64)

    if not np.array_equal(metadata_y, y.astype(np.int64)):
        raise ValueError(
            "metadata.csv 中的 label_id 顺序与 y.npy 不一致"
        )

    unique_labels = np.unique(y)

    if not np.array_equal(unique_labels, np.array([0, 1])):
        raise ValueError(
            f"标签必须只有 0 和 1，实际为：{unique_labels.tolist()}"
        )


def add_group_id(metadata: pd.DataFrame) -> pd.DataFrame:
    result = metadata.copy()

    result["base_recording_name"] = (
        result["file_name"]
        .astype(str)
        .map(get_base_recording_name)
    )

    # 加 source_class，避免不同目录里的同名文件被误合并
    result["group_id"] = (
        result["source_class"].astype(str)
        + "::"
        + result["base_recording_name"].astype(str)
    )

    group_label_count = (
        result.groupby("group_id")["label_id"].nunique()
    )

    mixed_groups = group_label_count[group_label_count > 1]

    if not mixed_groups.empty:
        error_path = PROJECT_ROOT / "mixed_label_groups.csv"

        result[
            result["group_id"].isin(mixed_groups.index)
        ].to_csv(
            error_path,
            index=False,
            encoding="utf-8-sig",
        )

        raise ValueError(
            "同一个原始录音组中出现多个标签。"
            f"详情已保存到：{error_path}"
        )

    return result


def choose_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    groups_per_label = []

    for label_id in np.unique(y):
        groups_per_label.append(
            len(np.unique(groups[y == label_id]))
        )

    actual_splits = min(
        N_SPLITS,
        min(groups_per_label),
    )

    if actual_splits < 2:
        raise ValueError(
            "每个标签至少需要两个独立原始录音组。"
            f"当前各类组数：{groups_per_label}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    target_test_fraction = 1.0 / actual_splits
    overall_leak_fraction = float(np.mean(y == 1))

    candidates = []

    for fold_number, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        test_fraction = len(test_index) / len(y)
        test_leak_fraction = float(np.mean(y[test_index] == 1))

        score = (
            abs(test_fraction - target_test_fraction)
            + abs(test_leak_fraction - overall_leak_fraction)
        )

        candidates.append(
            (
                score,
                fold_number,
                np.asarray(train_index, dtype=np.int64),
                np.asarray(test_index, dtype=np.int64),
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1]))

    _, fold_number, train_index, test_index = candidates[0]

    return train_index, test_index, fold_number


def check_isolation(
    train_index: np.ndarray,
    test_index: np.ndarray,
    groups: np.ndarray,
    sample_count: int,
) -> None:
    train_groups = set(groups[train_index].tolist())
    test_groups = set(groups[test_index].tolist())

    overlap = train_groups & test_groups

    if overlap:
        raise RuntimeError(
            "训练集和测试集存在同源组重叠："
            f"{sorted(overlap)[:10]}"
        )

    combined = np.concatenate([train_index, test_index])

    if len(combined) != sample_count:
        raise RuntimeError("训练集和测试集没有覆盖全部样本")

    if len(np.unique(combined)) != sample_count:
        raise RuntimeError("训练集和测试集之间存在重复样本")


def save_split(
    name: str,
    index: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> None:
    np.save(
        OUTPUT_DIR / f"X_{name}.npy",
        X[index].astype(np.float32),
    )

    np.save(
        OUTPUT_DIR / f"y_{name}.npy",
        y[index].astype(np.int64),
    )

    np.save(
        OUTPUT_DIR / f"{name}_indices.npy",
        index.astype(np.int64),
    )

    (
        metadata.iloc[index]
        .copy()
        .reset_index(drop=True)
        .to_csv(
            OUTPUT_DIR / f"{name}_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )


def main() -> None:
    print("=" * 76)
    print("只生成训练集和测试集")
    print(f"版本：{SCRIPT_VERSION}")
    print("=" * 76)

    X, y, metadata = load_data()

    check_data(X, y, metadata)

    metadata = add_group_id(metadata)

    groups = metadata["group_id"].astype(str).to_numpy()

    train_index, test_index, selected_fold = choose_split(
        X,
        y,
        groups,
    )

    check_isolation(
        train_index,
        test_index,
        groups,
        len(y),
    )

    # 每次运行都彻底重建全新目录，杜绝旧文件残留
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    save_split(
        "train",
        train_index,
        X,
        y,
        metadata,
    )

    save_split(
        "test",
        test_index,
        X,
        y,
        metadata,
    )

    split_name = np.empty(len(y), dtype=object)
    split_name[train_index] = "train"
    split_name[test_index] = "test"

    manifest = metadata.copy()

    manifest.insert(
        0,
        "original_row_index",
        np.arange(len(metadata), dtype=np.int64),
    )

    manifest["split"] = split_name

    manifest.to_csv(
        OUTPUT_DIR / "split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []

    for name, index in [
        ("train", train_index),
        ("test", test_index),
    ]:
        split_y = y[index]
        split_groups = groups[index]

        rows.append(
            {
                "split": name,
                "sample_count": int(len(index)),
                "sample_fraction": float(len(index) / len(y)),
                "group_count": int(len(np.unique(split_groups))),
                "no_leak_count": int(np.sum(split_y == 0)),
                "leak_count": int(np.sum(split_y == 1)),
                "leak_fraction": float(np.mean(split_y == 1)),
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"原始特征矩阵：{X.shape}")
    print(f"原始录音组总数：{len(np.unique(groups))}")
    print(f"选中的测试折：{selected_fold}")

    print()
    print(summary.to_string(index=False))

    print()
    print("隔离检查：")
    print("  训练集与测试集重叠组数：0")

    print()
    print("实际生成的 NPY 文件：")

    for path in sorted(OUTPUT_DIR.glob("*.npy")):
        print(f"  {path.name}")

    expected_npy_files = {
        "X_train.npy",
        "y_train.npy",
        "train_indices.npy",
        "X_test.npy",
        "y_test.npy",
        "test_indices.npy",
    }

    actual_npy_files = {
        path.name
        for path in OUTPUT_DIR.glob("*.npy")
    }

    if actual_npy_files != expected_npy_files:
        raise RuntimeError(
            "输出文件集合异常："
            f"{sorted(actual_npy_files)}"
        )

    print()
    print("确认：只存在 train 和 test。")


if __name__ == "__main__":
    main()
