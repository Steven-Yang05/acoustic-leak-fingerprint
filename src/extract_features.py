from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd


# ============================================================
# 路径配置
# ============================================================

# 当前脚本位置：
# <repo>/src/extract_features.py
#
# 项目根目录：
# <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据目录：
# <repo>/data
DATA_ROOT = PROJECT_ROOT / "data"

# 特征输出目录：
# <repo>/features
OUTPUT_ROOT = PROJECT_ROOT / "features"


# ============================================================
# 二分类标签配置
# ============================================================

# leak：泄漏，标签为 1
# no_leak：不泄漏，标签为 0
# noise：环境噪声，也归入不泄漏，标签为 0
CLASS_TO_LABEL = {
    "leak": 1,
    "no_leak": 0,
    "noise": 0,
}

LABEL_NAMES = {
    0: "no_leak",
    1: "leak",
}


# ============================================================
# 音频和特征参数
# ============================================================

EXPECTED_SAMPLE_RATE = 8000

# 8000 Hz 采样率下：
# 256 个采样点约为 32 ms
# 80 个采样点约为 10 ms
N_FFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 80

# Mel 频带数量
N_MELS = 40

# MFCC 系数数量
N_MFCC = 20

# 频率范围
FMIN = 20.0
FMAX = 4000.0


def summarize_feature(
    feature: np.ndarray,
    prefix: str,
) -> Tuple[List[float], List[str]]:
    """
    将时序特征转换为固定长度向量。

    对每个特征通道计算：
    1. 时间均值
    2. 时间标准差
    """
    feature = np.asarray(feature, dtype=np.float32)

    if feature.ndim == 1:
        feature = feature.reshape(1, -1)

    values: List[float] = []
    names: List[str] = []

    channel_count = feature.shape[0]

    for channel_index in range(channel_count):
        channel = feature[channel_index]

        if channel_count == 1:
            base_name = prefix
        else:
            base_name = f"{prefix}_{channel_index + 1:02d}"

        mean_value = float(np.mean(channel))
        std_value = float(np.std(channel))

        values.extend(
            [
                mean_value,
                std_value,
            ]
        )

        names.extend(
            [
                f"{base_name}_mean",
                f"{base_name}_std",
            ]
        )

    return values, names


def calculate_delta(
    feature: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    计算 MFCC 的一阶或二阶差分特征。
    """
    frame_count = feature.shape[1]

    if frame_count < 3:
        return np.zeros_like(feature)

    width = min(9, frame_count)

    # librosa 要求 width 为奇数
    if width % 2 == 0:
        width -= 1

    if width < 3:
        return np.zeros_like(feature)

    return librosa.feature.delta(
        feature,
        order=order,
        width=width,
        mode="nearest",
    )


def extract_features(
    wav_path: Path,
) -> Tuple[np.ndarray, List[str]]:
    """
    从一个 WAV 文件中提取固定长度特征向量。

    特征维度：

    MFCC：
        20 × 均值和标准差 = 40

    MFCC 一阶差分：
        20 × 均值和标准差 = 40

    MFCC 二阶差分：
        20 × 均值和标准差 = 40

    Log-Mel：
        40 × 均值和标准差 = 80

    其他特征：
        频谱重心          2
        频谱带宽          2
        频谱滚降点        2
        频谱平坦度        2
        过零率            2
        RMS 能量          2

    最终总维度：
        212
    """

    # sr=None：保留 WAV 原始采样率
    # mono=True：读取为单声道
    y, sample_rate = librosa.load(
        wav_path,
        sr=None,
        mono=True,
        dtype=np.float32,
    )

    if y.size == 0:
        raise ValueError("音频为空")

    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"采样率异常：期望 {EXPECTED_SAMPLE_RATE} Hz，"
            f"实际 {sample_rate} Hz"
        )

    if not np.all(np.isfinite(y)):
        raise ValueError("音频中存在 NaN 或无穷值")

    # 去除直流偏置
    # 不进行峰值归一化，以保留泄漏声音的能量差异
    y = y - np.mean(y)

    # ========================================================
    # 短时傅里叶变换
    # ========================================================

    stft = librosa.stft(
        y=y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=False,
    )

    magnitude = np.abs(stft)
    power = magnitude ** 2

    # ========================================================
    # Log-Mel 频谱
    # ========================================================

    mel_power = librosa.feature.melspectrogram(
        S=power,
        sr=sample_rate,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )

    log_mel = librosa.power_to_db(
        mel_power,
        ref=1.0,
        top_db=80.0,
    )

    # ========================================================
    # MFCC 及其差分
    # ========================================================

    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=N_MFCC,
        dct_type=2,
        norm="ortho",
    )

    mfcc_delta = calculate_delta(
        mfcc,
        order=1,
    )

    mfcc_delta2 = calculate_delta(
        mfcc,
        order=2,
    )

    # ========================================================
    # 频谱特征
    # ========================================================

    spectral_centroid = librosa.feature.spectral_centroid(
        S=magnitude,
        sr=sample_rate,
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude,
        sr=sample_rate,
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sample_rate,
        roll_percent=0.85,
    )

    spectral_flatness = librosa.feature.spectral_flatness(
        S=magnitude,
    )

    # ========================================================
    # 时域特征
    # ========================================================

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )

    # ========================================================
    # 转换为固定长度向量
    # ========================================================

    feature_values: List[float] = []
    feature_names: List[str] = []

    feature_groups = [
        ("mfcc", mfcc),
        ("mfcc_delta", mfcc_delta),
        ("mfcc_delta2", mfcc_delta2),
        ("log_mel", log_mel),
        ("spectral_centroid", spectral_centroid),
        ("spectral_bandwidth", spectral_bandwidth),
        ("spectral_rolloff", spectral_rolloff),
        ("spectral_flatness", spectral_flatness),
        ("zero_crossing_rate", zero_crossing_rate),
        ("rms", rms),
    ]

    for prefix, feature in feature_groups:
        values, names = summarize_feature(
            feature,
            prefix,
        )

        feature_values.extend(values)
        feature_names.extend(names)

    vector = np.asarray(
        feature_values,
        dtype=np.float32,
    )

    vector = np.nan_to_num(
        vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if vector.shape[0] != 212:
        raise ValueError(
            f"特征维度异常：期望 212，实际 {vector.shape[0]}"
        )

    return vector, feature_names


def collect_wav_files(
    class_directory: Path,
) -> List[Path]:
    """
    递归查找目录中的所有 WAV 文件。
    """
    wav_files = [
        path
        for path in class_directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]

    return sorted(wav_files)


def main() -> None:
    print("=" * 70)
    print("WAV 二分类特征提取")
    print("=" * 70)

    print(f"项目目录：{PROJECT_ROOT}")
    print(f"数据目录：{DATA_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"找不到数据目录：{DATA_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_vectors: List[np.ndarray] = []
    metadata_rows = []
    csv_rows = []
    failed_rows = []

    reference_feature_names: List[str] | None = None

    total_discovered = 0
    total_success = 0

    # ========================================================
    # 遍历 leak、no_leak、noise 三个原始文件夹
    # ========================================================

    for source_class, label_id in CLASS_TO_LABEL.items():
        class_directory = DATA_ROOT / source_class

        if not class_directory.exists():
            raise FileNotFoundError(
                f"找不到类别目录：{class_directory}"
            )

        wav_files = collect_wav_files(
            class_directory
        )

        total_discovered += len(wav_files)

        label_name = LABEL_NAMES[label_id]

        print()
        print("-" * 70)
        print(f"原始类别：{source_class}")
        print(f"二分类标签：{label_id} ({label_name})")
        print(f"WAV 文件数量：{len(wav_files)}")
        print("-" * 70)

        for index, wav_path in enumerate(
            wav_files,
            start=1,
        ):
            try:
                vector, feature_names = extract_features(
                    wav_path
                )

                if reference_feature_names is None:
                    reference_feature_names = feature_names

                elif feature_names != reference_feature_names:
                    raise ValueError(
                        "不同音频文件生成的特征名称不一致"
                    )

                relative_path = wav_path.relative_to(
                    PROJECT_ROOT
                )

                all_vectors.append(vector)

                metadata_rows.append(
                    {
                        "file_path": str(relative_path),
                        "file_name": wav_path.name,
                        "source_class": source_class,
                        "label": label_name,
                        "label_id": label_id,
                    }
                )

                feature_dictionary = {
                    name: float(value)
                    for name, value in zip(
                        feature_names,
                        vector,
                    )
                }

                csv_rows.append(
                    {
                        "file_path": str(relative_path),
                        "file_name": wav_path.name,
                        "source_class": source_class,
                        "label": label_name,
                        "label_id": label_id,
                        **feature_dictionary,
                    }
                )

                total_success += 1

            except Exception as error:
                failed_rows.append(
                    {
                        "file_path": str(wav_path),
                        "source_class": source_class,
                        "label_id": label_id,
                        "error": str(error),
                    }
                )

            if (
                index % 50 == 0
                or index == len(wav_files)
            ):
                print(
                    f"\r已处理：{index}/{len(wav_files)}",
                    end="",
                    flush=True,
                )

        print()

    if not all_vectors:
        raise RuntimeError(
            "没有成功提取任何音频特征"
        )

    if reference_feature_names is None:
        raise RuntimeError(
            "没有生成特征名称"
        )

    # ========================================================
    # 构建特征矩阵和标签数组
    # ========================================================

    X = np.vstack(
        all_vectors
    ).astype(np.float32)

    y = np.asarray(
        [
            row["label_id"]
            for row in metadata_rows
        ],
        dtype=np.int64,
    )

    # ========================================================
    # 保存 NumPy 文件
    # ========================================================

    np.save(
        OUTPUT_ROOT / "X.npy",
        X,
    )

    np.save(
        OUTPUT_ROOT / "y.npy",
        y,
    )

    # ========================================================
    # 保存完整特征 CSV
    # ========================================================

    feature_dataframe = pd.DataFrame(
        csv_rows
    )

    feature_dataframe.to_csv(
        OUTPUT_ROOT / "features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 保存文件和标签对应关系
    # ========================================================

    metadata_dataframe = pd.DataFrame(
        metadata_rows
    )

    metadata_dataframe.to_csv(
        OUTPUT_ROOT / "metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 保存二分类标签定义
    # ========================================================

    label_dataframe = pd.DataFrame(
        [
            {
                "label_id": label_id,
                "label": label_name,
            }
            for label_id, label_name in LABEL_NAMES.items()
        ]
    )

    label_dataframe.to_csv(
        OUTPUT_ROOT / "label_map.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 保存原始文件夹到二分类标签的映射
    # ========================================================

    source_class_dataframe = pd.DataFrame(
        [
            {
                "source_class": source_class,
                "label_id": label_id,
                "label": LABEL_NAMES[label_id],
            }
            for source_class, label_id in CLASS_TO_LABEL.items()
        ]
    )

    source_class_dataframe.to_csv(
        OUTPUT_ROOT / "source_class_map.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 保存特征列名称
    # ========================================================

    with open(
        OUTPUT_ROOT / "feature_names.txt",
        "w",
        encoding="utf-8",
    ) as file:
        for feature_name in reference_feature_names:
            file.write(feature_name + "\n")

    # ========================================================
    # 保存失败文件记录
    # ========================================================

    failure_path = OUTPUT_ROOT / "failed_files.csv"

    if failed_rows:
        pd.DataFrame(
            failed_rows
        ).to_csv(
            failure_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        failure_path.unlink(
            missing_ok=True
        )

    # ========================================================
    # 输出最终结果
    # ========================================================

    print()
    print("=" * 70)
    print("特征提取完成")
    print("=" * 70)

    print(f"发现文件数：{total_discovered}")
    print(f"成功文件数：{total_success}")
    print(f"失败文件数：{len(failed_rows)}")

    print()
    print(f"特征矩阵 X：{X.shape}")
    print(f"标签数组 y：{y.shape}")
    print(f"特征数据类型：{X.dtype}")
    print(f"标签数据类型：{y.dtype}")

    print()
    print("二分类标签数量：")

    unique_labels, counts = np.unique(
        y,
        return_counts=True,
    )

    for label_id, count in zip(
        unique_labels,
        counts,
    ):
        label_id = int(label_id)
        label_name = LABEL_NAMES[label_id]

        print(
            f"  {label_name} ({label_id})：{count}"
        )

    print()
    print("输出文件：")
    print(f"  {OUTPUT_ROOT / 'X.npy'}")
    print(f"  {OUTPUT_ROOT / 'y.npy'}")
    print(f"  {OUTPUT_ROOT / 'features.csv'}")
    print(f"  {OUTPUT_ROOT / 'metadata.csv'}")
    print(f"  {OUTPUT_ROOT / 'label_map.csv'}")
    print(f"  {OUTPUT_ROOT / 'source_class_map.csv'}")
    print(f"  {OUTPUT_ROOT / 'feature_names.txt'}")

    if failed_rows:
        print(f"  {failure_path}")

    print()
    print("前 10 个特征名称：")

    for name in reference_feature_names[:10]:
        print(f"  {name}")

    print()
    print("标签定义：")
    print("  0 = no_leak，不泄漏")
    print("  1 = leak，泄漏")
    print()
    print("原始类别合并规则：")
    print("  leak    -> 1")
    print("  no_leak -> 0")
    print("  noise   -> 0")


if __name__ == "__main__":
    main()
