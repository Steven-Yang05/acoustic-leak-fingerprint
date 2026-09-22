from collections import Counter
from pathlib import Path

import soundfile as sf


DATASET_DIR = Path(__file__).resolve().parents[1] / "data"

FOLDERS = {
    "leak": DATASET_DIR / "leak",
    "no_leak": DATASET_DIR / "no_leak",
    "noise": DATASET_DIR / "noise",
}


def find_wav_files(folder: Path) -> list[Path]:
    """递归查找所有 WAV 文件，兼容 .wav 和 .WAV。"""
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]


def inspect_folder(category: str, folder: Path) -> None:
    print("\n" + "=" * 60)
    print(f"类别：{category}")
    print(f"目录：{folder}")

    if not folder.exists():
        print("状态：目录不存在")
        return

    wav_files = find_wav_files(folder)
    print(f"WAV 文件数量：{len(wav_files)}")

    if not wav_files:
        print("没有发现 WAV 文件。")
        return

    sample_rates = Counter()
    channel_counts = Counter()
    subtypes = Counter()
    durations = []
    failed_files = []

    for wav_path in wav_files:
        try:
            info = sf.info(wav_path)

            sample_rates[info.samplerate] += 1
            channel_counts[info.channels] += 1
            subtypes[info.subtype] += 1
            durations.append(info.duration)

        except Exception as exc:
            failed_files.append((wav_path, str(exc)))

    print(f"成功读取数量：{len(durations)}")
    print(f"读取失败数量：{len(failed_files)}")
    print(f"采样率分布：{dict(sample_rates)}")
    print(f"声道数分布：{dict(channel_counts)}")
    print(f"音频格式分布：{dict(subtypes)}")

    if durations:
        total_seconds = sum(durations)

        print(f"最短时长：{min(durations):.3f} 秒")
        print(f"最长时长：{max(durations):.3f} 秒")
        print(f"平均时长：{total_seconds / len(durations):.3f} 秒")
        print(f"总时长：{total_seconds / 3600:.3f} 小时")

    if failed_files:
        print("\n前 10 个读取失败的文件：")

        for path, error in failed_files[:10]:
            print(f"- {path}")
            print(f"  原因：{error}")


def main() -> None:
    print(f"数据集根目录：{DATASET_DIR}")

    for category, folder in FOLDERS.items():
        inspect_folder(category, folder)


if __name__ == "__main__":
    main()
