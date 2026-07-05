"""
生成统一标签文件 labels.csv

扫描 processed/ 目录下所有 *_meta.json 文件，汇总为一张标签表。
供后续 DataLoader 使用。

使用方式:
    python generate_labels.py --data_dir ../data/processed --output ../data/labels.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd

# Fix Windows GBK encoding issue
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# labels.csv 的列定义
COLUMNS = [
    "subject_id",
    "dataset",
    "label",              # 0=对照, 1=患者
    "file_path",          # 预处理后 .npy 文件路径
    "n_epochs",           # 分段数
    "n_channels",
    "n_samples",          # 每段采样点数
    "original_fs",        # 原始采样率
    "duration_seconds",   # 原始总时长
    "diagnosis_type",     # "depression" / "adhd" / "control"
    "subtype",            # 数据集子类型
    "source_file",        # 原始文件路径
]


def scan_processed(data_dir: Path) -> List[Dict]:
    """扫描所有预处理后的 *_meta.json 文件。

    Args:
        data_dir: processed/ 目录路径。

    Returns:
        按 COLUMNS 组织的字典列表。
    """
    records = []
    meta_files = sorted(data_dir.rglob("*_meta.json"))

    if not meta_files:
        print(f"⚠ 在 {data_dir} 中未找到 *_meta.json 文件")
        return records

    for meta_file in meta_files:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 找到对应的 .npy 文件
        eeg_file = meta_file.parent / f"{meta['subject_id']}_eeg.npy"
        if not eeg_file.exists():
            print(f"⚠ 找不到对应 .npy: {eeg_file}")
            continue

        # 读取 .npy 获取形状信息（仅读 header，不加载全文）
        try:
            arr = __import__("numpy").load(str(eeg_file), mmap_mode="r")
            n_epochs, n_channels, n_samples = arr.shape
        except Exception:
            n_epochs, n_channels, n_samples = 0, meta.get("n_channels", 0), 0

        record = {
            "subject_id": meta["subject_id"],
            "dataset": meta.get("dataset", "unknown"),
            "label": meta.get("label", -1),
            "file_path": str(eeg_file.relative_to(data_dir.parent).as_posix()),
            "n_epochs": n_epochs,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "original_fs": meta.get("original_fs", 0),
            "duration_seconds": meta.get("duration_seconds", 0),
            "diagnosis_type": meta.get("diagnosis_type", "unknown"),
            "subtype": meta.get("subtype", ""),
            "source_file": meta.get("source_file", ""),
        }
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="生成统一标签文件 labels.csv"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/processed",
        help="预处理后数据目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/labels.csv",
        help="输出 CSV 文件路径",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    print(f"扫描 {data_dir} ...")
    records = scan_processed(data_dir)

    if not records:
        print("未找到任何记录，请先运行数据集 loader")
        return

    df = pd.DataFrame(records, columns=COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 打印统计信息
    print(f"\n✓ 标签文件已保存: {output_path}")
    print(f"  总记录数: {len(df)}")
    print(f"  数据集分布:")
    for ds, count in df["dataset"].value_counts().items():
        print(f"    {ds}: {count}")
    print(f"  标签分布:")
    for label, count in df["label"].value_counts().sort_index().items():
        label_name = {0: "对照", 1: "患者"}.get(label, "未知")
        print(f"    {label} ({label_name}): {count}")


if __name__ == "__main__":
    main()
