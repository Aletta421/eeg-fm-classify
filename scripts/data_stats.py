"""
处理后 EEG 数据统计报告

用法:
    python scripts/data_stats.py
    python scripts/data_stats.py --data_dir data/processed --detail
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import os

SEP = "=" * 72
SUB = "-" * 72


def get_size_str(size_bytes: float) -> str:
    """Human-readable file size."""
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024 ** 2)
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def collect_stats(data_dir: str) -> list:
    """Collect per-dataset statistics."""
    processed = Path(data_dir)
    datasets = []

    for ds_dir in sorted(processed.iterdir()):
        if not ds_dir.is_dir():
            continue

        ds_name = ds_dir.name
        unique_subs = set()
        labels = Counter()
        conditions = Counter()
        shapes = Counter()
        original_fs = Counter()
        total_size = 0
        ch_counts = Counter()
        epoch_counts = Counter()

        for mf in sorted(ds_dir.rglob("*_meta.json")):
            try:
                meta = json.loads(mf.read_text())
            except Exception:
                continue

            sid = meta.get("subject_id", "?")
            unique_subs.add(sid)
            labels[meta.get("diagnosis_type", "?")] += 1

            # Best-effort condition/task extraction
            cond = (
                meta.get("condition")
                or meta.get("task")
                or meta.get("subtype")
                or meta.get("format", "?")
            )
            conditions[cond] += 1

            orig_fs = meta.get("original_fs", "?")
            if isinstance(orig_fs, (int, float)):
                original_fs[f"{orig_fs} Hz"] += 1
            else:
                original_fs[str(orig_fs)] += 1

            # Check .npy file
            npy_f = mf.parent / f"{sid}_eeg.npy"
            if npy_f.exists():
                total_size += npy_f.stat().st_size
                import numpy as np
                shape = np.load(str(npy_f)).shape
                shapes[shape] += 1
                epoch_counts[shape[0]] += 1
                ch_counts[shape[1]] += 1

        datasets.append({
            "name": ds_name,
            "n_subjects": len(unique_subs),
            "n_files": sum(shapes.values()),
            "total_size_bytes": total_size,
            "original_fs": dict(original_fs),
            "actual_fs_hz": 200,  # All resampled to 200 Hz
            "labels": dict(labels),
            "conditions": dict(conditions),
            "shapes": dict(shapes),
            "epoch_range": (min(epoch_counts.keys()) if epoch_counts else 0,
                           max(epoch_counts.keys()) if epoch_counts else 0),
            "ch_range": (min(ch_counts.keys()) if ch_counts else 0,
                        max(ch_counts.keys()) if ch_counts else 0),
        })

    return datasets


def print_summary(datasets: list):
    """Print summary table."""
    print(f"\n{SEP}")
    print("  EEG 处理后数据统计")
    print(f"{SEP}")

    # ---- Table 1: Overview ----
    print(f"\n{'数据集':<16} {'受试者':>6} {'文件':>6} {'磁盘占用':>10} "
          f"{'原始采样率':>14} {'处理后采样率':>12}")
    print(SUB)

    total_subs = 0
    total_files = 0
    total_bytes = 0

    for ds in datasets:
        total_subs += ds["n_subjects"]
        total_files += ds["n_files"]
        total_bytes += ds["total_size_bytes"]
        orig_fs = ", ".join(ds["original_fs"].keys())
        print(f"{ds['name']:<16} {ds['n_subjects']:>6} {ds['n_files']:>6} "
              f"{get_size_str(ds['total_size_bytes']):>10} {orig_fs:>14} "
              f"{ds['actual_fs_hz']} Hz (统一)")

    print(SUB)
    print(f"{'总计':<16} {total_subs:>6} {total_files:>6} "
          f"{get_size_str(total_bytes):>10}")
    print(f"\n  注: 处理后所有数据统一为 200 Hz, 10 秒窗口 (2000 采样点)")

    # ---- Table 2: Labels ----
    print(f"\n{'数据集':<16} {'标签分布'}")
    print(SUB)
    for ds in datasets:
        label_str = ", ".join(f"{k}={v}" for k, v in sorted(ds["labels"].items()))
        print(f"{ds['name']:<16} {label_str}")

    # ---- Table 3: Detail ----
    print(f"\n{'数据集':<16} {'Epochs':>8} {'通道数':>8} {'Shape示例'}")
    print(SUB)
    for ds in datasets:
        ep_range = f"{ds['epoch_range'][0]}-{ds['epoch_range'][1]}"
        ch_range = f"{ds['ch_range'][0]}-{ds['ch_range'][1]}"
        shape_examples = ", ".join(
            f"{s}={c}" for s, c in sorted(ds["shapes"].items())[:3]
        )
        print(f"{ds['name']:<16} {ep_range:>8} {ch_range:>8} {shape_examples}")


def print_detailed(datasets: list):
    """Print per-dataset detailed breakdown."""
    for ds in datasets:
        print(f"\n{SEP}")
        print(f"  {ds['name']}")
        print(f"{SEP}")
        print(f"  受试者:     {ds['n_subjects']}")
        print(f"  文件:       {ds['n_files']}")
        print(f"  磁盘占用:   {get_size_str(ds['total_size_bytes'])}")
        print(f"  原始采样率: {', '.join(ds['original_fs'].keys())}")
        print(f"  处理后:     200 Hz, 10s window (2000 samples)")
        print(f"\n  标签分布:")
        for label, count in sorted(ds["labels"].items()):
            print(f"    {label:<20s} {count:>6d}")
        print(f"\n  条件/任务:")
        for cond, count in sorted(ds["conditions"].items()):
            print(f"    {cond:<40s} {count:>6d}")
        if ds["shapes"]:
            print(f"\n  Shape 分布:")
            for shape, count in sorted(ds["shapes"].items()):
                print(f"    {str(shape):<25s} {count:>6d}")


def main():
    parser = argparse.ArgumentParser(description="处理后 EEG 数据统计")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="处理后数据目录")
    parser.add_argument("--detail", action="store_true",
                        help="显示每个数据集的详细信息")
    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"错误: 目录不存在 {args.data_dir}")
        return

    print(f"数据目录: {os.path.abspath(args.data_dir)}")
    datasets = collect_stats(args.data_dir)
    print_summary(datasets)

    if args.detail:
        print_detailed(datasets)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
