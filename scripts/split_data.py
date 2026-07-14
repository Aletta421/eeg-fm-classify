"""
确定并保存 Train/Test Split（按受试者分层）

在 Step 5 (z-score + clip) 之后、Step 7 (滑窗) 之前运行。
扫描 data/processed/ 下所有 *_meta.json，按 label 分层划分，
将 split 信息持久化到 data/splits.json。

用法:
    python scripts/split_data.py
    python scripts/split_data.py --data_dir data/processed --output data/splits.json
    python scripts/split_data.py --test_split 0.2 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np

# Fix Windows GBK encoding issue
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def collect_subjects(data_dir: Path) -> List[Dict]:
    """扫描所有 *_meta.json，收集受试者信息。

    Returns:
        [{subject_id, dataset, source_dir, label, diagnosis_type}, ...]
        source_dir 是 data_dir 下的第一级子目录名，用于标识数据来源。
    """
    subjects = []
    seen = set()  # 去重（同一 subject 可能有多个 meta 文件）

    for meta_file in sorted(data_dir.rglob("*_meta.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠ 跳过损坏的 meta 文件: {meta_file} ({e})", file=sys.stderr)
            continue

        subject_id = meta.get("subject_id", "")
        if not subject_id:
            continue

        # 去重：同一 subject_id 只保留第一次出现的记录
        if subject_id in seen:
            continue
        seen.add(subject_id)

        # 获取 data_dir 下的第一级子目录名（即数据来源目录）
        try:
            source_dir = meta_file.relative_to(data_dir).parts[0]
        except (ValueError, IndexError):
            source_dir = "unknown"

        subjects.append({
            "subject_id": subject_id,
            "dataset": meta.get("dataset", "unknown"),
            "source_dir": source_dir,
            "label": meta.get("label", -1),
            "diagnosis_type": meta.get("diagnosis_type", "unknown"),
        })

    return subjects


def stratified_split(
    subjects: List[Dict],
    holdout_dataset: str = "testdata",
    test_split: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """按受试者分层划分 train/test。

    Args:
        subjects: 受试者列表。
        holdout_dataset: 始终作为独立测试集的数据集名称（不算在 train/test split 中）。
        test_split: 测试集比例（从非 holdout 数据中划分）。
        seed: 随机种子。

    Returns:
        {"train": [...], "test": [...], "holdout": [...]}
    """
    rng = np.random.RandomState(seed)

    # 分离 holdout（通过目录名识别，而非 meta 中的 dataset 字段）
    holdout_subjects = [s for s in subjects if s["source_dir"] == holdout_dataset]
    trainval_subjects = [s for s in subjects if s["source_dir"] != holdout_dataset]

    # 按 label 分组（分层 split）
    label_groups: Dict[int, List[Dict]] = defaultdict(list)
    for s in trainval_subjects:
        label_groups[s["label"]].append(s)

    train_ids: List[str] = []
    test_ids: List[str] = []

    for label, group in label_groups.items():
        ids = [s["subject_id"] for s in group]
        rng.shuffle(ids)
        n_test = max(1, int(len(ids) * test_split))
        test_ids.extend(ids[:n_test])
        train_ids.extend(ids[n_test:])

    holdout_ids = [s["subject_id"] for s in holdout_subjects]

    # 统计
    print(f"\n  分层 Split 结果 (seed={seed}, test_split={test_split}):")
    print(f"    Train:   {len(train_ids)} subjects")
    print(f"    Test:    {len(test_ids)} subjects")
    print(f"    Holdout: {len(holdout_ids)} subjects (holdout={holdout_dataset})")
    print(f"    总计:    {len(train_ids) + len(test_ids) + len(holdout_ids)} subjects")

    # 按数据集统计
    all_ids = train_ids + test_ids + holdout_ids
    id_to_subj = {s["subject_id"]: s for s in subjects}
    dataset_counts = defaultdict(lambda: {"train": 0, "test": 0, "holdout": 0})
    for sid in train_ids:
        ds = id_to_subj.get(sid, {}).get("source_dir", "?")
        dataset_counts[ds]["train"] += 1
    for sid in test_ids:
        ds = id_to_subj.get(sid, {}).get("source_dir", "?")
        dataset_counts[ds]["test"] += 1
    for sid in holdout_ids:
        ds = id_to_subj.get(sid, {}).get("source_dir", "?")
        dataset_counts[ds]["holdout"] += 1

    print(f"\n  按数据集分布:")
    for ds, counts in sorted(dataset_counts.items()):
        print(f"    {ds:<20s} train={counts['train']:>4d}  test={counts['test']:>4d}  holdout={counts['holdout']:>4d}")

    return {
        "train": sorted(train_ids),
        "test": sorted(test_ids),
        "holdout": sorted(holdout_ids),
    }


def main():
    parser = argparse.ArgumentParser(
        description="确定并保存 Train/Test Split（按受试者分层）"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="预处理后数据目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits.json",
        help="输出 split 文件路径",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="测试集比例 (默认: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--holdout_dataset",
        type=str,
        default="testdata",
        help="始终作为独立测试集的数据集名称 (默认: testdata)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    if not data_dir.exists():
        print(f"❌ 目录不存在: {data_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Train/Test Split")
    print(f"  数据目录: {data_dir}")
    print(f"{'='*60}")

    # 收集受试者
    subjects = collect_subjects(data_dir)
    if not subjects:
        print("❌ 未找到任何受试者数据，请先运行预处理")
        sys.exit(1)

    print(f"  扫描到 {len(subjects)} 个唯一受试者")

    # 分层 split
    split = stratified_split(
        subjects,
        holdout_dataset=args.holdout_dataset,
        test_split=args.test_split,
        seed=args.seed,
    )

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        **split,
        "seed": args.seed,
        "test_split": args.test_split,
        "holdout_dataset": args.holdout_dataset,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Split 已保存: {output_path.resolve()}")


if __name__ == "__main__":
    main()
