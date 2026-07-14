"""
EEG 数据预处理 — 一键运行 (7-Step Pipeline)

Step 1-5: 信号处理 (重采样→带通→陷波→zscore→clip)
Step 6:   确定 80% train / 20% test split (按受试者分层)
Step 7:   滑窗分段 10s (2000 samples)

用法:
    python scripts/run_preprocess.py                  # 两阶段: 信号处理 → split → 分段
    python scripts/run_preprocess.py --single-pass    # 单阶段: 逐数据集完整处理
    python scripts/run_preprocess.py --datasets MODMA,IEEE  # 只处理指定数据集
    python scripts/run_preprocess.py --dry-run         # 仅预览，不处理
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PREPROC = ROOT / "preprocessing"

DATASETS = {
    "MODMA": {
        "script": PREPROC / "load_modma.py",
        "desc": "MODMA — 抑郁症 (128导静息/ERP + 3导静息)",
        "args": [
            "--data_dir", str(ROOT / "data/MODMA"),
            "--output_dir", str(ROOT / "data/processed/MODMA"),
        ],
    },
    "IEEE": {
        "script": PREPROC / "load_ieee.py",
        "desc": "IEEE ADHD — 儿童ADHD (19导)",
        "args": [
            "--data_dir", str(ROOT / "data/IEEE_ADHD"),
            "--output_dir", str(ROOT / "data/processed/IEEE_ADHD"),
        ],
    },
    "Mendeley": {
        "script": PREPROC / "load_mendeley.py",
        "desc": "Mendeley — 成人ADHD (5通道, 11任务)",
        "args": [
            "--data_dir", str(ROOT / "data/Mendeley_ADHD"),
            "--output_dir", str(ROOT / "data/processed/Mendeley"),
        ],
    },
    "OpenNeuro": {
        "script": PREPROC / "load_openneuro.py",
        "desc": "OpenNeuro ds003478 — 抑郁症 (64导静息)",
        "args": [
            "--data_dir", str(ROOT / "data/OpenNeuro_ds003478"),
            "--output_dir", str(ROOT / "data/processed/OpenNeuro"),
        ],
    },
    "TDBRAIN": {
        "script": PREPROC / "load_tdbrain.py",
        "desc": "TDBRAIN V3 — 多病种临床数据 (26导)",
        "args": [
            "--data_dir", str(ROOT / "data/TDBRAIN/TDBRAIN_Dataset_V3_1"),
            "--output_dir", str(ROOT / "data/processed/TDBRAIN"),
            "--xlsx", str(ROOT / "data/TDBRAIN/TDBRAIN_participants_V3.xlsx"),
        ],
    },
}


def run_dataset(name: str, info: dict, extra_args: list = None,
                dry_run: bool = False) -> bool:
    """Run one dataset's preprocessing script. Returns True on success."""
    script = info["script"]
    if not script.exists():
        print(f"  [跳过] 脚本不存在: {script}")
        return False

    cmd = [sys.executable, str(script)] + info["args"]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  处理: {name} — {info['desc']}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry-run, 跳过)\n")
        return True

    result = subprocess.run(cmd, cwd=str(PREPROC))
    if result.returncode != 0:
        print(f"  [失败] {name} 返回码: {result.returncode}\n")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="EEG 数据一键预处理 (7-Step)")
    parser.add_argument("--datasets", type=str,
                        help="指定处理的数据集 (逗号分隔, 如 MODMA,IEEE)")
    parser.add_argument("--skip", type=str,
                        help="跳过的数据集 (逗号分隔, 如 Mendeley,TDBRAIN)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览不执行")
    parser.add_argument("--single-pass", action="store_true",
                        help="单阶段模式: 每个数据集独立完成全部步骤 (旧行为)")
    args = parser.parse_args()

    # 确定要处理的数据集列表
    if args.datasets:
        names = [n.strip() for n in args.datasets.split(",")]
    else:
        names = list(DATASETS.keys())

    if args.skip:
        skip_set = {n.strip() for n in args.skip.split(",")}
        names = [n for n in names if n not in skip_set]

    # 验证
    invalid = [n for n in names if n not in DATASETS]
    if invalid:
        print(f"未知数据集: {invalid}")
        print(f"可选: {list(DATASETS.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  EEG 数据预处理 (7-Step Pipeline)")
    print(f"  数据集: {', '.join(names)}")
    if args.dry_run:
        print(f"  模式: dry-run (仅预览)")
    elif args.single_pass:
        print(f"  模式: 单阶段 (逐数据集完整处理)")
    else:
        print(f"  模式: 两阶段 (信号处理 → split → 分段)")
    print(f"{'='*60}")

    if args.single_pass:
        # ---- 单阶段模式 (旧行为): 逐数据集完整处理 ----
        results = {}
        for name in names:
            ok = run_dataset(name, DATASETS[name], dry_run=args.dry_run)
            results[name] = ok
    else:
        # ---- 两阶段模式 (默认): Step 1-5 → Step 6 → Step 7 ----
        results = {}

        # 阶段 1: 信号处理 (Step 1-5), 跳过分段
        print(f"\n{'='*60}")
        print(f"  阶段 1/2: Step 1-5 信号处理 (skip_epoching)")
        print(f"{'='*60}")
        for name in names:
            ok = run_dataset(name, DATASETS[name],
                             extra_args=["--skip_epoching"],
                             dry_run=args.dry_run)
            results[name] = ok

        # 阶段间: 确定 split (Step 6) + 生成标签
        if not args.dry_run and any(results.values()):
            split_script = ROOT / "scripts/split_data.py"
            if split_script.exists():
                print(f"\n{'='*60}")
                print(f"  Step 6: 确定 Train/Test Split")
                print(f"{'='*60}")
                subprocess.run([sys.executable, str(split_script)])

            labels_script = ROOT / "preprocessing/generate_labels.py"
            if labels_script.exists():
                print(f"\n  生成标签文件 (labels.csv)...")
                subprocess.run([sys.executable, str(labels_script)])

        # 阶段 2: 滑窗分段 (Step 7)
        if not args.dry_run and any(results.values()):
            print(f"\n{'='*60}")
            print(f"  阶段 2/2: Step 7 滑窗分段 (epoch_only)")
            print(f"{'='*60}")
            for name in names:
                if results[name]:  # 只处理阶段 1 成功的数据集
                    run_dataset(name, DATASETS[name],
                                extra_args=["--epoch_only"],
                                dry_run=args.dry_run)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  预处理完成")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {status:4s}  {name}")

    # 统计 (仅在非 dry-run 且有成功处理时)
    if not args.dry_run and any(results.values()):
        stats_script = ROOT / "scripts/data_stats.py"
        if stats_script.exists():
            print(f"\n  正在生成数据统计...")
            subprocess.run([sys.executable, str(stats_script)])


if __name__ == "__main__":
    main()
