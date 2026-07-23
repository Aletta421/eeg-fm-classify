"""Run deterministic subject-wise EEG preprocessing.

The subject split is created before signal processing and reused by every
dataset loader. By default, signal processing and epoching run in two stages;
use --single-pass to perform both in one loader invocation.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING = ROOT / "preprocessing"
SPLITS = ROOT / "data" / "splits.csv"

DATASETS = {
    "MODMA": {
        "script": "load_modma.py",
        "description": "MODMA depression",
        "args": [
            "--data_dir", str(ROOT / "data/MODMA"),
            "--output_dir", str(ROOT / "data/processed/MODMA"),
        ],
    },
    "IEEE": {
        "script": "load_ieee.py",
        "description": "IEEE ADHD",
        "args": [
            "--data_dir", str(ROOT / "data/IEEE_ADHD"),
            "--output_dir", str(ROOT / "data/processed/IEEE_ADHD"),
        ],
    },
    "Mendeley": {
        "script": "load_mendeley.py",
        "description": "Mendeley ADHD",
        "args": [
            "--data_dir", str(ROOT / "data/Mendeley_ADHD"),
            "--output_dir", str(ROOT / "data/processed/Mendeley"),
        ],
    },
    "OpenNeuro": {
        "script": "load_openneuro.py",
        "description": "OpenNeuro ds003478 depression",
        "args": [
            "--data_dir", str(ROOT / "data/OpenNeuro_ds003478"),
            "--output_dir", str(ROOT / "data/processed/OpenNeuro"),
        ],
    },
    "TDBRAIN": {
        "script": "load_tdbrain.py",
        "description": "TDBRAIN clinical EEG",
        "args": [
            "--data_dir", str(ROOT / "data/TDBRAIN/TDBRAIN_Dataset_V3_1"),
            "--output_dir", str(ROOT / "data/processed/TDBRAIN"),
            "--xlsx", str(ROOT / "data/TDBRAIN/TDBRAIN_participants_V3.xlsx"),
        ],
    },
}


def run(command, cwd=ROOT, dry_run=False):
    print("\n$ " + " ".join(map(str, command)), flush=True)
    if dry_run:
        return True
    return subprocess.run(command, cwd=cwd).returncode == 0


def run_dataset(name, extra_args=None, dry_run=False):
    info = DATASETS[name]
    command = [
        sys.executable,
        str(PREPROCESSING / info["script"]),
        *info["args"],
        "--split_manifest",
        str(SPLITS),
    ]
    if extra_args:
        command.extend(extra_args)
    print(f"\nProcessing {name}: {info['description']}")
    return run(command, cwd=PREPROCESSING, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", help="comma-separated dataset names")
    parser.add_argument("--skip", help="comma-separated dataset names to skip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--reuse-splits", action="store_true")
    parser.add_argument("--single-pass", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = list(DATASETS) if not args.datasets else [
        name.strip() for name in args.datasets.split(",") if name.strip()
    ]
    if args.skip:
        skipped = {name.strip() for name in args.skip.split(",")}
        names = [name for name in names if name not in skipped]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        parser.error(f"unknown datasets: {unknown}; choose from {list(DATASETS)}")

    if args.reuse_splits:
        if not SPLITS.exists():
            parser.error(f"--reuse-splits requested but {SPLITS} does not exist")
    else:
        split_command = [
            sys.executable,
            str(ROOT / "scripts/create_splits.py"),
            "--output", str(SPLITS),
            "--seed", str(args.seed),
            "--test-fraction", str(args.test_fraction),
        ]
        if not run(split_command, dry_run=args.dry_run):
            sys.exit(1)

    results = {}
    if args.single_pass:
        for name in names:
            results[name] = run_dataset(name, dry_run=args.dry_run)
    else:
        print("\nStage 1/2: signal processing")
        for name in names:
            results[name] = run_dataset(
                name, extra_args=["--skip_epoching"], dry_run=args.dry_run
            )

        print("\nStage 2/2: epoching")
        for name in names:
            if results[name]:
                results[name] = run_dataset(
                    name, extra_args=["--epoch_only"], dry_run=args.dry_run
                )

    if any(results.values()):
        labels_command = [
            sys.executable,
            str(PREPROCESSING / "generate_labels.py"),
            "--data_dir", str(ROOT / "data/processed"),
            "--output", str(ROOT / "data/labels.csv"),
        ]
        run(labels_command, cwd=PREPROCESSING, dry_run=args.dry_run)

    print("\nPreprocessing summary")
    for name, success in results.items():
        print(f"  {name:12s} {'OK' if success else 'FAILED'}")

    if not args.dry_run and any(results.values()):
        stats_script = ROOT / "scripts/data_stats.py"
        if stats_script.exists():
            run([sys.executable, str(stats_script)])

    if not all(results.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
