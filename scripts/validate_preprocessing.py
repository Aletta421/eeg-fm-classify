"""Validate REVE preprocessing outputs and subject-level split isolation."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read_manifest(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default=str(ROOT / "data/processed"))
    parser.add_argument("--splits", default=str(ROOT / "data/splits.csv"))
    args = parser.parse_args()

    processed = Path(args.processed)
    split_rows = read_manifest(Path(args.splits))
    assignment = {}
    errors = []
    for row in split_rows:
        key = (row["dataset"], row["subject_id"])
        if key in assignment and assignment[key] != row["split"]:
            errors.append(f"split leakage in manifest: {key}")
        assignment[key] = row["split"]

    observed_subjects = defaultdict(set)
    counts = Counter()
    meta_files = sorted(processed.rglob("*_meta.json"))
    for meta_file in meta_files:
        with meta_file.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        subject_id = str(meta["subject_id"])
        dataset = meta["dataset"]
        split = meta.get("split")
        key = (dataset, subject_id)
        expected_split = assignment.get(key)
        if split != expected_split:
            errors.append(f"split mismatch: {meta_file}: {split} != {expected_split}")
        observed_subjects[key].add(split)
        if len(observed_subjects[key]) > 1:
            errors.append(f"subject appears in both output splits: {key}")

        eeg_file = meta_file.parent / f"{subject_id}_eeg.npy"
        pos_file = meta_file.parent / f"{subject_id}_ch_pos.npy"
        if not eeg_file.exists() or not pos_file.exists():
            errors.append(f"incomplete preprocessing output: {meta_file.parent}")
            continue
        eeg = np.load(eeg_file, mmap_mode="r")
        pos = np.load(pos_file)
        if eeg.ndim != 3 or eeg.shape[2] != 2000:
            errors.append(f"invalid EEG shape {eeg.shape}: {eeg_file}")
        if eeg.dtype != np.float32:
            errors.append(f"invalid EEG dtype {eeg.dtype}: {eeg_file}")
        if pos.shape != (eeg.shape[1], 3) or pos.dtype != np.float32:
            errors.append(f"invalid channel positions {pos.shape}/{pos.dtype}: {pos_file}")
        if not np.all(np.isfinite(pos)) or np.any(np.linalg.norm(pos, axis=1) == 0):
            errors.append(f"missing/non-finite channel position: {pos_file}")
        for start in range(0, eeg.shape[0], 64):
            if not np.all(np.isfinite(eeg[start:start + 64])):
                errors.append(f"non-finite EEG values: {eeg_file}")
                break
        if meta.get("target_fs") != 200:
            errors.append(f"target_fs is not 200: {meta_file}")
        if float(meta.get("duration_seconds", 0)) < 10:
            errors.append(f"recording shorter than 10s: {meta_file}")
        counts[(dataset, split)] += 1

    for row in split_rows:
        if row["payload_ready"].lower() != "true":
            continue
        key = (row["dataset"], row["subject_id"])
        if key not in observed_subjects:
            errors.append(f"available manifest subject has no output: {key}")

    if any(path.name.lower() == "testdata" for path in processed.rglob("*")):
        errors.append("testdata output exists, but testdata is excluded from Phase 2")

    print(f"Validated {len(meta_files)} recording outputs")
    for (dataset, split), count in sorted(counts.items()):
        print(f"  {dataset:22s} {split:5s}: {count}")
    if errors:
        print(f"FAILED with {len(errors)} issue(s):", file=sys.stderr)
        for error in errors[:100]:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    print("PASS: no subject leakage and all outputs match the REVE preprocessing contract")


if __name__ == "__main__":
    main()
