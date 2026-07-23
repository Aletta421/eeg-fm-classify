"""Helpers for applying the raw-subject train/test split manifest."""

import csv
from pathlib import Path
from typing import Dict


def load_subject_splits(manifest_path: str, dataset: str) -> Dict[str, str]:
    """Return subject_id -> split for one dataset, rejecting ambiguous rows."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"split manifest not found: {path}")

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != dataset:
                continue
            subject_id = row["subject_id"]
            split = row["split"]
            if split not in {"train", "test"}:
                raise ValueError(f"invalid split {split!r} for {dataset}/{subject_id}")
            if subject_id in mapping and mapping[subject_id] != split:
                raise ValueError(f"subject appears in both splits: {dataset}/{subject_id}")
            mapping[subject_id] = split

    if not mapping:
        raise ValueError(f"no split rows found for dataset {dataset!r}")
    return mapping


def set_result_split(result: dict, split: str) -> dict:
    result["meta"]["split"] = split
    return result
