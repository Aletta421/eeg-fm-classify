"""Subject-level EEG datasets and train/validation/test splitting."""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SubjectKey = Tuple[str, str]

DATASET_ALIASES = {
    "IEEE": "IEEE_ADHD",
    "OpenNeuro": "OpenNeuro_ds003478",
}


def _normalize_datasets(names: Optional[List[str]]) -> Optional[List[str]]:
    if not names:
        return names
    return [DATASET_ALIASES.get(name, name) for name in names]


def _stratified_subject_splits(
    frame: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Dict[SubjectKey, str]:
    """Keep recorded test subjects fixed and split recorded train into train/val."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")

    subject_rows = []
    for (dataset, subject_id), group in frame.groupby(
        ["dataset", "subject_id"], sort=True
    ):
        labels = set(group["label"].astype(int))
        source_splits = set(group["split"].astype(str))
        diagnoses = set(group["diagnosis_type"].astype(str))
        if len(labels) != 1 or len(source_splits) != 1 or len(diagnoses) != 1:
            raise ValueError(f"inconsistent subject metadata: {dataset}/{subject_id}")
        source_split = next(iter(source_splits))
        if source_split not in {"train", "test"}:
            raise ValueError(
                f"invalid recorded split {source_split!r}: {dataset}/{subject_id}"
            )
        subject_rows.append({
            "key": (str(dataset), str(subject_id)),
            "dataset": str(dataset),
            "label": next(iter(labels)),
            "diagnosis": next(iter(diagnoses)),
            "source_split": source_split,
        })

    assignments = {
        row["key"]: "test"
        for row in subject_rows
        if row["source_split"] == "test"
    }
    strata = defaultdict(list)
    for row in subject_rows:
        if row["source_split"] == "train":
            strata[(row["dataset"], row["diagnosis"], row["label"])].append(
                row["key"]
            )

    for stratum, keys in sorted(strata.items()):
        keys = sorted(keys)
        digest = hashlib.sha256(f"{seed}:{stratum}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        rng.shuffle(keys)
        if val_fraction == 0 or len(keys) < 2:
            n_val = 0
        else:
            n_val = max(1, min(len(keys) - 1, round(len(keys) * val_fraction)))
        for index, key in enumerate(keys):
            assignments[key] = "val" if index < n_val else "train"

    return assignments


class SubjectEEGDataset(Dataset):
    """One item per subject, containing that subject's selected EEG segments."""

    def __init__(
        self,
        labels_csv: str = "data/labels.csv",
        include_diagnosis: Optional[List[str]] = None,
        include_datasets: Optional[List[str]] = None,
        exclude_datasets: Optional[List[str]] = None,
        max_channels: int = -1,
        split: str = "train",
        val_fraction: float = 0.2,
        max_segments: int = -1,
        seed: int = 42,
        project_root: Optional[Path] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.split = split
        self.max_segments = max_segments
        self.seed = seed
        self.epoch = 0
        self.project_root = Path(project_root or Path.cwd())

        csv_path = Path(labels_csv)
        if not csv_path.is_absolute():
            csv_path = self.project_root / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"labels file not found: {csv_path}")
        self.labels_dir = csv_path.parent

        frame = pd.read_csv(csv_path)
        required = {
            "dataset", "subject_id", "split", "label", "diagnosis_type",
            "file_path", "n_epochs", "n_channels",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"labels file is missing columns: {sorted(missing)}")

        frame["dataset"] = frame["dataset"].astype(str)
        frame["subject_id"] = frame["subject_id"].astype(str)
        frame["label"] = (frame["label"].astype(int) > 0).astype(int)
        if include_diagnosis:
            frame = frame[frame["diagnosis_type"].isin(include_diagnosis)]
        included = _normalize_datasets(include_datasets)
        excluded = _normalize_datasets(exclude_datasets)
        if included:
            frame = frame[frame["dataset"].isin(included)]
        if excluded:
            frame = frame[~frame["dataset"].isin(excluded)]
        if max_channels > 0:
            frame = frame[frame["n_channels"].astype(int) <= max_channels]
        if frame.empty:
            raise ValueError("no records remain after applying data filters")

        assignments = _stratified_subject_splits(frame, val_fraction, seed)
        self._subject_groups = {}
        for key, group in frame.groupby(["dataset", "subject_id"], sort=True):
            normalized_key = (str(key[0]), str(key[1]))
            if assignments[normalized_key] == split:
                self._subject_groups[normalized_key] = group.copy()
        self.subject_keys = sorted(self._subject_groups)
        self._labels = {
            key: int(group["label"].iloc[0])
            for key, group in self._subject_groups.items()
        }
        self._segments = {
            key: self._build_segment_index(group)
            for key, group in self._subject_groups.items()
        }

    def _resolve_data_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        candidates = [self.project_root / path, self.labels_dir / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    def _build_segment_index(self, group: pd.DataFrame) -> List[Dict]:
        segments = []
        for _, row in group.iterrows():
            eeg_path = self._resolve_data_path(str(row["file_path"]))
            for epoch_index in range(int(row["n_epochs"])):
                segments.append({
                    "eeg_path": eeg_path,
                    "pos_path": eeg_path.parent / (
                        f"{row['subject_id']}_ch_pos.npy"
                    ),
                    "epoch_index": epoch_index,
                })
        if not segments:
            key = (group["dataset"].iloc[0], group["subject_id"].iloc[0])
            raise ValueError(f"subject has no EEG segments: {key}")
        return segments

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.subject_keys)

    def __getitem__(self, index: int) -> Dict:
        key = self.subject_keys[index]
        references = self._segments[key]
        if 0 < self.max_segments < len(references):
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, index])
            )
            selected = sorted(
                rng.choice(len(references), self.max_segments, replace=False).tolist()
            )
            references = [references[i] for i in selected]

        segments = []
        for reference in references:
            eeg_all = np.load(reference["eeg_path"], mmap_mode="r")
            epoch_index = reference["epoch_index"]
            if eeg_all.ndim != 3 or epoch_index >= eeg_all.shape[0]:
                raise ValueError(
                    f"invalid EEG segment {epoch_index}: {reference['eeg_path']}"
                )
            eeg = np.array(eeg_all[epoch_index], dtype=np.float32)
            pos = np.load(reference["pos_path"]).astype(np.float32)
            if pos.shape != (eeg.shape[0], 3):
                raise ValueError(
                    f"channel positions do not match EEG: {reference['pos_path']}"
                )
            segments.append({
                "eeg": torch.from_numpy(eeg),
                "pos": torch.from_numpy(pos),
            })

        return {
            "segments": segments,
            "label": self._labels[key],
            "dataset": key[0],
            "subject_id": key[1],
        }

    @property
    def n_classes(self) -> int:
        return 2

    @property
    def label_counts(self) -> Dict[int, int]:
        return dict(Counter(self._labels.values()))

    @property
    def class_weights(self) -> torch.Tensor:
        counts = self.label_counts
        if not counts:
            return torch.ones(self.n_classes)
        total = sum(counts.values())
        weights = torch.ones(self.n_classes)
        for label, count in counts.items():
            weights[label] = total / (len(counts) * count)
        return weights


def collate_subject_batch(batch: List[Dict]) -> Dict:
    """Pad channels, time, and segment counts while retaining a segment mask."""
    if not batch:
        return {}
    max_segments = max(len(item["segments"]) for item in batch)
    max_channels = max(
        segment["eeg"].shape[0]
        for item in batch
        for segment in item["segments"]
    )
    max_times = max(
        segment["eeg"].shape[1]
        for item in batch
        for segment in item["segments"]
    )
    batch_size = len(batch)
    eeg = torch.zeros(batch_size, max_segments, max_channels, max_times)
    pos = torch.zeros(batch_size, max_segments, max_channels, 3)
    segment_mask = torch.zeros(batch_size, max_segments, dtype=torch.bool)

    for subject_index, item in enumerate(batch):
        for segment_index, segment in enumerate(item["segments"]):
            channels, times = segment["eeg"].shape
            eeg[subject_index, segment_index, :channels, :times] = segment["eeg"]
            pos[subject_index, segment_index, :channels] = segment["pos"]
            segment_mask[subject_index, segment_index] = True

    return {
        "eeg": eeg,
        "pos": pos,
        "segment_mask": segment_mask,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "dataset": [item["dataset"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
    }


def create_subject_dataloaders(
    labels_csv: str = "data/labels.csv",
    include_diagnosis: Optional[List[str]] = None,
    include_datasets: Optional[List[str]] = None,
    exclude_datasets: Optional[List[str]] = None,
    max_channels: int = -1,
    val_fraction: float = 0.2,
    train_segments_per_subject: int = 8,
    eval_segments_per_subject: int = -1,
    batch_size: int = 4,
    seed: int = 42,
    num_workers: int = 0,
    project_root: Optional[Path] = None,
) -> Dict[str, DataLoader]:
    common = dict(
        labels_csv=labels_csv,
        include_diagnosis=include_diagnosis,
        include_datasets=include_datasets,
        exclude_datasets=exclude_datasets,
        max_channels=max_channels,
        val_fraction=val_fraction,
        seed=seed,
        project_root=project_root,
    )
    datasets = {
        "train": SubjectEEGDataset(
            split="train", max_segments=train_segments_per_subject, **common
        ),
        "val": SubjectEEGDataset(
            split="val", max_segments=eval_segments_per_subject, **common
        ),
        "test": SubjectEEGDataset(
            split="test", max_segments=eval_segments_per_subject, **common
        ),
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            collate_fn=collate_subject_batch,
            pin_memory=torch.cuda.is_available(),
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    return {
        **loaders,
        **{f"{split}_dataset": dataset for split, dataset in datasets.items()},
    }
