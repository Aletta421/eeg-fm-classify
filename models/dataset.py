"""
EEG PyTorch Dataset

读取 data/labels.csv，按受试者划分 train/test（80/20 分层 split），
testdata 目录始终作为独立测试集。

关键设计：
- 每个 .npy 文件包含多个 epochs (n_epochs, n_channels, 2000)
- Dataset 展开为单个 epoch，每次 __getitem__ 返回一个 epoch
- 按 subject_id 划分，确保同一受试者的所有 epochs 在同一集合中
- 二分类: label 0=HC, 1=患者
- 通过 include_diagnosis 区分疾病: ["control", "depression"] 或 ["control", "adhd"]
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """EEG 数据集。

    每个样本为单个 epoch: (eeg, pos, label)
    - eeg:   (n_channels, 2000) float32
    - pos:   (n_channels, 3) float32 电极坐标
    - label: int (0=对照, 1=患者)
    """

    def __init__(
        self,
        labels_csv: str = "data/labels.csv",
        include_diagnosis: Optional[List[str]] = None,
        include_datasets: Optional[List[str]] = None,
        exclude_datasets: Optional[List[str]] = None,
        holdout_dataset: str = "testdata",
        max_channels: int = -1,
        max_epochs_per_subject: int = -1,
        split: str = "train",
        seed: int = 42,
        project_root: Optional[Path] = None,
        splits_file: Optional[str] = "data/splits.json",
    ):
        """
        Args:
            labels_csv: 标签索引文件路径。
            include_diagnosis: 只包含特定 diagnosis_type（如 ["control", "depression"]）。
                              默认 None = 使用所有数据（HC + 所有患者类型）。
            include_datasets: 只包含指定数据集。
            exclude_datasets: 排除的数据集列表。
            holdout_dataset: 始终作为独立测试集的数据集目录名（testdata）。
            max_epochs_per_subject: 每个受试者最多使用的 epoch 数。
            split: "train" | "test"。
            seed: 随机种子（仅在无预计算 split 时使用）。
            project_root: 项目根目录，用于解析相对路径。
        """
        self.split = split
        self.max_epochs_per_subject = max_epochs_per_subject
        self.project_root = project_root or Path.cwd()

        # 读取标签
        csv_path = self.project_root / labels_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"Labels file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # 二分类: HC=0, 患者=1
        self.df["label"] = (self.df["label"] > 0).astype(int)
        self._n_classes = 2

        # 按疾病类型过滤（用于分别训练 depression 和 ADHD 模型）
        # include_diagnosis=["control", "depression"] → 抑郁症模型
        # include_diagnosis=["control", "adhd"]       → ADHD 模型
        if include_diagnosis:
            self.df = self.df[self.df["diagnosis_type"].isin(include_diagnosis)]

        if include_datasets:
            self.df = self.df[self.df["dataset"].isin(include_datasets)]

        if exclude_datasets:
            self.df = self.df[~self.df["dataset"].isin(exclude_datasets)]

        if max_channels > 0:
            self.df = self.df[self.df["n_channels"] <= max_channels]

        # 分离 holdout 数据集
        holdout_mask = self.df["dataset"] == holdout_dataset
        self.holdout_df = self.df[holdout_mask].copy()
        self.trainval_df = self.df[~holdout_mask].copy()

        logger.info(
            f"Total usable samples: {len(self.df)} "
            f"(trainval={len(self.trainval_df)}, holdout={len(self.holdout_df)})"
        )

        # 按受试者划分 train/test
        self._subject_split(seed)

        # 构建 epoch 索引（展开受试者的所有 epochs）
        target_df = self._get_split_df()
        self._epochs_index = self._build_epoch_index(target_df)

        logger.info(
            f"[{split}] {len(self._epochs_index)} epochs "
            f"from {target_df['subject_id'].nunique()} subjects"
        )

    def _subject_split(self, seed: int):
        """按受试者分层划分 train/test。

        优先级：
        1. labels.csv 中的 "split" 列（由 generate_labels.py + split_data.py 写入）
        2. splits.json 文件（由 split_data.py 生成）
        3. 随机分层 split（fallback，打印 warning）
        """
        # 尝试从 labels.csv 的 split 列读取
        if "split" in self.df.columns:
            split_map = {}
            for _, row in self.df[["subject_id", "split"]].drop_duplicates().iterrows():
                if row["split"] and str(row["split"]) in ("train", "test", "holdout"):
                    split_map[row["subject_id"]] = str(row["split"])

            if split_map:
                train_subjects = {s for s, sp in split_map.items() if sp == "train"}
                # test = 20% 非holdout + 全部 holdout
                test_subjects = {s for s, sp in split_map.items() if sp in ("test", "holdout")}
                holdout_subjects = {s for s, sp in split_map.items() if sp == "holdout"}
                logger.info(
                    f"Using pre-recorded split from labels.csv: "
                    f"train={len(train_subjects)}, test={len(test_subjects)} "
                    f"(holdout={len(holdout_subjects)})"
                )
                self._subject_splits = {
                    "train": train_subjects,
                    "test": test_subjects,
                }
                self.train_subjects = train_subjects
                self.test_subjects = test_subjects
                return

        # 尝试从 splits.json 读取
        splits_path = self.project_root / "data/splits.json"
        if splits_path.exists():
            try:
                with open(splits_path, "r", encoding="utf-8") as f:
                    splits_data = json.load(f)
                train_subjects = set(splits_data.get("train", []))
                test_subjects = set(splits_data.get("test", []))
                holdout_subjects = set(splits_data.get("holdout", []))
                # test split 包含 20% 非 holdout + 全部 holdout
                all_test = test_subjects | holdout_subjects
                logger.info(
                    f"Using pre-recorded split from {splits_path}: "
                    f"train={len(train_subjects)}, test={len(all_test)} "
                    f"(holdout={len(holdout_subjects)})"
                )
                self._subject_splits = {
                    "train": train_subjects,
                    "test": all_test,
                }
                self.train_subjects = train_subjects
                self.test_subjects = all_test
                return
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load splits file: {e}")

        # Fallback: 随机分层 split
        logger.warning(
            "No pre-recorded split found. Using random stratified split. "
            "Run 'python scripts/split_data.py' first to fix the split."
        )
        rng = np.random.RandomState(seed)

        subject_labels = (
            self.trainval_df.groupby("subject_id")["label"].first().to_dict()
        )
        unique_subjects = list(subject_labels.keys())

        train_subjects = set()
        test_subjects = set()

        for lbl in set(subject_labels.values()):
            label_subjects = [s for s in unique_subjects if subject_labels[s] == lbl]
            rng.shuffle(label_subjects)
            n_test = max(1, int(len(label_subjects) * 0.2))
            test_subjects.update(label_subjects[:n_test])
            train_subjects.update(label_subjects[n_test:])

        # holdout 也加入 test
        test_subjects |= set(self.holdout_df["subject_id"].unique())

        self._subject_splits = {
            "train": train_subjects,
            "test": test_subjects,
        }
        self.train_subjects = train_subjects
        self.test_subjects = test_subjects

    def _get_split_df(self) -> pd.DataFrame:
        """获取当前 split 对应的数据。"""
        split_subjects = self._subject_splits[self.split]
        combined_df = pd.concat([self.trainval_df, self.holdout_df])
        return combined_df[
            combined_df["subject_id"].isin(split_subjects)
        ].copy()

    def _build_epoch_index(self, df: pd.DataFrame) -> List[Dict]:
        """构建 epoch 索引，每个 epoch 一条记录。"""
        index = []
        for _, row in df.iterrows():
            n_epochs = int(row["n_epochs"])
            if self.max_epochs_per_subject > 0:
                n_epochs = min(n_epochs, self.max_epochs_per_subject)
            for epoch_i in range(n_epochs):
                index.append({
                    "file_path": row["file_path"],
                    "subject_id": row["subject_id"],
                    "dataset": row["dataset"],
                    "label": int(row["label"]),
                    "diagnosis_type": row["diagnosis_type"],
                    "epoch_idx": epoch_i,
                    "n_channels": int(row["n_channels"]),
                })
        return index

    def __len__(self) -> int:
        return len(self._epochs_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单个 epoch。

        Returns:
            {
                "eeg":   Tensor (n_channels, 2000),
                "pos":   Tensor (n_channels, 3),
                "label": int,
                "subject_id": str,
                "dataset": str,
            }
        """
        entry = self._epochs_index[idx]
        epoch_i = entry["epoch_idx"]

        npy_path = self.project_root / entry["file_path"]
        pos_path = npy_path.parent / f"{entry['subject_id']}_ch_pos.npy"

        if not npy_path.exists():
            raise FileNotFoundError(f"EEG file not found: {npy_path}")

        # pos 文件，不存在就用零填充
        try:
            pos = np.load(pos_path).astype(np.float32)
        except (FileNotFoundError, OSError):
            pos = np.zeros((entry["n_channels"], 3), dtype=np.float32)

        # 读取 epoch（lazy: 只读需要的 epoch）
        try:
            eeg_all = np.load(npy_path, mmap_mode="r")
            if epoch_i >= eeg_all.shape[0]:
                epoch_i = eeg_all.shape[0] - 1
            eeg = np.array(eeg_all[epoch_i], dtype=np.float32)
        except Exception:
            eeg_all = np.load(npy_path).astype(np.float32)
            if epoch_i >= eeg_all.shape[0]:
                epoch_i = eeg_all.shape[0] - 1
            eeg = eeg_all[epoch_i]

        # 确保 pos 和 eeg 通道数一致
        if pos.shape[0] != eeg.shape[0]:
            pos = np.zeros((eeg.shape[0], 3), dtype=np.float32)

        return {
            "eeg": torch.from_numpy(eeg),
            "pos": torch.from_numpy(pos),
            "label": entry["label"],
            "subject_id": entry["subject_id"],
            "dataset": entry["dataset"],
        }

    def get_subject_epochs(self, subject_id: str) -> List[int]:
        """获取指定受试者的所有 epoch 索引。"""
        return [
            i for i, entry in enumerate(self._epochs_index)
            if entry["subject_id"] == subject_id
        ]

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def label_counts(self) -> Dict[int, int]:
        df = self._get_split_df()
        return df["label"].value_counts().to_dict()

    @property
    def class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理不平衡）。"""
        counts = self.label_counts
        total = sum(counts.values())
        weights = torch.zeros(max(counts.keys()) + 1)
        for label, count in counts.items():
            weights[label] = total / (len(counts) * count)
        return weights


def collate_eeg_batch(batch: List[Dict]) -> Dict:
    """自定义 collate 函数，处理不同通道数的样本。

    将 batch 中不同通道数的 EEG 零填充到最大通道数。
    REVE 的 attention pooling 会自然忽略填充的零通道。
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    max_ch = max(b["eeg"].shape[0] for b in batch)
    max_t = max(b["eeg"].shape[1] for b in batch)

    eegs_padded = []
    poss_padded = []
    for b in batch:
        eeg = b["eeg"]
        pos = b["pos"]
        c, t = eeg.shape

        if c < max_ch or t < max_t:
            padded_eeg = torch.zeros(max_ch, max_t, dtype=eeg.dtype)
            padded_eeg[:c, :t] = eeg
            eegs_padded.append(padded_eeg)
        else:
            eegs_padded.append(eeg)

        if c < max_ch:
            padded_pos = torch.zeros(max_ch, 3, dtype=pos.dtype)
            padded_pos[:c, :] = pos
            poss_padded.append(padded_pos)
        else:
            poss_padded.append(pos)

    eegs = torch.stack(eegs_padded)   # (B, max_ch, max_t)
    poss = torch.stack(poss_padded)   # (B, max_ch, 3)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    subject_ids = [b["subject_id"] for b in batch]
    datasets = [b["dataset"] for b in batch]

    return {
        "eeg": eegs,
        "pos": poss,
        "label": labels,
        "subject_id": subject_ids,
        "dataset": datasets,
    }


def create_dataloaders(
    labels_csv: str = "data/labels.csv",
    include_diagnosis: Optional[List[str]] = None,
    include_datasets: Optional[List[str]] = None,
    exclude_datasets: Optional[List[str]] = None,
    holdout_dataset: str = "testdata",
    max_channels: int = -1,
    max_epochs_per_subject: int = -1,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 4,
    project_root: Optional[Path] = None,
) -> Dict[str, DataLoader]:
    """创建 train/test 两个 DataLoader。

    通过 include_diagnosis 区分疾病类型:
        create_dataloaders(include_diagnosis=["control", "depression"])  # 抑郁症模型
        create_dataloaders(include_diagnosis=["control", "adhd"])       # ADHD 模型

    Returns:
        {
            "train": DataLoader,
            "test": DataLoader,
            "train_dataset": EEGDataset,
            "test_dataset": EEGDataset,
        }
    """
    common_kwargs = dict(
        labels_csv=labels_csv,
        include_diagnosis=include_diagnosis,
        include_datasets=include_datasets,
        exclude_datasets=exclude_datasets,
        holdout_dataset=holdout_dataset,
        max_channels=max_channels,
        max_epochs_per_subject=max_epochs_per_subject,
        seed=seed,
        project_root=project_root,
    )

    train_dataset = EEGDataset(split="train", **common_kwargs)
    test_dataset = EEGDataset(split="test", **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_eeg_batch,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eeg_batch,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }


# ================================================================
# 快速测试
# ================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).parent.parent

    print("=" * 60)
    print("Testing EEGDataset...")
    print("=" * 60)

    # 全量二分类
    dataloaders = create_dataloaders(
        batch_size=4,
        num_workers=0,
        project_root=project_root,
    )

    for split_name, loader in [
        ("train", dataloaders["train"]),
        ("test", dataloaders["test"]),
    ]:
        print(f"\n--- {split_name} ---")
        print(f"  Batches: {len(loader)}")
        batch = next(iter(loader))
        if batch:
            print(f"  eeg shape:  {batch['eeg'].shape}")
            print(f"  pos shape:  {batch['pos'].shape}")
            print(f"  labels:     {batch['label'].tolist()}")
            print(f"  subjects:   {batch['subject_id']}")
            print(f"  datasets:   {batch['dataset']}")

    for split_name, ds in [
        ("train", dataloaders["train_dataset"]),
        ("test", dataloaders["test_dataset"]),
    ]:
        print(f"\n{split_name}: {len(ds)} epochs, labels={ds.label_counts}")
