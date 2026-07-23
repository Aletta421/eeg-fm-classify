import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.subject_dataset import (
    SubjectEEGDataset,
    collate_subject_batch,
    create_subject_dataloaders,
)
from models.subject_model import SubjectAggregationModel
from models.subject_trainer import EarlyStopping, SubjectTrainer


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, eeg, pos=None):
        return eeg[:, 0, :2] * self.scale


class SubjectDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = []
        subjects = [
            ("A", "a0", "train", 0),
            ("A", "a1", "train", 0),
            ("A", "a2", "train", 1),
            ("A", "a3", "train", 1),
            ("A", "a4", "test", 0),
            ("A", "a5", "test", 1),
            ("B", "a0", "test", 0),
        ]
        for dataset, subject_id, split, label in subjects:
            subject_dir = self.root / "processed" / dataset / split / subject_id
            subject_dir.mkdir(parents=True)
            eeg_path = subject_dir / f"{subject_id}_eeg.npy"
            pos_path = subject_dir / f"{subject_id}_ch_pos.npy"
            eeg = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
            np.save(eeg_path, eeg)
            np.save(pos_path, np.ones((2, 3), dtype=np.float32))
            rows.append({
                "subject_id": subject_id,
                "dataset": dataset,
                "split": split,
                "label": label,
                "diagnosis_type": "control" if label == 0 else "case",
                "file_path": eeg_path.relative_to(self.root).as_posix(),
                "n_epochs": 3,
                "n_channels": 2,
            })

        self.labels_csv = self.root / "labels.csv"
        with self.labels_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_subject_split_is_deterministic_and_keeps_test_fixed(self):
        kwargs = dict(
            labels_csv=str(self.labels_csv.relative_to(self.root)),
            project_root=self.root,
            val_fraction=0.5,
            seed=7,
        )
        train = SubjectEEGDataset(split="train", **kwargs)
        val = SubjectEEGDataset(split="val", **kwargs)
        test = SubjectEEGDataset(split="test", **kwargs)
        train_again = SubjectEEGDataset(split="train", **kwargs)

        train_keys = set(train.subject_keys)
        val_keys = set(val.subject_keys)
        test_keys = set(test.subject_keys)
        self.assertFalse(train_keys & val_keys)
        self.assertFalse(train_keys & test_keys)
        self.assertFalse(val_keys & test_keys)
        self.assertEqual(train_keys, set(train_again.subject_keys))
        self.assertEqual(
            test_keys, {("A", "a4"), ("A", "a5"), ("B", "a0")}
        )

    def test_one_dataset_item_and_one_batch_row_per_subject(self):
        dataset = SubjectEEGDataset(
            labels_csv=str(self.labels_csv.relative_to(self.root)),
            project_root=self.root,
            split="test",
            val_fraction=0.5,
            max_segments=2,
        )
        item = dataset[0]
        self.assertEqual(len(item["segments"]), 2)

        batch = collate_subject_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["eeg"].shape[:2], (2, 2))
        self.assertEqual(batch["label"].shape, (2,))
        self.assertTrue(batch["segment_mask"].all())

    def test_dataloaders_count_subjects_not_segments(self):
        loaders = create_subject_dataloaders(
            labels_csv=str(self.labels_csv.relative_to(self.root)),
            project_root=self.root,
            val_fraction=0.5,
            train_segments_per_subject=2,
            eval_segments_per_subject=-1,
            batch_size=2,
            num_workers=0,
        )
        total_subjects = sum(
            len(loaders[f"{split}_dataset"]) for split in ("train", "val", "test")
        )
        self.assertEqual(total_subjects, 7)

    def test_cpu_training_uses_validation_checkpoint_before_test(self):
        loaders = create_subject_dataloaders(
            labels_csv=str(self.labels_csv.relative_to(self.root)),
            project_root=self.root,
            val_fraction=0.5,
            train_segments_per_subject=2,
            eval_segments_per_subject=2,
            batch_size=2,
            num_workers=0,
        )
        model = SubjectAggregationModel(
            encoder=FakeEncoder(),
            embedding_dim=2,
            n_classes=2,
            aggregation="mean",
            freeze_encoder=True,
        )
        trainer = SubjectTrainer(
            model=model,
            device=torch.device("cpu"),
            lr=1e-2,
            patience=1,
        )
        result = trainer.fit(
            loaders["train"],
            loaders["val"],
            loaders["test"],
            epochs=2,
            checkpoint_dir=self.root / "checkpoints",
        )
        self.assertTrue(
            (self.root / "checkpoints" / "best_subject_model.pt").exists()
        )
        self.assertGreaterEqual(result["final"]["best_optimizer_step"], 1)
        self.assertIn("balanced_acc", result["final"]["test"])


class SubjectModelTests(unittest.TestCase):
    def test_mean_pooling_produces_one_embedding_per_subject(self):
        model = SubjectAggregationModel(
            encoder=FakeEncoder(),
            embedding_dim=2,
            n_classes=2,
            aggregation="mean",
            freeze_encoder=True,
        )
        eeg = torch.tensor([
            [[[1.0, 3.0]], [[5.0, 7.0]], [[9.0, 11.0]]],
            [[[2.0, 4.0]], [[6.0, 8.0]], [[0.0, 0.0]]],
        ])
        pos = torch.zeros(2, 3, 1, 3)
        mask = torch.tensor([[True, True, True], [True, True, False]])

        pooled = model.encode_subjects(eeg, pos, mask)
        expected = torch.tensor([[5.0, 7.0], [4.0, 6.0]])
        torch.testing.assert_close(pooled, expected)
        self.assertEqual(model(eeg, pos, mask).shape, (2, 2))

    def test_transform_and_classifier_train_while_encoder_stays_frozen(self):
        encoder = FakeEncoder()
        model = SubjectAggregationModel(
            encoder=encoder,
            embedding_dim=2,
            n_classes=2,
            aggregation="transform_mean",
            transform_hidden_dim=4,
            freeze_encoder=True,
        )
        eeg = torch.randn(2, 3, 1, 2)
        pos = torch.zeros(2, 3, 1, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)

        model(eeg, pos, mask).sum().backward()
        self.assertIsNone(encoder.scale.grad)
        self.assertTrue(any(p.grad is not None for p in model.transform.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.classifier.parameters()))


class EarlyStoppingTests(unittest.TestCase):
    def test_best_validation_step_controls_stopping(self):
        tracker = EarlyStopping(patience=2, mode="max")
        self.assertTrue(tracker.update(0.50, epoch=1, step=10))
        self.assertTrue(tracker.update(0.60, epoch=2, step=20))
        self.assertFalse(tracker.update(0.55, epoch=3, step=30))
        self.assertFalse(tracker.should_stop)
        self.assertFalse(tracker.update(0.54, epoch=4, step=40))
        self.assertTrue(tracker.should_stop)
        self.assertEqual((tracker.best_epoch, tracker.best_step), (2, 20))


if __name__ == "__main__":
    unittest.main()
