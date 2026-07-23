"""Validation-driven training for subject-level EEG classification."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "max"
    best_value: Optional[float] = None
    best_epoch: int = 0
    best_step: int = 0
    bad_epochs: int = 0

    def __post_init__(self):
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")

    def update(self, value: float, epoch: int, step: int) -> bool:
        improved = (
            self.best_value is None
            or (self.mode == "max" and value > self.best_value)
            or (self.mode == "min" and value < self.best_value)
        )
        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.best_step = int(step)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    y_pred = probabilities.argmax(axis=1)
    present_classes = np.unique(y_true)
    recalls = []
    f1_scores = []
    for label in present_classes:
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        recalls.append(recall)
        f1_scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
    metrics = {
        "balanced_acc": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1_scores)),
    }
    if probabilities.shape[1] == 2 and len(present_classes) > 1:
        positive = probabilities[y_true == 1, 1]
        negative = probabilities[y_true == 0, 1]
        comparisons = positive[:, None] - negative[None, :]
        metrics["auroc"] = float(
            (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
            / comparisons.size
        )
    else:
        metrics["auroc"] = 0.5
    return metrics


class SubjectTrainer:
    """Optimize one loss per subject and select the best step on validation."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        monitor_metric: str = "balanced_acc",
        monitor_mode: str = "max",
        patience: int = 5,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.monitor_metric = monitor_metric
        self.grad_clip = grad_clip
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("subject model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        self.early_stopping = EarlyStopping(patience, monitor_mode)
        self.optimizer_steps = 0

    def _move_batch(self, batch: Dict) -> Dict:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_logits, all_labels = [], []
        for batch in loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            logits = self.model(
                batch["eeg"], batch["pos"], batch["segment_mask"]
            )
            loss = self.criterion(logits, batch["label"])
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            self.optimizer.step()
            self.optimizer_steps += 1
            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(batch["label"].detach())
        metrics = classification_metrics(
            torch.cat(all_logits), torch.cat(all_labels)
        )
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []
        for batch in loader:
            batch = self._move_batch(batch)
            logits = self.model(
                batch["eeg"], batch["pos"], batch["segment_mask"]
            )
            total_loss += self.criterion(logits, batch["label"]).item()
            all_logits.append(logits)
            all_labels.append(batch["label"])
        metrics = classification_metrics(
            torch.cat(all_logits), torch.cat(all_labels)
        )
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        test_loader,
        epochs: int,
        checkpoint_dir: Path,
        config=None,
    ) -> Dict:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best_subject_model.pt"
        history = []

        for epoch in range(1, epochs + 1):
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            history.append({
                "epoch": epoch,
                "optimizer_step": self.optimizer_steps,
                "train": train_metrics,
                "val": val_metrics,
            })
            monitored = val_metrics[self.monitor_metric]
            if self.early_stopping.update(
                monitored, epoch, self.optimizer_steps
            ):
                torch.save({
                    "epoch": epoch,
                    "optimizer_step": self.optimizer_steps,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                }, best_path)
            if self.early_stopping.should_stop:
                break

        checkpoint = torch.load(
            best_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = self.evaluate(test_loader)
        result = {
            "best_epoch": self.early_stopping.best_epoch,
            "best_optimizer_step": self.early_stopping.best_step,
            "best_val_metric": self.early_stopping.best_value,
            "test": test_metrics,
        }
        with (checkpoint_dir / "subject_history.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump({"epochs": history, "final": result}, handle, indent=2)
        return {"history": history, "final": result}
