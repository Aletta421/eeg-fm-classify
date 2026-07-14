"""
REVE 模型训练脚本

支持：
- 线性探测（linear_probe）：冻结 encoder，仅训练分类头
- 微调（finetune）：解冻最后 N 层
- 自动保存最佳模型、日志记录
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    SequentialLR,
    LinearLR,
)
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dataset import create_dataloaders, collate_eeg_batch
from models.config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    get_exp_depression,
    get_exp_adhd,
    get_exp_single_dataset,
)
from models.heads import create_head, SklearnHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ================================================================
# 模型构建
# ================================================================


def build_model(
    config: ModelConfig,
    device: torch.device,
    n_classes: Optional[int] = None,
    training_config: Optional[TrainingConfig] = None,
) -> nn.Module:
    """构建 REVE 模型，可选替换分类头。

    支持 HuggingFace 镜像（设置 HF_ENDPOINT 环境变量或 config.hf_endpoint）。

    Args:
        config: 模型配置。
        device: 计算设备。
        n_classes: 输出类别数（用于替换分类头时）。
        training_config: 训练配置（包含 head_type 等参数）。

    Returns:
        REVE 模型实例。
    """
    import os
    from braindecode.models import REVE

    # 设置 HF 镜像（国内用户）
    if config.hf_endpoint:
        os.environ["HF_ENDPOINT"] = config.hf_endpoint
        logger.info(f"Using HF mirror: {config.hf_endpoint}")

    logger.info(f"Loading REVE model: {config.model_id}")

    model_kwargs = dict(
        n_outputs=config.n_outputs,
        n_chans=None,  # 可变通道数
        n_times=config.n_times,
        input_window_seconds=config.input_window_seconds,
        sfreq=config.sfreq,
        attention_pooling=config.use_attention_pooling,
    )

    if config.pretrained:
        try:
            model = REVE.from_pretrained(
                config.model_id,
                force_download=config.force_download,
                local_files_only=config.local_files_only,
                token=config.hf_token,
                **model_kwargs,
            )
        except Exception as e:
            # local_files_only=True 且无缓存 → 直接 fallback
            logger.warning(
                f"Cannot load pretrained weights: {e}\n"
                "Falling back to random initialization.\n"
                "To download weights: set HF_ENDPOINT=https://hf-mirror.com (China mirror)\n"
                "Then run: python -c \"from braindecode.models import REVE; "
                "REVE.from_pretrained('brain-bzh/reve-base')\""
            )
            logger.info("Creating REVE with random weights (no pretrained)...")
            model = REVE(**model_kwargs)
    else:
        model = REVE(**model_kwargs)

    model = model.to(device)

    # 替换分类头
    if training_config is not None and n_classes is not None:
        replace_classification_head(model, training_config, n_classes)

    return model


def replace_classification_head(
    model: nn.Module,
    training_config: TrainingConfig,
    n_classes: int,
    embed_dim: int = 512,
):
    """用自定义分类头替换 REVE 的 final_layer。

    Args:
        model: REVE 模型实例。
        training_config: 训练配置（包含 head_type 等参数）。
        n_classes: 输出类别数。
        embed_dim: REVE encoder 输出的 embedding 维度（默认 512）。

    Returns:
        修改后的 model（原地修改）。
    """
    head_type = training_config.head_type

    if head_type == "linear":
        # 使用默认配置即可，线性头等同于原 final_layer
        pass  # 保留原始 final_layer
        logger.info("Using default REVE linear head (final_layer)")
        return model

    logger.info(f"Replacing classification head: linear -> {head_type}")

    # 构建自定义 head kwargs
    head_kwargs = dict(
        in_features=embed_dim,
        n_classes=n_classes,
    )

    if head_type == "mlp":
        head_kwargs.update(
            hidden_dim=training_config.head_hidden_dim,
            num_layers=training_config.head_num_layers,
            dropout=training_config.head_dropout,
            use_batchnorm=training_config.head_use_batchnorm,
        )
    elif head_type == "cnn1d":
        head_kwargs.update(
            conv_channels=[64, 128],
            kernel_sizes=[7, 5],
            dropout=training_config.head_dropout,
        )
    elif head_type == "attention":
        head_kwargs.update(
            num_heads=8,
            ff_dim=training_config.head_hidden_dim,
            dropout=training_config.head_dropout,
        )
    elif head_type.startswith("sklearn_"):
        # sklearn 头：传递 random_state
        head_kwargs["random_state"] = 42

    new_head = create_head(head_type, **head_kwargs)
    new_head.reset_parameters()
    new_head = new_head.to(next(model.parameters()).device)  # 对齐设备

    # 替换
    model.final_layer = new_head
    logger.info(
        f"  Head params: {new_head.count_params():,}  "
        f"(type: {head_type}, embed_dim={embed_dim}, n_classes={n_classes})"
    )

    return model


def setup_linear_probe(model: nn.Module, training_config: TrainingConfig):
    """设置线性探测：冻结 encoder，只训练分类头 (model.final_layer)。"""
    logger.info("Setting up linear probe: freezing encoder")

    # 直接通过 model.final_layer 定位分类头参数
    head_param_ids = set()
    if hasattr(model, "final_layer"):
        head_param_ids = {id(p) for p in model.final_layer.parameters()}

    if head_param_ids:
        # PyTorch head：只解冻 head 参数
        for name, param in model.named_parameters():
            param.requires_grad = id(param) in head_param_ids
    elif isinstance(model.final_layer, SklearnHead):
        # sklearn head：无 PyTorch 参数，全部冻结即可
        for param in model.parameters():
            param.requires_grad = False
        logger.info("  Sklearn head: no PyTorch params to train")
    else:
        # Fallback: 关键词匹配
        logger.warning("No 'final_layer' found, falling back to keyword matching")
        for name, param in model.named_parameters():
            if any(kw in name for kw in ["fc", "classifier", "head", "final"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # 打印可训练参数数量
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def setup_finetune(model: nn.Module, training_config: TrainingConfig):
    """设置微调：解冻最后 N 层 + 分类头。"""
    n_unfreeze = training_config.unfreeze_layers
    logger.info(f"Setting up finetune: unfreezing last {n_unfreeze} layers")

    # 先冻结所有
    for param in model.parameters():
        param.requires_grad = False

    # 解冻分类头 (model.final_layer)
    head_param_ids = set()
    if hasattr(model, "final_layer"):
        head_param_ids = {id(p) for p in model.final_layer.parameters()}
        for pid in head_param_ids:
            for name, param in model.named_parameters():
                if id(param) == pid:
                    param.requires_grad = True
    else:
        # Fallback: 关键词匹配
        for name, param in model.named_parameters():
            if any(kw in name for kw in ["fc", "classifier", "head", "final"]):
                param.requires_grad = True

    # 解冻最后 N 层 transformer
    if n_unfreeze > 0:
        # REVE 的 transformer 层在 "transformer.layers" 中
        # 层索引在 transformer.layers.N 的 N 位置
        max_layer = -1
        for name, _ in model.named_parameters():
            if "transformer.layers." in name:
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            max_layer = max(max_layer, layer_idx)
                        except ValueError:
                            pass
                        break

        if max_layer >= 0:
            unfreeze_from = max_layer - n_unfreeze + 1
            logger.info(f"  Unfreezing transformer layers {unfreeze_from}-{max_layer}")
            for name, param in model.named_parameters():
                if "transformer.layers." in name:
                    parts = name.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                if layer_idx >= unfreeze_from:
                                    param.requires_grad = True
                            except ValueError:
                                pass
                            break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# ================================================================
# 训练循环
# ================================================================


class Trainer:
    """REVE 模型训练器。"""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.training_cfg = config.training

        # 优化器
        self.optimizer = self._build_optimizer()
        self.scaler = GradScaler(device.type, enabled=config.training.use_amp and device.type == "cuda")

        # 学习率调度器
        self.scheduler = self._build_scheduler()

        # 损失函数 (带类别权重)
        if class_weights is not None and config.training.use_class_weights:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            logger.info(f"Using class weights: {class_weights.tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 最佳模型追踪（仅用于保存最佳 checkpoint，不做早停）
        self.best_metric = 0.0 if config.training.monitor_mode == "max" else float("inf")
        self.best_epoch = 0

    def _build_optimizer(self):
        """构建优化器。"""
        cfg = self.training_cfg

        # 微调模式下，encoder 和 head 使用不同学习率
        if cfg.mode == "finetune":
            encoder_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any(kw in name for kw in ["fc", "classifier", "head", "final"]):
                        head_params.append(param)
                    else:
                        encoder_params.append(param)

            param_groups = [
                {"params": head_params, "lr": cfg.lr},
                {"params": encoder_params, "lr": cfg.lr * cfg.encoder_lr_mult},
            ]
        else:
            param_groups = [
                {"params": [p for p in self.model.parameters() if p.requires_grad]}
            ]

        if cfg.optimizer == "adamw":
            return AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adam":
            return Adam(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            return SGD(param_groups, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _build_scheduler(self):
        """构建学习率调度器。"""
        cfg = self.training_cfg
        if cfg.lr_scheduler is None:
            return None

        # 预热
        if cfg.warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=cfg.warmup_epochs,
            )
            if cfg.lr_scheduler == "cosine":
                main = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs - cfg.warmup_epochs)
            elif cfg.lr_scheduler == "step":
                main = StepLR(self.optimizer, step_size=10, gamma=0.5)
            else:
                return warmup
            return SequentialLR(self.optimizer, [warmup, main], milestones=[cfg.warmup_epochs])

        if cfg.lr_scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        elif cfg.lr_scheduler == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif cfg.lr_scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode=cfg.monitor_mode, factor=0.5, patience=3
            )
        return None

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        pbar = tqdm(train_loader, desc="Training", leave=False)

        grad_accum = max(1, self.training_cfg.gradient_accumulation_steps)
        self.optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            eeg = batch["eeg"].to(self.device)
            pos = batch["pos"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(self.device.type, enabled=self.training_cfg.use_amp and self.device.type == "cuda"):
                logits = self.model(eeg, pos=pos)
                loss = self.criterion(logits, labels) / grad_accum

            self.scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                if self.training_cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.training_cfg.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.3f}"})

        # 计算指标
        from sklearn.metrics import balanced_accuracy_score

        avg_loss = total_loss / len(train_loader)
        acc = balanced_accuracy_score(all_labels, all_preds)

        return {"loss": avg_loss, "balanced_acc": acc}

    @torch.no_grad()
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """验证一个 epoch。"""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        all_probs_list = []  # 存储完整概率矩阵 (N, n_classes)
        pbar = tqdm(val_loader, desc="Validating", leave=False)

        for batch in pbar:
            eeg = batch["eeg"].to(self.device)
            pos = batch["pos"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(eeg, pos=pos)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs_list.append(probs.cpu().numpy())

        # 计算指标
        import numpy as np
        from sklearn.metrics import (
            balanced_accuracy_score,
            roc_auc_score,
            f1_score,
        )

        all_probs = np.concatenate(all_probs_list, axis=0)  # (N, n_classes)
        n_classes = all_probs.shape[1]

        avg_loss = total_loss / len(val_loader)
        acc = balanced_accuracy_score(all_labels, all_preds)

        metrics = {
            "loss": avg_loss,
            "balanced_acc": acc,
            "f1": f1_score(all_labels, all_preds, average="weighted"),
        }

        # AUROC: 二分类用正类概率，多分类用 ovr macro
        if len(set(all_labels)) > 1:
            if n_classes == 2:
                metrics["auroc"] = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                metrics["auroc"] = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr", average="macro"
                )
        else:
            metrics["auroc"] = 0.5

        return metrics

    @staticmethod
    def _extract_all_embeddings(model, dataloader, device) -> tuple:
        """提取全量 embedding（final_layer 之前）+ labels。"""
        import numpy as np

        # 注册 hook 捕获 final_layer 输入
        embeddings = []
        labels = []

        def hook_fn(module, input, output):
            embeddings.append(input[0].detach().cpu().numpy())

        handle = model.final_layer.register_forward_hook(hook_fn)

        try:
            for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
                eeg = batch["eeg"].to(device)
                pos = batch["pos"].to(device)
                model(eeg, pos=pos)
                labels.extend(batch["label"].tolist())
        finally:
            handle.remove()

        X = np.concatenate(embeddings, axis=0)
        y = np.array(labels)
        return X, y

    def _fit_sklearn(self, train_loader, test_loader, checkpoint_dir: Path):
        """sklearn 头训练路径：提取 embedding → fit sklearn → 评估。"""
        import numpy as np
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

        logger.info("Sklearn head detected — extracting embeddings for fit...")

        # 1. 提取 train embedding 并拟合
        X_train, y_train = self._extract_all_embeddings(
            self.model, train_loader, self.device
        )
        logger.info(
            f"  Train embeddings: {X_train.shape}, labels: {np.bincount(y_train).tolist()}"
        )
        self.model.final_layer.fit(X_train, y_train)
        logger.info("  Sklearn classifier fitted.")

        # 2. 在 train 上评估（作为 train_metrics）
        y_train_pred = self.model.final_layer.clf.predict(X_train)
        try:
            y_train_prob = self.model.final_layer.clf.predict_proba(X_train)[:, 1]
        except (AttributeError, IndexError):
            y_train_prob = y_train_pred.astype(float)

        train_metrics = {
            "loss": 0.0,  # sklearn 无 loss
            "balanced_acc": float(balanced_accuracy_score(y_train, y_train_pred)),
            "auroc": float(roc_auc_score(y_train, y_train_prob))
                if len(set(y_train)) > 1 else 0.5,
        }

        # 3. 在 test 上评估
        X_test, y_test = self._extract_all_embeddings(
            self.model, test_loader, self.device
        )
        logger.info(
            f"  Test embeddings: {X_test.shape}, labels: {np.bincount(y_test).tolist()}"
        )

        y_test_pred = self.model.final_layer.clf.predict(X_test)
        try:
            y_test_prob = self.model.final_layer.clf.predict_proba(X_test)[:, 1]
        except (AttributeError, IndexError):
            y_test_prob = y_test_pred.astype(float)

        test_metrics = {
            "loss": 0.0,
            "balanced_acc": float(balanced_accuracy_score(y_test, y_test_pred)),
            "f1": float(f1_score(y_test, y_test_pred, average="weighted")),
        }
        if len(set(y_test)) > 1:
            test_metrics["auroc"] = float(roc_auc_score(y_test, y_test_prob))
        else:
            test_metrics["auroc"] = 0.5

        logger.info(
            f"Final test: bal_acc={test_metrics['balanced_acc']:.4f} "
            f"auroc={test_metrics.get('auroc', 'N/A'):.4f}"
        )

        # 保存 checkpoint (sklearn 头不保存 PyTorch state)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history = [{"epoch": 1, "train": train_metrics, "lr": 0.0}]

        # 保存 sklearn 模型用 joblib
        try:
            import joblib
            joblib.dump(self.model.final_layer.clf, str(checkpoint_dir / "sklearn_head.joblib"))
            logger.info(f"Sklearn model saved → {checkpoint_dir / 'sklearn_head.joblib'}")
        except ImportError:
            logger.warning("joblib not installed, sklearn model not saved")

        final_info = {
            "best_epoch": 1,
            "best_train_metric": train_metrics["balanced_acc"],
            "final_test_metrics": test_metrics,
        }
        history_path = checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump({"epochs": history, "final": final_info}, f, indent=2, default=str)

        # 仍然保存 config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "name": self.config.name,
                    "datasets": self.config.datasets,
                    "mode": self.config.training.mode,
                    "head_type": self.config.training.head_type,
                    "best_metric": train_metrics["balanced_acc"],
                    "best_epoch": 1,
                },
                f, indent=2, default=str,
            )

        return history

    def fit(self, train_loader, test_loader, checkpoint_dir: Path):
        """完整训练流程。

        训练固定 epoch 数（不做早停），训练结束后在 test 上最终评估。
        每个 epoch 在 train 上计算指标用于监控，最佳 checkpoint 基于 train 指标保存。
        """
        # sklearn 头走单独的 fit 路径
        if isinstance(self.model.final_layer, SklearnHead):
            return self._fit_sklearn(train_loader, test_loader, checkpoint_dir)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        history = []

        for epoch in range(self.training_cfg.epochs):
            # 训练 + 在 train 上计算指标（仅用于监控）
            train_metrics = self.train_epoch(train_loader)

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(train_metrics[self.config.training.monitor_metric])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # 记录
            epoch_info = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "lr": current_lr,
            }
            history.append(epoch_info)

            logger.info(
                f"Epoch {epoch+1:3d}/{self.training_cfg.epochs} | "
                f"lr={current_lr:.2e} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['balanced_acc']:.4f}"
            )

            # 检查是否最佳（基于 train 指标，仅用于保存 checkpoint）
            monitor_val = train_metrics[self.config.training.monitor_metric]
            is_best = (
                monitor_val > self.best_metric
                if self.config.training.monitor_mode == "max"
                else monitor_val < self.best_metric
            )

            if is_best:
                self.best_metric = monitor_val
                self.best_epoch = epoch + 1
                self._save_checkpoint(checkpoint_dir, epoch + 1, train_metrics, is_best=True)
                logger.info(f"  ✓ New best model! {self.config.training.monitor_metric}={monitor_val:.4f}")

        # 训练完成后，在 test 上最终评估
        logger.info("Training complete. Running final evaluation on test set...")
        test_metrics = self.validate_epoch(test_loader)
        logger.info(
            f"Final test: loss={test_metrics['loss']:.4f} "
            f"bal_acc={test_metrics['balanced_acc']:.4f} "
            f"auroc={test_metrics.get('auroc', 'N/A'):.4f}"
        )

        # 保存训练历史和最终 test 指标
        final_info = {
            "best_epoch": self.best_epoch,
            "best_train_metric": self.best_metric,
            "final_test_metrics": test_metrics,
        }
        history_path = checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump({"epochs": history, "final": final_info}, f, indent=2, default=str)

        logger.info(f"Training complete. Best train {self.config.training.monitor_metric}={self.best_metric:.4f} at epoch {self.best_epoch}")
        return history

    def _save_checkpoint(
        self, checkpoint_dir: Path, epoch: int, metrics: Dict, is_best: bool = False
    ):
        """保存模型检查点。"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        if is_best:
            path = checkpoint_dir / "best_model.pt"
        else:
            path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"

        torch.save(checkpoint, path)

        # 删除旧的非最佳 checkpoint
        if not is_best:
            old_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            for old in old_checkpoints[:-2]:  # 保留最近 2 个
                old.unlink()


# ================================================================
# 主训练函数
# ================================================================


def train(config: ExperimentConfig):
    """执行完整训练流程。

    Args:
        config: 实验配置。
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        logger.warning(
            "⚠️  No GPU detected! REVE has 69M parameters and will be extremely slow on CPU.\n"
            "   For practical training, a GPU (CUDA) is strongly recommended.\n"
            "   For quick smoke testing, use --quick to limit data to 100 samples."
        )

    logger.info(f"Using device: {device}")
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Datasets: {config.datasets}")

    # 创建数据加载器
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        labels_csv=config.data.labels_csv,
        include_diagnosis=config.data.include_diagnosis or None,
        include_datasets=config.datasets or None,
        exclude_datasets=config.data.exclude_datasets or None,
        max_channels=config.data.max_channels,
        max_epochs_per_subject=config.data.max_epochs_per_subject,
        batch_size=config.training.batch_size,
        seed=config.data.seed,
        num_workers=config.data.num_workers,
        project_root=config.data.project_root,
    )

    train_dataset = dataloaders["train_dataset"]
    test_dataset = dataloaders["test_dataset"]

    logger.info(
        f"Train: {len(train_dataset)} epochs, {len(train_dataset.train_subjects)} subjects"
    )
    logger.info(
        f"Test: {len(test_dataset)} epochs, {len(test_dataset.test_subjects)} subjects"
    )

    # 根据数据自动设置输出类别数
    n_classes = train_dataset.n_classes
    config.model.n_outputs = n_classes
    logger.info(f"Binary classification, {n_classes} classes")

    # 构建模型
    logger.info("Building model...")
    model = build_model(
        config.model, device,
        n_classes=n_classes,
        training_config=config.training,
    )

    # 设置训练模式
    if config.training.mode == "linear_probe":
        setup_linear_probe(model, config.training)
    elif config.training.mode == "finetune":
        setup_finetune(model, config.training)

    # 训练
    class_weights = (
        train_dataset.class_weights if config.training.use_class_weights else None
    )
    trainer = Trainer(model, config, device, class_weights=class_weights)
    history = trainer.fit(
        dataloaders["train"],
        dataloaders["test"],
        config.checkpoint_dir,
    )

    # 保存配置
    config_path = config.checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "name": config.name,
                "datasets": config.datasets,
                "mode": config.training.mode,
                "epochs": config.training.epochs,
                "lr": config.training.lr,
                "batch_size": config.training.batch_size,
                "best_val_metric": trainer.best_metric,
                "best_epoch": trainer.best_epoch,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Results saved to {config.checkpoint_dir}")
    return model, history


# ================================================================
# CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="REVE EEG Model Training")

    # 实验预设
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=["depression", "adhd"],
        help="预定义实验: depression (HC vs MDD) 或 adhd (HC vs ADHD)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单数据集名称 (如 MODMA, TDBRAIN)",
    )

    # 训练模式
    parser.add_argument(
        "--mode",
        type=str,
        default="linear_probe",
        choices=["linear_probe", "finetune"],
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="linear",
        choices=[
            "linear", "mlp", "cnn1d", "attention",
            "sklearn_lr", "sklearn_rf", "sklearn_svm_rbf", "sklearn_svm_linear",
            "sklearn_knn", "sklearn_gbdt", "sklearn_adaboost",
            "sklearn_xgb", "sklearn_lgbm",
        ],
        help="分类头类型: linear, mlp, cnn1d, attention, sklearn_lr, sklearn_rf, ...",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--amp", action="store_true", default=False,
                        help="启用混合精度训练 (fp16，节省显存)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="梯度累积步数（小batch_size时用于保持effective batch size）")

    # 模型
    parser.add_argument(
        "--model",
        type=str,
        default="reve",
        choices=["reve"],
    )

    # 设备
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    # 数据
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="逗号分隔的数据集列表，如 'MODMA,IEEE_ADHD'",
    )
    parser.add_argument(
        "--diagnosis",
        type=str,
        default=None,
        choices=["depression", "adhd"],
        help="疾病类型: depression (HC vs MDD) 或 adhd (HC vs ADHD)",
    )
    parser.add_argument(
        "--max_channels", type=int, default=-1,
        help="最大通道数 (-1=不限制, 3=只取3通道数据)",
    )

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--name", type=str, default=None, help="实验名称")

    # 离线模式 / 镜像
    parser.add_argument("--local_files_only", action="store_true", help="只使用本地缓存")
    parser.add_argument("--hf_mirror", type=str, default=None,
                        help="HuggingFace 镜像 (如 https://hf-mirror.com)")

    # 快速验证模式
    parser.add_argument("--quick", action="store_true",
                        help="快速验证：仅用 1 epoch + 少量数据")

    return parser.parse_args()


def main():
    args = parse_args()

    # 选择配置
    if args.experiment == "depression":
        config = get_exp_depression()
    elif args.experiment == "adhd":
        config = get_exp_adhd()
    elif args.dataset and args.diagnosis:
        config = get_exp_single_dataset(args.dataset, args.diagnosis)
    elif args.dataset:
        config = get_exp_single_dataset(args.dataset)
    else:
        # 手动构建配置
        datasets = []
        if args.datasets:
            datasets = [d.strip() for d in args.datasets.split(",")]

        config = ExperimentConfig(
            name=args.name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            datasets=datasets if datasets else [],
            training=TrainingConfig(
                mode=args.mode,
                head_type=args.head_type,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            ),
            output_dir=args.output_dir,
            device=args.device,
        )
        config.data.num_workers = args.num_workers
        config.model.local_files_only = args.local_files_only
        config.training.head_type = args.head_type
        config.training.use_amp = args.amp
        config.training.gradient_accumulation_steps = args.grad_accum
        if args.diagnosis:
            config.data.include_diagnosis = ["control", args.diagnosis.strip()]

    # 快速验证模式
    if args.quick:
        logger.info("Quick mode: limiting data, 1 epoch")
        config.training.epochs = 1
        config.training.batch_size = 16
        config.data.max_epochs_per_subject = 1
        config.data.exclude_datasets = ["TDBRAIN"]
        config.model.pretrained = True
        config.model.local_files_only = True
        config.name = (config.name or "exp") + "_quick"

    # HF 镜像
    if args.hf_mirror:
        config.model.hf_endpoint = args.hf_mirror

    # 覆盖预设参数（对所有配置路径生效）
    if args.datasets and args.experiment:
        config.datasets = [d.strip() for d in args.datasets.split(",")]
    if args.name:
        config.name = args.name
        config.checkpoint_dir = config.output_root / "checkpoints" / config.name
        config.log_dir = config.output_root / "logs" / config.name
        config.result_dir = config.output_root / "results" / config.name
    if args.max_channels > 0:
        config.data.max_channels = args.max_channels
    if args.diagnosis:
        config.data.include_diagnosis = ["control", args.diagnosis.strip()]
    # head_type 覆盖 (对所有配置路径生效)
    if args.head_type:
        config.training.head_type = args.head_type

    train(config)


if __name__ == "__main__":
    main()
