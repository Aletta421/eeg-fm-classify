"""
模型训练配置文件

所有可调参数集中管理，支持命令行覆盖。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal


@dataclass
class DataConfig:
    """数据加载配置。"""

    # 标签文件路径（相对于项目根目录）
    labels_csv: str = "data/labels.csv"

    # 随机种子（用于 subject-wise split）
    seed: int = 42

    # 排除的数据集（可选）
    exclude_datasets: List[str] = field(default_factory=list)

    # 只包含特定的 diagnosis_type:
    #   ["control", "depression"] → 抑郁症二分类
    #   ["control", "adhd"]       → ADHD 二分类
    #   空列表 = 使用所有数据 (HC + 全部患者类型)
    include_diagnosis: List[str] = field(default_factory=list)

    # 每个受试者最多使用的 epoch 数（-1 表示全部使用）
    max_epochs_per_subject: int = -1

    # 最大通道数（-1 表示不限制，>0 过滤掉大于该值的文件）
    max_channels: int = -1

    # 数据加载子进程数
    num_workers: int = 4

    def __post_init__(self):
        # 确保 labels_csv 是相对于项目根目录的路径
        self.project_root = Path(__file__).parent.parent


@dataclass
class ModelConfig:
    """REVE 模型配置。"""

    # 预训练模型 ID
    model_id: str = "brain-bzh/reve-base"

    # 输出类别数（二分类）
    n_outputs: int = 2

    # 是否使用注意力池化（支持可变通道数/时间长度）
    use_attention_pooling: bool = True

    # 输入窗口秒数（10s）
    input_window_seconds: float = 10.0

    # 采样率
    sfreq: float = 200.0

    # 输入时间点数
    n_times: int = 2000

    # 是否加载预训练权重
    pretrained: bool = True

    # 是否强制下载
    force_download: bool = False

    # 离线模式（只使用本地缓存）
    local_files_only: bool = False

    # HuggingFace token
    hf_token: Optional[str] = None

    # HuggingFace 镜像端点（国内用户: https://hf-mirror.com）
    hf_endpoint: Optional[str] = None


@dataclass
class TrainingConfig:
    """训练配置。"""

    # 训练模式
    mode: Literal["linear_probe", "finetune"] = "linear_probe"

    # 分类头类型 (替换 REVE 的 final_layer)
    #   神经网络: linear, mlp, cnn1d, attention
    #   sklearn:  sklearn_lr, sklearn_rf, sklearn_svm_rbf, sklearn_svm_linear,
    #             sklearn_knn, sklearn_gbdt, sklearn_adaboost, sklearn_xgb, sklearn_lgbm
    head_type: str = "linear"

    # MLP 头参数
    head_hidden_dim: int = 256
    head_num_layers: int = 2
    head_dropout: float = 0.3
    head_use_batchnorm: bool = True

    # 训练轮数
    epochs: int = 20

    # 批次大小
    batch_size: int = 256

    # 学习率
    lr: float = 1e-3

    # 权重衰减
    weight_decay: float = 1e-4

    # 优化器
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"

    # 学习率调度 (None=恒定学习率)
    lr_scheduler: Optional[Literal["cosine", "step", "plateau"]] = None

    # 学习率预热轮数
    warmup_epochs: int = 0

    # 梯度裁剪
    grad_clip: float = 1.0

    # 微调时解冻的层数（仅 mode="finetune" 时有效）
    # -1 表示解冻全部层
    unfreeze_layers: int = -1

    # 微调时 encoder 的学习率乘数（仅 mode="finetune" 时有效）
    encoder_lr_mult: float = 0.01

    # 混合精度训练
    use_amp: bool = False

    # 类别权重（处理不平衡数据）
    use_class_weights: bool = True

    # 梯度累积步数（用于小 batch 模拟大批次）
    gradient_accumulation_steps: int = 1

    # 早停（已弃用：当前训练固定 epoch，不做早停）
    early_stopping_patience: int = 0

    # 评估指标
    monitor_metric: str = "balanced_acc"

    # 指标方向
    monitor_mode: Literal["min", "max"] = "max"


@dataclass
class ExperimentConfig:
    """实验配置（组合所有子配置）。"""

    # 实验名称
    name: str = "exp_001_single_modma"

    # 使用的数据集列表
    datasets: List[str] = field(default_factory=list)

    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # 输出目录
    output_dir: str = "outputs"

    # 设备
    device: str = "cuda"

    # 日志间隔（步数）
    log_interval: int = 50

    def __post_init__(self):
        self.output_root = Path(self.output_dir)
        self.checkpoint_dir = self.output_root / "checkpoints" / self.name
        self.log_dir = self.output_root / "logs" / self.name
        self.result_dir = self.output_root / "results" / self.name


# ================================================================
# 预定义实验模板
# ================================================================

def get_exp_depression() -> ExperimentConfig:
    """抑郁症二分类: HC vs MDD (MODMA + OpenNeuro + TDBRAIN 的 depression 数据)。"""
    return ExperimentConfig(
        name="exp_depression",
        datasets=["MODMA", "OpenNeuro", "TDBRAIN"],
        data=DataConfig(include_diagnosis=["control", "depression"]),
        training=TrainingConfig(mode="linear_probe", epochs=20, lr=1e-3, batch_size=256),
    )


def get_exp_adhd() -> ExperimentConfig:
    """ADHD 二分类: HC vs ADHD (IEEE + Mendeley + TDBRAIN 的 ADHD 数据)。"""
    return ExperimentConfig(
        name="exp_adhd",
        datasets=["IEEE_ADHD", "Mendeley", "TDBRAIN"],
        data=DataConfig(include_diagnosis=["control", "adhd"]),
        training=TrainingConfig(mode="linear_probe", epochs=20, lr=1e-3, batch_size=256),
    )


def get_exp_single_dataset(dataset: str, diagnosis: str = None) -> ExperimentConfig:
    """单数据集训练。

    Args:
        dataset: 数据集名称 (MODMA, IEEE_ADHD, Mendeley, OpenNeuro, TDBRAIN)
        diagnosis: "depression" 或 "adhd"，None = 全部患者类型
    """
    diag_types = ["control", diagnosis] if diagnosis else None
    return ExperimentConfig(
        name=f"exp_{dataset.lower()}" + (f"_{diagnosis}" if diagnosis else ""),
        datasets=[dataset],
        data=DataConfig(include_diagnosis=diag_types),
        training=TrainingConfig(mode="linear_probe", epochs=20, lr=1e-3, batch_size=256),
    )
