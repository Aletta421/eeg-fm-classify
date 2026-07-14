"""
自定义神经网络分类头

用于替换 REVE 的 final_layer (Sequential(LayerNorm, Linear(512, n_outputs)))。
所有 head 接受 (B, 512) embedding，输出 (B, n_classes) logits。

用法:
    from models.heads import create_head
    head = create_head("mlp", in_features=512, n_classes=2,
                       hidden_dim=256, num_layers=2, dropout=0.3)
    model.final_layer = head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, List


# ================================================================
# 基础类
# ================================================================


class BaseHead(nn.Module):
    """分类头抽象基类。

    所有子类需实现 forward(x) 其中 x.shape = (B, in_features)。
    """

    def __init__(self, in_features: int = 512, n_classes: int = 2):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes

    def reset_parameters(self):
        """初始化参数（子类可覆盖）。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ================================================================
# LinearHead — 等价原始 REVE 分类头
# ================================================================


class LinearHead(BaseHead):
    """单层线性分类头（等同 REVE 原始 final_layer）。

    REVE 原始: Sequential(LayerNorm(512), Linear(512, n_outputs))
    """

    def __init__(self, in_features: int = 512, n_classes: int = 2, **kwargs):
        super().__init__(in_features, n_classes)
        self.norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# ================================================================
# MLPHead — 多层感知机分类头
# ================================================================


class MLPHead(BaseHead):
    """2-3 层 MLP 分类头。

    Architecture: LayerNorm → Linear → ReLU → [BN?] → [Dropout] → ... → Linear

    Args:
        in_features: 输入维度 (REVE 默认 512)
        n_classes: 输出类别数
        hidden_dim: 隐藏层维度 (默认 256)
        num_layers: MLP 层数（含最终输出层，即 ≥2）
        dropout: Dropout 比例 (默认 0.3)
        use_batchnorm: 是否在每层后加 BatchNorm1d
    """

    def __init__(
        self,
        in_features: int = 512,
        n_classes: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        **kwargs,
    ):
        super().__init__(in_features, n_classes)
        assert num_layers >= 2, "MLPHead requires at least 2 layers"

        layers: List[nn.Module] = [nn.LayerNorm(in_features)]

        prev_dim = in_features
        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 最终输出层
        layers.append(nn.Linear(prev_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ================================================================
# CNN1DHead — 1D 卷积分类头
# ================================================================


class CNN1DHead(BaseHead):
    """1D CNN 分类头 — 将 (B, 512) 视为 (B, 1, 512) 做时序卷积。

    Architecture:
        reshape → Conv1d blocks → AdaptiveAvgPool1d → Linear

    Args:
        in_features: 输入维度 (512)
        n_classes: 输出类别数
        conv_channels: 各层输出通道列表 (如 [64, 128])
        kernel_sizes: 各层 kernel size (如 [7, 5])
        dropout: Dropout 比例
    """

    def __init__(
        self,
        in_features: int = 512,
        n_classes: int = 2,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_features, n_classes)

        if conv_channels is None:
            conv_channels = [64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5]

        assert len(conv_channels) == len(kernel_sizes), (
            f"conv_channels and kernel_sizes must have same length, "
            f"got {len(conv_channels)} vs {len(kernel_sizes)}"
        )

        conv_layers: List[nn.Module] = []
        in_ch = 1  # 将 512 视为单通道
        for i, (out_ch, ks) in enumerate(zip(conv_channels, kernel_sizes)):
            padding = ks // 2
            conv_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=padding, bias=False)
            )
            conv_layers.append(nn.BatchNorm1d(out_ch))
            conv_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 计算 flatten 后的维度
        self.fc = nn.Linear(conv_channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 512) → (B, 1, 512)
        x = x.unsqueeze(1)
        x = self.conv(x)        # (B, conv_channels[-1], 512)
        x = self.pool(x)        # (B, conv_channels[-1], 1)
        x = x.squeeze(-1)       # (B, conv_channels[-1])
        return self.fc(x)


# ================================================================
# AttentionHead — 自注意力分类头
# ================================================================


class AttentionHead(BaseHead):
    """轻量 Self-Attention 分类头。

    将 (B, 512) reshape 为 (B, num_heads, head_dim) 序列，
    做完 self-attention 后 pooling + Linear。

    Args:
        in_features: 输入维度 (512)
        n_classes: 输出类别数
        num_heads: 注意力头数 (必须整除 in_features)
        ff_dim: 前馈网络隐藏维度
        dropout: Dropout 比例
    """

    def __init__(
        self,
        in_features: int = 512,
        n_classes: int = 2,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(in_features, n_classes)
        assert in_features % num_heads == 0, (
            f"in_features ({in_features}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        self.norm1 = nn.LayerNorm(in_features)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(in_features)
        self.ff = nn.Sequential(
            nn.Linear(in_features, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, in_features),
            nn.Dropout(dropout),
        )

        self.norm_out = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) → (B, 1, D)
        x = x.unsqueeze(1)

        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual

        # Pool & classify
        x = x.squeeze(1)       # (B, D)
        x = self.norm_out(x)
        return self.fc(x)


# ================================================================
# 工厂函数
# ================================================================

HEAD_REGISTRY = {
    "linear": LinearHead,
    "mlp": MLPHead,
    "cnn1d": CNN1DHead,
    "attention": AttentionHead,
}


# ================================================================
# SklearnHead — sklearn 分类器包装为 head
# ================================================================


class SklearnHead(BaseHead):
    """将 sklearn 分类器包装为 REVE 分类头。

    由 train.py 在训练前提取全量 embedding，调用 clf.fit()，
    之后 forward() 走 clf.predict_proba() / clf.predict()。

    参数通过 build_sklearn_head() 工厂函数自动构建。
    """

    def __init__(self, clf, in_features: int = 512, n_classes: int = 2):
        super().__init__(in_features, n_classes)
        self.clf = clf
        self._fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        if self._fitted:
            proba = self.clf.predict_proba(x_np)
            return torch.from_numpy(proba).float().to(x.device)
        else:
            # 未拟合时（如 embedding 提取阶段）：返回伪 logits
            # hook 已经捕获了输入，这里的输出仅用于让 forward 走通
            return torch.zeros(x.shape[0], self.n_classes, device=x.device)

    def fit(self, X: "np.ndarray", y: "np.ndarray"):
        """用全量 embedding 拟合 sklearn 分类器。"""
        self.clf.fit(X, y)
        self._fitted = True

    def count_params(self) -> int:
        return 0  # sklearn 分类器不算 PyTorch 参数


# ---- sklearn 工厂函数 ----

def _build_sklearn_lr(in_features: int = 512, n_classes: int = 2,
                      random_state: int = 42, **kwargs) -> SklearnHead:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, random_state=random_state,
    )
    return SklearnHead(clf, in_features, n_classes)


def _build_sklearn_rf(in_features: int = 512, n_classes: int = 2,
                      random_state: int = 42, **kwargs) -> SklearnHead:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_leaf=5, random_state=random_state, n_jobs=-1,
    )
    return SklearnHead(clf, in_features, n_classes)


def _build_sklearn_svm_rbf(in_features: int = 512, n_classes: int = 2,
                           random_state: int = 42, **kwargs) -> SklearnHead:
    from sklearn.svm import SVC
    clf = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, random_state=random_state,
    )
    return SklearnHead(clf, in_features, n_classes)


def _build_sklearn_svm_linear(in_features: int = 512, n_classes: int = 2,
                              random_state: int = 42, **kwargs) -> SklearnHead:
    from sklearn.svm import SVC
    clf = SVC(
        kernel="linear", C=1.0,
        probability=True, random_state=random_state,
    )
    return SklearnHead(clf, in_features, n_classes)


def _build_sklearn_knn(in_features: int = 512, n_classes: int = 2, **kwargs) -> SklearnHead:
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    return SklearnHead(clf, in_features, n_classes)


def _build_sklearn_gbdt(in_features: int = 512, n_classes: int = 2,
                        random_state: int = 42, **kwargs) -> SklearnHead:
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=random_state,
    )
    return SklearnHead(clf, in_features, n_classes)


def _build_sklearn_adaboost(in_features: int = 512, n_classes: int = 2,
                            random_state: int = 42, **kwargs) -> SklearnHead:
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5,
        random_state=random_state,
    )
    return SklearnHead(clf, in_features, n_classes)


def _build_sklearn_xgb(in_features: int = 512, n_classes: int = 2,
                       random_state: int = 42, **kwargs) -> SklearnHead:
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=random_state,
        )
        return SklearnHead(clf, in_features, n_classes)
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")


def _build_sklearn_lgbm(in_features: int = 512, n_classes: int = 2,
                        random_state: int = 42, **kwargs) -> SklearnHead:
    try:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=random_state,
        )
        return SklearnHead(clf, in_features, n_classes)
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")


# 注册所有 sklearn head 类型
SKLEARN_HEAD_BUILDERS = {
    "sklearn_lr":         _build_sklearn_lr,
    "sklearn_rf":         _build_sklearn_rf,
    "sklearn_svm_rbf":    _build_sklearn_svm_rbf,
    "sklearn_svm_linear": _build_sklearn_svm_linear,
    "sklearn_knn":        _build_sklearn_knn,
    "sklearn_gbdt":       _build_sklearn_gbdt,
    "sklearn_adaboost":   _build_sklearn_adaboost,
    "sklearn_xgb":        _build_sklearn_xgb,
    "sklearn_lgbm":       _build_sklearn_lgbm,
}

HEAD_REGISTRY.update(SKLEARN_HEAD_BUILDERS)


def create_head(
    head_type: str,
    in_features: int = 512,
    n_classes: int = 2,
    **kwargs,
) -> BaseHead:
    """创建分类头实例。

    Args:
        head_type: 神经网络头 (linear, mlp, cnn1d, attention)
                   或 sklearn 头 (sklearn_lr, sklearn_rf, sklearn_svm_rbf, ...)
        in_features: 输入 embedding 维度
        n_classes: 输出类别数
        **kwargs: 传递给具体 head 的参数

    Returns:
        BaseHead 子类实例

    Raises:
        ValueError: 未知 head_type
    """
    head_type = head_type.lower().strip()
    if head_type not in HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head_type '{head_type}'. "
            f"Available: {sorted(HEAD_REGISTRY.keys())}"
        )

    builder = HEAD_REGISTRY[head_type]
    return builder(in_features=in_features, n_classes=n_classes, **kwargs)


# ================================================================
# 快速测试
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing custom classification heads")
    print("=" * 60)

    batch_size = 4
    in_features = 512
    n_classes = 2
    x = torch.randn(batch_size, in_features)

    for head_type in ["linear", "mlp", "cnn1d", "attention"]:
        head = create_head(head_type, in_features, n_classes)
        out = head(x)
        params = head.count_params()
        print(f"\n{head_type}:")
        print(f"  Input:  {x.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Params: {params:,}")
        assert out.shape == (batch_size, n_classes), (
            f"Shape mismatch: expected {(batch_size, n_classes)}, got {out.shape}"
        )

    # 测试 reset_parameters
    print("\n--- reset_parameters ---")
    for head_type in ["linear", "mlp", "cnn1d", "attention"]:
        head = create_head(head_type, in_features, n_classes)
        head.reset_parameters()
        print(f"  {head_type}: OK")

    print("\n[OK] All heads pass forward + reset test")
