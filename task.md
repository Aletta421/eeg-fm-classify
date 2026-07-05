# 任务文档 — Phase 2: REVE 模型训练

> 状态：Phase 1 数据预处理 ✅ | Phase 2 待开始

---

## Phase 1 完成情况

### 数据预处理 ✅

所有原始数据已处理为统一格式：**200 Hz, 10s 窗口, z-score 归一化**

| 数据集 | 受试者 | 文件 | 磁盘 | 标签 |
|--------|--------|------|------|------|
| MODMA | 68 | 161 | 11.7 GB | 87 对照 + 74 抑郁 |
| IEEE ADHD | 121 | 121 | 473 MB | 60 对照 + 61 ADHD |
| Mendeley | 80 | 880 | 73 MB | 42 对照 + 38 ADHD |
| OpenNeuro | 120 | 239 | 7.8 GB | 75 对照 + 45 抑郁 |
| TDBRAIN | 1,046 | 1,175 | 6.7 GB | 308 对照 + 345 抑郁 + 217 ADHD + 176 其他 |
| testdata | 60 | 120 | 703 MB | 30 对照 + 30 抑郁 |
| **总计** | **1,495** | **2,696** | **27.3 GB** | |

### 输出格式

```
data/processed/
├── MODMA/              # resting_3ch, resting_128ch, erp_128ch
├── IEEE_ADHD/          # ADHD_part1/2, Control_part1/2
├── Mendeley/           # 11 tasks × 80 subjects
├── OpenNeuro/          # run01/run02 × 120 subjects
├── TDBRAIN/            # restEC + oddball × 1046 subjects
└── testdata/           # restEC + restEO × 60 subjects (独立评估集)

每个目录下:
  {subject_id}_eeg.npy     # (n_epochs, n_channels, 2000) float64
  {subject_id}_ch_pos.npy  # (n_channels, 3) 电极3D坐标
  {subject_id}_meta.json   # 元数据 (标签/采样率/通道名等)

全局:
  data/labels.csv          # 2,696 行，索引所有处理后的文件
```

---

## Phase 2 目标

### 核心任务

使用 **REVE** 预训练模型对 EEG 数据进行抑郁症/ADHD 二分类。

### 目标指标

| 指标 | 目标 |
|------|------|
| 平衡准确率 | ≥ 80% |
| AUROC | ≥ 0.85 |
| 跨数据集泛化 | 训练集 vs testdata 性能差距 ≤ 10% |

---

## 任务分解

### 2.1 环境搭建

```bash
pip install braindecode[hug]   # REVE 模型 + HuggingFace Hub
pip install mne                 # EEG 处理
pip install torch torchvision   # PyTorch
pip install huggingface_hub     # 下载预训练权重
```

注册 HuggingFace 账号，登录后获取 `brain-bzh/reve-base` 访问权限：
```bash
huggingface-cli login
```

### 2.2 数据加载器

需要实现一个 PyTorch Dataset，能够：

1. 读取 `data/labels.csv`，遍历所有文件
2. 每次返回 `(eeg, pos, label)` 三元组
3. EEG shape: `(n_channels, 2000)` — 单个 epoch
4. pos shape: `(n_channels, 3)` — 电极 3D 坐标
5. 按受试者划分 train/val/test，避免同一受试者的 epochs 跨集合泄露
6. testdata 始终作为独立测试集

**关键点：每个 .npy 文件包含多个 epochs（如 12 epochs），Dataset 应该展开为单个 epoch。**

### 2.3 通道位置对齐

REVE 的位置编码需要标准电极名称来查找 3D 坐标。不同数据集的通道命名体系不同：

| 数据集 | 通道命名 | 对齐方式 |
|--------|---------|---------|
| MODMA 128ch | E1-E128 (EGI) | `biosemi128_` 前缀 → REVE position bank |
| MODMA 3ch | Fp1, Fpz, Fp2 | 标准 10-20 名称，直接匹配 |
| IEEE ADHD | 标准 10-20 | 直接匹配 |
| Mendeley | Cz, F4, O1, F3, Fz | 标准 10-20，直接匹配 |
| OpenNeuro | 标准 10-20 大写 | 直接匹配 |
| TDBRAIN / testdata | 标准 10-20 | 直接匹配 |

```python
from braindecode.models import REVE

model = REVE.from_pretrained("brain-bzh/reve-base")

# 方式一：使用位置坐标 (n_channels, 3)
output = model(eeg_batch, pos=pos_batch)

# 方式二：使用电极名称 (str)
output = model(eeg_batch, pos=["C3", "C4", "Fz", ...])
```

### 2.4 模型加载与配置

```python
model = REVE.from_pretrained(
    "brain-bzh/reve-base",
    n_outputs=2,          # 二分类
    use_attention_pooling=True,  # 可变长度输入
)
```

**线性探测策略（推荐先试）：**
- 冻结 REVE encoder 所有参数
- 仅训练最后的分类头
- 学习率 1e-3 ~ 1e-4，AdamW
- 验证效果后，可选微调最后几层

### 2.5 训练策略

#### 实验 1：单数据集线性探测

逐数据集验证 REVE 特征质量：

```
数据: MODMA, IEEE ADHD, TDBRAIN
训练: 80%, 验证: 20%
epochs: 20
lr: 1e-3 (分类头)
batch_size: 256
```

#### 实验 2：多数据集混合训练

合并所有可用数据，统一训练：

```
数据: MODMA + IEEE + Mendeley + OpenNeuro + TDBRAIN
训练: 80%, 验证: 20%
testdata: 始终独立
epochs: 20
lr: 1e-3
```

#### 实验 3：跨数据集泛化测试

```
训练: MODMA + OpenNeuro (抑郁症)
测试: testdata (抑郁症)
评估跨数据集泛化能力
```

### 2.6 评估指标

```python
from sklearn.metrics import (
    balanced_accuracy_score,  # 平衡准确率 (核心指标)
    roc_auc_score,            # AUROC
    f1_score,                 # F1
    classification_report,    # 分类报告
)
```

每轮实验记录：

```yaml
experiment_name: exp_001_single_modma
model: REVE-base
mode: linear_probe
datasets: [MODMA]
epochs: 20
lr: 1e-3
batch_size: 256
val_balanced_acc: 0.xxx
val_auroc: 0.xxx
```

### 2.7 消融实验（可选）

- 不同 epoch 数量对性能的影响
- 注意力池化 vs. 全局平均池化
- 是否冻结 encoder vs. 微调最后 N 层
- 使用原始采样率 vs 统一 200 Hz（当前已统一）

---

## 文件结构

```
eeg/
├── task.md                          # 本文件
├── README.md
├── data/labels.csv                  # 标签索引
│
├── data/                            # 原始数据 (gitignore)
├── data/processed/                  # 预处理后数据 (gitignore)
│
├── preprocessing/                   # Phase 1 的预处理脚本
├── scripts/
│   ├── run_preprocess.py            # 一键预处理
│   └── data_stats.py                # 数据统计
│
├── models/                          # Phase 2 模型代码 (待实现)
│   ├── dataset.py                   # EEG Dataset
│   ├── train.py                     # 训练脚本
│   ├── evaluate.py                  # 评估脚本
│   └── config.py                    # 模型/训练配置
│
├── configs/                         # 配置文件
├── outputs/                         # 训练产物 (gitignore)
│   ├── checkpoints/
│   ├── logs/
│   └── results/
└── docs/
```

---

## 下一步行动

1. [ ] 安装依赖 (`braindecode`, `huggingface_hub`)
2. [ ] 注册 HuggingFace 获取 REVE 权重访问
3. [ ] 实现 `models/dataset.py` — EEG 数据加载器
4. [ ] 实现 `models/train.py` — 线性探测训练
5. [ ] 运行实验 1：单数据集验证 REVE 效果
6. [ ] 运行实验 2：多数据集混合训练
7. [ ] 运行实验 3：testdata 跨数据集泛化

---

*Phase 1 完成于 2026-07-03，Phase 2 开始于 —*
