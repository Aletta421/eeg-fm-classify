# EEG Foundation Model — 抑郁症与ADHD分类预测

> 基于 REVE (NeurIPS 2025) 预训练模型实现抑郁症和 ADHD 的 EEG 二分类，目标准确率 ≥80%。

---

## 项目状态

| Phase | 内容 | 状态 |
|:---|:---|:---|
| 1. 数据预处理 | 下载、清洗、统一格式、生成标签 | ✅ 完成 |
| 2. 模型训练 | REVE 线性探测 → 多数据集混合 → 泛化测试 | 🔜 进行中 |

---

## 处理后数据总览

**格式：200 Hz, 10s 窗口, z-score 归一化, 2,696 个 .npy 文件**

| 数据集 | 受试者 | 文件 | 标签 |
|--------|--------|------|------|
| MODMA | 68 | 161 | 87 HC + 74 MDD |
| IEEE ADHD | 121 | 121 | 60 HC + 61 ADHD |
| Mendeley | 80 | 880 | 42 HC + 38 ADHD |
| OpenNeuro | 120 | 239 | 75 HC + 45 MDD |
| TDBRAIN | 1,046 | 1,175 | 308 HC + 345 MDD + 217 ADHD + ... |
| testdata | 60 | 120 | 30 HC + 30 MDD (独立评估) |
| **总计** | **1,495** | **2,696** | |

---

## 快速开始

### 1. 环境

```bash
python -m venv eeg_env
source eeg_env/Scripts/activate   # Windows Git Bash
# 或 source eeg_env/bin/activate  # Linux/macOS

pip install braindecode[hug] mne torch huggingface_hub
pip install numpy scipy openpyxl pyyaml scikit-learn
```

### 2. 下载数据

| 数据集 | 链接 | 疾病 |
|:---|:---|:---|
| MODMA | [modma.lzu.edu.cn](https://modma.lzu.edu.cn/data/index/) | 抑郁症 |
| IEEE DataPort | [ieee-dataport.org](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children) | ADHD 儿童 |
| Mendeley Data | [data.mendeley.com](https://data.mendeley.com/datasets/6k4g25fhzg/1) | ADHD 成人 |
| OpenNeuro ds003478 | [openneuro.org](https://openneuro.org/datasets/ds003478/versions/1.1.0/download) | 抑郁症 |
| TDBRAIN | [brainclinics.com](https://brainclinics.com/resources/) | 多病种 |

将数据放入 `data/` 对应子目录（见[协作指南](docs/collaboration.md)）。

### 3. 数据预处理

```bash
# 一键预处理全部数据集
python scripts/run_preprocess.py

# 仅处理指定数据集
python scripts/run_preprocess.py --datasets MODMA,IEEE

# 查看处理后统计
python scripts/data_stats.py
python scripts/data_stats.py --detail
```

### 4. 训练模型

```bash
cd models

# 线性探测（冻结 encoder，仅训练分类头）
python train.py --mode linear_probe --model reve --epochs 20 --lr 1e-3

# 微调最后几层
python train.py --mode finetune --model reve --epochs 50 --lr 1e-5
```

### 5. 评估

```bash
python evaluate.py --checkpoint ../outputs/checkpoints/best_model.ckpt --test_data ../data/processed/testdata
```

---

## 项目结构

```
eeg/
├── README.md                    # 本文件
├── task.md                      # 任务文档（Phase 2 详细计划）
├── requirements.txt             # Python 依赖
│
├── data/                        # 原始数据（gitignore）
│   ├── MODMA/
│   ├── IEEE_ADHD/
│   ├── Mendeley_ADHD/
│   ├── TDBRAIN/
│   ├── OpenNeuro_ds003478/
│   ├── processed/               # 预处理输出（gitignore）
│   │   ├── MODMA/
│   │   ├── IEEE_ADHD/
│   │   ├── Mendeley/
│   │   ├── OpenNeuro/
│   │   ├── TDBRAIN/
│   │   └── testdata/            # 独立评估集
│   └── labels.csv               # 标签索引文件
│
├── scripts/
│   ├── run_preprocess.py        # 一键预处理
│   └── data_stats.py            # 数据统计报告
│
├── preprocessing/               # 预处理 loader（数据集→npy）
│   ├── base_loader.py           # 基类（滤波/重采样/分段）
│   ├── load_modma.py
│   ├── load_ieee.py
│   ├── load_mendeley.py
│   ├── load_openneuro.py
│   ├── load_tdbrain.py
│   └── generate_labels.py
│
├── models/                      # 模型代码（Phase 2）
│   ├── dataset.py
│   ├── train.py
│   └── config.py
│
├── configs/                     # 配置文件
│   └── preprocess_config.yaml
│
├── outputs/                     # 训练产物（gitignore）
│   ├── checkpoints/
│   ├── logs/
│   └── results/
│
└── docs/
    └── collaboration.md
```

---

## 技术方案

### 为什么选 REVE

- **4D 位置编码**（电极 x,y,z + 时间 t），支持任意通道数（2→128），解决我们跨数据集通道不一致问题
- **最大规模预训练**：92 数据集、25,000 受试者、60,000 小时 EEG
- **开箱即用**：`brain-bzh/reve-base` on HuggingFace，Braindecode 集成
- NeurIPS 2025，10 个下游任务 SOTA

```
EEG [B, C, 2000] + pos [B, C, 3]
  → Patch 分割 → Linear Embedding
  → 4D Fourier 位置编码
  → Transformer (22层, 72M 参数)
  → 分类头
```

### 模型流程

```
原始 EEG (.bdf/.mat/.set)
  → 滤波器 (0.5-99.5 Hz 带通 + 50 Hz 陷波)
  → Z-score 归一化 (clip ±15σ)
  → 分段 (10s 不重叠窗口)
  → 输出: .npy (epochs, channels, 2000)
  → REVE encoder → 线性分类头 → 预测 (HC/MDD/ADHD)
```

---

## 实施路线图

| 阶段 | 任务 | 状态 |
|:---|:---|:---|
| **1. 数据准备** | 下载 5 个数据集、统一格式、生成标签 | ✅ 完成 |
| **2. 环境搭建** | 安装 braindecode、注册 HuggingFace | 🔜 |
| **3. 单集线性探测** | REVE 冻结 encoder，逐数据集验证 | 🔜 |
| **4. 多集混合训练** | 跨数据集联合训练，testdata 独立评估 | 🔜 |
| **5. 优化消融** | 微调、超参搜索、通道对齐优化 | 🔜 |

---

## 关键参考

| 文献 | 说明 |
|:---|:---|
| [REVE](https://arxiv.org/abs/2510.21585) | EEG Foundation Model，4D 位置编码，NeurIPS 2025 |
| [Braindecode](https://braindecode.org/) | EEG 深度学习库，REVE 模型集成 |
| [brain-bzh/reve-base](https://huggingface.co/brain-bzh/reve-base) | 预训练权重（HuggingFace） |

---

*项目初始化于 2026-06-30，Phase 1 完成于 2026-07-03。*
