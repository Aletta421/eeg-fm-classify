# 任务文档 — Phase 2: REVE 模型训练

> 状态：Phase 1 ✅ | 数据质量审计 ✅ | 死通道检测 ✅ | 重新训练 🔄

---

## 任务目标

对 EEG 数据进行 **二分类**（HC vs MDD / HC vs ADHD）和 **三分类**（HC / MDD / ADHD）

| 指标 | 目标 | 二分类最佳 | 三分类最佳 |
|------|------|-----------|-----------|
| 平衡准确率 | ≥ 80% | 0.940 (TDBRAIN-depression) | 0.751 (TDBRAIN) |
| AUROC | ≥ 0.85 | 0.969 (TDBRAIN-depression) | 0.926 (TDBRAIN) ✅ |
| 跨数据集泛化 | 差距 ≤ 10% | — | 待测 |

---

## 数据

### 处理后统计（2026-07-13 重新处理，新增 train/test split）

| 数据集 | 受试者 | 文件 | 磁盘 | 通道数 | Epochs | 标签分布 |
|--------|--------|------|------|--------|--------|---------|
| IEEE_ADHD | 121 | 121 | 236 MB | 19 | 6-33 | adhd=61, control=60 |
| Mendeley | 80 | 880 | 37 MB | 2 | 1-4 | adhd=418, control=462 |
| MODMA | 68 | 161 | 5.8 GB | 3-128 | 7-181 | control=87, depression=74 |
| OpenNeuro | 120 | 239 | 3.9 GB | 64 | 13-58 | control=149, depression=90 |
| TDBRAIN | 1,046 | 1,175 | 3.3 GB | 26 | 9-36 | control=308, depression=345, adhd=217, other=306 |
| **总计** | **1,435** | **2,576** | **13.3 GB** | | | |

- 统一 200 Hz, 10s 窗口 (2000 采样点)，无重叠
- Train/Test = 80/20，按受试者分层划分 (`data/splits.json`)
- Train: 2,005 文件 / Test: 571 文件

### 数据质量审计（2026-07-13）

逐文件、逐通道扫描 2,576 条记录，检测死通道（全记录 std < 0.01）：

| 数据集 | 评级 | 死通道% | 坏文件% | std 中位数 | 说明 |
|--------|------|---------|---------|-----------|------|
| IEEE_ADHD | 🟢 A | 0% | 0% | 1.000 | 质量最优，无任何问题 |
| MODMA | 🟢 A | 0% | 1% | 0.929 | 高质量，偶发单通道 |
| Mendeley | 🟢 A | 0.6% | 1% | 1.000 | 仅 FADHD_07 已知损坏 |
| OpenNeuro | 🔴 C | 12.5% | 35% | 0.968 | ⚠️ 84/239 文件有死通道 |
| TDBRAIN | 🟡 B | 0% | 0% | 0.745 | 无死通道，方差偏低（临床数据正常特征） |

死通道集中在 OpenNeuro：最严重文件 40/64 通道全平，均为原始采集电极接触不良。已在 `dataset.py` 中添加**死通道自动检测与置零**（`detect_dead_channels=True`，lazy 检测 + 缓存）。

三分类标签映射：`control → 0`, `depression → 1`, `adhd → 2`

---

## 数据处理流水线

### 7-Step 统一预处理

所有 5 个数据集经过**完全相同的信号处理流水线**（`preprocessing/base_loader.py`），确保不同来源 EEG 数据可比：

| 步骤 | 操作 | 参数 | 目的 |
|------|------|------|------|
| 1. 重采样 | `scipy.signal.resample` | 原始 128~500 Hz → **200 Hz** | 匹配 REVE 模型输入 |
| 2. 带通滤波 | 4 阶 Butterworth | **0.5–99.5 Hz** | 去除直流漂移和高频噪声 |
| 3. 陷波滤波 | IIR Notch, Q=30 | **50 Hz** | 去除工频干扰 |
| 4. Z-score 归一化 | per recording, per channel | mean=0, std=1 | 消除幅值差异 |
| 5. 极端值裁剪 | clip at **±15σ** | 参考 REVE 论文 | 防止异常尖峰 |
| 6. Train/Test Split | 按受试者分层 | **80/20** (`scripts/split_data.py`) | 独立测试集 |
| 7. 滑窗分段 | 10s 窗口, **无重叠** | 2000 采样点/epoch | 固定长度训练样本 |

> Step 1-5 由基类 `BaseEEGLoader._preprocess_signal()` 统一执行。每个数据集的 loader 只负责三个差异操作：**读取原始格式** + **提取标签** + **提取通道坐标**。

### 两阶段预处理流程

```
阶段 1: 信号处理 (Step 1-5)
  process(file, skip_epoching=True) → 保存为 (n_channels, n_samples) 连续数据
    ↓
Step 6: 确定 Train/Test Split
  扫描所有 *_meta.json → 按受试者分层划分 → 写入 data/splits.json + labels.csv
    ↓
阶段 2: 滑窗分段 (Step 7)
  epoch_output_dir() → 加载连续数据 → 切为 10s 窗口 → 保存为 (n_epochs, n_channels, 2000)
```

两阶段设计的原因：split 必须在分段**之前**确定（按受试者，非按 epoch），否则同一受试者的不同 epoch 可能泄漏到 train 和 test。

### 各数据集差异化处理

| 数据集 | 原始格式 | 原始采样率 | 通道 | 关键适配 |
|--------|---------|-----------|------|---------|
| **MODMA** | `.mat` / `.raw` / `.txt` | 250 Hz | 128 / 3 | 过滤 E129+ 辅助通道；去除全零 trigger 行；EGI 蒙太奇坐标 |
| **IEEE** | `.mat` (int16) | 128 Hz | 19 | 标准 10-20；标签从目录结构推断 |
| **Mendeley** | `.mat` (1×11 cell) | 256 Hz | 2/任务 | 跳过 FADHD_07（已知损坏）；跳过全零段；每任务不同通道对 |
| **OpenNeuro** | EEGLAB `.set/.fdt` | 500 Hz | 67→64 | 过滤 HEOG/VEOG/EKG；排除 BDI 8-13 灰区；BDI≥14→抑郁 |
| **TDBRAIN** | BioSemi `.bdf` | ~500 Hz | 26→26 | 过滤 Status/EOG/ECG/EMG；优先 indication 列，回退 formal_status；仅 restEC+oddball |

### 输出格式

```
data/processed/{dataset}/{subject_id}/{run_id}/
├── {subject_id}_eeg.npy       # (n_epochs, n_channels, 2000) float32
├── {subject_id}_ch_pos.npy    # (n_channels, 3) 3D 电极坐标
└── {subject_id}_meta.json     # 标签、诊断类型、原始采样率、任务等
```

---

## 环境

| 项目 | 值 |
|------|-----|
| GPU | RTX 3070 Ti 8GB, CUDA 12.1 |
| PyTorch | 2.5.1+cu121 |
| braindecode | 1.6.1 |
| REVE 权重 | `brain-bzh/reve-base` 本地缓存 |
| 训练速度 | ~30s/epoch (单数据集, <30ch) |

---

## 快速开始

```bash
# 二分类（5 数据集 × diagnosis）
bash scripts/run_binary_experiments.sh

# 评估模型
python scripts/evaluate_model.py --checkpoint <path> --datasets <name>

# 128 通道训练（需 AMP + 小 batch + 梯度累积）
python models/train.py --dataset MODMA --diagnosis depression --batch_size 4 --amp --grad_accum 16

# 对新数据进行诊断推理
python scripts/predict.py --checkpoint <path> --input_dir data/new_patients/

# 数据统计
python scripts/data_stats.py
```

---

## 实验

### 实验 1: 单数据集二分类 🔄 重新训练中

> ⚠️ 以下为旧 split（无独立 test set）的结果。重新训练后将更新。

| 数据集 | 诊断 | Bal Acc | AUROC | F1 | Subj Acc | 备注 |
|--------|------|---------|-------|----|---------|------|
| IEEE | ADHD | 0.765 | 0.812 | 0.795 | 0.758 | 19ch, 121人 |
| MODMA | Depression | 0.610 | 0.633 | 0.534 | 0.541 | 仅 3ch, 128ch 待跑 |
| Mendeley | ADHD | 0.863 | 0.910 | 0.856 | 0.854 | 5ch ✅ |
| OpenNeuro | Depression | 0.719 | 0.832 | 0.831 | 0.821 | ~70ch, 死通道问题 |
| TDBRAIN | Depression | **0.940** | **0.969** | 0.935 | 0.895 | ✅✅ 最佳 |
| TDBRAIN | ADHD | 0.828 | 0.946 | 0.808 | 0.735 | ✅ |

> 旧结果: `outputs/results/binary_summary_20260710_1609.csv`

### 实验 2: 三分类 🔄 待重新训练

| 数据集 | Bal Acc | AUROC (macro) | F1 | Subj Acc | 备注 |
|--------|---------|---------------|----|---------|------|
| TDBRAIN | 0.751 | **0.926** ✅ | 0.806 | 0.717 | 26ch, 三类 |
| 5-dataset mix | 0.607 | 0.754 | 0.687 | 0.694 | 各数据集标签不全 |

**混淆矩阵 (TDBRAIN)**：
```
        预测→ HC   MDD  ADHD
真实 HC      1311   53   64    (92%)
真实 MDD      89   623  116    (75%)
真实 ADHD     13   203  300    (58%)
```

> ADHD ↔ MDD 互混严重，HC 识别 92%。AUROC 0.926 达标，BalAcc 0.751 距 0.80 差 ~5%。

> 旧结果: `outputs/results/multiclass_summary_20260710_1903.csv`

### 实验 3: 多数据集混合 ⬜

### 实验 4: 跨数据集泛化 ⬜

### 实验 5: MODMA 128 通道 ⬜

> 需要 AMP + batch_size=4 + grad_accum=16，预计 ~1h/epoch。

---

## 已解决问题

| 问题 | 修复 |
|------|------|
| Cosine LR 提前熄火 | 默认恒定 lr |
| `--dataset` 不生效 | 添加 `include_datasets` |
| HF 401 | 本地缓存 + `--local_files_only` |
| TDBRAIN ADHD BalAcc=0.5 | label 映射 + 分层 split + class_weights |
| MODMA 128ch OOM | `--amp` + `--grad_accum` |
| 多数据集三分类差 | 各数据集标签不全，不适合混 |
| **无独立 test set** | 按受试者 80/20 分层 split (`data/splits.json`) |
| **OpenNeuro 死通道** | `dataset.py` 自动检测 + 置零 + channel_mask |

---

## 已知限制

- **128 通道极慢**: RTX 3070 Ti 8GB 上 ~1h/epoch，需 AMP + grad_accum
- **三分类 ADHD 召回低**: 58%，需更多数据或针对性调参
- **OpenNeuro 死通道**: 已自动检测置零，但 35% 文件受影响，可能限制其性能上限

---

## 关键文件

```
eeg/
├── task.md
├── preprocessing/
│   ├── base_loader.py              # 基类（7-step 预处理流水线）
│   ├── load_ieee.py / load_modma.py / load_mendeley.py / load_openneuro.py / load_tdbrain.py
│   └── generate_labels.py          # 生成 labels.csv
├── models/
│   ├── config.py                   # 实验配置（Data/Model/Training）
│   ├── dataset.py                  # EEG Dataset（死通道检测, 分层 split, 可变通道 batching）
│   └── train.py                    # 训练脚本（AMP, grad_accum, class_weights）
├── scripts/
│   ├── run_preprocess.py           # 一键预处理（两阶段: 信号处理 → split → 分段）
│   ├── split_data.py               # Train/Test 分层划分
│   ├── data_stats.py               # 数据统计报告
│   ├── evaluate_model.py           # 独立评估入口（加载 checkpoint → 评估 → JSON）
│   ├── predict.py                  # 新数据诊断推理
│   └── run_binary_experiments.sh   # 二分类批量实验
├── data/
│   ├── labels.csv                  # 标签索引（含 split 列）
│   ├── splits.json                 # Train/Test 受试者划分
│   └── processed/                  # 预处理后数据 (*_eeg.npy, *_ch_pos.npy, *_meta.json)
└── outputs/
    ├── checkpoints/
    │   ├── exp1_IEEE_adhd/
    │   ├── exp1_MODMA_depression/
    │   ├── exp1_Mendeley_adhd/
    │   ├── exp1_OpenNeuro_depression/
    │   ├── exp1_TDBRAIN_depression/
    │   ├── exp1_TDBRAIN_adhd/
    │   ├── exp1_TDBRAIN_multiclass/
    │   └── exp2_multidataset_multiclass/
    └── results/
        ├── binary_summary_20260710_1609.csv
        └── multiclass_summary_20260710_1903.csv
```

---

*Phase 1: 2026-07-03 | 二分类: 2026-07-10 | 三分类: 2026-07-10 | 数据质量审计 + 死通道修复: 2026-07-13*
