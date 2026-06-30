# EEG Foundation Model — 抑郁症与ADHD分类预测

> 基于 EEG Foundation Model（REVE/LUNA/LaBraM）实现抑郁症和 ADHD 患者与健康对照的二分类预测，目标准确率 **≥80%**。

---

## 项目概述

- **任务**：利用预训练 EEG Foundation Model，对 EEG 信号进行抑郁症/ADHD 二分类
- **数据**：5 个公开 EEG 数据集（MODMA、TDBRAIN、IEEE DataPort、Mendeley Data、OpenNeuro）
- **模型**：REVE（首选）、LUNA（次选）、LaBraM（备选）
- **指标**：平衡准确率 ≥80%，AUROC ≥0.85

详细技术方案见 [`task.md`](task.md)，数据集资源汇总见 [`content.md`](content.md)。

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv eeg_env
source eeg_env/Scripts/activate   # Windows Git Bash
# 或 source eeg_env/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据集

| 数据集 | 链接 | 疾病类型 | 优先级 |
|:---|:---|:---|:---|
| MODMA | [modma.lzu.edu.cn](https://modma.lzu.edu.cn/data/index/) | 抑郁症 | 高 |
| IEEE DataPort | [ieee-dataport.org](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children) | ADHD（儿童） | 高 |
| Mendeley Data | [data.mendeley.com](https://data.mendeley.com/datasets/6k4g25fhzg/1) | ADHD（成人） | 中 |
| OpenNeuro ds003478 | [openneuro.org](https://openneuro.org/datasets/ds003478/versions/1.1.0/download) | 抑郁症 | 中 |
| TDBRAIN | [brainclinics.com](https://brainclinics.com/resources/) | 抑郁+ADHD | 高（需申请） |

将下载的数据放入 `data/` 对应子目录。

### 3. 数据预处理

```bash
cd preprocessing

# 示例：处理单个数据集
python load_modma.py --data_dir ../data/MODMA --output_dir ../data/processed

# 统一采样率到 200Hz
python resample.py --input_dir ../data/processed --target_fs 200

# 生成统一标签文件
python generate_labels.py --data_dir ../data/processed --output ../data/labels.csv
```

### 4. 训练模型

```bash
cd models

# 线性探测（冻结编码器，仅训练分类头）
python train.py --mode linear_probe --model luna --epochs 20 --lr 1e-4

# 全模型微调
python train.py --mode finetune --model luna --epochs 50 --lr 1e-5
```

### 5. 评估

```bash
python evaluate.py --checkpoint ../outputs/checkpoints/best_model.ckpt --test_data ../data/test
```

---

## 项目结构

```
eeg/
├── README.md                    # 本文件
├── content.md                   # 数据集资源汇总
├── task.md                      # 技术方案详情
├── requirements.txt             # Python 依赖
├── .gitignore
│
├── data/                        # 原始数据（gitignore）
│   ├── MODMA/
│   ├── IEEE_ADHD/
│   ├── Mendeley_ADHD/
│   ├── TDBRAIN/
│   └── OpenNeuro_ds003478/
│
├── preprocessing/               # 数据预处理脚本
├── models/                      # 模型定义与训练脚本
├── configs/                     # 配置文件
├── notebooks/                   # Jupyter 探索性分析
├── outputs/                     # 训练产物（gitignore）
│   ├── checkpoints/
│   ├── logs/
│   └── results/
└── docs/                        # 额外文档
```

---

## 协作指南

### 分支策略

```bash
git checkout -b feature/xxx    # 新功能分支
git checkout -b fix/xxx        # 修复分支
```

`main` 分支保持稳定，所有开发在 feature 分支进行，完成后合并。

### 代码规范

- 使用 `black` 格式化代码
- 关键函数添加类型注解
- 数据路径通过配置文件传入，**禁止硬编码绝对路径**

### 数据共享

数据集为公开资源，**每位合作者需自行从源链接下载**，不通过 Git 共享。确保所有人的 `data/` 目录结构一致。

### 实验记录

每次训练在 `outputs/logs/` 下记录：
- 使用的数据集组合
- 超参数（学习率、批大小、epoch 数）
- 验证集/测试集性能指标

---

## 实施路线图

| 阶段 | 任务 | 预计时间 |
|:---|:---|:---|
| **1. 数据准备** | 下载数据、统一格式、生成标签 | 1-2 周 |
| **2. 基线建立** | 加载预训练模型、单数据集验证 | 1 周 |
| **3. 模型优化** | 多数据集混合、超参数调优 | 2-4 周 |
| **4. 评估部署** | 测试集评估、可解释性分析 | 1 周 |

---

## 关键参考文献

| 文献 | 说明 | 链接 |
|:---|:---|:---|
| **REVE** | 最大规模 EEG-FM（60k小时），4D位置编码 | arXiv |
| **LUNA** | 拓扑无关编码器，TUAR SOTA | [GitHub](https://github.com/pulp-bio/BioFoundation) |
| **LaBraM** | 开源多规模 EEG-FM | [GitHub](https://github.com/935963004/LaBraM) |
| **CBraMod** | 交叉注意力 EEG 解码 | arXiv |

---

## 注意事项

- 按受试者划分训练/验证/测试集，避免数据泄露
- TDBRAIN 需单独申请访问，提前准备
- 注意各数据集许可条款，仅供研究使用
- 参与者隐私均已匿名化，但仍需合规使用

---

*项目初始化于 2026-06-30，欢迎贡献。*
