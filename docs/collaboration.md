# 协作规范

> 本文档约定项目协作流程、代码规范和沟通方式，所有合作者请遵守。

---

## Git 工作流

### 分支命名

```
feature/<功能名>     # 新功能开发，如 feature/load_modma
fix/<问题名>         # Bug 修复，如 fix/sampling_rate
exp/<实验名>         # 实验性分支，如 exp/luna_finetune
```

### 工作流程

```bash
# 1. 从 main 拉取最新代码
git checkout main
git pull origin main

# 2. 创建新分支
git checkout -b feature/xxx

# 3. 开发并提交（小步提交，有意义的 commit message）
git add .
git commit -m "feat: add MODMA data loader with resampling"

# 4. 推送分支
git push origin feature/xxx

# 5. 发起 Pull Request（或团队内通知合并）
```

### Commit Message 规范

```
feat: <简短描述>     # 新功能
fix: <简短描述>      # Bug 修复
docs: <简短描述>     # 文档更新
refactor: <简短描述> # 代码重构
exp: <简短描述>      # 实验相关
```

---

## 代码规范

### 格式化

安装并使用 Black：

```bash
pip install black
black --line-length 100 .
```

### 类型注解

关键函数建议添加类型注解：

```python
def load_eeg(file_path: str, target_fs: int = 200) -> np.ndarray:
    """加载 EEG 数据并重采样到目标采样率。

    Args:
        file_path: EEG 文件路径（.mat/.edf/.set）
        target_fs: 目标采样率 (Hz)

    Returns:
        形状为 (n_channels, n_samples) 的数组
    """
    ...
```

### 路径管理

**禁止硬编码绝对路径**，使用配置文件或环境变量：

```python
# 错误:
data = np.load("C:/Users/张三/Desktop/data/subject_001.npy")

# 正确:
from configs import DATA_DIR
data = np.load(DATA_DIR / "subject_001.npy")
```

---

## 数据规范

### 目录结构

每位合作者的 `data/` 目录必须一致：

```
data/
├── MODMA/
│   └── ...           # 原始 MODMA 文件
├── IEEE_ADHD/
│   └── ...           # 原始 IEEE DataPort 文件
├── Mendeley_ADHD/
│   └── ...
├── TDBRAIN/
│   └── ...
├── OpenNeuro_ds003478/
│   └── ...
└── processed/        # 预处理后的数据（脚本生成）
    └── ...
```

### 标签文件格式

统一使用 `data/labels.csv`：

```csv
subject_id,dataset,label,file_path,duration_seconds,sampling_rate,n_channels,diagnosis_type
subj_001,MODMA,1,data/processed/MODMA/subj_001.npy,300.0,200,128,depression
subj_002,MODMA,0,data/processed/MODMA/subj_002.npy,300.0,200,128,control
```

---

## 实验记录

每次实验在 `outputs/logs/` 下创建记录文件，格式：

```yaml
# experiment_001.yaml
date: 2026-07-01
model: LUNA-Base
datasets: [MODMA, IEEE_ADHD]
mode: linear_probe
hyperparams:
  lr: 1e-4
  batch_size: 128
  epochs: 20
  weight_decay: 0.05
results:
  balanced_accuracy: 0.782
  auroc: 0.851
  f1_weighted: 0.796
```

---

## 沟通

- 每周至少一次进度同步
- 遇到阻塞问题及时沟通，不要独自纠结超过半天
- 重要决策（模型选择、架构变更）在群内讨论后确定

---

*最后更新：2026-06-30*
