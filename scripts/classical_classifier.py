"""
REVE 特征提取 + 传统分类器对比

流程:
  1. 加载预训练 REVE，冻结 encoder 提取特征（embedding）
  2. 用 sklearn 分类器（LR, RF, XGBoost, LightGBM）替代线性头
  3. 对比 AUROC / BalAcc

用法:
  python scripts/classical_classifier.py --dataset IEEE_ADHD --diagnosis adhd
  python scripts/classical_classifier.py -d TDBRAIN -g depression --classifiers lr,rf,xgb
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dataset import create_dataloaders, collate_eeg_batch
from models.config import ModelConfig


# ================================================================
# 特征提取
# ================================================================


def load_reve_encoder(
    device: torch.device,
    local_files_only: bool = True,
) -> nn.Module:
    """加载预训练 REVE，返回去掉分类头的 encoder。"""
    from braindecode.models import REVE

    model = REVE.from_pretrained(
        "brain-bzh/reve-base",
        n_outputs=2,
        n_chans=None,
        n_times=2000,
        input_window_seconds=10.0,
        sfreq=200.0,
        attention_pooling=True,
        local_files_only=local_files_only,
    )
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, dataloader, device) -> tuple:
    """从 REVE 的 logits 层前提取 embedding。

    通过 forward hook 捕获全连接层之前的特征向量。
    """
    # 找到最后一个全连接层，在它之前插入 hook
    embedding_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            embedding_layer = name  # 最后一个是分类头

    if embedding_layer is None:
        raise ValueError("Cannot find classification head in REVE model")

    embeddings = []
    labels = []
    subject_ids = []
    datasets = []

    # 注册 hook 捕获分类头前的特征
    def hook_fn(module, input, output):
        embeddings.append(input[0].detach().cpu().numpy())

    target_module = dict(model.named_modules())[embedding_layer]
    handle = target_module.register_forward_hook(hook_fn)

    try:
        for batch in tqdm(dataloader, desc="Extracting features"):
            eeg = batch["eeg"].to(device)
            pos = batch["pos"].to(device)

            model(eeg, pos=pos)  # hook 自动捕获 embedding

            labels.extend(batch["label"].tolist())
            subject_ids.extend(batch["subject_id"])
            datasets.extend(batch["dataset"])
    finally:
        handle.remove()

    X = np.concatenate(embeddings, axis=0)
    y = np.array(labels)
    return X, y, subject_ids, datasets


# ================================================================
# 分类器训练 & 评估
# ================================================================


def build_classifiers(random_state: int = 42) -> Dict:
    """构建待对比的分类器字典。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    classifiers = {
        "lr": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=2000, random_state=random_state,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, random_state=random_state, n_jobs=-1,
        ),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=random_state,
        ),
        "svm_linear": SVC(
            kernel="linear", C=1.0,
            probability=True, random_state=random_state,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1,
        ),
        "gbdt": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=random_state,
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5,
            random_state=random_state,
        ),
    }

    # XGBoost (可选)
    try:
        from xgboost import XGBClassifier
        classifiers["xgb"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=random_state,
        )
    except ImportError:
        pass

    # LightGBM (可选)
    try:
        from lightgbm import LGBMClassifier
        classifiers["lgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=random_state,
        )
    except ImportError:
        pass

    return classifiers


def train_torch_head(
    head_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """使用 PyTorch 自定义分类头（MLP/CNN1D/Attention）训练 + 评估。

    在 REVE embeddings 上训练一个小的神经网络分类头，
    用于与传统 sklearn 分类器对比。
    """
    from models.heads import create_head
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    in_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # 构建 head
    head = create_head(
        head_type,
        in_features=in_features,
        n_classes=n_classes,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        use_batchnorm=True,
    ).to(device)

    # 转 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    # 计算类别权重
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        [sum(class_counts) / (len(class_counts) * c) for c in class_counts],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    # 训练
    head.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = head(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    # 评估
    head.eval()
    with torch.no_grad():
        logits = head(X_test_t)
        y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()

    return {
        "balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }


def train_and_eval(clf, X_train, y_train, X_test, y_test):
    """训练 + 评估单个分类器。"""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }


# ================================================================
# 主函数
# ================================================================


def main():
    parser = argparse.ArgumentParser(
        description="REVE 特征提取 + 传统分类器对比",
    )
    parser.add_argument("--dataset", "-d", required=True, help="数据集名")
    parser.add_argument("--diagnosis", "-g", required=True,
                        choices=["depression", "adhd"], help="诊断类型")
    parser.add_argument("--classifiers", default="lr,rf,svm_rbf,svm_linear,knn,gbdt,adaboost,xgb,lgbm",
                        help="分类器列表，逗号分隔 (默认: lr,rf,svm_rbf,svm_linear,knn,gbdt,adaboost,xgb,lgbm)")
    parser.add_argument("--torch_heads", default=None,
                        help="PyTorch 自定义分类头，逗号分隔 (如 mlp,cnn1d,attention)")
    parser.add_argument("--max_channels", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).parent.parent

    print("=" * 55)
    print(f" 古典分类器对比: {args.dataset} / {args.diagnosis}")
    print("=" * 55)

    # ---- 1. 加载数据 ----
    print("\n[1/3] Loading data...")
    dataloaders = create_dataloaders(
        include_diagnosis=["control", args.diagnosis],
        include_datasets=[args.dataset],
        max_channels=args.max_channels,
        batch_size=args.batch_size,
        num_workers=0,
        project_root=project_root,
    )

    train_ds, test_ds = dataloaders["train_dataset"], dataloaders["test_dataset"]
    print(f"  Train: {len(train_ds)} epochs, {len(train_ds.train_subjects)} subjects")
    print(f"  Test:  {len(test_ds)} epochs, {len(test_ds.test_subjects)} subjects")

    # ---- 2. 加载 REVE 并提取特征 ----
    print("\n[2/3] Loading REVE + extracting embeddings...")
    model = load_reve_encoder(device, local_files_only=True)

    X_train, y_train, _, _ = extract_embeddings(model, dataloaders["train"], device)
    X_test, y_test, test_subjs, test_ds_names = extract_embeddings(
        model, dataloaders["test"], device
    )

    print(f"  Embedding dim: {X_train.shape[1]}")
    print(f"  Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")
    print(f"  Train label dist: {np.bincount(y_train)}")
    print(f"  Test  label dist: {np.bincount(y_test)}")

    # ---- 3. 训练 + 对比 ----
    print("\n[3/3] Training classifiers...")
    print(f"\n{'='*55}")
    print(f" {'Classifier':<20} {'BalAcc':>8} {'AUROC':>8} {'F1':>8}")
    print(f"{'-'*55}")

    clf_names = [c.strip() for c in args.classifiers.split(",")]
    classifiers = build_classifiers(args.seed)
    results = {}

    for name in clf_names:
        if name not in classifiers:
            print(f"  ⚠ '{name}' not available (install xgboost/lightgbm?)")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = train_and_eval(
                classifiers[name], X_train, y_train, X_test, y_test
            )

        results[name] = metrics
        print(f" {name:<20} {metrics['balanced_acc']:>8.4f} {metrics['auroc']:>8.4f} {metrics['f1_weighted']:>8.4f}")

    # ---- 3b. 训练 PyTorch 自定义头 ----
    if args.torch_heads:
        head_names = [h.strip() for h in args.torch_heads.split(",")]
        for head_name in head_names:
            try:
                metrics = train_torch_head(
                    head_name, X_train, y_train, X_test, y_test,
                    device=device, seed=args.seed,
                )
                results[f"torch_{head_name}"] = metrics
                print(f" torch_{head_name:<15} {metrics['balanced_acc']:>8.4f} {metrics['auroc']:>8.4f} {metrics['f1_weighted']:>8.4f}")
            except Exception as e:
                print(f"  ⚠ torch head '{head_name}' failed: {e}")

    # 对比 REVE 线性头
    print(f"\n{'='*55}")
    print(" (对比: REVE 线性头需跑 train.py 获取)")

    # 保存结果
    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        out_path = project_root / "outputs" / "results" / f"classical_{args.dataset}_{args.diagnosis}"
    out_path.mkdir(parents=True, exist_ok=True)

    result_file = out_path / "classifier_comparison.json"
    with open(result_file, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "diagnosis": args.diagnosis,
            "train_n": X_train.shape[0],
            "test_n": X_test.shape[0],
            "embedding_dim": X_train.shape[1],
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved → {result_file}")


if __name__ == "__main__":
    main()
