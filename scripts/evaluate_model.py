"""
独立模型评估脚本

用法:
    # 自动从 checkpoint 推断配置
    python scripts/evaluate_model.py --checkpoint outputs/checkpoints/exp_single_tdbrain/best_model.pt

    # 指定数据集和任务
    python scripts/evaluate_model.py --checkpoint outputs/checkpoints/xxx/best_model.pt --datasets TDBRAIN --task multiclass

    # 只在测试集上评估
    python scripts/evaluate_model.py --checkpoint xxx.pt --split test
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dataset import create_dataloaders, EEGDataset, collate_eeg_batch
from models.config import ModelConfig


# ================================================================
# 加载模型
# ================================================================


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """加载 checkpoint 并打印摘要。"""
    from braindecode.models import REVE

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 恢复配置
    exp_cfg = ckpt.get("config")
    if exp_cfg is not None:
        model_cfg = exp_cfg.model
        data_cfg = exp_cfg.data
        datasets = exp_cfg.datasets if hasattr(exp_cfg, 'datasets') else []
        # 恢复 diagnosis 过滤条件（二分类：HC vs MDD 或 HC vs ADHD）
        include_diagnosis = getattr(data_cfg, "include_diagnosis", None) if data_cfg else None
        if include_diagnosis is not None and len(include_diagnosis) == 0:
            include_diagnosis = None  # 空列表 = 不过滤
    else:
        model_cfg = ModelConfig()
        datasets = []
        include_diagnosis = None

    n_outputs = getattr(model_cfg, "n_outputs", 2)
    n_times = getattr(model_cfg, "n_times", 2000)
    input_window = getattr(model_cfg, "input_window_seconds", 10.0)
    sfreq = getattr(model_cfg, "sfreq", 200.0)
    use_attn = getattr(model_cfg, "use_attention_pooling", True)

    print(f"\n{'='*55}")
    print(f"  Checkpoint: {Path(checkpoint_path).name}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  n_outputs: {n_outputs}")
    print(f"  Datasets: {datasets}")
    print(f"  Diagnosis filter: {include_diagnosis}")
    print(f"  Saved metrics: {ckpt.get('metrics', {})}")
    print(f"{'='*55}")

    model = REVE(
        n_outputs=n_outputs,
        n_chans=None,
        n_times=n_times,
        input_window_seconds=input_window,
        sfreq=sfreq,
        attention_pooling=use_attn,
    )

    # 如果 checkpoint 使用了自定义分类头，替换 final_layer
    head_type = "linear"
    if exp_cfg is not None and hasattr(exp_cfg, 'training'):
        training_cfg = exp_cfg.training
        head_type = getattr(training_cfg, 'head_type', 'linear')

    if head_type != "linear":
        from models.heads import create_head, SklearnHead
        print(f"  Head type: {head_type} (replacing final_layer)")
        model.final_layer = create_head(
            head_type,
            in_features=512,
            n_classes=n_outputs,
        ).to(device)

    # 加载 state_dict（sklearn 头无参数，用 non-strict）
    strict = not head_type.startswith("sklearn_")
    state_dict = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        print(f"  ⚠ Strict load failed, trying non-strict: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    # sklearn 头：加载 joblib 模型
    if head_type.startswith("sklearn_"):
        import joblib
        joblib_path = Path(checkpoint_path).parent / "sklearn_head.joblib"
        if joblib_path.exists():
            model.final_layer.clf = joblib.load(str(joblib_path))
            model.final_layer._fitted = True
            print(f"  Loaded sklearn model: {joblib_path}")
        else:
            print(f"  ⚠ sklearn_head.joblib not found at {joblib_path}")

    model = model.to(device)
    model.eval()
    return model, {"datasets": datasets, "n_outputs": n_outputs, "include_diagnosis": include_diagnosis}


# ================================================================
# 评估核心
# ================================================================


@torch.no_grad()
def evaluate(model, dataloader, device, n_classes):
    """在 DataLoader 上评估模型。"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_subjects, all_datasets = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        eeg = batch["eeg"].to(device)
        pos = batch["pos"].to(device)
        labels = batch["label"].to(device)

        logits = model(eeg, pos=pos)
        total_loss += criterion(logits, labels).item()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.append(probs.cpu().numpy())
        all_subjects.extend(batch["subject_id"])
        all_datasets.extend(batch["dataset"])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)
    n = len(y_true)

    # 核心指标
    results = {
        "n_samples": n,
        "loss": total_loss / max(len(dataloader), 1),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

    # AUROC
    if n_classes == 2:
        results["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    elif n_classes > 2 and len(set(y_true)) > 1:
        try:
            results["auroc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            results["auroc_macro"] = None
    else:
        results["auroc"] = None

    # 混淆矩阵
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    # 每类指标
    label_names = {0: "HC", 1: "MDD", 2: "ADHD"}
    present_labels = sorted(set(y_true))
    target_names = [label_names.get(l, f"Class-{l}") for l in present_labels]
    results["per_class"] = classification_report(
        y_true, y_pred, labels=present_labels,
        target_names=target_names, output_dict=True, zero_division=0,
    )

    # 按受试者聚合
    subj_accs = []
    for s in set(all_subjects):
        mask = np.array([x == s for x in all_subjects])
        if mask.sum():
            subj_accs.append((y_true[mask] == y_pred[mask]).mean())
    results["subject_balanced_acc"] = float(np.mean(subj_accs)) if subj_accs else 0.0

    # 按数据集分组
    results["per_dataset"] = {}
    for ds in sorted(set(all_datasets)):
        mask = np.array([d == ds for d in all_datasets])
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        ds_subjects = [all_subjects[i] for i in range(len(all_subjects)) if mask[i]]
        ds_r = {
            "n": int(mask.sum()),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)),
            "f1_weighted": float(f1_score(yt, yp, average="weighted")),
        }
        # 按受试者聚合
        subj_accs = []
        for s in set(ds_subjects):
            sm = np.array([x == s for x in ds_subjects])
            if sm.sum():
                subj_accs.append((yt[sm] == yp[sm]).mean())
        ds_r["subject_acc"] = float(np.mean(subj_accs)) if subj_accs else 0.0
        if n_classes == 2 and len(set(yt)) > 1:
            ds_r["auroc"] = float(roc_auc_score(yt, y_prob[mask][:, 1]))
        results["per_dataset"][ds] = ds_r

    return results


# ================================================================
# 主函数
# ================================================================


def main():
    parser = argparse.ArgumentParser(
        description="EEG 模型独立评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/evaluate_model.py --checkpoint outputs/checkpoints/xxx/best_model.pt
  python scripts/evaluate_model.py --checkpoint xxx.pt --datasets TDBRAIN --task multiclass
  python scripts/evaluate_model.py --checkpoint xxx.pt --split test
        """,
    )
    parser.add_argument("--checkpoint", required=True, help="模型 .pt 路径")
    parser.add_argument("--datasets", default=None, help="数据集 (逗号分隔)，默认用 checkpoint 中记录的")
    parser.add_argument("--diagnosis", default=None,
                        help="诊断类型过滤 (如 'depression' 或 'adhd')，默认从 checkpoint 配置恢复")
    parser.add_argument("--split", default="all", choices=["all", "train", "test"])
    parser.add_argument("--max_channels", type=int, default=-1, help="最大通道数过滤")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None, help="结果保存目录")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).parent.parent

    # 加载模型和元信息
    model, meta = load_checkpoint(args.checkpoint, device)

    # 确定 datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = meta["datasets"] if meta["datasets"] else None

    # 确定 diagnosis 过滤: CLI 参数优先，否则从 checkpoint 恢复
    if args.diagnosis:
        include_diagnosis = ["control", args.diagnosis.strip()]
        print(f"Using diagnosis filter (CLI): {include_diagnosis}")
    elif meta.get("include_diagnosis"):
        include_diagnosis = meta["include_diagnosis"]
        print(f"Using diagnosis filter (from checkpoint): {include_diagnosis}")
    else:
        include_diagnosis = None
        print("No diagnosis filter — using all data")

    n_classes = meta["n_outputs"]

    # 创建数据加载器
    print(f"\nLoading data (datasets={datasets}, diagnosis={include_diagnosis})...")
    loader_dict = create_dataloaders(
        labels_csv="data/labels.csv",
        include_diagnosis=include_diagnosis,
        include_datasets=datasets,
        max_channels=args.max_channels,
        batch_size=args.batch_size,
        num_workers=0,
        project_root=project_root,
    )

    # 评估
    splits = ["train", "test"] if args.split == "all" else [args.split]
    all_results = {}

    for split_name in splits:
        print(f"Evaluating {split_name}...", end=" ", flush=True)
        results = evaluate(model, loader_dict[split_name], device, n_classes)
        all_results[split_name] = results

        auroc = results.get("auroc") or results.get("auroc_macro")
        auroc_str = f"{auroc:.4f}" if auroc else "N/A"
        print(f"BalAcc={results['balanced_acc']:.4f}  AUROC={auroc_str}  "
              f"SubjAcc={results['subject_balanced_acc']:.4f}")

    # 输出详细报告
    print(f"\n{'='*55}")
    print(" SUMMARY")
    print(f"{'='*55}")
    for split_name in splits:
        r = all_results[split_name]
        print(f"\n--- {split_name.upper()} (n={r['n_samples']}) ---")
        print(f"  Balanced Acc:  {r['balanced_acc']:.4f}")
        auroc = r.get("auroc") or r.get("auroc_macro")
        if auroc:
            print(f"  AUROC:         {auroc:.4f}")
        print(f"  F1 weighted:   {r['f1_weighted']:.4f}")
        print(f"  F1 macro:      {r['f1_macro']:.4f}")
        print(f"  Subject Acc:   {r['subject_balanced_acc']:.4f}")

        if r.get("per_dataset"):
            print("  Per dataset:")
            for ds_name, ds_r in r["per_dataset"].items():
                a = ds_r.get("auroc")
                a_str = f" AUROC={a:.4f}" if a else ""
                print(f"    {ds_name:15s}: n={ds_r['n']:5d}  BalAcc={ds_r['balanced_acc']:.4f}{a_str}")

        # 混淆矩阵
        cm = np.array(r["confusion_matrix"])
        if cm.size:
            print(f"  Confusion Matrix:\n{cm}")

    if "generalization_gap" in all_results:
        g = all_results["generalization_gap"]
        ok = "OK" if g["ok"] else "WARN"
        print(f"\n--- Generalization Gap: {g['gap']:.4f} [{ok}] ---")

    # 保存 JSON
    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        out_path = project_root / "outputs" / "results" / Path(args.checkpoint).stem
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "evaluation.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved → {json_path}")


if __name__ == "__main__":
    main()
