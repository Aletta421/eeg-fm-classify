"""
对新 EEG 数据进行诊断推理

用法:
    # 单个文件
    python scripts/predict.py --checkpoint outputs/checkpoints/exp1_TDBRAIN_depression/best_model.pt \\
        --eeg data/new_subject/subject_001.npy --pos data/new_subject/subject_001_ch_pos.npy

    # 整个目录（批量诊断）
    python scripts/predict.py --checkpoint outputs/checkpoints/exp1_TDBRAIN_multiclass/best_model.pt \\
        --input_dir data/new_patients/

    # 指定设备
    python scripts/predict.py --checkpoint xxx.pt --input_dir data/new/ --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ================================================================
# 加载模型
# ================================================================


def load_model(checkpoint_path: str, device: torch.device):
    """加载训练好的模型。"""
    from braindecode.models import REVE

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    exp_cfg = ckpt.get("config")
    if exp_cfg is None:
        raise ValueError("Checkpoint 缺少 config，请使用训练时保存的 best_model.pt")

    model_cfg = exp_cfg.model
    n_outputs = getattr(model_cfg, "n_outputs", 2)
    n_times = getattr(model_cfg, "n_times", 2000)
    input_window = getattr(model_cfg, "input_window_seconds", 10.0)
    sfreq = getattr(model_cfg, "sfreq", 200.0)
    use_attn = getattr(model_cfg, "use_attention_pooling", True)
    task = exp_cfg.data.task

    # label 名称
    if task == "multiclass" or n_outputs == 3:
        label_names = {0: "HC (健康)", 1: "MDD (抑郁症)", 2: "ADHD (多动症)"}
    else:
        label_names = {0: "HC (健康)", 1: "患者"}

    print(f"模型: {n_outputs} 分类 ({task}), epoch={ckpt.get('epoch', '?')}")
    print(f"训练集: {exp_cfg.datasets}")
    print(f"保存时指标: {ckpt.get('metrics', {})}")

    model = REVE(
        n_outputs=n_outputs,
        n_chans=None,
        n_times=n_times,
        input_window_seconds=input_window,
        sfreq=sfreq,
        attention_pooling=use_attn,
    )

    # 如果 checkpoint 使用了自定义分类头，替换 final_layer
    training_cfg = getattr(exp_cfg, 'training', None)
    if training_cfg is not None:
        head_type = getattr(training_cfg, 'head_type', 'linear')
        if head_type != 'linear':
            from models.heads import create_head
            print(f"Head type: {head_type} (replacing final_layer)")
            model.final_layer = create_head(
                head_type,
                in_features=512,
                n_classes=n_outputs,
            ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    return model, label_names, task


# ================================================================
# 推理
# ================================================================


@torch.no_grad()
def predict_single(model, eeg: np.ndarray, pos: np.ndarray, device: torch.device):
    """
    对单个被试的所有 epochs 进行推理。

    Args:
        eeg:  (n_epochs, n_channels, 2000) float32
        pos:  (n_channels, 3) float32
        device: torch device

    Returns:
        {
            "predicted_class": int,
            "predicted_label": str,
            "probabilities": [float, ...],       # 各类别平均概率
            "per_epoch": [
                {"epoch": 0, "pred": 0, "probs": [0.9, 0.1]},
                ...
            ],
            "consensus": float,                   # 多数投票比例
        }
    """
    model.eval()
    n_epochs = eeg.shape[0]

    all_probs = []
    all_preds = []

    for i in range(n_epochs):
        epoch_data = torch.from_numpy(eeg[i]).unsqueeze(0).to(device)  # (1, C, T)
        epoch_pos = torch.from_numpy(pos).unsqueeze(0).to(device)      # (1, C, 3)

        logits = model(epoch_data, pos=epoch_pos)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (n_classes,)
        pred = int(probs.argmax())

        all_probs.append(probs.tolist())
        all_preds.append(pred)

    # 平均概率
    avg_probs = np.mean(all_probs, axis=0).tolist()

    # 多数投票
    from collections import Counter
    vote_counts = Counter(all_preds)
    majority_class = vote_counts.most_common(1)[0][0]
    consensus = vote_counts[majority_class] / n_epochs

    return {
        "predicted_class": majority_class,
        "probabilities": avg_probs,
        "per_epoch": [
            {"epoch": i, "pred": p, "probs": pr}
            for i, (p, pr) in enumerate(zip(all_preds, all_probs))
        ],
        "consensus": round(consensus, 4),
    }


# ================================================================
# 主函数
# ================================================================


def main():
    parser = argparse.ArgumentParser(description="EEG 诊断推理")
    parser.add_argument("--checkpoint", required=True, help="训练好的模型 .pt")
    parser.add_argument("--eeg", default=None, help="单个 .npy 文件路径")
    parser.add_argument("--pos", default=None, help="ch_pos.npy 路径（与 --eeg 搭配）")
    parser.add_argument("--input_dir", default=None, help="批量推理：包含 .npy 的目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="结果保存 JSON 路径（可选）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    model, label_names, task = load_model(args.checkpoint, device)

    # 收集待推理文件
    if args.eeg:
        eeg_path = Path(args.eeg)
        pos_path = Path(args.pos) if args.pos else eeg_path.parent / f"{eeg_path.stem.split('_epochs')[0]}_ch_pos.npy"
        files = [(eeg_path, pos_path)]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        npy_files = sorted(input_dir.glob("*.npy"))
        # 配对 eeg 和 pos 文件
        eeg_files = [f for f in npy_files if not f.name.endswith("_ch_pos.npy") and "_meta" not in f.name]
        files = []
        for ef in eeg_files:
            pos_f = ef.parent / f"{ef.stem}_ch_pos.npy"
            files.append((ef, pos_f if pos_f.exists() else None))
        if not files:
            print(f"错误: {input_dir} 中未找到 .npy 文件")
            sys.exit(1)
    else:
        print("请指定 --eeg 或 --input_dir")
        sys.exit(1)

    # 推理
    all_results = {}
    for eeg_path, pos_path in files:
        subject_id = eeg_path.stem
        print(f"\n{'='*50}")
        print(f"推理: {subject_id}")
        print(f"  EEG: {eeg_path}")
        print(f"  Pos: {pos_path}")

        if not eeg_path.exists():
            print(f"  ❌ EEG 文件不存在，跳过")
            continue

        eeg = np.load(eeg_path).astype(np.float32)
        if eeg.ndim == 2:
            eeg = eeg[np.newaxis, :, :]  # 单 epoch → (1, C, T)

        if pos_path and pos_path.exists():
            pos = np.load(pos_path).astype(np.float32)
        else:
            print(f"  ⚠ ch_pos.npy 未找到，使用零填充")
            pos = np.zeros((eeg.shape[1], 3), dtype=np.float32)

        # 对齐通道数
        if pos.shape[0] != eeg.shape[1]:
            print(f"  ⚠ 通道数不匹配 (eeg={eeg.shape[1]}, pos={pos.shape[0]})，截断/填充")
            if pos.shape[0] < eeg.shape[1]:
                pad = np.zeros((eeg.shape[1] - pos.shape[0], 3), dtype=np.float32)
                pos = np.concatenate([pos, pad], axis=0)
            else:
                pos = pos[:eeg.shape[1]]

        result = predict_single(model, eeg, pos, device)
        result["subject_id"] = subject_id
        result["n_epochs"] = eeg.shape[0]
        result["n_channels"] = eeg.shape[1]

        # 打印结果
        print(f"  Epochs: {result['n_epochs']}, Channels: {result['n_channels']}")
        print(f"  预测: {label_names[result['predicted_class']]}")
        print(f"  共识: {result['consensus']:.1%} ({int(result['consensus'] * result['n_epochs'])}/{result['n_epochs']} epochs)")
        print(f"  各类概率:")
        for cls_id, prob in enumerate(result["probabilities"]):
            bar = "█" * int(prob * 40) + "░" * (40 - int(prob * 40))
            print(f"    {label_names.get(cls_id, f'Class-{cls_id}')}: {bar} {prob:.2%}")

        all_results[subject_id] = result

    # 保存 JSON
    if args.output:
        out_path = Path(args.output)
    elif args.input_dir:
        out_path = Path(args.input_dir) / "predictions.json"
    else:
        out_path = Path(args.eeg).parent / f"{Path(args.eeg).stem}_prediction.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
