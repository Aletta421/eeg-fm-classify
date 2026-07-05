"""
IEEE DataPort 儿童 ADHD 数据集加载器

数据来源: IEEE DataPort Children ADHD dataset
格式: .mat 文件，每个文件一个受试者
标签: 由目录结构确定 — ADHD_part1/ADHD_part2 → 1 (ADHD), Control_part1/Control_part2 → 0 (对照)
采样率: 128 Hz, 19 通道（标准 10-20 系统）

使用方式:
    python load_ieee.py --data_dir ../data/IEEE_ADHD --output_dir ../data/processed/IEEE_ADHD
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
from scipy.io import loadmat as sio_loadmat

# Fix Windows GBK encoding issue
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from base_loader import BaseEEGLoader


class IEEELoader(BaseEEGLoader):
    """IEEE DataPort 儿童 ADHD 数据集加载器。

    数据按目录组织:
      - ADHD_part1/, ADHD_part2/ → 61 个 ADHD 受试者 (label=1)
      - Control_part1/, Control_part2/ → 60 个对照受试者 (label=0)

    每个 .mat 文件: (n_samples, 19) int16 矩阵，需转置为 (19, n_samples)。
    """

    dataset_name = "IEEE_ADHD"

    # 标准 10-20 系统 19 通道名称（参考 IEEE 数据集文档）
    CHANNEL_NAMES = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "Fz", "Cz", "Pz",
    ]

    def __init__(self, config_path: str = "../configs/preprocess_config.yaml"):
        super().__init__(config_path)

    # ================================================================
    # 子类实现
    # ================================================================

    def _read_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取单个 .mat 文件。

        Returns:
            (eeg_data, meta): data shape (19, n_samples)
        """
        file_path = Path(file_path)

        mat = sio_loadmat(str(file_path))
        # .mat 文件只有一个数据键（如 "v1p"），跳过 __ 开头的元数据键
        data_key = [k for k in mat if not k.startswith("__")][0]
        raw = mat[data_key]  # (n_samples, 19) int16

        # 转置为 (19, n_samples)
        data = raw.T.astype(np.float64)

        # 从文件名提取 subject_id（如 "v1p" → "v1p"）
        subject_id = file_path.stem

        # 采样率 128 Hz（IEEE 数据集标准）
        fs = 128.0

        # 标签由目录决定
        parent_dir = file_path.parent.name.lower()
        if "adhd" in parent_dir:
            diagnosis = "adhd"
        else:
            diagnosis = "control"

        return data, {
            "subject_id": subject_id,
            "fs": fs,
            "diagnosis": diagnosis,
            "format": "ieee_mat",
        }

    def _parse_label(self, meta: Dict[str, Any]) -> Tuple[int, str]:
        """标签由 _read_file 中解析的 diagnosis 字段决定。"""
        diagnosis = meta.get("diagnosis", "control")
        if diagnosis == "adhd":
            return 1, "adhd"
        else:
            return 0, "control"

    def _extract_channels(
        self, data: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[List[str], np.ndarray]:
        """19 通道标准 10-20 系统。

        尝试从 MNE 获取坐标，失败则返回全零坐标。
        """
        import mne

        n_channels = data.shape[0]
        ch_names = self.CHANNEL_NAMES[:n_channels]

        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            pos = montage.get_positions()["ch_pos"]
            positions = np.array([
                pos.get(ch, [0.0, 0.0, 0.0]) for ch in ch_names
            ])
        except Exception:
            positions = np.zeros((n_channels, 3))

        return ch_names, positions


# ================================================================
# 命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="IEEE ADHD 数据集预处理")
    parser.add_argument(
        "--data_dir", type=str, default="../data/IEEE_ADHD",
        help="IEEE ADHD 原始数据目录",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/processed/IEEE_ADHD",
        help="预处理后数据输出目录",
    )
    parser.add_argument(
        "--config", type=str, default="../configs/preprocess_config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 四个子目录 → 标签
    subdirs = {
        "ADHD_part1": "adhd",
        "ADHD_part2": "adhd",
        "Control_part1": "control",
        "Control_part2": "control",
    }

    loader = IEEELoader(args.config)

    total_files = 0
    total_subjects = 0
    skipped = 0

    for subdir_name, label_type in subdirs.items():
        sub_path = data_dir / subdir_name
        if not sub_path.exists():
            print(f"⚠ 跳过不存在的目录: {sub_path}")
            continue

        mat_files = sorted(sub_path.glob("*.mat"))
        print(f"\n{'='*60}")
        print(f"处理 IEEE_ADHD/{subdir_name}: {len(mat_files)} 个受试者")
        print(f"{'='*60}")

        for f in mat_files:
            try:
                result = loader.process(str(f))
                subj_id = result["meta"]["subject_id"]
                out_subj_dir = output_dir / subdir_name / subj_id
                loader.save(result, str(out_subj_dir))
                total_files += 1
                total_subjects += 1
            except Exception as e:
                print(f"  ✗ {f.name}: {e}", file=sys.stderr)
                skipped += 1
                continue

            if total_subjects % 20 == 0:
                print(f"  已处理 {total_subjects} 个受试者...")

    if skipped:
        print(f"\n  {skipped} 个文件处理失败")

    print(f"\n✓ IEEE_ADHD 预处理完成: {total_subjects} 个受试者, {total_files} 个文件")
    print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
