"""
Mendeley Adult ADHD 数据集加载器

数据来源: https://data.mendeley.com/datasets/6k4g25fhzg/1
论文: 79 人 (42 HC + 37 ADHD)，成人 20-68 岁

格式: 4 个 .mat 文件，每个文件是 1x11 cell array
  - FC.mat    : 13 个女性对照
  - MC.mat    : 29 个男性对照
  - FADHD.mat : 11 个女性 ADHD
  - MADHD.mat : 27 个男性 ADHD

每个 cell 对应一个任务条件，shape = (n_subjects, n_samples, 2):
  0: Eyes open baseline  — Cz+F4,  30s (7680 samples @256Hz)
  1: Eyes closed         — Cz+F4,  20s (5120 samples)
  2: Eyes open           — Cz+F4,  20s (5120 samples)
  3: Cognitive Challenge — Cz+F4,  45s (11520 samples)
  4: Pre-Omni baseline   — Cz+F4,  15s (3840 samples)
  5: Omni harmonic       — Cz+F4,  30s (7680 samples)
  6: Eyes open baseline  — O1+F4,  30s (7680 samples)
  7: Eyes closed         — O1+F4,  30s (7680 samples)
  8: Eyes open           — O1+F4,  30s (7680 samples)
  9: Eyes closed         — F3+F4,  45s (11520 samples)
 10: Eyes closed         — Fz+F4,  45s (11520 samples)

注: FADHD 第 7 号受试者数据损坏（全零）。

使用方式:
    python load_mendeley.py --data_dir ../data/Mendeley_ADHD --output_dir ../data/processed/Mendeley
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
from scipy.io import loadmat as sio_loadmat

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from base_loader import BaseEEGLoader

FS = 256.0  # Mendeley 数据集固定采样率

# 11 个任务的元数据: (索引, 任务名, 通道列表)
TASKS = [
    (0,  "eyes_open_baseline_cz_f4",   ["Cz", "F4"]),
    (1,  "eyes_closed_cz_f4",          ["Cz", "F4"]),
    (2,  "eyes_open_cz_f4",            ["Cz", "F4"]),
    (3,  "cognitive_challenge_cz_f4",  ["Cz", "F4"]),
    (4,  "pre_omni_baseline_cz_f4",    ["Cz", "F4"]),
    (5,  "omni_harmonic_cz_f4",        ["Cz", "F4"]),
    (6,  "eyes_open_baseline_o1_f4",   ["O1", "F4"]),
    (7,  "eyes_closed_o1_f4",          ["O1", "F4"]),
    (8,  "eyes_open_o1_f4",            ["O1", "F4"]),
    (9,  "eyes_closed_f3_f4",          ["F3", "F4"]),
    (10, "eyes_closed_fz_f4",          ["Fz", "F4"]),
]

# 四个分组: 文件名 → (标签, 诊断类型, 显示名)
GROUPS = {
    "FC":     (0, "control", "Female Control"),
    "MC":     (0, "control", "Male Control"),
    "FADHD":  (1, "adhd",   "Female ADHD"),
    "MADHD":  (1, "adhd",   "Male ADHD"),
}

# 10-20 电极近似 3D 坐标
CH_COORDS = {
    "Cz":  (0.000,  0.000,  0.090),
    "F4":  (0.060,  0.040,  0.055),
    "O1":  (-0.030, -0.100, 0.020),
    "F3":  (-0.060, 0.040,  0.055),
    "Fz":  (0.000,  0.050,  0.065),
}


class MendeleyLoader(BaseEEGLoader):
    """Mendeley Adult ADHD 数据集加载器。

    每个 .mat 文件是 1×11 cell array，每个 cell 对应一个任务，
    每个 cell shape = (n_subjects_in_group, n_samples, 2)。
    """

    dataset_name = "Mendeley"

    # ----------------------------------------------------------------
    # 子类实现（简化，实际处理走 process_epoch_array）
    # ----------------------------------------------------------------

    def _read_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError("请使用 process_epoch_array()")

    def _parse_label(self, meta: Dict[str, Any]) -> Tuple[int, str]:
        return meta.get("label", 0), meta.get("diagnosis_type", "control")

    def _extract_channels(
        self, data: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[List[str], np.ndarray]:
        ch_names = meta.get("ch_names", ["Ch1", "Ch2"])
        positions = np.array([CH_COORDS.get(ch, [0.0, 0.0, 0.0]) for ch in ch_names])
        return ch_names, positions

    # ----------------------------------------------------------------
    # 处理单条 EEG 记录
    # ----------------------------------------------------------------

    def process_array(
        self, data: np.ndarray, subject_id: str, ch_names: List[str],
        label: int, diag_type: str, group: str, task_name: str,
        skip_epoching: bool = False,
    ) -> Dict[str, Any]:
        """处理 (n_channels, n_samples) 的 EEG 数组。

        Args:
            data: (n_channels, n_samples)，经预处理前为原始 EEG。
            skip_epoching: 若 True，跳过 Step 7 滑窗分段。
            ...（描述信息）

        Returns:
            标准结果字典。
        """
        raw_meta = {
            "subject_id": subject_id,
            "fs": FS,
            "ch_names": ch_names,
            "group": group,
            "task": task_name,
            "label": label,
            "diagnosis_type": diag_type,
            "format": "mendeley_mat",
        }

        channels, ch_positions = self._extract_channels(data, raw_meta)
        eeg = self._preprocess_signal(data, raw_meta)

        if skip_epoching:
            epochs = eeg  # 连续数据 (n_channels, n_samples)
        else:
            epochs = self._epoch(eeg)  # (n_epochs, n_channels, n_samples)

        meta = {
            "subject_id": subject_id,
            "dataset": self.dataset_name,
            "original_fs": FS,
            "n_channels": len(ch_names),
            "diagnosis_type": diag_type,
            "source_file": f"mendeley://{group}/{task_name}/{subject_id}",
            "group": group,
            "task": task_name,
            "ch_names": ch_names,
            "format": "mat_cell",
        }

        return {
            "eeg": epochs,
            "channels": channels,
            "ch_positions": ch_positions,
            "label": label,
            "meta": meta,
        }


# ================================================================
# 命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Mendeley ADHD 数据集预处理")
    parser.add_argument("--data_dir", type=str, default="../data/Mendeley_ADHD",
                        help="Mendeley 原始数据目录")
    parser.add_argument("--output_dir", type=str, default="../data/processed/Mendeley",
                        help="预处理后输出目录")
    parser.add_argument("--config", type=str, default="../configs/preprocess_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--skip_epoching", action="store_true",
                        help="只跑 Step 1-5, 跳过 Step 7 滑窗分段")
    parser.add_argument("--epoch_only", action="store_true",
                        help="只对已有连续数据执行 Step 7 分段，不重新预处理")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / "EEG"
    output_dir = Path(args.output_dir)

    mat_files = sorted(data_dir.glob("*.mat"))
    if not mat_files:
        print(f"❌ 未找到 .mat 文件: {data_dir}")
        sys.exit(1)

    print(f"找到 {len(mat_files)} 个 .mat 文件: {[f.name for f in mat_files]}")
    print(f"配置: target_fs=200 Hz, window=10s, zscore\n")

    loader = MendeleyLoader(args.config)

    # --epoch_only 模式：只对已有连续数据分段
    if args.epoch_only:
        print(f"Step 7 分段模式: {output_dir}")
        loader.epoch_output_dir(str(output_dir))
        print(f"\n✓ 分段完成")
        return

    total_subjects = 0
    total_files = 0
    total_skipped = 0

    for mat_file in mat_files:
        fname = mat_file.stem  # FC / MC / FADHD / MADHD
        if fname not in GROUPS:
            print(f"⚠ 跳过未知文件: {mat_file.name}")
            continue

        label, diag_type, display_name = GROUPS[fname]

        print(f"\n{'='*60}")
        print(f"加载 {fname}.mat ({display_name})")
        print(f"{'='*60}")

        mat = sio_loadmat(str(mat_file))
        key = [k for k in mat if not k.startswith("__")][0]
        cell_array = mat[key]  # (1, 11) object array

        # 获取该组的受试者人数（从第一个 cell 推断）
        n_subjects = cell_array[0, 0].shape[0]
        print(f"  {n_subjects} 个受试者, {cell_array.shape[1]} 个任务")

        for subj_idx in range(n_subjects):
            subj_id = f"{fname}_{subj_idx+1:02d}"

            for task_idx, task_name, ch_names in TASKS:
                cell = cell_array[0, task_idx]  # (n_subjects, n_samples, 2)

                # 提取该受试者的数据: (n_samples, 2) → (2, n_samples)
                subj_data = cell[subj_idx].T.astype(np.float64)

                # 跳过全零数据（FADHD subject 7 已知损坏）
                if np.all(subj_data == 0):
                    if fname == "FADHD" and subj_idx == 6:
                        continue  # 已知损坏
                    total_skipped += 1
                    continue

                try:
                    result = loader.process_array(
                        subj_data, subj_id, ch_names,
                        label, diag_type, fname, task_name,
                        skip_epoching=args.skip_epoching,
                    )

                    out_dir = output_dir / task_name / subj_id
                    loader.save(result, str(out_dir))
                    total_files += 1

                except Exception as e:
                    print(f"  ✗ {subj_id}/{task_name}: {e}", file=sys.stderr)
                    total_skipped += 1

            total_subjects += 1
            if total_subjects % 20 == 0:
                print(f"  已处理 {total_subjects} 个受试者 ({total_files} 个文件)...")

    n_groups = len(mat_files)
    print(f"\n✓ Mendeley 预处理完成: {total_subjects} 个受试者, "
          f"{total_files} 个任务文件 ({n_groups} 组 × ~11 任务)")
    print(f"  跳过: {total_skipped}")
    print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
