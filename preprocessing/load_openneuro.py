"""
OpenNeuro ds003478 抑郁症数据集加载器

数据来源: OpenNeuro ds003478 "EEG Depression Rest"
格式: BIDS 兼容，EEGLAB .set/.fdt 文件
标签: BDI (Beck Depression Inventory) — BDI >= 14 → 抑郁 (1), BDI <= 7 → 对照 (0)

通道: 67 (含 HEOG/VEOG/EKG，需过滤)
采样率: 500 Hz
每个受试者 1-2 个 run（task-Rest）

使用方式:
    python load_openneuro.py --data_dir ../data/OpenNeuro_ds003478 --output_dir ../data/processed/OpenNeuro
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np

# Fix Windows GBK encoding issue
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from base_loader import BaseEEGLoader

# 非 EEG 通道（眼电、心电），需从数据中排除
NON_EEG_CHANNELS = {"HEOG", "VEOG", "EKG", "EOG", "ECG", "EMG", "Trigger", "Status"}


class OpenNeuroLoader(BaseEEGLoader):
    """OpenNeuro ds003478 数据集加载器。

    BDI 阈值:
      - BDI >= 14 → 抑郁组 (label=1)
      - BDI <= 7  → 对照组 (label=0)
      - BDI 8-13  → 排除（灰区）
    """

    dataset_name = "OpenNeuro_ds003478"

    def __init__(self, config_path: str = "../configs/preprocess_config.yaml"):
        super().__init__(config_path)
        self._bdi_labels: Dict[str, Tuple[int, str]] = {}  # sub-XXX → (label, diagnosis)

    # ================================================================
    # 子类实现
    # ================================================================

    def _read_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取 EEGLAB .set 文件。

        .set 文件引用同目录下的 .fdt 二进制数据。
        """
        import mne
        import warnings

        file_path = Path(file_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_eeglab(
                str(file_path), preload=False, verbose=False
            )
            raw.load_data()

        all_data = raw.get_data()          # (n_channels, n_samples)
        fs = raw.info["sfreq"]
        all_ch_names = raw.info["ch_names"]

        # 过滤非 EEG 通道（HEOG, VEOG, EKG 等）
        eeg_indices = [
            i for i, ch in enumerate(all_ch_names)
            if ch.upper() not in NON_EEG_CHANNELS
        ]
        data = all_data[eeg_indices]
        ch_names = [all_ch_names[i] for i in eeg_indices]

        # 从路径解析 subject_id 和 run
        # 路径格式: sub-078/eeg/sub-078_task-Rest_run-01_eeg.set
        # 注意: 目录名 "sub-078" 和文件名 "sub-078_task-..." 都以 sub- 开头，
        # 需要精确定位到目录级别的 "sub-XXX"
        parts = file_path.parts
        subject_id = None
        run_id = "01"
        for p in parts:
            # 匹配目录级别的 sub-XXX（不含下划线后的任务/条件信息）
            if p.startswith("sub-") and len(p) <= 10 and "_" not in p:
                subject_id = p
                break
        if subject_id is None:
            # fallback: 从文件名前缀提取
            subject_id = file_path.stem[:7]
        if "run-" in file_path.name:
            run_id = file_path.name.split("run-")[1].split("_")[0]

        return data, {
            "subject_id": subject_id,
            "fs": fs,
            "ch_names": ch_names,
            "run_id": run_id,
            "format": "eeglab_set",
        }

    def _parse_label(self, meta: Dict[str, Any]) -> Tuple[int, str]:
        """从 participants.tsv 中查找 BDI 分数并转为标签。"""
        subject_id = meta["subject_id"]

        if not self._bdi_labels:
            self._load_bdi_labels()

        return self._bdi_labels.get(
            subject_id, (-1, "excluded")
        )

    def _extract_channels(
        self, data: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[List[str], np.ndarray]:
        """提取通道名，尝试匹配标准 10-20 蒙太奇坐标。"""
        import mne

        n_channels = data.shape[0]
        ch_names = meta.get("ch_names", [f"Ch{i+1}" for i in range(n_channels)])

        # 尝试从标准蒙太奇获取坐标
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            pos = montage.get_positions()["ch_pos"]
            positions = np.array([
                pos.get(ch.upper(), [0.0, 0.0, 0.0]) for ch in ch_names
            ])
        except Exception:
            positions = np.zeros((n_channels, 3))

        return ch_names, positions

    # ================================================================
    # 标签加载
    # ================================================================

    def _load_bdi_labels(self):
        """从 participants.tsv 加载 BDI 标签。"""
        data_root = Path(__file__).parent.parent / "data" / "OpenNeuro_ds003478"
        tsv_path = data_root / "participants.tsv"

        if not tsv_path.exists():
            print(f"⚠ participants.tsv 不存在: {tsv_path}", file=sys.stderr)
            return

        bdi_14_plus = 0
        bdi_7_minus = 0
        bdi_excluded = 0

        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                subj_id = row["participant_id"]
                try:
                    bdi = int(row["BDI"])
                except (ValueError, KeyError):
                    bdi_excluded += 1
                    continue

                if bdi >= 14:
                    self._bdi_labels[subj_id] = (1, "depression")
                    bdi_14_plus += 1
                elif bdi <= 7:
                    self._bdi_labels[subj_id] = (0, "control")
                    bdi_7_minus += 1
                else:
                    bdi_excluded += 1
                    # 不加入 _bdi_labels，后续 process 会跳过

        print(
            f"  [OpenNeuro] BDI 标签: {bdi_14_plus} 抑郁 (BDI>=14), "
            f"{bdi_7_minus} 对照 (BDI<=7), {bdi_excluded} 排除 (BDI 8-13)"
        )


# ================================================================
# 命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="OpenNeuro ds003478 数据集预处理")
    parser.add_argument(
        "--data_dir", type=str, default="../data/OpenNeuro_ds003478",
        help="OpenNeuro 原始数据目录",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/processed/OpenNeuro",
        help="预处理后数据输出目录",
    )
    parser.add_argument(
        "--config", type=str, default="../configs/preprocess_config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 扫描所有 .set 文件
    set_files = sorted(data_dir.rglob("*_eeg.set"))
    if not set_files:
        print(f"⚠ 未在 {data_dir} 中找到 .set 文件")
        return

    loader = OpenNeuroLoader(args.config)

    # 按受试者分组
    subjects: Dict[str, List[Path]] = {}
    for f in set_files:
        # 从路径提取 sub-XXX
        for p in f.parts:
            if p.startswith("sub-"):
                sid = p
                break
        else:
            sid = f.stem[:7]
        if sid not in subjects:
            subjects[sid] = []
        subjects[sid].append(f)

    print(f"\n{'='*60}")
    print(f"处理 OpenNeuro ds003478: {len(subjects)} 个受试者, {len(set_files)} 个文件")
    print(f"{'='*60}")

    total_files = 0
    total_subjects = 0
    skipped = 0

    for sid in sorted(subjects.keys()):
        # 先检查该受试者是否有有效标签
        label, diag_type = loader._parse_label({"subject_id": sid})
        if label == -1:
            skipped += 1
            continue

        for run_file in subjects[sid]:
            # 将不同 run 存到不同子目录，避免覆盖
            run_id = "run01"
            if "run-" in run_file.name:
                run_id = "run" + run_file.name.split("run-")[1].split("_")[0]
            out_subj_dir = output_dir / sid / run_id

            try:
                result = loader.process(str(run_file))
                loader.save(result, str(out_subj_dir))
                total_files += 1
            except Exception as e:
                print(f"  ✗ {run_file.name}: {e}", file=sys.stderr)
                skipped += 1
                continue

        total_subjects += 1
        if total_subjects % 10 == 0:
            print(f"  已处理 {total_subjects} 个受试者...")

    if skipped:
        print(f"\n  {skipped} 个文件/受试者被跳过（标签缺失或处理失败）")

    print(f"\n✓ OpenNeuro 预处理完成: {total_subjects} 个受试者, {total_files} 个文件")
    print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
