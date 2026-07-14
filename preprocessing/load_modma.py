"""
MODMA 数据集加载器

支持三种数据子类型:
  - EEG_128channels_resting_lanzhou_2015  (128导静息态, .mat)
  - EEG_128channels_ERP_lanzhou_2015      (128导任务态, .raw)
  - EEG_3channels_resting_lanzhou_2015    (3导静息态, .txt)

标签: MDD (Major Depressive Disorder) → 1, HC (Healthy Control) → 0

使用方式:
    python load_modma.py --data_dir ../data/MODMA --output_dir ../data/processed/MODMA
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import warnings

import numpy as np
import openpyxl

# Fix Windows GBK encoding issue for Unicode characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from base_loader import BaseEEGLoader


class ModmaLoader(BaseEEGLoader):
    """MODMA 数据集加载器。

    处理流程:
        1. 读取 .mat (静息态128导) / .raw (EGI任务态) / .txt (3导) 文件
        2. 从 xlsx 元数据表中提取 MDD/HC 标签
        3. 提取通道名和标准 10-20 3D 坐标
    """

    dataset_name = "MODMA"

    # 128导 EGI HydroCel 电极名称 → 10-20 标准名映射（常见对照）
    EGI_TO_1020 = {
        "E1": "Fp1", "E2": "Fp2", "E3": "F3", "E4": "F4",
        "E5": "C3", "E6": "C4", "E7": "P3", "E8": "P4",
        "E9": "O1", "E10": "O2", "E11": "F7", "E12": "F8",
        "E13": "T3", "E14": "T4", "E15": "T5", "E16": "T6",
        "E17": "Fz", "E18": "Cz", "E19": "Pz",
        # 完整 128 导映射可在此扩展
    }

    def __init__(self, config_path: str = "../configs/preprocess_config.yaml"):
        super().__init__(config_path)
        self._label_cache: Dict[str, Tuple[int, str]] = {}  # 缓存 subject_id → (label, diagnosis)
        self._loaded_subtypes: set = set()  # 已加载标签的 subtype

    # ================================================================
    # 子类实现
    # ================================================================

    def _read_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取 MODMA .raw / .mat / .txt 文件。

        Returns:
            (eeg_data, meta): data shape (n_channels, n_samples)
        """
        file_path = Path(file_path)

        if file_path.suffix == ".raw":
            return self._read_raw_egi(str(file_path))
        elif file_path.suffix == ".txt":
            return self._read_txt_3ch(str(file_path))
        elif file_path.suffix == ".mat":
            return self._read_mat_resting(str(file_path))
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _parse_label(self, meta: Dict[str, Any]) -> Tuple[int, str]:
        """从 xlsx 元数据表中查找受试者标签。

        Args:
            meta: 必须包含 "subject_id" 和 "subtype" 字段。

        Returns:
            (label, diagnosis_type): label=1(MDD)/0(HC)，diagnosis_type="depression"/"control"
        """
        subject_id = str(meta["subject_id"])
        subtype = meta.get("subtype", "resting_128ch")

        # 使用缓存避免重复读取 xlsx
        if subtype not in self._loaded_subtypes:
            self._load_labels(subtype)

        label, diag_type = self._label_cache.get(
            subject_id, (0, "control")
        )
        return label, diag_type

    def _extract_channels(
        self, data: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[List[str], np.ndarray]:
        """提取通道名和 3D 坐标。

        128导: 尝试匹配 EGI HydroCel 128 蒙太奇 → MNE 标准 3D 坐标
        3导: 使用手动近似坐标
        """
        import mne
        n_channels = data.shape[0]

        if n_channels == 128:
            try:
                montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
                ch_names = montage.ch_names
                pos = montage.get_positions()["ch_pos"]
                positions = np.array([pos.get(ch, [0, 0, 0]) for ch in ch_names])
            except Exception:
                # 兜底：全零坐标
                ch_names = [f"E{i+1}" for i in range(n_channels)]
                positions = np.zeros((n_channels, 3))
        elif n_channels == 3:
            # MODMA 3导: Fp1, Fpz, Fp2（前额可穿戴设备）
            ch_names = ["Fp1", "Fpz", "Fp2"]
            # 近似 3D 坐标（10-20 系统）
            positions = np.array([
                [-0.028, 0.085, -0.010],   # Fp1
                [0.000, 0.095, 0.000],     # Fpz
                [0.028, 0.085, -0.010],    # Fp2
            ])
        else:
            ch_names = [f"Ch{i+1}" for i in range(n_channels)]
            positions = np.zeros((n_channels, 3))

        return ch_names, positions

    # ================================================================
    # 文件读取实现
    # ================================================================

    def _read_raw_egi(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取 EGI NetStation .raw 文件（使用 MNE）。

        .raw 文件是 EGI 专有格式，MNE 的 read_raw_egi() 可直接读取。
        数据包含 150 个通道，其中只有 E1-E128 是 EEG 信号，
        E129 及 CELL/SESS/.../swrp 等为辅助/事件通道，需过滤掉。
        """
        import mne

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_egi(file_path, preload=False, verbose=False)
            raw.load_data()

        all_data = raw.get_data()          # (n_channels, n_samples)
        fs = raw.info["sfreq"]
        all_ch_names = raw.info["ch_names"]

        # 只保留 E1–E128（EEG 通道），过滤 E129 和辅助通道
        eeg_mask = [
            ch.startswith("E") and ch[1:].isdigit() and 1 <= int(ch[1:]) <= 128
            for ch in all_ch_names
        ]
        eeg_indices = [i for i, m in enumerate(eeg_mask) if m]
        data = all_data[eeg_indices]
        ch_names = [all_ch_names[i] for i in eeg_indices]

        # 从文件名解析 subject_id
        # 文件名示例: "02010002erp 20150416 1131.raw" 或 "02010008_erp(n) 20150619 1709.raw"
        filename = Path(file_path).stem
        subject_id = filename[:8]  # 前 8 位是受试者编号

        # 推断数据子类型
        file_path_lower = str(file_path).lower()
        if "erp" in file_path_lower:
            subtype = "erp_128ch"
        else:
            subtype = "resting_128ch"

        return data, {
            "subject_id": subject_id,
            "fs": fs,
            "ch_names": ch_names,
            "subtype": subtype,
            "format": "egi_raw",
        }

    def _read_txt_3ch(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取 3 通道 .txt 文件。

        格式: 每行 3 个整数，空格分隔，约 300k 行。
        采样率约 250 Hz（参考 MODMA 文档）。
        """
        # 读取原始数据
        raw_values = np.loadtxt(file_path, dtype=np.float64)  # (n_samples, 3)

        # 转置为 (3, n_samples)
        data = raw_values.T

        # 从文件名解析 subject_id
        # 文件名示例: "02010001_still.txt"
        filename = Path(file_path).stem
        subject_id = filename[:8]

        # 采样率为 250 Hz（MODMA 3 导设备）
        fs = 250.0

        return data, {
            "subject_id": subject_id,
            "fs": fs,
            "subtype": "resting_3ch",
            "format": "txt_3ch",
        }

    def _read_mat_resting(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取静息态 128 导 .mat 文件。

        .mat 文件结构:
          - 数据键名: 动态（如 "a02010002rest_20150416_1017mat"），
            是唯一的一个 (129, N) 数组 — 前 128 行为 EEG，第 129 行为全零。
          - samplingRate: (1, 1) 标量，采样率 250 Hz。
        """
        from scipy.io import loadmat as sio_loadmat

        mat = sio_loadmat(file_path)

        # 找到数据键：忽略 __header__/__version__/__globals__ 和元数据键，
        # 在剩余的 2D 数组中取列数（时间样本）最多的那个作为 EEG 数据。
        data = None
        fs = 250.0  # 默认值
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            if key == "samplingRate":
                fs = float(value.flat[0])
            elif key == "Impedances_0":
                continue  # 阻抗值，不是 EEG 数据
            elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] > 100:
                if data is None or value.shape[1] > data.shape[1]:
                    data = value

        if data is None:
            raise ValueError(f"无法在 .mat 文件中找到 EEG 数据: {file_path}")

        # 去掉全零行（通常是最后一行，即 E129 / trigger 通道）
        valid_rows = ~np.all(data == 0, axis=1)
        if not valid_rows.all():
            data = data[valid_rows]

        # 从文件名解析 subject_id
        # 文件名示例: "02010002rest 20150416 1017..mat"
        filename = Path(file_path).stem
        subject_id = filename[:8]

        return data, {
            "subject_id": subject_id,
            "fs": fs,
            "subtype": "resting_128ch",
            "format": "mat_resting",
        }

    # ================================================================
    # 标签解析
    # ================================================================

    def _load_labels(self, subtype: str):
        """从 xlsx 文件批量加载受试者标签到缓存。"""
        # 子类型 → xlsx 路径映射
        type_map = {
            "resting_128ch": "EEG_128channels_resting_lanzhou_2015/subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx",
            "erp_128ch": "EEG_128channels_ERP_lanzhou_2015/subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx",
            "resting_3ch": "EEG_3channels_resting_lanzhou_2015/subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx",
        }

        xlsx_rel = type_map.get(subtype)
        if xlsx_rel is None:
            return

        # MODMA 数据在 data/MODMA/ 下，loader 在 preprocessing/ 下
        # 向上是项目根目录
        data_root = Path(__file__).parent.parent / "data" / "MODMA"
        xlsx_path = data_root / xlsx_rel

        if not xlsx_path.exists():
            print(f"⚠ 标签文件不存在: {xlsx_path}", file=sys.stderr)
            return

        wb = openpyxl.load_workbook(str(xlsx_path))
        ws = wb.active

        for row in ws.iter_rows(values_only=True):
            if row[0] is None or row[0] == "subject id":
                continue
            subject_id = str(row[0])
            type_label = str(row[1]) if row[1] is not None else "unknown"

            if type_label == "MDD":
                self._label_cache[subject_id] = (1, "depression")
            elif type_label == "HC":
                self._label_cache[subject_id] = (0, "control")
            else:
                # 未知标签 → 跳过该受试者（不缓存，后续会返回默认值）
                pass

        print(f"  [MODMA] 已加载 {len(self._label_cache)} 条标签 ({subtype})")
        self._loaded_subtypes.add(subtype)


# ================================================================
# 命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="MODMA 数据集预处理")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/MODMA",
        help="MODMA 原始数据目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/processed/MODMA",
        help="预处理后数据输出目录",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        choices=["resting_128ch", "erp_128ch", "resting_3ch", "all"],
        default="all",
        help="要处理的数据子类型 (默认: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/preprocess_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument("--skip_epoching", action="store_true",
                        help="只跑 Step 1-5, 跳过 Step 7 滑窗分段")
    parser.add_argument("--epoch_only", action="store_true",
                        help="只对已有连续数据执行 Step 7 分段，不重新预处理")
    args = parser.parse_args()

    # 子类型与目录的映射
    subtype_dirs = {
        "resting_128ch": "EEG_128channels_resting_lanzhou_2015",
        "erp_128ch": "EEG_128channels_ERP_lanzhou_2015",
        "resting_3ch": "EEG_3channels_resting_lanzhou_2015",
    }

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.subtype != "all":
        subtype_dirs = {args.subtype: subtype_dirs[args.subtype]}

    loader = ModmaLoader(args.config)

    # --epoch_only 模式：只对已有连续数据分段
    if args.epoch_only:
        print(f"Step 7 分段模式: {output_dir}")
        loader.epoch_output_dir(str(output_dir))
        print(f"\n✓ 分段完成")
        return

    total_files = 0
    total_subjects = 0

    for subtype, subdir in subtype_dirs.items():
        sub_path = data_dir / subdir
        if not sub_path.exists():
            print(f"⚠ 跳过不存在的目录: {sub_path}")
            continue

        # 查找所有数据文件
        eeg_files = list(sub_path.glob("*.raw")) + list(sub_path.glob("*.txt")) + list(sub_path.glob("*.mat"))

        # 按受试者分组（同一受试者可能有多个文件）
        subjects: Dict[str, List[Path]] = {}
        for f in eeg_files:
            sid = f.stem[:8]  # 前 8 位
            if sid not in subjects:
                subjects[sid] = []
            subjects[sid].append(f)

        print(f"\n{'='*60}")
        print(f"处理 MODMA/{subtype}: {len(subjects)} 个受试者, {len(eeg_files)} 个文件")
        print(f"{'='*60}")

        skipped = 0
        for sid, files in subjects.items():
            out_subj_dir = output_dir / subtype / sid

            for f in files:
                try:
                    result = loader.process(str(f), skip_epoching=args.skip_epoching)
                    loader.save(result, str(out_subj_dir))
                    total_files += 1
                except Exception as e:
                    print(f"  ✗ {f.name}: {e}", file=sys.stderr)
                    skipped += 1
                    continue

            total_subjects += 1
            if total_subjects % 10 == 0:
                print(f"  已处理 {total_subjects} 个受试者...")

        if skipped:
            print(f"  {skipped} 个文件处理失败")

    print(f"\n✓ MODMA 预处理完成: {total_subjects} 个受试者, {total_files} 个文件")
    print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
