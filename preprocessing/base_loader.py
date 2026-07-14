"""
EEG 数据集加载器基类

所有数据集 loader 必须继承此类，实现统一的输入输出接口。

7-Step 预处理流程:
  1. 重采样 → 200 Hz
  2. 带通滤波 0.5-99.5 Hz (4阶 Butterworth)
  3. 陷波滤波 50 Hz (IIR Notch)
  4. Z-score 归一化 (per recording, per channel)
  5. 极端值裁剪 ±15σ
  6. 确定 80% train / 20% test split (按受试者分层, split_data.py)
  7. 滑窗分段 10s (2000 samples, 无重叠)

两阶段预处理:
    # 阶段 1: 信号处理 (Step 1-5), 跳过分段
    result = loader.process(file, skip_epoching=True)
    loader.save(result, output_dir)   # 保存为 (n_channels, n_samples)

    # 确定 split (Step 6) — 由 run_preprocess.py 调用 split_data.py

    # 阶段 2: 滑窗分段 (Step 7)
    loader.epoch_output_dir(output_dir)  # 加载连续数据 → 分段 → 保存

使用方式:
    from preprocessing.base_loader import BaseEEGLoader

    class ModmaLoader(BaseEEGLoader):
        def _read_file(self, file_path):
            ...
        def _parse_label(self, meta):
            ...
        def _extract_channels(self, raw_data):
            ...
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import yaml
from scipy import signal


class BaseEEGLoader(ABC):
    """EEG 数据加载器抽象基类。

    子类只需实现三个方法：
        _read_file()    — 读取原始文件
        _parse_label()  — 从元数据中提取 0/1 标签
        _extract_channels() — 提取通道名和 3D 坐标
    """

    def __init__(self, config_path: str = "../configs/preprocess_config.yaml"):
        """初始化加载器。

        Args:
            config_path: 预处理配置文件路径。
        """
        self.config = self._load_config(config_path)
        self.signal_cfg = self.config["signal"]
        self.norm_cfg = self.config["normalize"]
        self.epoch_cfg = self.config["epoching"]
        self.chan_cfg = self.config["channels"]
        self.out_cfg = self.config["output"]

    # ================================================================
    # 公共接口（子类不要覆盖）
    # ================================================================

    def process(self, file_path: str, skip_epoching: bool = False) -> Dict[str, Any]:
        """完整处理流程：读取 → 滤波 → 重采样 → 归一化 → (可选分段)。

        两阶段预处理:
          阶段 1: process(file, skip_epoching=True) → 只跑 Step 1-5
          阶段 2: epoch_output_dir() → 加载连续数据, 跑 Step 7

        Args:
            file_path: 原始数据文件路径。
            skip_epoching: 若为 True，跳过 Step 7 (滑窗分段)，返回连续数据。
                           用于两阶段预处理：先跑 Step 1-5，确定 split，再跑 Step 7。

        Returns:
            {
                "eeg": np.ndarray,          # (n_epochs, n_channels, n_samples) 或 (n_channels, n_samples)
                "channels": list[str],       # 通道名称列表
                "ch_positions": np.ndarray,  # (n_channels, 3) 3D坐标
                "label": int,                # 0=对照, 1=患者
                "meta": {
                    "subject_id": str,
                    "dataset": str,
                    "original_fs": float,
                    "n_channels": int,
                    "diagnosis_type": str,
                    "source_file": str,
                }
            }
        """
        file_path = Path(file_path)

        # Step 1: 读取原始文件
        raw_data, raw_meta = self._read_file(str(file_path))

        # Step 2: 解析标签
        label, diagnosis_type = self._parse_label(raw_meta)

        # Step 3: 提取通道信息
        channels, ch_positions = self._extract_channels(raw_data, raw_meta)

        # Step 1-5: 重采样/滤波/陷波/zscore/clip
        eeg = self._preprocess_signal(raw_data, raw_meta)

        # Step 7: 滑窗分段（可选跳过，用于两阶段预处理）
        if skip_epoching:
            epochs = eeg  # 返回连续数据 (n_channels, n_samples)
        else:
            epochs = self._epoch(eeg)  # 返回 (n_epochs, n_channels, n_samples)

        # 组装元数据
        meta = {
            "subject_id": raw_meta.get("subject_id", file_path.stem),
            "dataset": self.dataset_name,
            "original_fs": raw_meta.get("fs", self.signal_cfg["target_fs"]),
            "n_channels": eeg.shape[0],
            "diagnosis_type": diagnosis_type,
            "source_file": str(file_path),
            **raw_meta,
        }

        return {
            "eeg": epochs,
            "channels": channels,
            "ch_positions": ch_positions,
            "label": label,
            "meta": meta,
        }

    def save(self, result: Dict[str, Any], output_dir: str):
        """保存预处理结果。

        Args:
            result: process() 返回的字典。
            output_dir: 输出目录。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subj_id = result["meta"]["subject_id"]

        if self.out_cfg["format"] == "npy":
            np.save(output_dir / f"{subj_id}_eeg.npy", result["eeg"])
            np.save(output_dir / f"{subj_id}_ch_pos.npy", result["ch_positions"])
        elif self.out_cfg["format"] == "pt":
            import torch
            torch.save(torch.tensor(result["eeg"]), output_dir / f"{subj_id}_eeg.pt")

        # 通道名和元数据单独保存
        import json
        with open(output_dir / f"{subj_id}_meta.json", "w", encoding="utf-8") as f:
            save_meta = {k: v for k, v in result["meta"].items() if not isinstance(v, np.ndarray)}
            save_meta["channels"] = result["channels"]
            save_meta["label"] = result["label"]
            json.dump(save_meta, f, indent=2, default=str, ensure_ascii=False)

    def apply_epoching(self, continuous_data: np.ndarray) -> np.ndarray:
        """对已预处理（step 1-5）的连续数据应用滑窗分段。

        用于两阶段预处理流程：
            阶段 1: process(file, skip_epoching=True) → save_continuous()
            阶段 2: 确定 split → apply_epoching() → save()

        Args:
            continuous_data: (n_channels, n_samples) 连续 EEG 数据。

        Returns:
            (n_epochs, n_channels, window_samples) 分段后数据。
        """
        return self._epoch(continuous_data)

    def save_continuous(self, result: Dict[str, Any], output_dir: str):
        """保存未分段的连续预处理数据（两阶段预处理的中间产物）。

        保存格式与 save() 相同，但 eeg 为 (n_channels, n_samples) 而非
        (n_epochs, n_channels, n_samples)。后续可调用 epoch_output_dir() 完成分段。

        Args:
            result: process(file, skip_epoching=True) 返回的字典。
            output_dir: 输出目录。
        """
        self.save(result, output_dir)

    def epoch_output_dir(self, output_dir: str):
        """Step 7: 对输出目录中所有连续数据进行滑窗分段。

        扫描 output_dir 下所有 *_eeg.npy 文件:
          - 若已是 3D (n_epochs, n_channels, n_samples) → 跳过
          - 若是 2D (n_channels, n_samples) → 应用 _epoch() → 保存为 3D

        用于两阶段预处理的阶段 2。

        Args:
            output_dir: 包含连续 *_eeg.npy 文件的目录。
        """
        import json

        output_dir = Path(output_dir)
        npy_files = sorted(output_dir.rglob("*_eeg.npy"))

        epoched = 0
        skipped = 0
        for npy_path in npy_files:
            data = np.load(str(npy_path), mmap_mode="r")
            if data.ndim == 3:
                skipped += 1
                continue  # 已分段，跳过

            # 加载全量数据并分段
            data = np.array(data, dtype=np.float64)
            epochs = self._epoch(data)  # (n_epochs, n_channels, window_samples)
            np.save(str(npy_path), epochs.astype(np.float32))

            # 更新 meta 文件中的 n_epochs
            meta_path = npy_path.parent / f"{npy_path.stem.replace('_eeg', '')}_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["n_epochs"] = int(epochs.shape[0])
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

            epoched += 1

        print(f"  [epoch_output_dir] {output_dir}: {epoched} 文件分段, {skipped} 跳过 (已分段)")

    # ================================================================
    # 子类必须实现的方法
    # ================================================================

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """数据集名称，如 "MODMA", "IEEE_ADHD"."""
        ...

    @abstractmethod
    def _read_file(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """读取原始文件。

        Args:
            file_path: 原始文件路径。

        Returns:
            (eeg_data, meta_dict)
            eeg_data: (n_channels, n_samples) EEG 数值矩阵。
            meta_dict: 至少包含 {"subject_id", "fs"}。
        """
        ...

    @abstractmethod
    def _parse_label(self, meta: Dict[str, Any]) -> Tuple[int, str]:
        """从元数据中解析标签。

        Args:
            meta: _read_file 返回的元数据字典。

        Returns:
            (label, diagnosis_type)
            label: 0=对照, 1=患者。
            diagnosis_type: "depression" / "adhd" / "control"。
        """
        ...

    @abstractmethod
    def _extract_channels(
        self, data: np.ndarray, meta: Dict[str, Any]
    ) -> Tuple[List[str], np.ndarray]:
        """提取通道名称和 3D 电极坐标。

        Args:
            data: (n_channels, n_samples) EEG 数据。
            meta: _read_file 返回的元数据字典。

        Returns:
            (channel_names, positions)
            channel_names: 通道名列表，长度 = n_channels。
            positions: (n_channels, 3) 电极 3D 坐标数组。
                       若无位置信息，返回全零数组。
        """
        ...

    # ================================================================
    # 内部方法（统一实现，子类通常不需要覆盖）
    # ================================================================

    def _preprocess_signal(
        self, data: np.ndarray, meta: Dict[str, Any]
    ) -> np.ndarray:
        """Step 1-5: 信号预处理流水线。"""
        original_fs = meta.get("fs", self.signal_cfg["target_fs"])
        target_fs = self.signal_cfg["target_fs"]

        # Step 1: 重采样
        if original_fs != target_fs:
            data = self._resample(data, original_fs, target_fs)

        # Step 2: 带通滤波 (0.5-99.5 Hz)
        data = self._bandpass_filter(data, target_fs)

        # Step 3: 陷波滤波 (50 Hz)
        data = self._notch_filter(data, target_fs)

        # Step 4: Z-score 归一化
        if self.norm_cfg["method"] == "zscore":
            data = self._zscore_normalize(data)

        # Step 5: 极端值裁剪 (±15σ)
        data = self._clip_extremes(data)

        return data

    def _resample(self, data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
        """重采样到目标采样率。"""
        if orig_fs == target_fs:
            return data
        n_target = int(data.shape[1] * target_fs / orig_fs)
        return signal.resample(data, n_target, axis=1)

    def _bandpass_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """0.5-99.5 Hz 带通滤波。"""
        nyq = fs / 2
        low = self.signal_cfg["lowcut"] / nyq
        high = self.signal_cfg["highcut"] / nyq
        if high >= 1.0:
            high = 0.99
        b, a = signal.butter(4, [low, high], btype="band")
        return signal.filtfilt(b, a, data, axis=1)

    def _notch_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """陷波滤波去除工频干扰。"""
        notch = self.signal_cfg["notch"]
        if notch is None:
            return data
        q = 30.0
        b, a = signal.iirnotch(notch, q, fs)
        return signal.filtfilt(b, a, data, axis=1)

    def _zscore_normalize(self, data: np.ndarray) -> np.ndarray:
        """按记录独立做 Z-score 归一化。"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std < 1e-8] = 1.0
        return (data - mean) / std

    def _clip_extremes(self, data: np.ndarray) -> np.ndarray:
        """裁剪极端值 (REVE: >15σ)。"""
        clip_val = self.norm_cfg["clip_sigma"]
        return np.clip(data, -clip_val, clip_val)

    def _epoch(self, data: np.ndarray) -> np.ndarray:
        """Step 7: 滑窗分段 — 将连续数据切为固定窗口。"""
        window_samples = int(self.epoch_cfg["window_sec"] * self.signal_cfg["target_fs"])
        overlap = self.epoch_cfg["overlap_pct"]
        stride = int(window_samples * (1 - overlap))

        n_channels, n_total = data.shape
        if n_total < window_samples:
            # 数据太短：垫零到至少一个窗口
            padded = np.zeros((n_channels, window_samples))
            padded[:, :n_total] = data
            return padded[np.newaxis, ...]

        epochs = []
        for start in range(0, n_total - window_samples + 1, stride):
            epochs.append(data[:, start : start + window_samples])

        return np.stack(epochs, axis=0)  # (n_epochs, n_channels, n_samples)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """加载 YAML 配置文件。"""
        config_file = Path(__file__).parent / config_path
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
