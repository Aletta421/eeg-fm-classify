"""
EEG 数据预处理模块

为每个数据集编写 load_<dataset>.py，继承 BaseEEGLoader 基类。

当前状态:
    [✓] base_loader.py     — 基类（已完成）
    [✓] load_modma.py      — MODMA（抑郁症，128导+3导）
    [✓] load_ieee.py       — IEEE DataPort（ADHD儿童，19导）
    [✓] load_mendeley.py   — Mendeley Data（ADHD成人，11-29导）
    [✓] load_openneuro.py  — OpenNeuro ds003478（抑郁症，67导）
    [✓] load_tdbrain.py    — TDBRAIN（抑郁+ADHD，33导）
    [✓] generate_labels.py — 统一标签生成
"""

from .base_loader import BaseEEGLoader
