"""CellPhoneDB 结果整理与可视化适配层。

该目录主要用于衔接 CellPhoneDB / ktplotspy 的输出结果与项目内部分析流程，
包括：

- 结果文件读取、整理、筛选与聚类
- 输入文件准备
- 常见的 CPDB 热图、dotplot、chordplot 等可视化包装
- 若干用于下游组合图的辅助工具函数

Notes:
    1. 该层以保留 CellPhoneDB 原始输出语义为主，同时尽量简化常见使用流程。
    2. 高频公开函数应优先参考其 `Example` 段进行调用。
"""

from .cellphone_inspector import *
from .plot import *
from .toolkit import *
