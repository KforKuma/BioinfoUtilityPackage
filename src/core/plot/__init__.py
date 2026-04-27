"""`src.core.plot` 的绘图工具集合。

该模块主要整理围绕 AnnData / Scanpy 的常用可视化函数，包括：

- 基础图形，如 stacked bar、piechart 和 stacked violin
- Scanpy 相关图形，如 UMAP、dotplot、matrixplot
- 特定分析场景下的 PCA、回归和表达相关性可视化

Notes:
    1. 该层以“化繁为简”为目标，尽量保持与上层分析流程兼容。
    2. 具体函数的使用示例请参考各子模块中的详细 `Example` 段。
"""
