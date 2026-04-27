"""XGBoost 训练、结果分析与可视化适配层。

该目录主要用于衔接 AnnData / scVI / PCA 特征与 XGBoost 多分类流程，包含：

1. 数据集准备与 Leave-One-Donor-Out 数据导出。
2. 模型训练、评估结果回读与基础统计。
3. 常用结果图、稳定性图和 UMAP 展示。

Notes:
    1. 该层以保守兼容为优先，尽量保持原有公开函数名不变。
    2. 高频函数建议优先参考各自的 `Example` 段进行调用。
"""

from .compute import *
from .outcome_analysis import *
from .plot import *
