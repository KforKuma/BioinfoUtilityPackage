from .dm import simulate_DM_data
from .ln import simulate_LogisticNormal_hierarchical
from .resample import simulate_CLR_resample_data
# 为了方便你在 stats/__init__.py 里一次性导出
__all__ = [
    "simulate_DM_data",
    "simulate_LogisticNormal_hierarchical",
    "simulate_CLR_resample_data"
]