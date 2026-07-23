from .anova import run_ANOVA_naive, run_ANOVA_transformed
from .clr import run_CLR_LMM, run_CLR_LMM_with_LFC, run_pCLR_LMM, run_pCLR_OLS
from .dirichlet import run_Dirichlet_Wald, run_Dirichlet_Multinomial_Wald
from .dkd import run_DKD
from .lmm import run_LMM
from .perm import run_Perm_Mixed
# from .sccoda import run_scCODA

def run_PyDESeq2(*args, **kwargs):
    """Lazily import the optional PyDESeq2 implementation."""
    from .deseq2 import run_PyDESeq2 as _run_pydeseq2

    return _run_pydeseq2(*args, **kwargs)

# 在 stats.engine 中集中导出丰度统计方法。
# scCODA 依赖较重且容易在导入阶段产生大量 warning，因此默认不导出。
__all__ = [
    "run_ANOVA_naive", "run_ANOVA_transformed",
    "run_CLR_LMM", "run_CLR_LMM_with_LFC", "run_pCLR_LMM", "run_pCLR_OLS",
    "run_PyDESeq2",
    "run_Dirichlet_Wald", "run_Dirichlet_Multinomial_Wald",
    "run_DKD",
    "run_LMM",
    "run_Perm_Mixed",
    # 'run_scCODA'
]
