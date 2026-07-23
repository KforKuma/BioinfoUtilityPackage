"""Lightweight statistical core API with lazy legacy exports.

Importing :mod:`src.stats` does not initialise plotting, Scanpy or optional
method dependencies. Established entry points are resolved on first access.
"""

from importlib import import_module

from .benchmark_metrics import (
    ConfusionCounts,
    calculate_binary_metrics,
    confusion_counts,
    estimate_fdr_from_replicates,
)
from .contrasts import WaldContrastResult, wald_linear_contrast
from .multiple_testing import apply_bh_by_family
from .validation import parse_boolean_series, parse_boolean_value


_LAZY_EXPORTS = {
    "run_ANOVA_naive": ("src.stats.engine.anova", "run_ANOVA_naive"),
    "run_ANOVA_transformed": ("src.stats.engine.anova", "run_ANOVA_transformed"),
    "run_CLR_LMM": ("src.stats.engine.clr", "run_CLR_LMM"),
    "run_CLR_LMM_with_LFC": ("src.stats.engine.clr", "run_CLR_LMM_with_LFC"),
    "run_pCLR_LMM": ("src.stats.engine.clr", "run_pCLR_LMM"),
    "run_pCLR_OLS": ("src.stats.engine.clr", "run_pCLR_OLS"),
    "run_PyDESeq2": ("src.stats.engine.deseq2", "run_PyDESeq2"),
    "run_Dirichlet_Wald": ("src.stats.engine.dirichlet", "run_Dirichlet_Wald"),
    "run_Dirichlet_Multinomial_Wald": (
        "src.stats.engine.dirichlet",
        "run_Dirichlet_Multinomial_Wald",
    ),
    "run_DKD": ("src.stats.engine.dkd", "run_DKD"),
    "run_LMM": ("src.stats.engine.lmm", "run_LMM"),
    "run_Perm_Mixed": ("src.stats.engine.perm", "run_Perm_Mixed"),
    "simulate_DM_data": ("src.stats.simulation.dm", "simulate_DM_data"),
    "simulate_LogisticNormal_hierarchical": (
        "src.stats.simulation.ln",
        "simulate_LogisticNormal_hierarchical",
    ),
    "simulate_CLR_resample_data": (
        "src.stats.simulation.resample",
        "simulate_CLR_resample_data",
    ),
    "collect_real_data_results": (
        "src.stats.real_data_analysis",
        "collect_real_data_results",
    ),
    "run_Meta_Ensemble": ("src.stats.meta_engine.Tri_anchor", "run_Meta_Ensemble"),
    "run_Meta_Ensemble_adaptive": (
        "src.stats.meta_engine.Tri_anchor",
        "run_Meta_Ensemble_adaptive",
    ),
    "make_input": ("src.stats.support", "make_input"),
    "make_result": ("src.stats.support", "make_result"),
    "run_abundance_pipeline": ("src.stats.pipeline", "run_abundance_pipeline"),
}

__all__ = [
    "ConfusionCounts",
    "WaldContrastResult",
    "apply_bh_by_family",
    "calculate_binary_metrics",
    "confusion_counts",
    "estimate_fdr_from_replicates",
    "parse_boolean_series",
    "parse_boolean_value",
    "wald_linear_contrast",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
