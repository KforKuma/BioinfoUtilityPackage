"""Deprecated compatibility import for the canonical pipeline package."""

import warnings

warnings.warn(
    "src.stats.runners is deprecated; import runner APIs from src.stats.pipeline.",
    DeprecationWarning,
    stacklevel=2,
)

from src.stats.pipeline import DifferentialAbundanceRunner, RunnerResult

__all__ = ["DifferentialAbundanceRunner", "RunnerResult"]
