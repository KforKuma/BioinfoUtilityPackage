"""Deprecated compatibility wrapper for evaluation metric primitives."""

import warnings

warnings.warn(
    "src.stats.benchmark_metrics is deprecated; use src.stats.evaluation.benchmark_metrics.",
    DeprecationWarning,
    stacklevel=2,
)

from src.stats.evaluation.benchmark_metrics import *  # noqa: F401,F403
