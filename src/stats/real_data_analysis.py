"""Deprecated compatibility wrapper for the canonical real-data pipeline helpers."""

import warnings

warnings.warn(
    "src.stats.real_data_analysis is deprecated; use src.stats.pipeline.real_data_analysis.",
    DeprecationWarning,
    stacklevel=2,
)

from src.stats.pipeline.real_data_analysis import *  # noqa: F401,F403
