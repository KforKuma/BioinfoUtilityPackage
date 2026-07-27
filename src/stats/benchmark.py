"""Deprecated compatibility wrapper for :mod:`src.stats.evaluation.benchmark`."""

import warnings

warnings.warn(
    "src.stats.benchmark is deprecated; use src.stats.evaluation.benchmark.",
    DeprecationWarning,
    stacklevel=2,
)

from src.stats.evaluation.benchmark import *  # noqa: F401,F403
from src.stats.evaluation.benchmark import main


if __name__ == "__main__":
    main()
