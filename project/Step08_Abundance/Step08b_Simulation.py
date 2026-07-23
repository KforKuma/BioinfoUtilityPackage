"""Deprecated Step08 simulation wrapper for the canonical phase-2 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.stats.pipeline import run_abundance_pipeline


def main() -> None:
    warnings.warn(
        "Step08b_Simulation.py is now a thin compatibility wrapper; "
        "statistical logic lives in src.stats.pipeline and src.stats.evaluation.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config" / "phase2_simulation.yml",
    )
    args = parser.parse_args()
    result = run_abundance_pipeline(args.config)
    print(result.output_root)


if __name__ == "__main__":
    main()
