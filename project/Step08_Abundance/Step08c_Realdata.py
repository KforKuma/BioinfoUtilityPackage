"""Step08c: run canonical real-data DA from an immutable Step08a manifest."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from project.Step08_Abundance.phase4_shared import load_prepared_input
from src.stats.pipeline import PipelineRunResult, run_abundance_pipeline


# %% Configuration
CONFIG_PATH = REPOSITORY_ROOT / "config" / "phase4_step08_real_data.yml"
METHODS = ("propeller", "sccomp", "tri_anchor")
PREPARATION_MANIFEST: Path | None = None


# %% Reusable functions
def build_config(
    preparation_manifest: str | Path,
    config_path: str | Path = CONFIG_PATH,
    *,
    run_id: str | None = None,
    output_root: str | Path | None = None,
) -> dict:
    canonical, manifest = load_prepared_input(preparation_manifest)
    with Path(config_path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    config = deepcopy(document)
    if tuple(config["methods"]) != METHODS:
        raise ValueError(f"Phase-4 Step08c default methods must be {METHODS!r}.")
    contrast = canonical.contrast_specification.iloc[0]
    config["contrast"].update({
        key: contrast[key]
        for key in (
            "contrast_id", "factor", "group_1", "group_2", "reference_group",
            "reference_cell_type", "reference_selection_reason",
        )
    })
    config["real_data"] = {
        "prepared_manifest": str(Path(preparation_manifest).resolve()),
    }
    config["lineage"] = {
        "preparation_run_id": manifest["run_id"],
        "preparation_analysis_id": manifest["analysis_id"],
        "preparation_input_hash": manifest["canonical_input_hash"],
    }
    if run_id is not None:
        config["run_id"] = run_id
    if output_root is not None:
        config["output_root"] = str(Path(output_root))
    return config


def run_analysis(config: dict) -> PipelineRunResult:
    """Run methods once and emit only real-data summaries and figures."""
    return run_abundance_pipeline(config)


def main(argv: list[str] | None = None) -> PipelineRunResult:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    config = build_config(
        args.preparation_manifest,
        args.config,
        run_id=args.run_id,
        output_root=args.output_root,
    )
    pipeline_result = run_analysis(config)
    print(pipeline_result.output_root)
    return pipeline_result


# %% Interactive execution
result: PipelineRunResult | None = None

if __name__ == "__main__":
    result = main()
