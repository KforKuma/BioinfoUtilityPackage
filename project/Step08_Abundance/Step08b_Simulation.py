"""Step08b: run the canonical small simulation workflow, optionally linked to Step08a."""

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
CONFIG_PATH = REPOSITORY_ROOT / "config" / "phase4_step08_simulation.yml"
METHODS = ("propeller", "sccomp", "tri_anchor")
PREPARATION_MANIFEST: Path | None = None


# %% Reusable functions
def build_config(
    config_path: str | Path = CONFIG_PATH,
    *,
    preparation_manifest: str | Path | None = PREPARATION_MANIFEST,
    run_id: str | None = None,
    output_root: str | Path | None = None,
) -> dict:
    with Path(config_path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    config = deepcopy(document)
    if tuple(config["methods"]) != METHODS:
        raise ValueError(f"Phase-4 Step08b default methods must be {METHODS!r}.")
    if preparation_manifest is not None:
        canonical, manifest = load_prepared_input(preparation_manifest)
        included = canonical.cell_type_manifest["inclusion_status"].eq("included")
        n_celltypes = int(included.sum())
        # The DM generator uses synthetic CT labels; the protected final CT is its
        # preregistered reference. No real-data effect estimate is transferred.
        reference = f"CT{n_celltypes}"
        totals = canonical.abundance_long.groupby("sample_id")["count"].sum()
        parameters = config["simulation"]["parameters"]
        parameters["n_celltypes"] = n_celltypes
        parameters["total_count_mean"] = float(totals.median())
        parameters["total_count_sd"] = float(min(totals.std(ddof=1), 500.0))
        parameters["protected_cell_types"] = [reference]
        parameters["population_reference_cell_type"] = reference
        config["contrast"]["reference_cell_type"] = reference
        config["lineage"] = {
            "preparation_manifest": str(Path(preparation_manifest).resolve()),
            "preparation_run_id": manifest["run_id"],
            "preparation_input_hash": manifest["canonical_input_hash"],
        }
    if run_id is not None:
        config["run_id"] = run_id
    if output_root is not None:
        config["output_root"] = str(Path(output_root))
    return config


def run_analysis(config: dict | str | Path = CONFIG_PATH) -> PipelineRunResult:
    """Run DM -> adapters/Tri_anchor -> canonical evaluation -> plot-ready figure."""
    return run_abundance_pipeline(config)


def main(argv: list[str] | None = None) -> PipelineRunResult:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--preparation-manifest", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    config = build_config(
        args.config,
        preparation_manifest=args.preparation_manifest,
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
