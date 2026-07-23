from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats.validation import parse_boolean_series


REQUIRED_TRUTH_COLUMNS = {
    "run_id",
    "analysis_id",
    "scenario_id",
    "replicate_id",
    "simulation_seed",
    "contrast_id",
    "cell_type",
    "effect_component",
    "truth_source",
    "is_true_effect",
    "effect_value",
    "effect_estimand",
    "effect_scale",
    "reference_cell_type",
}

TRUTH_KEY_COLUMNS = [
    "run_id",
    "analysis_id",
    "contrast_id",
    "cell_type",
    "effect_component",
    "truth_source",
]

TRUTH_SOURCES = {"injected", "population", "observed"}


def validate_truth_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate independently persisted simulation truth without choosing a source."""
    missing = REQUIRED_TRUTH_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Truth table is missing required columns: {sorted(missing)}")
    result = frame.copy()
    identity = [
        "run_id", "analysis_id", "scenario_id", "replicate_id", "simulation_seed",
        "contrast_id", "cell_type", "effect_component", "truth_source",
    ]
    if result[identity].isna().any().any():
        raise ValueError("Truth identity and simulation-lineage columns must be non-missing.")
    if result["run_id"].astype(str).eq(result["analysis_id"].astype(str)).any():
        raise ValueError("Truth `run_id` and `analysis_id` must remain distinct.")
    invalid_sources = set(result["truth_source"].astype(str)) - TRUTH_SOURCES
    if invalid_sources:
        raise ValueError(f"Unknown truth sources: {sorted(invalid_sources)}")
    if result.duplicated(TRUTH_KEY_COLUMNS).any():
        duplicates = result.loc[result.duplicated(TRUTH_KEY_COLUMNS, keep=False), TRUTH_KEY_COLUMNS]
        raise ValueError(f"Duplicate truth keys detected: {duplicates.head().to_dict('records')}")

    result["is_true_effect"] = parse_boolean_series(result["is_true_effect"])
    effect = pd.to_numeric(result["effect_value"], errors="coerce")
    status = result.get("truth_status", pd.Series("available", index=result.index)).astype(str)
    available = status.eq("available")
    if result.loc[available, "is_true_effect"].isna().any():
        raise ValueError("Available truth rows require a non-missing is_true_effect.")
    if (effect.loc[available].isna() | ~np.isfinite(effect.loc[available])).any():
        raise ValueError("Available truth rows require a finite effect_value.")
    if result.loc[available, ["effect_estimand", "effect_scale"]].isna().any().any():
        raise ValueError("Available truth rows require effect estimand and scale.")
    result["effect_value"] = effect
    result["truth_status"] = status
    return result
