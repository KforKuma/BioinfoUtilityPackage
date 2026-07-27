from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


ROLE_METHODS = {"S": "sccomp", "B1": "dcats", "B2": "clr_lmm"}
VALID_DIRECTIONS = {"group_1_higher", "group_2_higher"}
SCHEMA_VERSION = "rce-production-v1"


@dataclass(frozen=True)
class ProductionClassification:
    primary_decision: bool | None
    support_tier: str
    support_direction_status: str
    supporting_methods: tuple[str, ...]
    positive_supporting_methods: tuple[str, ...]
    conflicting_methods: tuple[str, ...]
    reason_code: str


def default_production_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / "rce_production_v1.yaml"


def load_production_registry(path: str | Path | None = None) -> Mapping[str, Any]:
    source = Path(path) if path else default_production_registry_path()
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("The production registry must use rce-production-v1.")
    if document.get("production_registry_methods") != ["rce_consensus"]:
        raise ValueError("Only rce_consensus may remain in the formal production registry.")
    alias = document.get("historical_aliases", {}).get("rce_supported", {})
    if alias.get("canonical_method") != "rce_consensus":
        raise ValueError("rce_supported must be a historical alias of rce_consensus.")
    return document


def _state(states: Mapping[str, Mapping[str, Any]], role: str) -> dict[str, Any]:
    value = dict(states.get(role, {}))
    return {
        "method": ROLE_METHODS[role],
        "usable": bool(value.get("usable", False)),
        "decision": value.get("decision") if value.get("usable", False) else None,
        "direction": value.get("direction") if value.get("usable", False) else None,
    }


def classify_production_states(states: Mapping[str, Mapping[str, Any]]) -> ProductionClassification:
    anchors = {role: _state(states, role) for role in ROLE_METHODS}
    supporting = tuple(value["method"] for value in anchors.values() if value["usable"])
    positive_roles = tuple(
        role for role, value in anchors.items() if value["usable"] and value["decision"] is True
    )
    positive_methods = tuple(anchors[role]["method"] for role in positive_roles)
    positive_directions = {
        role: anchors[role]["direction"] for role in positive_roles
        if anchors[role]["direction"] in VALID_DIRECTIONS
    }
    if len(positive_directions) != len(positive_roles):
        return ProductionClassification(
            None, "unavailable", "unavailable", supporting, positive_methods, (),
            "positive_anchor_direction_unavailable",
        )
    conflicting: tuple[str, ...] = ()
    if len(set(positive_directions.values())) > 1:
        conflicting = positive_methods
        return ProductionClassification(
            None, "direction_conflict", "conflict", supporting, positive_methods,
            conflicting, "positive_anchor_direction_conflict",
        )

    s, b1, b2 = (anchors[role] for role in ("S", "B1", "B2"))
    if not s["usable"]:
        return ProductionClassification(
            None, "unavailable", "unavailable", supporting, positive_methods, (),
            "sccomp_unavailable",
        )
    if s["decision"] is True:
        if s["direction"] not in VALID_DIRECTIONS:
            return ProductionClassification(
                None, "unavailable", "unavailable", supporting, positive_methods, (),
                "sccomp_direction_unavailable",
            )
        b1_positive = b1["usable"] and b1["decision"] is True
        b2_positive = b2["usable"] and b2["decision"] is True
        if b1_positive and b2_positive:
            return ProductionClassification(
                True, "tier_1", "consistent", supporting, positive_methods, (),
                "sccomp_dcats_clr_lmm_same_direction",
            )
        if b1_positive:
            return ProductionClassification(
                True, "tier_2_dcats", "consistent", supporting, positive_methods, (),
                "sccomp_dcats_same_direction",
            )
        if b2_positive:
            return ProductionClassification(
                True, "tier_2_clr_lmm", "consistent", supporting, positive_methods, (),
                "sccomp_clr_lmm_same_direction",
            )
        if not b1["usable"] or not b2["usable"]:
            return ProductionClassification(
                None, "unavailable", "unavailable", supporting, positive_methods, (),
                "support_evidence_insufficient",
            )
        return ProductionClassification(
            False, "sccomp_only", "unsupported", supporting, positive_methods, (),
            "sccomp_only_exploratory",
        )

    if b1["usable"] and b1["decision"] is True and b2["usable"] and b2["decision"] is True:
        return ProductionClassification(
            False, "rescue_only", "support_only_consistent", supporting, positive_methods, (),
            "dcats_clr_lmm_rescue_observation",
        )
    return ProductionClassification(
        False, "no_discovery", "not_applicable", supporting, positive_methods, (),
        "rce_consensus_negative",
    )


def _effect_direction(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return "unavailable"
    if numeric > 0:
        return "group_1_higher"
    if numeric < 0:
        return "group_2_higher"
    return "no_effect"


def validate_rce_production_v1(frame: pd.DataFrame) -> pd.DataFrame:
    registry = load_production_registry()
    required = set(registry["required_columns"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"rce-production-v1 missing columns: {sorted(missing)}")
    if frame.duplicated(["run_id", "contrast_id", "cell_type"]).any():
        raise ValueError("Duplicate rce-production-v1 keys.")
    negative = frame["primary_decision"].astype("boolean").eq(False) & frame["production_eligible"]
    if not frame.loc[negative, "production_abundance_FC"].eq(1.0).all():
        raise ValueError("Production-negative eligible rows must use FC=1.")
    unresolved = frame["rce_support_tier"].isin(["direction_conflict", "unavailable"])
    if frame.loc[unresolved, "production_abundance_FC"].notna().any():
        raise ValueError("Conflict/unavailable production values must be NA.")
    positive = frame["primary_decision"].astype("boolean").eq(True) & frame["production_eligible"]
    expected = 0.5 * pd.to_numeric(frame.loc[positive, "posterior_mean_log2FC"], errors="coerce")
    if not np.allclose(frame.loc[positive, "production_abundance_log2FC"], expected, atol=1e-12):
        raise ValueError("Production log2FC must be 0.5 * posterior mean log2FC.")
    reference = frame["is_common_benchmark_reference"].astype(bool)
    if frame.loc[reference, "benchmark_eligible"].astype(bool).any():
        raise ValueError("Common benchmark references cannot be benchmark eligible.")
    return frame


def export_rce_production_v1(
    phase10_table: pd.DataFrame,
    provenance: pd.DataFrame,
    *,
    data_context: str = "simulation_benchmark",
    allow_reference_production: bool = False,
) -> pd.DataFrame:
    if data_context not in {"simulation_benchmark", "real_data"}:
        raise ValueError("data_context must be simulation_benchmark or real_data")
    provenance = provenance.loc[provenance["method"].astype(str).eq("rce_consensus"), [
        "run_id", "cell_type", "anchor_states", "decision_path"
    ]].drop_duplicates(["run_id", "cell_type"])
    frame = phase10_table.merge(
        provenance, on=["run_id", "cell_type"], how="left", validate="one_to_one"
    )
    classifications = []
    for raw in frame["anchor_states"]:
        if pd.isna(raw):
            classifications.append(classify_production_states({}))
        else:
            classifications.append(classify_production_states(json.loads(str(raw))))
    frame["primary_decision"] = pd.Series(
        [item.primary_decision for item in classifications], dtype="boolean"
    )
    frame["rce_support_tier"] = [item.support_tier for item in classifications]
    frame["rce_support_direction_status"] = [item.support_direction_status for item in classifications]
    frame["supporting_methods"] = [json.dumps(item.supporting_methods) for item in classifications]
    frame["positive_supporting_methods"] = [json.dumps(item.positive_supporting_methods) for item in classifications]
    frame["conflicting_methods"] = [json.dumps(item.conflicting_methods) for item in classifications]
    frame["rce_reason_code"] = [item.reason_code for item in classifications]

    frame = frame.rename(columns={
        "raw_abundance_FC": "phase9_point_abundance_FC",
        "raw_abundance_log2FC": "phase9_point_abundance_log2FC",
        "raw_log2FC_mean": "posterior_mean_log2FC",
        "raw_log2FC_sd": "posterior_log2FC_sd",
        "raw_log2FC_lower": "posterior_log2FC_lower",
        "raw_log2FC_upper": "posterior_log2FC_upper",
        "raw_proportion_control_mean": "posterior_mean_proportion_control",
        "raw_proportion_case_mean": "posterior_mean_proportion_case",
    })
    frame["production_effect_direction"] = frame["posterior_mean_log2FC"].map(_effect_direction)
    frame["production_direction_source"] = "sccomp_posterior_prediction"
    frame["production_interval_scope"] = "conditional_on_rce_decision_and_fixed_shrinkage"
    frame["shrinkage_selection_source"] = "selected_on_phase10_validation_using_phase9_inputs"
    frame["is_common_benchmark_reference"] = frame["is_reference_cell_type"].astype(bool)
    frame["benchmark_eligible"] = ~frame["is_common_benchmark_reference"]
    frame["outside_benchmark_validation"] = (
        frame["is_common_benchmark_reference"] & (data_context == "real_data")
    )
    uncertainty_columns = [
        "posterior_mean_log2FC", "posterior_log2FC_sd",
        "posterior_log2FC_lower", "posterior_log2FC_upper",
    ]
    available = frame[uncertainty_columns].notna().all(axis=1) & frame["primary_decision"].notna()
    if data_context == "simulation_benchmark":
        frame["production_eligible"] = available & ~frame["is_common_benchmark_reference"]
        frame["reference_policy"] = "common_benchmark_reference_excluded"
    else:
        frame["production_eligible"] = available & (
            ~frame["is_common_benchmark_reference"] | bool(allow_reference_production)
        )
        frame["reference_policy"] = np.where(
            frame["is_common_benchmark_reference"],
            "real_data_reference_explicitly_enabled" if allow_reference_production else
            "real_data_reference_disabled_by_configuration",
            "not_a_common_benchmark_reference",
        )

    positive = frame["primary_decision"].astype("boolean").eq(True) & frame["production_eligible"]
    negative = frame["primary_decision"].astype("boolean").eq(False) & frame["production_eligible"]
    frame["production_abundance_log2FC"] = np.where(
        positive, 0.5 * frame["posterior_mean_log2FC"], np.where(negative, 0.0, np.nan)
    )
    frame["production_abundance_FC"] = np.exp2(frame["production_abundance_log2FC"])
    frame["production_log2FC_lower"] = np.where(
        positive, 0.5 * frame["posterior_log2FC_lower"], np.nan
    )
    frame["production_log2FC_upper"] = np.where(
        positive, 0.5 * frame["posterior_log2FC_upper"], np.nan
    )
    frame["production_FC_lower"] = np.exp2(frame["production_log2FC_lower"])
    frame["production_FC_upper"] = np.exp2(frame["production_log2FC_upper"])
    frame["production_reason_code"] = np.select(
        [frame["is_common_benchmark_reference"] & ~frame["production_eligible"],
         frame["rce_support_tier"].eq("direction_conflict"),
         frame["rce_support_tier"].eq("unavailable"), negative, positive],
        ["benchmark_reference_excluded", "direction_conflict", "unavailable",
         "rce_negative_gated_to_neutral", "rce_positive_fixed_half_shrinkage"],
        default="not_production_eligible",
    )
    frame["schema_version"] = SCHEMA_VERSION
    frame["production_rule_id"] = "rce-consensus-production-v1.0.0"
    frame["estimator_version"] = "sccomp-2.4.0;fixed-exponent-0.5;rce-production-v1"
    drop = ["direction_status", "rce_consensus_decision", "rce_confidence_tier",
            "is_reference_cell_type", "is_evaluation_eligible", "is_production_weight_eligible",
            "shrinkage_method", "shrinkage_weight", "reason_code", "anchor_states"]
    frame = frame.drop(columns=[column for column in drop if column in frame])
    return validate_rce_production_v1(frame)
