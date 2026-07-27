from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
import yaml

from src.stats.schemas import validate_contrast_public_view, validate_evidence_layer
from src.stats.validation import parse_boolean_series


_KEY = ["run_id", "analysis_id", "contrast_id", "cell_type", "effect_component"]
_DIRECTIONS = {"group_1_higher", "group_2_higher"}
_LOGICS = {"baseline", "primary", "high_recall", "consensus", "supported"}


@dataclass(frozen=True)
class RCEModule:
    method_id: str
    logic: str
    required_roles: tuple[str, ...]
    decision_rule_id: str
    description: str

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "RCEModule":
        module = cls(
            method_id=str(record["method_id"]),
            logic=str(record["logic"]),
            required_roles=tuple(map(str, record["required_roles"])),
            decision_rule_id=str(record["decision_rule_id"]),
            description=str(record["description"]),
        )
        if module.logic not in _LOGICS:
            raise ValueError(f"Unknown RCE logic: {module.logic!r}")
        if not module.required_roles or not module.decision_rule_id:
            raise ValueError("RCE modules require explicit roles and a decision rule ID.")
        return module


@dataclass(frozen=True)
class RCERegistry:
    schema_version: str
    rule_version: str
    roles: Mapping[str, str]
    modules: tuple[RCEModule, ...]

    def validate(self) -> "RCERegistry":
        if set(self.roles) != {"S", "B1", "B2", "D"}:
            raise ValueError("RCE roles must be exactly S, B1, B2, and D.")
        if len({module.method_id for module in self.modules}) != len(self.modules):
            raise ValueError("RCE module IDs must be unique.")
        for module in self.modules:
            unknown = set(module.required_roles) - set(self.roles)
            if unknown:
                raise ValueError(f"RCE module {module.method_id!r} has unknown roles: {sorted(unknown)}")
        return self


def default_rce_registry_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "project" / "Step08_Abundance" / "configs" / "rce_module_registry.yaml"
    )


def load_rce_registry(path: str | Path | None = None) -> RCERegistry:
    source = Path(path) if path is not None else default_rce_registry_path()
    with source.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("RCE registry must be a YAML mapping.")
    return RCERegistry(
        schema_version=str(document["schema_version"]),
        rule_version=str(document["rule_version"]),
        roles={str(key): str(value) for key, value in document["roles"].items()},
        modules=tuple(RCEModule.from_mapping(item) for item in document["modules"]),
    ).validate()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchor_state(group: pd.DataFrame, registry: RCERegistry) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role, method in registry.roles.items():
        rows = group.loc[group["method"].astype(str).eq(method)]
        if len(rows) > 1:
            raise ValueError(f"Duplicate RCE anchor row for role={role}, method={method}.")
        if rows.empty:
            result[role] = {"method": method, "usable": False, "decision": None,
                            "direction": None, "reason": "missing_result"}
            continue
        row = rows.iloc[0]
        decision = parse_boolean_series(pd.Series([row["primary_decision"]])).iloc[0]
        usable = bool(
            row["contrast_status"] == "success"
            and bool(row["is_available"])
            and bool(row["is_valid"])
            and pd.notna(decision)
        )
        result[role] = {
            "method": method,
            "usable": usable,
            "decision": bool(decision) if usable else None,
            "direction": str(row["effect_direction"]) if usable else None,
            "reason": None if usable else (
                str(row["failure_reason"]) if pd.notna(row.get("failure_reason"))
                else "anchor_unavailable_or_invalid"
            ),
        }
    return result


def _same_positive_direction(states: dict[str, dict[str, Any]], roles: tuple[str, ...]) -> tuple[bool, str | None]:
    selected = [states[role] for role in roles]
    if not all(item["usable"] and item["decision"] for item in selected):
        return False, None
    directions = [item["direction"] for item in selected]
    if any(direction not in _DIRECTIONS for direction in directions):
        return False, None
    return len(set(directions)) == 1, directions[0] if len(set(directions)) == 1 else None


def _evaluate_module(
    module: RCEModule,
    states: dict[str, dict[str, Any]],
) -> tuple[bool | None, str, str]:
    """Return decision, decision path, and limitation reason using three-valued logic."""
    s, b1, b2, d = (states[role] for role in ("S", "B1", "B2", "D"))
    if module.logic == "baseline":
        if not s["usable"]:
            return None, "S unavailable", "required_anchor_unavailable"
        return bool(s["decision"]), "S positive" if s["decision"] else "S negative", ""

    if module.logic == "primary":
        rescue, _ = _same_positive_direction(states, ("B1", "B2"))
        if s["usable"] and s["decision"]:
            return True, "S positive", ""
        if rescue:
            return True, "B1 AND B2 positive with matching direction", ""
        if s["usable"] and not s["decision"] and b1["usable"] and b2["usable"]:
            return False, "S negative; B1/B2 rescue condition not met", ""
        return None, "primary rule unresolved", "required_anchor_unavailable"

    if module.logic == "high_recall":
        usable = [item for item in (s, d) if item["usable"]]
        if any(item["decision"] for item in usable):
            return True, "S OR D positive", ""
        if len(usable) == 2:
            return False, "S AND D negative", ""
        return None, "S OR D unresolved", "required_anchor_unavailable"

    if module.logic == "consensus":
        anchors = (s, b1, b2)
        positive_roles = tuple(
            role for role in ("S", "B1", "B2")
            if states[role]["usable"] and states[role]["decision"]
        )
        for left_index, left in enumerate(positive_roles):
            for right in positive_roles[left_index + 1:]:
                agreed, _ = _same_positive_direction(states, (left, right))
                if agreed:
                    return True, f"two-of-three positive with matching direction: {left}+{right}", ""
        unavailable_count = sum(not item["usable"] for item in anchors)
        if len(positive_roles) + unavailable_count >= 2:
            return None, "two-of-three rule unresolved", "required_anchor_unavailable_or_direction_missing"
        return False, "fewer than two possible positive anchors", ""

    if module.logic == "supported":
        if not s["usable"]:
            return None, "S unavailable", "required_anchor_unavailable"
        if not s["decision"]:
            return False, "S negative", ""
        for role in ("B1", "B2"):
            agreed, _ = _same_positive_direction(states, ("S", role))
            if agreed:
                return True, f"S positive supported by {role} with matching direction", ""
        if b1["usable"] and b2["usable"]:
            return False, "S positive but unsupported by B1/B2", ""
        return None, "S support rule unresolved", "support_anchor_unavailable"
    raise AssertionError(module.logic)


def _direction_status(
    states: dict[str, dict[str, Any]], decision: bool | None, roles: tuple[str, ...]
) -> tuple[str, str]:
    positive = [states[role] for role in roles if states[role]["usable"] and states[role]["decision"]]
    directions = [item["direction"] for item in positive if item["direction"] in _DIRECTIONS]
    if decision is None:
        return "unavailable", "not_applicable"
    if not decision:
        return "not_applicable", "not_applicable"
    if len(set(directions)) > 1:
        return "conflict", "undetermined"
    if len(directions) == 1:
        return ("single_anchor" if len(positive) == 1 else "consistent"), directions[0]
    return "direction_missing", "undetermined"


def _confidence_tier(
    states: dict[str, dict[str, Any]], decision: bool | None, status: str,
    roles: tuple[str, ...],
) -> str:
    if decision is None:
        return "unavailable"
    if not decision:
        return "no_discovery"
    count = sum(states[role]["usable"] and states[role]["decision"] for role in roles)
    if status == "conflict":
        return "direction_conflict"
    if count >= 3:
        return "high_support"
    if count == 2:
        return "supported"
    return "single_anchor"


def combine_rce_modules(
    public_view: pd.DataFrame,
    registry: RCERegistry | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine canonical public decisions into all configured RCE modules.

    Returns schema-valid public and evidence layers plus an explicit provenance table.
    Native evidence values are never read or numerically converted.
    """
    registry = (registry or load_rce_registry()).validate()
    anchors = public_view.loc[
        public_view["method"].astype(str).isin(registry.roles.values())
        & public_view["effect_component"].astype(str).eq("composition")
    ].copy()
    if anchors.empty:
        raise ValueError("No configured RCE anchor methods are present.")
    # A heterogeneous all-method CSV may deserialize numeric decision fields as
    # strings because another method uses a native Boolean metric. Restore the
    # declared numeric type before schema validation; no evidence value changes.
    numeric_rule = anchors["decision_operator"].isin(["<", "<=", ">", ">="])
    for column in ("decision_value", "decision_threshold"):
        anchors.loc[numeric_rule, column] = pd.to_numeric(
            anchors.loc[numeric_rule, column], errors="coerce"
        )
    anchors = validate_contrast_public_view(anchors)
    if anchors.duplicated(["method", *_KEY]).any():
        raise ValueError("Duplicate canonical RCE anchor keys detected.")

    public_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for key, group in anchors.groupby(_KEY, dropna=False, sort=False):
        states = _anchor_state(group, registry)
        template = group.sort_values("method").iloc[0].to_dict()
        is_reference = str(template["contrast_status"]) == "reference" or (
            pd.notna(template.get("reference_cell_type"))
            and str(template["cell_type"]) == str(template["reference_cell_type"])
        )
        for module in registry.modules:
            decision, decision_path, limitation = _evaluate_module(module, states)
            if is_reference:
                decision, decision_path, limitation = None, "common reference exclusion", ""
            direction_status, effect_direction = _direction_status(
                states, decision, module.required_roles
            )
            result_id = str(uuid5(NAMESPACE_URL, ":".join(map(str, (*key, module.method_id)))))
            evidence_id = str(uuid5(NAMESPACE_URL, f"rce-evidence:{result_id}"))
            module_states = [states[role] for role in module.required_roles]
            positive_methods = [item["method"] for item in module_states if item["usable"] and item["decision"]]
            supporting_methods = [item["method"] for item in module_states if item["usable"]]
            unavailable_methods = [item["method"] for item in module_states if not item["usable"]]
            confidence_tier = _confidence_tier(
                states, decision, direction_status, module.required_roles
            )

            row = dict(template)
            row.update({
                "result_id": result_id,
                "evidence_id": evidence_id if decision is not None else pd.NA,
                "method": module.method_id,
                "method_version": registry.rule_version,
                "estimate": float("nan"),
                "effect_estimand": "multi_method_decision_consensus",
                "effect_scale": "not_applicable",
                "effect_null": float("nan"),
                "effect_direction": effect_direction,
                "direction_basis": "canonical positive-anchor directions; no effect-scale conversion",
                "effect_estimate_source": "not_applicable",
                "result_interpretation": "RCE decision-only Boolean combination of canonical anchor decisions.",
                "primary_decision": decision if decision is not None else pd.NA,
                "decision_metric": "rce_boolean_decision" if decision is not None else pd.NA,
                "decision_value": decision if decision is not None else pd.NA,
                "decision_operator": "==" if decision is not None else pd.NA,
                "decision_threshold": True if decision is not None else pd.NA,
                "decision_rule_id": module.decision_rule_id if decision is not None else pd.NA,
                "decision_rule_description": module.description if decision is not None else pd.NA,
                "is_available": decision is not None,
                "is_valid": decision is not None,
                "contrast_status": "reference" if is_reference else ("success" if decision is not None else "unavailable"),
                "failure_reason": pd.NA if decision is not None or is_reference else limitation,
                "is_benchmark_eligible": bool(decision is not None and not is_reference),
                "estimand_compatibility": "decision_only" if decision is not None else "unavailable",
                "derived_from_native_effect": False,
            })
            if is_reference:
                row.update({"effect_direction": "not_applicable", "failure_reason": pd.NA,
                            "is_available": False, "is_valid": False,
                            "is_benchmark_eligible": False, "estimand_compatibility": "unavailable"})
            public_rows.append(row)

            state_json = json.dumps(states, sort_keys=True)
            provenance = {
                **dict(zip(_KEY, key)), "method": module.method_id,
                "result_id": result_id, "evidence_id": evidence_id if decision is not None else pd.NA,
                "rule_version": registry.rule_version, "decision": decision,
                "decision_path": decision_path, "limitation_reason": limitation or pd.NA,
                "supporting_methods": json.dumps(supporting_methods),
                "positive_methods": json.dumps(positive_methods),
                "unavailable_methods": json.dumps(unavailable_methods),
                "confidence_tier": confidence_tier, "direction_status": direction_status,
                "anchor_states": state_json,
            }
            provenance_rows.append(provenance)
            if decision is not None:
                evidence_rows.append({
                    "evidence_id": evidence_id, "result_id": result_id,
                    "evidence_paradigm": "other_native", "native_decision": decision,
                    "native_decision_metric": "rce_boolean_decision",
                    "native_decision_value": decision,
                    "native_decision_rule_id": module.decision_rule_id,
                    "decision_path": decision_path,
                    "supporting_methods": json.dumps(supporting_methods),
                    "positive_methods": json.dumps(positive_methods),
                    "unavailable_methods": json.dumps(unavailable_methods),
                    "confidence_tier": confidence_tier, "direction_status": direction_status,
                    "anchor_states": state_json,
                })

    result_public = validate_contrast_public_view(pd.DataFrame(public_rows))
    result_evidence = validate_evidence_layer(pd.DataFrame(evidence_rows))
    return result_public, result_evidence, pd.DataFrame(provenance_rows)
