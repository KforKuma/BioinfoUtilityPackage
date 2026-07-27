from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd
import numpy as np
from scipy.stats import norm
import yaml

from src.stats.adapters._shared import public_row
from src.stats.adapters.base import AdapterResult, BaseDifferentialAbundanceAdapter
from src.stats.engine import *
from src.stats.schemas import CanonicalDAInput, MethodDiagnostics
from src.stats.validation import parse_boolean_series
from src.utils.env_utils import call_with_compatible_args
from src.utils.warnings import deprecated


_ANCHOR_ROLES = {"compatible", "direction_only", "decision_only", "incompatible"}
_DIRECTIONAL_VALUES = {"group_1_higher", "group_2_higher", "no_effect"}


@dataclass(frozen=True)
class TriAnchorRule:
    """Versioned rule for combining canonical anchor decisions without score conversion."""

    rule_version: str
    anchor_methods: tuple[str, ...]
    anchor_roles: Mapping[str, str]
    minimum_valid_anchors: int
    minimum_positive_anchors: int
    require_direction_agreement: bool
    require_evidence_agreement: bool
    allow_decision_only: bool
    conflict_policy: str
    missing_anchor_policy: str
    tie_handling: str
    target_estimand: str
    target_scale: str
    effect_component: str
    primary_decision_rule_id: str
    reference_cell_type: str | None = None

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "TriAnchorRule":
        methods = tuple(str(value) for value in record["anchor_methods"])
        roles = {str(key): str(value) for key, value in record["anchor_roles"].items()}
        rule = cls(
            rule_version=str(record["rule_version"]),
            anchor_methods=methods,
            anchor_roles=roles,
            minimum_valid_anchors=int(record["minimum_valid_anchors"]),
            minimum_positive_anchors=int(record["minimum_positive_anchors"]),
            require_direction_agreement=bool(record["require_direction_agreement"]),
            require_evidence_agreement=bool(record.get("require_evidence_agreement", False)),
            allow_decision_only=bool(record["allow_decision_only"]),
            conflict_policy=str(record["conflict_policy"]),
            missing_anchor_policy=str(record["missing_anchor_policy"]),
            tie_handling=str(record["tie_handling"]),
            target_estimand=str(record["target_estimand"]),
            target_scale=str(record["target_scale"]),
            effect_component=str(record.get("effect_component", "composition")),
            primary_decision_rule_id=str(record["primary_decision_rule_id"]),
            reference_cell_type=(
                str(record["reference_cell_type"])
                if record.get("reference_cell_type") not in {None, ""} else None
            ),
        )
        rule.validate()
        return rule

    def validate(self) -> "TriAnchorRule":
        if len(self.anchor_methods) < 2 or len(set(self.anchor_methods)) != len(self.anchor_methods):
            raise ValueError("Tri_anchor requires at least two unique anchor methods.")
        if set(self.anchor_roles) != set(self.anchor_methods):
            raise ValueError("Tri_anchor anchor_roles must exactly cover anchor_methods.")
        invalid_roles = set(self.anchor_roles.values()) - _ANCHOR_ROLES
        if invalid_roles:
            raise ValueError(f"Unknown Tri_anchor roles: {sorted(invalid_roles)}")
        if not 2 <= self.minimum_valid_anchors <= len(self.anchor_methods):
            raise ValueError("minimum_valid_anchors must be between 2 and the anchor count.")
        if not 1 <= self.minimum_positive_anchors <= self.minimum_valid_anchors:
            raise ValueError("minimum_positive_anchors must not exceed minimum_valid_anchors.")
        if self.conflict_policy != "negative":
            raise ValueError("The conservative v1 conflict_policy must be 'negative'.")
        if self.missing_anchor_policy != "exclude":
            raise ValueError("Invalid or missing anchors must use missing_anchor_policy='exclude'.")
        if self.tie_handling != "negative":
            raise ValueError("The conservative v1 tie_handling must be 'negative'.")
        if not self.target_estimand or not self.target_scale or not self.primary_decision_rule_id:
            raise ValueError("Tri_anchor target semantics and decision rule must be explicit.")
        return self


def default_tri_anchor_rule_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "project" / "Step08_Abundance" / "configs" / "tri_anchor_rules.yaml"
    )


def load_tri_anchor_rule(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> TriAnchorRule:
    source = Path(path) if path is not None else default_tri_anchor_rule_path()
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[3] / source
    with source.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping) or not isinstance(document.get("tri_anchor"), Mapping):
        raise ValueError("Tri_anchor rule YAML requires a tri_anchor mapping.")
    record = dict(document["tri_anchor"])
    record.update(dict(overrides or {}))
    return TriAnchorRule.from_mapping(record)


def prepare_anchor_inputs(
    public_view: pd.DataFrame,
    evidence_layer: pd.DataFrame,
    *,
    run_id: str,
    analysis_id: str,
    contrast_id: str,
    effect_component: str,
    anchor_methods: Sequence[str],
) -> pd.DataFrame:
    """Align canonical anchor results and link native decisions for provenance only."""
    required = {
        "run_id", "analysis_id", "method", "contrast_id", "cell_type",
        "effect_component", "primary_decision", "effect_direction", "estimate",
        "effect_estimand", "effect_scale", "reference_cell_type", "is_available",
        "is_valid", "contrast_status", "evidence_id",
    }
    if missing := required - set(public_view.columns):
        raise ValueError(f"Canonical anchor public view is missing: {sorted(missing)}")
    selected = public_view.loc[
        public_view["run_id"].astype(str).eq(str(run_id))
        & public_view["analysis_id"].astype(str).eq(str(analysis_id))
        & public_view["contrast_id"].astype(str).eq(str(contrast_id))
        & public_view["effect_component"].astype(str).eq(str(effect_component))
        & public_view["method"].astype(str).isin(tuple(anchor_methods))
    ].copy()
    duplicate_key = ["method", "contrast_id", "cell_type", "effect_component"]
    if selected.duplicated(duplicate_key).any():
        raise ValueError("Tri_anchor received duplicate canonical anchor keys.")
    unknown = set(selected["method"].astype(str)) - set(map(str, anchor_methods))
    if unknown:
        raise ValueError(f"Tri_anchor received unconfigured anchors: {sorted(unknown)}")

    if evidence_layer.empty:
        selected["anchor_native_decision"] = pd.NA
    else:
        evidence_required = {"evidence_id", "native_decision"}
        if missing := evidence_required - set(evidence_layer.columns):
            raise ValueError(f"Canonical anchor evidence is missing: {sorted(missing)}")
        native = evidence_layer[["evidence_id", "native_decision"]].copy()
        if native["evidence_id"].duplicated().any():
            raise ValueError("Tri_anchor evidence IDs must be unique.")
        native = native.rename(columns={"native_decision": "anchor_native_decision"})
        selected = selected.merge(native, on="evidence_id", how="left", validate="one_to_one")
    selected["primary_decision"] = parse_boolean_series(selected["primary_decision"])
    selected["anchor_native_decision"] = parse_boolean_series(
        selected["anchor_native_decision"], errors="coerce"
    )
    return selected


def assess_anchor_compatibility(
    prepared: pd.DataFrame,
    rule: TriAnchorRule,
) -> pd.DataFrame:
    """Classify each anchor for the configured target without converting effect scales."""
    result = prepared.copy()
    compatibility: list[str] = []
    reasons: list[str] = []
    for _, row in result.iterrows():
        method = str(row["method"])
        configured_role = rule.anchor_roles.get(method, "incompatible")
        available = bool(row["is_available"]) if pd.notna(row["is_available"]) else False
        valid = bool(row["is_valid"]) if pd.notna(row["is_valid"]) else False
        if not available or not valid or str(row["contrast_status"]) != "success":
            compatibility.append("unavailable")
            reasons.append("anchor_unavailable_or_invalid")
            continue
        if pd.isna(row["primary_decision"]):
            compatibility.append("unavailable")
            reasons.append("primary_decision_unavailable")
            continue
        if configured_role == "incompatible":
            compatibility.append("incompatible")
            reasons.append("configured_incompatible")
            continue
        if configured_role == "compatible":
            exact_effect = (
                str(row["effect_estimand"]) == rule.target_estimand
                and str(row["effect_scale"]) == rule.target_scale
                and np.isfinite(pd.to_numeric(pd.Series([row["estimate"]]), errors="coerce").iloc[0])
            )
            row_reference = row.get("reference_cell_type", pd.NA)
            exact_reference = (
                pd.isna(row_reference) if rule.reference_cell_type is None
                else str(row_reference) == str(rule.reference_cell_type)
            )
            if exact_effect and exact_reference:
                compatibility.append("compatible")
                reasons.append("exact_target_effect_semantics")
            else:
                compatibility.append("incompatible")
                reasons.append("target_effect_semantics_mismatch")
            continue
        if configured_role == "direction_only":
            if str(row["effect_direction"]) in _DIRECTIONAL_VALUES:
                compatibility.append("direction_only")
                reasons.append("configured_direction_support")
            else:
                compatibility.append("decision_only")
                reasons.append("direction_unavailable_decision_retained")
            continue
        compatibility.append("decision_only")
        reasons.append("configured_decision_support")
    result["anchor_compatibility"] = compatibility
    result["anchor_compatibility_reason"] = reasons
    return result


def combine_anchor_evidence(
    assessed: pd.DataFrame,
    rule: TriAnchorRule,
    *,
    cell_types: Sequence[str],
) -> pd.DataFrame:
    """Apply majority and direction consensus to canonical decisions only."""
    records: list[dict[str, Any]] = []
    order = {method: index for index, method in enumerate(rule.anchor_methods)}
    for cell_type in map(str, cell_types):
        group = assessed.loc[assessed["cell_type"].astype(str).eq(cell_type)].copy()
        group["_order"] = group["method"].astype(str).map(order)
        group = group.sort_values("_order")
        usable = group.loc[group["anchor_compatibility"].isin(
            {"compatible", "direction_only", "decision_only"}
        )].copy()
        valid_count = len(usable)
        positive = usable.loc[usable["primary_decision"].eq(True)].copy()
        positive_count = len(positive)
        positive_directions = positive.loc[
            positive["effect_direction"].isin(_DIRECTIONAL_VALUES), "effect_direction"
        ].astype(str)
        direction_conflict = positive_directions.nunique() > 1
        direction_missing = len(positive_directions) < positive_count
        evidence_disagreement = False
        if rule.require_evidence_agreement and not usable.empty:
            comparable = usable["anchor_native_decision"].notna()
            evidence_disagreement = bool(
                (~comparable).any()
                or (
                    usable.loc[comparable, "anchor_native_decision"].astype(bool).to_numpy()
                    != usable.loc[comparable, "primary_decision"].astype(bool).to_numpy()
                ).any()
            )

        sufficient = valid_count >= rule.minimum_valid_anchors
        tied = valid_count > 0 and positive_count * 2 == valid_count
        consensus: Any = pd.NA
        limitation = "insufficient_valid_anchors"
        if sufficient:
            consensus = bool(positive_count >= rule.minimum_positive_anchors)
            limitation = ""
            if tied and rule.tie_handling == "negative":
                consensus = False
                limitation = "tie_resolved_negative"
            if rule.require_direction_agreement and (direction_conflict or direction_missing):
                consensus = False
                limitation = "direction_conflict_or_missing"
            if evidence_disagreement:
                consensus = False
                limitation = "native_primary_evidence_disagreement"

        directional = usable.loc[
            usable["effect_direction"].isin(_DIRECTIONAL_VALUES), "effect_direction"
        ].astype(str)
        consensus_direction = (
            directional.iloc[0] if len(directional) and directional.nunique() == 1
            else ("undetermined" if len(directional) else "not_applicable")
        )
        compatible_effects = usable.loc[usable["anchor_compatibility"].eq("compatible")].copy()
        compatible_values = pd.to_numeric(compatible_effects["estimate"], errors="coerce").dropna()
        estimate = float(compatible_values.median()) if len(compatible_values) else np.nan
        if not np.isfinite(estimate) and not rule.allow_decision_only and sufficient:
            consensus = pd.NA
            limitation = "unified_effect_unavailable_and_decision_only_disabled"

        native_effect_row = usable.loc[
            pd.to_numeric(usable["estimate"], errors="coerce").notna()
        ].head(1)
        records.append({
            "cell_type": cell_type,
            "valid_anchor_count": valid_count,
            "positive_anchor_count": positive_count,
            "valid_anchor_methods": json.dumps(usable["method"].astype(str).tolist()),
            "positive_anchor_methods": json.dumps(positive["method"].astype(str).tolist()),
            "compatibility_by_method": json.dumps(dict(zip(
                group["method"].astype(str), group["anchor_compatibility"].astype(str), strict=True
            )), sort_keys=True),
            "direction_conflict": direction_conflict,
            "direction_missing": direction_missing,
            "evidence_disagreement": evidence_disagreement,
            "consensus_direction": consensus_direction,
            "tri_anchor_consensus": consensus,
            "limitation_reason": limitation or pd.NA,
            "estimate": estimate,
            "estimand_compatibility": "compatible" if np.isfinite(estimate) else "decision_only",
            "anchor_effect_method": (
                str(native_effect_row.iloc[0]["method"]) if not native_effect_row.empty else pd.NA
            ),
            "anchor_native_effect": (
                native_effect_row.iloc[0]["estimate"] if not native_effect_row.empty else np.nan
            ),
            "anchor_effect_estimand": (
                native_effect_row.iloc[0]["effect_estimand"] if not native_effect_row.empty else pd.NA
            ),
            "anchor_effect_scale": (
                native_effect_row.iloc[0]["effect_scale"] if not native_effect_row.empty else pd.NA
            ),
        })
    return pd.DataFrame(records)


def build_tri_anchor_result(
    combined: pd.DataFrame,
    canonical_input: CanonicalDAInput,
    contrast: pd.Series,
    rule: TriAnchorRule,
    *,
    analysis_id: str,
    diagnostic_id: str,
    method_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build schema-valid public/evidence rows with no cross-paradigm score."""
    public_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    reference = rule.reference_cell_type or (
        str(contrast["reference_cell_type"])
        if pd.notna(contrast.get("reference_cell_type")) else None
    )
    for _, native in combined.iterrows():
        cell_type = str(native["cell_type"])
        direction_basis = (
            f"canonical_anchor_direction_consensus;target_estimand={rule.target_estimand};"
            f"reference_cell_type={reference or 'not_applicable'}"
        )
        estimate = pd.to_numeric(pd.Series([native["estimate"]]), errors="coerce").iloc[0]
        row, result_id, evidence_id = public_row(
            method_id="tri_anchor",
            method_version=method_version,
            analysis_id=analysis_id,
            diagnostic_id=diagnostic_id,
            contrast=contrast,
            cell_type=cell_type,
            effect_component=rule.effect_component,
            estimate=estimate,
            effect_estimand=rule.target_estimand,
            effect_scale=rule.target_scale,
            direction_basis=direction_basis,
            decision_rule_id=rule.primary_decision_rule_id,
            reference_cell_type=reference if reference is not None else pd.NA,
            effect_estimate_source="derived" if np.isfinite(estimate) else "not_applicable",
            result_interpretation=(
                "Tri_anchor decision consensus. A public effect is present only when anchors "
                "have exactly compatible target semantics."
            ),
            reference_strategy="common_exclusion" if reference else "not_applicable",
            reference_selection_reason=contrast.get("reference_selection_reason", pd.NA),
            reference_is_fixed=bool(reference),
            benchmark_estimand=rule.target_estimand,
            derived_from_native_effect=bool(np.isfinite(estimate)),
        )
        if cell_type == reference:
            row.update({
                "evidence_id": pd.NA,
                "estimate": np.nan,
                "effect_direction": "not_applicable",
                "primary_decision": pd.NA,
                "decision_metric": pd.NA,
                "decision_value": pd.NA,
                "decision_operator": pd.NA,
                "decision_threshold": pd.NA,
                "decision_rule_id": pd.NA,
                "decision_rule_description": pd.NA,
                "is_available": False,
                "is_valid": False,
                "contrast_status": "reference",
                "failure_reason": pd.NA,
                "is_benchmark_eligible": False,
                "estimand_compatibility": "unavailable",
                "derived_from_native_effect": False,
                "effect_estimate_source": "not_applicable",
            })
            public_rows.append(row)
            continue
        if pd.isna(native["tri_anchor_consensus"]):
            row.update({
                "evidence_id": pd.NA,
                "estimate": np.nan,
                "effect_direction": "not_applicable",
                "primary_decision": pd.NA,
                "decision_metric": pd.NA,
                "decision_value": pd.NA,
                "decision_operator": pd.NA,
                "decision_threshold": pd.NA,
                "decision_rule_id": pd.NA,
                "decision_rule_description": pd.NA,
                "is_available": False,
                "is_valid": False,
                "contrast_status": "unavailable",
                "failure_reason": native["limitation_reason"],
                "is_benchmark_eligible": False,
                "estimand_compatibility": "unavailable",
                "derived_from_native_effect": False,
                "effect_estimate_source": "not_applicable",
            })
            public_rows.append(row)
            continue
        if not np.isfinite(estimate):
            row["effect_direction"] = native["consensus_direction"]
            row["estimand_compatibility"] = "decision_only"
        public_rows.append(row)
        consensus = bool(native["tri_anchor_consensus"])
        evidence_rows.append({
            "evidence_id": evidence_id,
            "result_id": result_id,
            "evidence_paradigm": "other_native",
            "native_decision": consensus,
            "native_decision_metric": "tri_anchor_consensus",
            "native_decision_value": consensus,
            "native_decision_rule_id": f"tri-anchor-combination-{rule.rule_version}",
            "tri_anchor_consensus": consensus,
            "tri_anchor_rule_version": rule.rule_version,
            "anchor_methods": json.dumps(rule.anchor_methods),
            "valid_anchor_methods": native["valid_anchor_methods"],
            "positive_anchor_methods": native["positive_anchor_methods"],
            "valid_anchor_count": native["valid_anchor_count"],
            "positive_anchor_count": native["positive_anchor_count"],
            "direction_conflict": native["direction_conflict"],
            "evidence_disagreement": native["evidence_disagreement"],
            "compatibility_by_method": native["compatibility_by_method"],
            "anchor_effect_method": native["anchor_effect_method"],
            "anchor_native_effect": native["anchor_native_effect"],
            "anchor_effect_estimand": native["anchor_effect_estimand"],
            "anchor_effect_scale": native["anchor_effect_scale"],
        })
    return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)


class TriAnchorAdapter(BaseDifferentialAbundanceAdapter):
    """Dependent adapter that consumes canonical results produced earlier by the runner."""

    method_id = "tri_anchor"
    consumes_canonical_results = True

    def __init__(
        self,
        *,
        rule: TriAnchorRule | None = None,
        method_version: str = "1.0.0",
    ) -> None:
        super().__init__(method_version=method_version)
        self.rule = (rule or load_tri_anchor_rule()).validate()

    def prepare_native_input(self, canonical_input, contrast):  # pragma: no cover - guarded by runner
        raise RuntimeError("TriAnchorAdapter requires canonical anchor results.")

    def execute_native(self, native_input, contrast):  # pragma: no cover - guarded by runner
        raise RuntimeError("TriAnchorAdapter requires canonical anchor results.")

    def transform_native_output(self, native_output, canonical_input, contrast, **kwargs):
        raise RuntimeError("TriAnchorAdapter uses build_tri_anchor_result().")

    def run_from_anchor_results(
        self,
        canonical_input: CanonicalDAInput,
        contrast: pd.Series,
        *,
        anchor_public: pd.DataFrame,
        anchor_evidence: pd.DataFrame,
        analysis_id: str,
        run_id: str,
        native_output_dir: Path,
    ) -> AdapterResult:
        diagnostic_id = str(uuid4())
        diagnostics = MethodDiagnostics(
            diagnostic_id=diagnostic_id,
            analysis_id=analysis_id,
            method=self.method_id,
            method_version=self.method_version,
            status="running",
            input_hash=canonical_input.input_hash(),
            started_at=datetime.now(timezone.utc).isoformat(),
            details={
                "rule_version": self.rule.rule_version,
                "anchor_methods": list(self.rule.anchor_methods),
                "target_estimand": self.rule.target_estimand,
                "target_scale": self.rule.target_scale,
            },
        )
        rule_record = dict(self.rule.__dict__)
        if self.rule.reference_cell_type is None and pd.notna(contrast.get("reference_cell_type")):
            rule_record["reference_cell_type"] = str(contrast["reference_cell_type"])
        effective_rule = TriAnchorRule.from_mapping(rule_record)
        try:
            prepared = prepare_anchor_inputs(
                anchor_public,
                anchor_evidence,
                run_id=run_id,
                analysis_id=analysis_id,
                contrast_id=str(contrast["contrast_id"]),
                effect_component=effective_rule.effect_component,
                anchor_methods=effective_rule.anchor_methods,
            )
            assessed = assess_anchor_compatibility(prepared, effective_rule)
            cell_types = canonical_input.cell_type_manifest.loc[
                canonical_input.cell_type_manifest["inclusion_status"].eq("included"), "cell_type"
            ].astype(str)
            combined = combine_anchor_evidence(assessed, effective_rule, cell_types=cell_types)
            native_path = self.save_native_output(
                combined,
                native_output_dir,
                analysis_id=analysis_id,
                contrast_id=str(contrast["contrast_id"]),
            )
            diagnostics.native_output_path = str(native_path)
            diagnostics.details.update({
                "number_rows": len(combined),
                "number_decision_available": int(combined["tri_anchor_consensus"].notna().sum()),
                "number_decision_unavailable": int(combined["tri_anchor_consensus"].isna().sum()),
            })
            public, evidence = build_tri_anchor_result(
                combined,
                canonical_input,
                contrast,
                effective_rule,
                analysis_id=analysis_id,
                diagnostic_id=diagnostic_id,
                method_version=self.method_version,
            )
            diagnostics.converged = True
            diagnostics.finish(status="success")
            return AdapterResult(public, evidence, diagnostics)
        except Exception as exc:
            diagnostics.error_type = type(exc).__name__
            diagnostics.error_message = str(exc)
            diagnostics.converged = False
            diagnostics.finish(status="failed")
            public = self.unavailable_public_rows(
                canonical_input,
                contrast,
                analysis_id=analysis_id,
                diagnostic_id=diagnostic_id,
                failure_reason="canonical_conversion_error",
            )
            return AdapterResult(public, pd.DataFrame(), diagnostics)


@deprecated(alternative="TriAnchorAdapter through DifferentialAbundanceRunner")
def run_Meta_Ensemble(df_all: pd.DataFrame,
                      cell_type: str,
                      formula: str,
                      main_variable: str = "disease",
                      alpha: float = 0.05,
                      coef_threshold: float = 0.2,  # 降低硬门槛，依赖共识压制 FPR
                      **kwargs) -> Dict[str, Any]:
    """运行三方法共识 meta engine。

    该集合方法同时调用 Dirichlet-Multinomial Wald、CLR-LMM 和 PyDESeq2，
    对同一个 cell subtype/subpopulation 的 contrast_table 做对齐。最终显著性依赖
    2/3 多数投票、显著方法方向一致、中位数效应量达到阈值，以及 DMW 的宽松
    veto 检查。设计目标是降低单一统计方法失效时带来的假阳性。

    Args:
        df_all: 长表丰度数据。
        cell_type: 目标 cell subtype/subpopulation。
        formula: 传给子方法的右侧公式。
        main_variable: 主要解释变量。
        alpha: 显著性阈值。
        coef_threshold: meta 中位数效应量最低阈值。
        **kwargs: 透传给子方法的兼容参数。

    Returns:
        字典，包含 ``contrast_table``、``summary`` 和 ``raw_results``。``raw_results``
        保存每个子方法的原始结果，便于排查具体方法失败。

    Example:
        >>> res = run_Meta_Ensemble(
        ...     df_all=count_df,
        ...     cell_type="CD4_Tcm",
        ...     formula="disease + tissue",
        ...     main_variable="disease",
        ...     ref_label="HC",
        ... )
        >>> res["contrast_table"][["Coef.", "P>|z|", "method_agreement"]]
        # method_agreement 表示三个子方法中有几个支持该对比。
    """
    sub_methods = {
        'dmw': run_Dirichlet_Multinomial_Wald,
        'clr': run_CLR_LMM,
        'deseq2': run_PyDESeq2
    }
    
    base_kwargs = {'df_all': df_all, 'cell_type': cell_type, 'formula': formula,
                   'main_variable': main_variable, 'alpha': alpha, **kwargs}
    
    # 1. 运行所有子方法
    results = {}
    for name, func in sub_methods.items():
        try:
            res = call_with_compatible_args(func, **base_kwargs)
            if res and isinstance(res.get('contrast_table'), pd.DataFrame) and not res['contrast_table'].empty:
                results[name] = res
            else:
                results[name] = None
        except Exception:
            results[name] = None
    
    # 2. 提取并对齐数据
    all_indices = [r['contrast_table'].index for r in results.values() if r is not None]
    if not all_indices:
        return {'contrast_table': pd.DataFrame(), 'summary': 'All Methods Failed'}
    
    # 取并集索引，确保即便某个方法没算出来，我们也能对比其他方法
    common_idx = all_indices[0].union(all_indices[1]) if len(all_indices) > 1 else all_indices[0]
    for i in range(2, len(all_indices)):
        common_idx = common_idx.union(all_indices[i])
    
    def get_standardized_data(res_obj, target_index):
        """将单个子方法结果对齐到共同 contrast index。"""
        if res_obj is None:
            return (pd.Series(False, index=target_index), pd.Series(0, index=target_index),
                    pd.Series(1.0, index=target_index), pd.Series(0.0, index=target_index))
        
        df = res_obj['contrast_table'].reindex(target_index)
        sig = parse_boolean_series(df['significant']).fillna(False).astype(bool)
        dir_map = {'other_greater': 1, 'ref_greater': -1}
        direction = df['direction'].map(dir_map).fillna(0).astype(int)
        pvals = df['P>|z|'].fillna(1.0)
        c_col = 'Coef.' if 'Coef.' in df.columns else 'Coef'
        coefs = df[c_col].fillna(0.0)
        return sig, direction, pvals, coefs
    
    s1, d1, p1, c1 = get_standardized_data(results.get('dmw'), common_idx)
    s2, d2, p2, c2 = get_standardized_data(results.get('clr'), common_idx)
    s3, d3, p3, c3 = get_standardized_data(results.get('deseq2'), common_idx)
    
    # 3. 核心集成逻辑
    # A. 显著性计数
    sig_count = s1.astype(int) + s2.astype(int) + s3.astype(int)
    
    # B. 方向一致性 (非常关键：只检查那些判定为显著的方法是否方向一致)
    # 计算显著方法的方向和：如果 2 个方法显著且方向一致，绝对值应为 2
    actual_dir_sum = (s1 * d1) + (s2 * d2) + (s3 * d3)
    is_direction_coherent = (actual_dir_sum.abs() == sig_count) & (sig_count > 0)
    
    # 增加共识
    is_not_dmw_veto = p1 < 0.2
    
    # 如果是叠加效应，Meta 估计的 Coef. 符号必须与单方法中最显著的那个一致
    anchor_dir = pd.Series(np.where(p2 < p3, d2, d3), index=common_idx)
    
    # 4. P值聚合：采用 Stouffer's 思想的简化版 (中位数 P 在集成中通常表现最稳)
    # 或者用极小值 P (如果你追求 Power)
    combined_p = pd.concat([p1, p2, p3], axis=1).median(axis=1)
    
    # 5. 构造结果表
    meta_dir_val = actual_dir_sum.apply(np.sign).astype(int)
    rev_map = {1: 'other_greater', -1: 'ref_greater', 0: 'None'}
    
    # C. 效应量中位数 (比单用 DMW 更稳健)
    median_coef = pd.concat([c1, c2, c3], axis=1).median(axis=1)
    
    # D. 最终判定：多数原则 (>=2) 且 方向一致 且 满足最小效应门槛
    # 不再强制要求 DMW (s1) 必须为 True
    meta_significant = (
            (sig_count >= 2) &  # 多数投票
            is_direction_coherent &  # 方向一致
            (median_coef.abs() >= coef_threshold) &
            is_not_dmw_veto &
            (combined_p < alpha) &
            (meta_dir_val == anchor_dir)  # 集成方向必须与最可靠的单方法方向一致
    )
    
    
    # 寻找一个非空的参考列
    ref_col = "Unknown"
    for r in results.values():
        if r is not None:
            ref_col = r['contrast_table']['ref'].iloc[0]
            break
    
    meta_table = pd.DataFrame({
        'ref': ref_col,
        'Coef.': median_coef,
        'P>|z|': combined_p,
        'direction': meta_dir_val.map(rev_map),
        'significant': meta_significant,
        'method_agreement': sig_count
    }, index=common_idx)
    
    return {
        'contrast_table': meta_table,
        'summary': f"Consensus Meta. Hits: {meta_significant.sum()}, Agreement: {sig_count.mean():.2f}",
        'raw_results': results
    }


@deprecated(alternative="TriAnchorAdapter through DifferentialAbundanceRunner")
def run_Meta_Ensemble_adaptive(df_all: pd.DataFrame,
                               cell_type: str,
                               formula: str,
                               main_variable: str = "disease",
                               alpha: float = 0.05,
                               **kwargs) -> Dict[str, Any]:
    """运行自适应效应量阈值的三方法共识 meta engine。

    与 ``run_Meta_Ensemble`` 相同，本函数集成 DMW、CLR-LMM 和 PyDESeq2；
    区别是效应量阈值会根据三个方法估计系数的整体尺度动态调整，避免在低波动数据
    中过度保守，也避免在高波动数据中门槛过低。

    Args:
        df_all: 长表丰度数据。
        cell_type: 目标 cell subtype/subpopulation。
        formula: 传给子方法的右侧公式。
        main_variable: 主要解释变量。
        alpha: 显著性阈值。
        **kwargs: 透传给子方法的兼容参数。

    Returns:
        字典，包含 meta ``contrast_table``、摘要和子方法原始结果。

    Example:
        >>> res = run_Meta_Ensemble_adaptive(
        ...     df_all=count_df,
        ...     cell_type="Treg",
        ...     formula="disease + C(tissue, Treatment(reference='nif'))",
        ...     main_variable="disease",
        ... )
        >>> res["summary"]
        # 查看 meta 命中数量和平均方法一致度。
    """
    sub_methods = {
        'dmw': run_Dirichlet_Multinomial_Wald,
        'clr': run_CLR_LMM,
        'deseq2': run_PyDESeq2
    }
    
    base_kwargs = {'df_all': df_all, 'cell_type': cell_type, 'formula': formula,
                   'main_variable': main_variable, 'alpha': alpha, **kwargs}
    
    # 1. 运行所有子方法
    results = {}
    for name, func in sub_methods.items():
        try:
            res = call_with_compatible_args(func, **base_kwargs)
            if res and isinstance(res.get('contrast_table'), pd.DataFrame) and not res['contrast_table'].empty:
                results[name] = res
            else:
                results[name] = None
        except Exception:
            results[name] = None
    
    # 2. 提取并对齐数据
    all_indices = [r['contrast_table'].index for r in results.values() if r is not None]
    if not all_indices:
        return {'contrast_table': pd.DataFrame(), 'summary': 'All Methods Failed'}
    
    # 取并集索引，确保即便某个方法没算出来，我们也能对比其他方法
    common_idx = all_indices[0].union(all_indices[1]) if len(all_indices) > 1 else all_indices[0]
    for i in range(2, len(all_indices)):
        common_idx = common_idx.union(all_indices[i])
    
    def get_standardized_data(res_obj, target_index):
        """将单个子方法结果对齐到共同 contrast index。"""
        if res_obj is None:
            return (pd.Series(False, index=target_index), pd.Series(0, index=target_index),
                    pd.Series(1.0, index=target_index), pd.Series(0.0, index=target_index))
        
        df = res_obj['contrast_table'].reindex(target_index)
        sig = parse_boolean_series(df['significant']).fillna(False).astype(bool)
        dir_map = {'other_greater': 1, 'ref_greater': -1}
        direction = df['direction'].map(dir_map).fillna(0).astype(int)
        pvals = df['P>|z|'].fillna(1.0)
        c_col = 'Coef.' if 'Coef.' in df.columns else 'Coef'
        coefs = df[c_col].fillna(0.0)
        return sig, direction, pvals, coefs
    
    s1, d1, p1, c1 = get_standardized_data(results.get('dmw'), common_idx)
    s2, d2, p2, c2 = get_standardized_data(results.get('clr'), common_idx)
    s3, d3, p3, c3 = get_standardized_data(results.get('deseq2'), common_idx)
    
    # 3. 核心集成逻辑
    # A. 显著性计数
    sig_count = s1.astype(int) + s2.astype(int) + s3.astype(int)
    
    # B. 方向一致性 (非常关键：只检查那些判定为显著的方法是否方向一致)
    # 计算显著方法的方向和：如果 2 个方法显著且方向一致，绝对值应为 2
    actual_dir_sum = (s1 * d1) + (s2 * d2) + (s3 * d3)
    is_direction_coherent = (actual_dir_sum.abs() == sig_count) & (sig_count > 0)
    
    # C. 效应量中位数 (比单用 DMW 更稳健)
    median_coef = pd.concat([c1, c2, c3], axis=1).median(axis=1)
    
    # 计算全局中位绝对偏差 (Median Absolute Deviation, MAD)
    # MAD 是比标准差更稳健的离散度衡量
    all_coefs = pd.concat([c1, c2, c3])
    data_scale = all_coefs.abs().median()
    
    # 动态设置门槛：基础门槛 + 比例增益
    # 基础值 0.1 保证低波动下的敏感度，0.5 * data_scale 保证高波动下的拦截力
    dynamic_threshold = 0.1 + 0.3 * data_scale
    
    # 限制门槛范围，防止极端情况下门槛过高或过低
    dynamic_threshold = np.clip(dynamic_threshold, 0.15, 0.8)
    
    # 增加共识
    is_not_dmw_veto = p1 < 0.2
    
    # 如果是叠加效应，Meta 估计的 Coef. 符号必须与单方法中最显著的那个一致
    anchor_dir = pd.Series(np.where(p2 < p3, d2, d3), index=common_idx)
    
    # 4. P值聚合：采用 Stouffer's 思想的简化版 (中位数 P 在集成中通常表现最稳)
    # 或者用极小值 P (如果你追求 Power)
    combined_p = pd.concat([p1, p2, p3], axis=1).median(axis=1)
    
    
    # 5. 构造结果表
    meta_dir_val = actual_dir_sum.apply(np.sign).astype(int)
    rev_map = {1: 'other_greater', -1: 'ref_greater', 0: 'None'}
    
    
    # 6. 最终判定：多数原则 (>=2) 且 方向一致 且 满足最小效应门槛
    # 不再强制要求 DMW (s1) 必须为 True
    meta_significant = (
            (sig_count >= 2) &  # 多数投票
            is_direction_coherent &  # 方向一致
            (median_coef.abs() >= dynamic_threshold) &
            is_not_dmw_veto &
            (combined_p < alpha) &
            (meta_dir_val == anchor_dir)  # 集成方向必须与最可靠的单方法方向一致
    )
    
    # 寻找一个非空的参考列
    ref_col = "Unknown"
    for r in results.values():
        if r is not None:
            ref_col = r['contrast_table']['ref'].iloc[0]
            break
    
    meta_table = pd.DataFrame({
        'ref': ref_col,
        'Coef.': median_coef,
        'P>|z|': combined_p,
        'direction': meta_dir_val.map(rev_map),
        'significant': meta_significant,
        'method_agreement': sig_count
    }, index=common_idx)
    
    return {
        'contrast_table': meta_table,
        'summary': f"Consensus Meta. Hits: {meta_significant.sum()}, Agreement: {sig_count.mean():.2f}",
        'raw_results': results
    }


@deprecated(alternative="run_Meta_Ensemble_adaptive")
def run_Meta_Ensemble_dynamic(
        df_all: pd.DataFrame,
        cell_type: str,
        formula: str,
        main_variable: str = "disease",
        alpha: float = 0.05,
        coef_threshold: float = 0.2,
        # 新增动态超参数
        k_penalty: float = 2.0,  # 功能1：P值惩罚强度（越小越保守）
        inflation_factor: float = 1.0,  # 功能2：手动或预估的基因组膨胀因子 lambda
        diversity_weight: float = 0.5,  # 功能3：分歧度惩罚权重
        **kwargs
) -> Dict[str, Any]:
    """运行已废弃的动态 meta engine。

    该版本保留三个历史实验性策略：基于效应量的 p 值软惩罚、类似 genomic
    control 的 lambda 校正，以及基于方法间 z-score CV 的共识收缩。当前推荐使用
    ``run_Meta_Ensemble_adaptive``，本函数保留是为了兼容旧脚本。

    Args:
        df_all: 长表丰度数据。
        cell_type: 目标 cell subtype/subpopulation。
        formula: 传给子方法的右侧公式。
        main_variable: 主要解释变量。
        alpha: 显著性阈值。
        coef_threshold: 触发软惩罚的效应量阈值。
        k_penalty: p 值软惩罚强度。
        inflation_factor: 经验零分布膨胀系数。
        diversity_weight: 方法分歧度惩罚权重。
        **kwargs: 透传给子方法的兼容参数。

    Returns:
        字典，包含 meta ``contrast_table``、摘要和子方法原始结果。

    Example:
        >>> res = run_Meta_Ensemble_dynamic(
        ...     df_all=count_df,
        ...     cell_type="B_memory",
        ...     formula="disease + tissue",
        ...     inflation_factor=1.2,
        ... )
        >>> res["contrast_table"].head()
        # 仅建议用于复现历史结果。
    """
    sub_methods = {
        'dmw': run_Dirichlet_Multinomial_Wald,
        'clr': run_CLR_LMM,
        'deseq2': run_PyDESeq2
    }
    
    base_kwargs = {'df_all': df_all, 'cell_type': cell_type, 'formula': formula,
                   'main_variable': main_variable, 'alpha': alpha, **kwargs}
    
    # 1. 运行所有子方法
    results = {}
    for name, func in sub_methods.items():
        try:
            res = call_with_compatible_args(func, **base_kwargs)
            if res and isinstance(res.get('contrast_table'), pd.DataFrame) and not res['contrast_table'].empty:
                results[name] = res
            else:
                results[name] = None
        except Exception:
            results[name] = None
    
    # 2. 提取并对齐数据
    all_indices = [r['contrast_table'].index for r in results.values() if r is not None]
    if not all_indices:
        return {'contrast_table': pd.DataFrame(), 'summary': 'All Methods Failed'}
    
    # 取并集索引，确保即便某个方法没算出来，我们也能对比其他方法
    common_idx = all_indices[0].union(all_indices[1]) if len(all_indices) > 1 else all_indices[0]
    for i in range(2, len(all_indices)):
        common_idx = common_idx.union(all_indices[i])
    
    def get_standardized_data(res_obj, target_index):
        """将单个子方法结果对齐到共同 contrast index。"""
        if res_obj is None:
            return (pd.Series(False, index=target_index), pd.Series(0, index=target_index),
                    pd.Series(1.0, index=target_index), pd.Series(0.0, index=target_index))
        
        df = res_obj['contrast_table'].reindex(target_index)
        sig = parse_boolean_series(df['significant']).fillna(False).astype(bool)
        dir_map = {'other_greater': 1, 'ref_greater': -1}
        direction = df['direction'].map(dir_map).fillna(0).astype(int)
        pvals = df['P>|z|'].fillna(1.0)
        c_col = 'Coef.' if 'Coef.' in df.columns else 'Coef'
        coefs = df[c_col].fillna(0.0)
        return sig, direction, pvals, coefs
    
    s1, d1, p1, c1 = get_standardized_data(results.get('dmw'), common_idx)
    s2, d2, p2, c2 = get_standardized_data(results.get('clr'), common_idx)
    s3, d3, p3, c3 = get_standardized_data(results.get('deseq2'), common_idx)
    
    # --- 功能 1: 基于 coef_threshold 的 P 值软惩罚 (Conservative Design) ---
    # 目的：在大样本下，如果效应量达不到门槛，即使P值很小也要拉高它。
    # 采用指数缓冲函数：如果 |beta| >= threshold，惩罚为1；如果越小，惩罚越大。
    median_coef = pd.concat([c1, c2, c3], axis=1).median(axis=1)
    abs_beta = median_coef.abs()
    
    # 只有当 abs_beta < coef_threshold 时才触发惩罚
    # penalty = exp( k * (threshold - |beta|) )，且最小为1
    soft_penalty = np.exp(np.maximum(0, k_penalty * (coef_threshold - abs_beta)))
    
    # --- 功能 2: 基于过离散/膨胀的校正 (Empirical Null / Lambda Correction) ---
    # 目的：模拟高 scale_factor 下零假设分布变宽的情况。
    # 如果外部传入了 inflation_factor (lambda > 1)，则修正 Z-score
    def adjust_p_by_lambda(p_series, lam):
        """按经验膨胀系数缩小 z-score 后重新计算双侧 p 值。"""
        if lam <= 1.0: return p_series
        # 将 P 换算回 Z，缩小 Z 后再换回 P
        z = norm.ppf(1 - p_series / 2)
        z_adj = z / np.sqrt(lam)
        return 2 * norm.sf(np.abs(z_adj))
    
    p1_adj = adjust_p_by_lambda(p1, inflation_factor)
    p2_adj = adjust_p_by_lambda(p2, inflation_factor)
    p3_adj = adjust_p_by_lambda(p3, inflation_factor)
    
    # --- 功能 3: P 值的“共识多样性”收缩 (Consensus Diversity Shrinkage) ---
    # 目的：如果三个方法 $P$ 值高度一致且都很小，警惕系统性偏误。
    # 计算 Z-score 的变异系数 (CV)
    z_matrix = np.array([norm.ppf(1 - p_adj.clip(upper=0.999) / 2) for p_adj in [p1_adj, p2_adj, p3_adj]])
    z_mean = np.mean(z_matrix, axis=0)
    z_std = np.std(z_matrix, axis=0)
    # CV 越小（一致性越高），惩罚因子越大
    # 当一致性极高时，我们将 P 值向中值方向收缩
    cv = z_std / (np.abs(z_mean) + 1e-6)
    diversity_penalty = 1 + diversity_weight * np.exp(-cv * 3)  # CV越小，penalty越高
    
    # 3. 核心集成逻辑
    # A. 显著性计数
    sig_count = (pd.concat([p1_adj, p2_adj, p3_adj], axis=1) < alpha).sum(axis=1)
    
    # B. 方向一致性 (非常关键：只检查那些判定为显著的方法是否方向一致)
    actual_dir_sum = (s1 * d1) + (s2 * d2) + (s3 * d3)
    is_direction_coherent = (actual_dir_sum.abs() == sig_count) & (sig_count > 0)
    
    # 如果是叠加效应，Meta 估计的 Coef. 符号必须与单方法中最显著的那个一致
    anchor_dir = pd.Series(np.where(p2_adj < p3_adj, d2, d3), index=p1.index)
    
    # 4. P值聚合：采用 Stouffer's 思想的简化版 (中位数 P 在集成中通常表现最稳)
    # 或者用极小值 P (如果你追求 Power)
    combined_p_raw = pd.concat([p1_adj, p2_adj, p3_adj], axis=1).median(axis=1)
    
    # 应用功能 1 和 功能 3 的联合惩罚
    final_p = combined_p_raw * soft_penalty * diversity_penalty
    final_p = final_p.clip(upper=1.0)
    
    # 5. 构造结果表
    meta_dir_val = actual_dir_sum.apply(np.sign).astype(int)
    rev_map = {1: 'other_greater', -1: 'ref_greater', 0: 'None'}
    
    
    
    meta_significant = (
            (sig_count >= 2) &
            is_direction_coherent &
            (final_p < alpha) &  # 使用校正后的 final_p
            (abs_beta >= 0.1) &  # 保留一个极小的底线门槛
            (meta_dir_val == anchor_dir)  # 方向锁
    )
    
    # 寻找一个非空的参考列
    ref_col = "Unknown"
    for r in results.values():
        if r is not None:
            ref_col = r['contrast_table']['ref'].iloc[0]
            break

    meta_table = pd.DataFrame({
        'ref': ref_col,
        'Coef.': median_coef,
        'P>|z|': final_p,
        'direction': meta_dir_val.map(rev_map),
        'significant': meta_significant,
        'method_agreement': sig_count
    }, index=common_idx)
    
    return {
        'contrast_table': meta_table,
        'summary': f"Consensus Meta. Hits: {meta_significant.sum()}, Agreement: {sig_count.mean():.2f}",
        'raw_results': results
    }
