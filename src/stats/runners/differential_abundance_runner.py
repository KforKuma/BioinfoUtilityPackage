from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
from uuid import NAMESPACE_URL, uuid4, uuid5

import pandas as pd

from src.stats.adapters.base import BaseDifferentialAbundanceAdapter
from src.stats.schemas import (
    CanonicalDAInput,
    DecisionRuleRegistry,
    validate_contrast_public_view,
    validate_evidence_layer,
)


@dataclass
class RunnerResult:
    public_view: pd.DataFrame
    evidence_layer: pd.DataFrame
    diagnostics: pd.DataFrame


class DifferentialAbundanceRunner:
    def __init__(self, output_root: str | Path, rule_registry: DecisionRuleRegistry) -> None:
        self.output_root = Path(output_root)
        self.rule_registry = rule_registry

    def _ensure_output_layout(self) -> dict[str, Path]:
        paths = {
            name: self.output_root / name
            for name in (
                "figures", "contrast_tables", "method_native_outputs", "benchmark",
                "logs", "diagnostics", "environment_reports",
            )
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    @staticmethod
    def _concat_records(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
        # Object dtype gives nullable Bayesian/frequentist union columns deterministic semantics
        # and avoids pandas inferring a dtype from an all-NA method-specific column.
        aligned = [frame.reindex(columns=columns).astype(object) for frame in frames]
        return pd.concat(aligned, ignore_index=True)

    @staticmethod
    def _attach_run_identity(
        public: pd.DataFrame,
        evidence: pd.DataFrame,
        diagnostics: pd.DataFrame,
        *,
        run_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Attach immutable run identity and re-key result/evidence IDs centrally."""
        public = public.copy()
        evidence = evidence.copy()
        diagnostics = diagnostics.copy()
        if public.empty:
            return public, evidence, diagnostics
        public["run_id"] = run_id
        old_to_new_result: dict[str, str] = {}
        old_to_new_evidence: dict[str, str] = {}
        for index, row in public.iterrows():
            old_result_id = str(row["result_id"])
            new_result_id = str(uuid5(
                NAMESPACE_URL,
                ":".join((
                    run_id,
                    str(row["analysis_id"]),
                    str(row["method"]),
                    str(row["contrast_id"]),
                    str(row["cell_type"]),
                    str(row["effect_component"]),
                )),
            ))
            old_to_new_result[old_result_id] = new_result_id
            public.at[index, "result_id"] = new_result_id
            if pd.notna(row["evidence_id"]):
                old_evidence_id = str(row["evidence_id"])
                new_evidence_id = str(uuid5(NAMESPACE_URL, f"evidence:{new_result_id}"))
                old_to_new_evidence[old_evidence_id] = new_evidence_id
                public.at[index, "evidence_id"] = new_evidence_id
        if not evidence.empty:
            evidence["result_id"] = evidence["result_id"].astype(str).map(old_to_new_result)
            evidence["evidence_id"] = evidence["evidence_id"].astype(str).map(old_to_new_evidence)
            if evidence[["result_id", "evidence_id"]].isna().any().any():
                raise ValueError("Evidence rows could not be re-keyed to the supplied run_id.")
            evidence["run_id"] = run_id
        if not diagnostics.empty:
            diagnostics["run_id"] = run_id
        return public, evidence, diagnostics

    def _apply_primary_decisions(
        self,
        public: pd.DataFrame,
        evidence: pd.DataFrame,
        contrast_specs: pd.DataFrame,
        diagnostics: pd.DataFrame,
    ) -> pd.DataFrame:
        result = public.copy()
        if result.empty:
            return result
        spec_rules = (
            contrast_specs.set_index("contrast_id")["decision_rule_id"]
            if "decision_rule_id" in contrast_specs.columns
            else None
        )
        evidence_by_id = evidence.set_index("evidence_id") if not evidence.empty else pd.DataFrame()
        diagnostics_by_id = (
            diagnostics.set_index("diagnostic_id") if not diagnostics.empty else pd.DataFrame()
        )

        def invalidate(index: int, reason: str) -> None:
            result.at[index, "is_valid"] = False
            result.at[index, "primary_decision"] = pd.NA
            result.at[index, "is_benchmark_eligible"] = False
            result.at[index, "contrast_status"] = "invalid"
            result.at[index, "failure_reason"] = reason
            for column in (
                "decision_metric", "decision_value", "decision_operator",
                "decision_threshold", "decision_rule_id", "decision_rule_description",
            ):
                result.at[index, column] = pd.NA

        for index, row in result.iterrows():
            if row["contrast_status"] != "success":
                continue
            rule_id = row["decision_rule_id"]
            if pd.isna(rule_id) and spec_rules is not None:
                rule_id = spec_rules.loc[row["contrast_id"]]
            if pd.isna(rule_id):
                invalidate(index, "decision_rule_unavailable")
                continue
            try:
                rule = self.rule_registry.get(str(rule_id))
            except KeyError:
                invalidate(index, "decision_rule_unregistered")
                continue
            if (
                rule.method not in {"unspecified", str(row["method"])}
                or rule.effect_component != str(row["effect_component"])
            ):
                invalidate(index, "decision_rule_scope_mismatch")
                continue
            if "fallback" in rule.threshold_source and not rule.fallback_enabled:
                invalidate(index, "decision_rule_fallback_disabled")
                continue
            if rule.requires_valid_diagnostics:
                diagnostic_id = row["diagnostic_id"]
                if diagnostics.empty or diagnostic_id not in diagnostics_by_id.index:
                    invalidate(index, "diagnostics_unavailable")
                    continue
                diagnostic = diagnostics_by_id.loc[diagnostic_id]
                converged = diagnostic["converged"]
                if diagnostic["status"] != "success" or pd.isna(converged) or not bool(converged):
                    invalidate(index, "diagnostics_invalid")
                    continue
            evidence_id = row["evidence_id"]
            if evidence.empty or evidence_id not in evidence_by_id.index or rule.metric not in evidence_by_id.columns:
                invalidate(index, "decision_metric_unavailable")
                continue
            decision_value = evidence_by_id.loc[evidence_id, rule.metric]
            try:
                decision = rule.evaluate(decision_value)
            except ValueError:
                invalidate(index, "decision_metric_unavailable")
                continue
            result.loc[index, "primary_decision"] = decision
            result.loc[index, "decision_metric"] = rule.metric
            result.loc[index, "decision_value"] = decision_value
            result.loc[index, "decision_operator"] = rule.operator
            result.loc[index, "decision_threshold"] = rule.threshold
            result.loc[index, "decision_rule_id"] = rule.rule_id
            result.loc[index, "decision_rule_description"] = rule.description
            result.loc[index, "is_benchmark_eligible"] = (
                bool(result.loc[index, "is_benchmark_eligible"]) and rule.benchmark_enabled
            )
        return result

    def run(
        self,
        canonical_input: CanonicalDAInput,
        adapters: list[BaseDifferentialAbundanceAdapter],
        *,
        analysis_id: str,
        run_id: str | None = None,
    ) -> RunnerResult:
        if run_id is None:
            warnings.warn(
                "Calling DifferentialAbundanceRunner.run without run_id is deprecated; "
                "a distinct generated run_id was used.",
                DeprecationWarning,
                stacklevel=2,
            )
            run_id = f"run-{uuid4()}"
        run_id = str(run_id)
        if not run_id or run_id == str(analysis_id):
            raise ValueError("`run_id` must be non-empty and distinct from `analysis_id`.")
        canonical_input.validate()
        paths = self._ensure_output_layout()
        native_adapters = [
            adapter for adapter in adapters
            if not bool(getattr(adapter, "consumes_canonical_results", False))
        ]
        meta_adapters = [
            adapter for adapter in adapters
            if bool(getattr(adapter, "consumes_canonical_results", False))
        ]
        if meta_adapters and not native_adapters:
            raise ValueError("Canonical meta adapters require at least one ordinary anchor adapter.")

        public_frames, evidence_frames, diagnostic_rows = [], [], []
        for adapter in native_adapters:
            for _, contrast in canonical_input.contrast_specification.iterrows():
                adapter_result = adapter.run(
                    canonical_input,
                    contrast,
                    analysis_id=analysis_id,
                    native_output_dir=paths["method_native_outputs"],
                )
                public_frames.append(adapter_result.public_view)
                if not adapter_result.evidence_layer.empty:
                    evidence_frames.append(adapter_result.evidence_layer)
                diagnostic_rows.append(adapter_result.diagnostics.to_record())

        anchor_public = self._concat_records(public_frames)
        anchor_evidence = self._concat_records(evidence_frames)
        anchor_diagnostics = pd.DataFrame(diagnostic_rows)
        anchor_public, anchor_evidence, anchor_diagnostics = self._attach_run_identity(
            anchor_public, anchor_evidence, anchor_diagnostics, run_id=run_id
        )
        anchor_public = self._apply_primary_decisions(
            anchor_public,
            anchor_evidence,
            canonical_input.contrast_specification,
            anchor_diagnostics,
        )
        if not anchor_evidence.empty:
            anchor_evidence = validate_evidence_layer(anchor_evidence)
        anchor_public = validate_contrast_public_view(anchor_public)

        meta_public_frames, meta_evidence_frames, meta_diagnostic_rows = [], [], []
        for adapter in meta_adapters:
            for _, contrast in canonical_input.contrast_specification.iterrows():
                adapter_result = adapter.run_from_anchor_results(
                    canonical_input,
                    contrast,
                    anchor_public=anchor_public,
                    anchor_evidence=anchor_evidence,
                    analysis_id=analysis_id,
                    run_id=run_id,
                    native_output_dir=paths["method_native_outputs"],
                )
                meta_public_frames.append(adapter_result.public_view)
                if not adapter_result.evidence_layer.empty:
                    meta_evidence_frames.append(adapter_result.evidence_layer)
                meta_diagnostic_rows.append(adapter_result.diagnostics.to_record())

        meta_public = self._concat_records(meta_public_frames)
        meta_evidence = self._concat_records(meta_evidence_frames)
        meta_diagnostics = pd.DataFrame(meta_diagnostic_rows)
        if not meta_public.empty:
            meta_public, meta_evidence, meta_diagnostics = self._attach_run_identity(
                meta_public, meta_evidence, meta_diagnostics, run_id=run_id
            )

        public = self._concat_records([anchor_public, meta_public] if not meta_public.empty else [anchor_public])
        evidence = self._concat_records(
            [anchor_evidence, meta_evidence] if not meta_evidence.empty else [anchor_evidence]
        )
        diagnostics = self._concat_records(
            [anchor_diagnostics, meta_diagnostics] if not meta_diagnostics.empty else [anchor_diagnostics]
        )
        public = self._apply_primary_decisions(
            public, evidence, canonical_input.contrast_specification, diagnostics
        )
        if not evidence.empty:
            evidence = validate_evidence_layer(evidence)
        public = validate_contrast_public_view(public)

        public.to_csv(paths["contrast_tables"] / f"{analysis_id}.public.csv", index=False)
        evidence.to_csv(paths["contrast_tables"] / f"{analysis_id}.evidence.csv", index=False)
        diagnostics.to_csv(paths["diagnostics"] / f"{analysis_id}.diagnostics.csv", index=False)
        return RunnerResult(public, evidence, diagnostics)
