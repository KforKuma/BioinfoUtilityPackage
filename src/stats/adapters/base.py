from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import pandas as pd

from src.stats.schemas import CanonicalDAInput, MethodDiagnostics


@dataclass
class AdapterResult:
    public_view: pd.DataFrame
    evidence_layer: pd.DataFrame
    diagnostics: MethodDiagnostics


class BaseDifferentialAbundanceAdapter(ABC):
    method_id: str
    method_version: str

    def __init__(self, *, method_version: str = "unknown") -> None:
        if not getattr(self, "method_id", None):
            raise ValueError("Adapters must define a stable `method_id`.")
        self.method_version = method_version

    @abstractmethod
    def prepare_native_input(
        self,
        canonical_input: CanonicalDAInput,
        contrast: pd.Series,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def execute_native(self, native_input: Any, contrast: pd.Series) -> Any:
        raise NotImplementedError

    @abstractmethod
    def transform_native_output(
        self,
        native_output: Any,
        canonical_input: CanonicalDAInput,
        contrast: pd.Series,
        *,
        analysis_id: str,
        diagnostic_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def run(
        self,
        canonical_input: CanonicalDAInput,
        contrast: pd.Series,
        *,
        analysis_id: str,
        native_output_dir: Path,
    ) -> AdapterResult:
        canonical_input.validate()
        diagnostic_id = str(uuid4())
        diagnostics = MethodDiagnostics(
            diagnostic_id=diagnostic_id,
            analysis_id=analysis_id,
            method=self.method_id,
            method_version=self.method_version,
            status="running",
            input_hash=canonical_input.input_hash(),
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            native_input = self.prepare_native_input(canonical_input, contrast)
            native_output = self.execute_native(native_input, contrast)
            native_diagnostics = getattr(native_output, "attrs", {}).get("diagnostics", {})
            if isinstance(native_diagnostics, dict):
                diagnostics.details.update(native_diagnostics)
                native_warnings = native_diagnostics.get("warnings", [])
                if isinstance(native_warnings, str):
                    native_warnings = [native_warnings]
                diagnostics.warnings.extend(str(value) for value in native_warnings)
            native_path = self.save_native_output(
                native_output,
                native_output_dir,
                analysis_id=analysis_id,
                contrast_id=str(contrast["contrast_id"]),
            )
            diagnostics.native_output_path = str(native_path)
            public, evidence = self.transform_native_output(
                native_output,
                canonical_input,
                contrast,
                analysis_id=analysis_id,
                diagnostic_id=diagnostic_id,
            )
            diagnostics.converged = bool(native_diagnostics.get("converged", True))
            diagnostics.finish(status="success" if diagnostics.converged else "diagnostics_invalid")
            return AdapterResult(public, evidence, diagnostics)
        except Exception as exc:
            diagnostics.error_type = type(exc).__name__
            diagnostics.error_message = str(exc)
            diagnostics.converged = False
            diagnostics.finish(status="failed")
            failure_reason = str(getattr(exc, "reason", type(exc).__name__))
            public = self.unavailable_public_rows(
                canonical_input,
                contrast,
                analysis_id=analysis_id,
                diagnostic_id=diagnostic_id,
                failure_reason=failure_reason,
            )
            return AdapterResult(public, pd.DataFrame(), diagnostics)

    def save_native_output(
        self,
        native_output: Any,
        native_output_dir: Path,
        *,
        analysis_id: str,
        contrast_id: str,
    ) -> Path:
        target_dir = native_output_dir / self.method_id / analysis_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = contrast_id.replace(":", "_").replace("/", "_").replace("\\", "_")
        if isinstance(native_output, pd.DataFrame):
            target = target_dir / f"{stem}.csv"
            native_output.to_csv(target, index=False)
        else:
            target = target_dir / f"{stem}.json"
            with target.open("w", encoding="utf-8") as handle:
                json.dump(native_output, handle, ensure_ascii=False, indent=2, default=str)
        return target

    def unavailable_public_rows(
        self,
        canonical_input: CanonicalDAInput,
        contrast: pd.Series,
        *,
        analysis_id: str,
        diagnostic_id: str,
        failure_reason: str,
    ) -> pd.DataFrame:
        rows = []
        cell_types = canonical_input.cell_type_manifest.loc[
            canonical_input.cell_type_manifest["inclusion_status"].eq("included"), "cell_type"
        ]
        for cell_type in cell_types:
            reference_is_fixed = contrast.get("reference_is_fixed", False)
            reference_is_fixed = bool(reference_is_fixed) if pd.notna(reference_is_fixed) else False
            result_id = str(uuid5(
                NAMESPACE_URL,
                f"{analysis_id}:{self.method_id}:{contrast['contrast_id']}:{cell_type}:composition",
            ))
            rows.append(
                {
                    "result_id": result_id,
                    "evidence_id": pd.NA,
                    "method": self.method_id,
                    "method_version": self.method_version,
                    "analysis_id": analysis_id,
                    "cell_type": cell_type,
                    "contrast_id": contrast["contrast_id"],
                    "contrast_definition": contrast["contrast_definition"],
                    "contrast_type": contrast["contrast_type"],
                    "result_scope": "cell_type_specific",
                    "group_1": contrast["group_1"],
                    "group_2": contrast["group_2"],
                    "reference_group": contrast.get("reference_group", contrast["group_2"]),
                    "reference_cell_type": pd.NA,
                    "effect_component": "composition",
                    "estimate": float("nan"),
                    "effect_estimand": pd.NA,
                    "effect_scale": pd.NA,
                    "effect_null": pd.NA,
                    "effect_direction": "not_applicable",
                    "direction_basis": pd.NA,
                    "reference_strategy": contrast.get("reference_strategy", "not_applicable"),
                    "reference_selection_reason": contrast.get("reference_selection_reason", pd.NA),
                    "reference_is_fixed": reference_is_fixed,
                    "is_benchmark_eligible": False,
                    "estimand_compatibility": "unavailable",
                    "derived_from_native_effect": False,
                    "primary_decision": pd.NA,
                    "decision_metric": pd.NA,
                    "decision_value": pd.NA,
                    "decision_operator": pd.NA,
                    "decision_threshold": pd.NA,
                    "decision_rule_id": pd.NA,
                    "decision_rule_description": pd.NA,
                    "is_available": False,
                    "is_valid": False,
                    "contrast_status": "failed",
                    "failure_reason": failure_reason,
                    "diagnostic_id": diagnostic_id,
                }
            )
        return pd.DataFrame(rows)
