from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from src.stats.adapters.base import BaseDifferentialAbundanceAdapter
from src.stats.schemas import CanonicalDAInput


def _direction(estimate: float) -> str:
    return "group_1_higher" if estimate > 0 else ("group_2_higher" if estimate < 0 else "no_effect")


class _MockBase(BaseDifferentialAbundanceAdapter):
    primary_decision_rule_id: str

    def prepare_native_input(self, canonical_input: CanonicalDAInput, contrast: pd.Series) -> pd.DataFrame:
        return canonical_input.abundance_long.copy()

    def _ids(self, analysis_id: str, contrast_id: str, cell_type: str) -> tuple[str, str]:
        result_id = str(uuid5(NAMESPACE_URL, f"{analysis_id}:{self.method_id}:{contrast_id}:{cell_type}:composition"))
        evidence_id = str(uuid5(NAMESPACE_URL, f"evidence:{result_id}"))
        return result_id, evidence_id

    def _public_row(
        self,
        *,
        result_id: str,
        evidence_id: str,
        analysis_id: str,
        diagnostic_id: str,
        contrast: pd.Series,
        cell_type: str,
        estimate: float,
    ) -> dict[str, Any]:
        return {
            "result_id": result_id,
            "evidence_id": evidence_id,
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
            "estimate": estimate,
            "effect_estimand": "proportion_difference",
            "effect_scale": "proportion",
            "effect_null": 0.0,
            "effect_direction": _direction(estimate),
            "direction_basis": "signed_difference:null=0",
            "reference_strategy": "not_applicable",
            "reference_selection_reason": pd.NA,
            "reference_is_fixed": False,
            "is_benchmark_eligible": True,
            "estimand_compatibility": "compatible",
            "derived_from_native_effect": False,
            "effect_estimate_source": "method_native",
            "primary_decision": pd.NA,
            "decision_metric": pd.NA,
            "decision_value": pd.NA,
            "decision_operator": pd.NA,
            "decision_threshold": pd.NA,
            "decision_rule_id": self.primary_decision_rule_id,
            "decision_rule_description": pd.NA,
            "is_available": True,
            "is_valid": True,
            "contrast_status": "success",
            "failure_reason": pd.NA,
            "diagnostic_id": diagnostic_id,
        }


class MockFrequentistAdapter(_MockBase):
    method_id = "mock_frequentist"
    primary_decision_rule_id = "mock-frequentist-primary-v1"

    def execute_native(self, native_input: pd.DataFrame, contrast: pd.Series) -> pd.DataFrame:
        means = native_input.groupby(["cell_type", "sample_id"], as_index=False)["proportion"].sum()
        return means.groupby("cell_type", as_index=False)["proportion"].mean().rename(columns={"proportion": "estimate"})

    def transform_native_output(
        self, native_output, canonical_input, contrast, *, analysis_id, diagnostic_id
    ):
        public_rows, evidence_rows = [], []
        for index, row in native_output.reset_index(drop=True).iterrows():
            estimate = float(row["estimate"] - native_output["estimate"].mean())
            result_id, evidence_id = self._ids(analysis_id, contrast["contrast_id"], row["cell_type"])
            pvalue = min(0.99, 0.01 + index * 0.2)
            public_rows.append(self._public_row(
                result_id=result_id, evidence_id=evidence_id, analysis_id=analysis_id,
                diagnostic_id=diagnostic_id, contrast=contrast, cell_type=row["cell_type"], estimate=estimate,
            ))
            evidence_rows.append({
                "evidence_id": evidence_id, "result_id": result_id,
                "evidence_paradigm": "frequentist", "native_decision": pvalue < 0.05,
                "native_decision_metric": "pvalue_adjusted", "native_decision_value": pvalue,
                "native_decision_rule_id": "mock-frequentist-native-v1",
                "pvalue_raw": pvalue, "pvalue": pvalue, "pvalue_adjusted": pvalue,
                "pvalue_type": "adjusted", "adjustment_method": "BH",
                "adjustment_family": f"mock:{contrast['contrast_id']}", "nominal_alpha": 0.05,
            })
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)


class MockBayesianAdapter(_MockBase):
    method_id = "mock_bayesian"
    primary_decision_rule_id = "mock-bayesian-primary-v1"

    def execute_native(self, native_input: pd.DataFrame, contrast: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"cell_type": sorted(native_input["cell_type"].unique())})

    def transform_native_output(
        self, native_output, canonical_input, contrast, *, analysis_id, diagnostic_id
    ):
        public_rows, evidence_rows = [], []
        for index, row in native_output.iterrows():
            probability = 0.98 if index == 0 else 0.5
            estimate = 0.2 if index == 0 else -0.1
            result_id, evidence_id = self._ids(analysis_id, contrast["contrast_id"], row["cell_type"])
            public_rows.append(self._public_row(
                result_id=result_id, evidence_id=evidence_id, analysis_id=analysis_id,
                diagnostic_id=diagnostic_id, contrast=contrast, cell_type=row["cell_type"], estimate=estimate,
            ))
            evidence_rows.append({
                "evidence_id": evidence_id, "result_id": result_id,
                "evidence_paradigm": "bayesian", "native_decision": probability >= 0.95,
                "native_decision_metric": "posterior_inclusion_probability",
                "native_decision_value": probability,
                "native_decision_rule_id": "mock-bayesian-native-v1",
                "posterior_probability": probability,
                "posterior_probability_type": "inclusion",
                "posterior_inclusion_probability": probability,
                "pvalue_raw": pd.NA, "pvalue": pd.NA, "pvalue_adjusted": pd.NA,
                "pvalue_type": "not_applicable",
            })
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)


class MockFailureAdapter(_MockBase):
    method_id = "mock_failure"

    def execute_native(self, native_input: pd.DataFrame, contrast: pd.Series):
        raise RuntimeError("intentional mock failure")

    def transform_native_output(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("Failure adapter must not transform output.")
