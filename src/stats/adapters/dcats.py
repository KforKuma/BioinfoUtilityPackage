from __future__ import annotations

from typing import Any

import pandas as pd

from src.stats.adapters._shared import (
    NativeExecutor,
    NativeInput,
    frequentist_evidence,
    prepare_pairwise_input,
    public_row,
    require_columns,
)
from src.stats.adapters.base import BaseDifferentialAbundanceAdapter
from src.stats.adapters.r_bridge import RScriptBridge
from src.stats.schemas import CanonicalDAInput, DecisionRule, load_default_decision_rules


class DCATSAdapter(BaseDifferentialAbundanceAdapter):
    method_id = "dcats"
    primary_decision_rule_id = "dcats-bh-0.05-v1"

    def __init__(
        self,
        *,
        method_version: str = "unknown",
        bridge: RScriptBridge | None = None,
        executor: NativeExecutor | None = None,
        nominal_alpha: float = 0.05,
        pseudo_count: float | None = None,
        base_model: str = "NULL",
        reference_cell_types: list[str] | None = None,
        fix_phi: float | None = None,
        covariates: list[str] | None = None,
    ) -> None:
        super().__init__(method_version=method_version)
        if nominal_alpha != 0.05:
            raise ValueError("The v1 DCATS primary rule is registered at nominal_alpha=0.05.")
        if bridge is None and executor is None:
            raise ValueError("DCATSAdapter requires an RScriptBridge or executor.")
        if base_model not in {"NULL", "FULL"}:
            raise ValueError("DCATS base_model must be 'NULL' or 'FULL'.")
        self.bridge = bridge
        self.executor = executor
        self.nominal_alpha = nominal_alpha
        self.pseudo_count = pseudo_count
        self.base_model = base_model
        self.reference_cell_types = reference_cell_types
        self.fix_phi = fix_phi
        self.covariates = list(covariates or [])

    @classmethod
    def decision_rules(cls, alpha: float = 0.05) -> list[DecisionRule]:
        if alpha != 0.05:
            raise ValueError("Phase 1.5 decision rules are fixed at nominal_alpha=0.05.")
        return load_default_decision_rules().for_method(cls.method_id)

    def prepare_native_input(self, canonical_input: CanonicalDAInput, contrast: pd.Series) -> NativeInput:
        return prepare_pairwise_input(canonical_input, contrast, options={
            "nominal_alpha": self.nominal_alpha,
            "pseudo_count": self.pseudo_count,
            "base_model": self.base_model,
            "reference_cell_types": self.reference_cell_types,
            "fix_phi": self.fix_phi,
            "covariates": self.covariates,
        })

    def execute_native(self, native_input: NativeInput, contrast: pd.Series) -> pd.DataFrame:
        if self.executor is not None:
            return self.executor(native_input, contrast)
        assert self.bridge is not None
        return self.bridge.run(self.method_id, native_input)

    def transform_native_output(
        self,
        native_output: pd.DataFrame,
        canonical_input: CanonicalDAInput,
        contrast: pd.Series,
        *,
        analysis_id: str,
        diagnostic_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        require_columns(
            native_output,
            {"cell_type", "estimate", "pvalue_raw", "pvalue_adjusted"},
            self.method_id,
        )
        family = f"{self.method_id}:{analysis_id}:{contrast['contrast_id']}:composition"
        public_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        for _, native in native_output.iterrows():
            row, result_id, evidence_id = public_row(
                method_id=self.method_id,
                method_version=self.method_version,
                analysis_id=analysis_id,
                diagnostic_id=diagnostic_id,
                contrast=contrast,
                cell_type=str(native["cell_type"]),
                effect_component="composition",
                estimate=native["estimate"],
                effect_estimand="dcats_beta_binomial_log_odds_coefficient",
                effect_scale="log_odds",
                direction_basis="signed_beta_binomial_log_odds_contrast:null=0",
                decision_rule_id=self.primary_decision_rule_id,
                effect_estimate_source="method_native",
                result_interpretation=(
                    "Native DCATS beta-binomial coefficient for the requested factor; "
                    "not an absolute abundance difference."
                ),
            )
            public_rows.append(row)
            evidence_rows.append(frequentist_evidence(
                evidence_id=evidence_id,
                result_id=result_id,
                pvalue_raw=native["pvalue_raw"],
                pvalue_adjusted=native["pvalue_adjusted"],
                native_rule_id="dcats-native-bh-v1",
                adjustment_family=family,
                alpha=self.nominal_alpha,
                test_name="dcats_glm_lrt",
                statistic=native.get("statistic", pd.NA),
                statistic_type="LRT",
                standard_error=native.get("standard_error", pd.NA),
                extra={
                    "base_model": self.base_model,
                    "pseudo_count_policy": "native_default" if self.pseudo_count is None else "fixed",
                    "pseudo_count": self.pseudo_count,
                    "normalization_reference": (
                        "total_count" if self.reference_cell_types is None else ";".join(self.reference_cell_types)
                    ),
                    "fix_phi": self.fix_phi,
                    "test_scope": "cell_type_specific",
                    "distribution_family": "DCATS_GLM_native",
                    "native_fdr": native.get("native_fdr", pd.NA),
                },
            ))
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)
