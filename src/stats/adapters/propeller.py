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


class PropellerAdapter(BaseDifferentialAbundanceAdapter):
    method_id = "propeller"
    primary_decision_rule_id = "propeller-bh-0.05-v1"

    def __init__(
        self,
        *,
        method_version: str = "unknown",
        bridge: RScriptBridge | None = None,
        executor: NativeExecutor | None = None,
        transform: str = "logit",
        robust: bool = True,
        trend: bool = False,
        nominal_alpha: float = 0.05,
        covariates: list[str] | None = None,
    ) -> None:
        super().__init__(method_version=method_version)
        if nominal_alpha != 0.05:
            raise ValueError("The v1 Propeller primary rule is registered at nominal_alpha=0.05.")
        if transform not in {"logit", "asin"}:
            raise ValueError("Propeller transform must be 'logit' or 'asin'.")
        if bridge is None and executor is None:
            raise ValueError("PropellerAdapter requires an RScriptBridge or executor.")
        self.bridge = bridge
        self.executor = executor
        self.transform = transform
        self.robust = robust
        self.trend = trend
        self.nominal_alpha = nominal_alpha
        self.covariates = list(covariates or [])

    @classmethod
    def decision_rules(cls, alpha: float = 0.05) -> list[DecisionRule]:
        if alpha != 0.05:
            raise ValueError("Phase 1.5 decision rules are fixed at nominal_alpha=0.05.")
        return load_default_decision_rules().for_method(cls.method_id)

    def prepare_native_input(self, canonical_input: CanonicalDAInput, contrast: pd.Series) -> NativeInput:
        return prepare_pairwise_input(canonical_input, contrast, options={
            "transform": self.transform,
            "robust": self.robust,
            "trend": self.trend,
            "nominal_alpha": self.nominal_alpha,
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
        effect_scale = "logit" if self.transform == "logit" else "arcsine_sqrt"
        effect_estimand = f"sample_mean_{effect_scale}_proportion_difference"
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
                effect_estimand=effect_estimand,
                effect_scale=effect_scale,
                direction_basis=f"signed_transformed_proportion_difference:null=0;transform={self.transform}",
                decision_rule_id=self.primary_decision_rule_id,
            )
            public_rows.append(row)
            evidence_rows.append(frequentist_evidence(
                evidence_id=evidence_id,
                result_id=result_id,
                pvalue_raw=native["pvalue_raw"],
                pvalue_adjusted=native["pvalue_adjusted"],
                native_rule_id="propeller-native-bh-v1",
                adjustment_family=family,
                alpha=self.nominal_alpha,
                test_name="propeller_moderated_t",
                statistic=native.get("statistic", pd.NA),
                statistic_type="moderated_t",
                standard_error=native.get("standard_error", pd.NA),
                extra={
                    "transformation": self.transform,
                    "robust": self.robust,
                    "trend": self.trend,
                    "native_contrast": f"{contrast['group_1']} - {contrast['group_2']}",
                    "design_formula": contrast.get("design_formula", f"~0+{contrast.get('factor')}")
                },
            ))
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)
