from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.stats.adapters._shared import (
    NativeExecutor,
    NativeInput,
    bayesian_evidence,
    prepare_pairwise_input,
    public_row,
    require_columns,
)
from src.stats.adapters.base import BaseDifferentialAbundanceAdapter
from src.stats.adapters.r_bridge import RScriptBridge
from src.stats.schemas import CanonicalDAInput, DecisionRule, load_default_decision_rules


class SccompAdapter(BaseDifferentialAbundanceAdapter):
    method_id = "sccomp"
    composition_rule_id = "sccomp-composition-fdr-0.05-v1"
    variability_rule_id = "sccomp-variability-fdr-0.05-v1"

    def __init__(
        self,
        *,
        method_version: str = "unknown",
        bridge: RScriptBridge | None = None,
        executor: NativeExecutor | None = None,
        nominal_alpha: float = 0.05,
        effect_threshold: float = 0.1,
        percent_false_positive: float = 5.0,
        cores: int = 2,
        include_variability: bool = True,
        covariates: list[str] | None = None,
        model_cache_dir: str | None = None,
        inference_method: str = "hmc",
        max_sampling_iterations: int = 1000,
        adapt_delta: float = 0.95,
    ) -> None:
        super().__init__(method_version=method_version)
        if nominal_alpha != 0.05:
            raise ValueError("The v1 sccomp primary rules are registered at nominal_alpha=0.05.")
        if bridge is None and executor is None:
            raise ValueError("SccompAdapter requires an RScriptBridge or executor.")
        self.bridge = bridge
        self.executor = executor
        self.nominal_alpha = nominal_alpha
        self.effect_threshold = effect_threshold
        self.percent_false_positive = percent_false_positive
        self.cores = cores
        self.include_variability = include_variability
        self.covariates = list(covariates or [])
        self.model_cache_dir = model_cache_dir
        self.inference_method = inference_method
        self.max_sampling_iterations = max_sampling_iterations
        self.adapt_delta = adapt_delta

    @classmethod
    def decision_rules(cls, alpha: float = 0.05) -> list[DecisionRule]:
        if alpha != 0.05:
            raise ValueError("Phase 1.5 decision rules are fixed at nominal_alpha=0.05.")
        return load_default_decision_rules().for_method(cls.method_id)

    def prepare_native_input(self, canonical_input: CanonicalDAInput, contrast: pd.Series) -> NativeInput:
        prepared = prepare_pairwise_input(canonical_input, contrast, options={
            "nominal_alpha": self.nominal_alpha,
            "effect_threshold": self.effect_threshold,
            "percent_false_positive": self.percent_false_positive,
            "cores": self.cores,
            "include_variability": self.include_variability,
            "covariates": self.covariates,
            "model_cache_dir": self.model_cache_dir,
            "inference_method": self.inference_method,
            "max_sampling_iterations": self.max_sampling_iterations,
            "adapt_delta": self.adapt_delta,
        })
        sample_ids = prepared.sample_manifest["sample_id"].astype(str).tolist()
        cell_types = prepared.cell_type_manifest["cell_type"].astype(str).tolist()
        sample_to_native = {value: f"sample_{index:04d}" for index, value in enumerate(sample_ids, 1)}
        cell_to_native = {value: f"celltype_{index:04d}" for index, value in enumerate(cell_types, 1)}

        abundance = prepared.abundance.copy()
        abundance["sample_id"] = abundance["sample_id"].astype(str).map(sample_to_native)
        abundance["cell_type"] = abundance["cell_type"].astype(str).map(cell_to_native)
        sample_manifest = prepared.sample_manifest.copy()
        sample_manifest["sample_id"] = sample_manifest["sample_id"].astype(str).map(sample_to_native)
        cell_manifest = prepared.cell_type_manifest.copy()
        cell_manifest["cell_type"] = cell_manifest["cell_type"].astype(str).map(cell_to_native)
        options = dict(prepared.options)
        options["sample_id_map"] = {native: original for original, native in sample_to_native.items()}
        options["cell_type_id_map"] = {native: original for original, native in cell_to_native.items()}
        return NativeInput(abundance, sample_manifest, cell_manifest, prepared.contrast, options)

    def execute_native(self, native_input: NativeInput, contrast: pd.Series) -> pd.DataFrame:
        if self.executor is not None:
            return self.executor(native_input, contrast)
        assert self.bridge is not None
        output = self.bridge.run(self.method_id, native_input)
        mapping = native_input.options.get("cell_type_id_map", {})
        if "cell_type" in output.columns and mapping:
            output["native_cell_type_id"] = output["cell_type"].astype(str)
            output["cell_type"] = output["native_cell_type_id"].map(mapping)
            if output["cell_type"].isna().any():
                raise ValueError("sccomp returned an unmapped native cell-type identifier.")
        return output

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
            {"cell_type", "c_effect", "c_lower", "c_upper", "c_pH0", "c_FDR"},
            self.method_id,
        )
        public_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        for _, native in native_output.iterrows():
            components = [
                ("composition", "c", self.composition_rule_id, "composition_logit_contrast"),
            ]
            if self.include_variability and all(
                column in native_output.columns for column in ("v_effect", "v_lower", "v_upper", "v_pH0", "v_FDR")
            ) and np.isfinite(pd.to_numeric(pd.Series([native.get("v_effect")]), errors="coerce").iloc[0]):
                components.append(("variability", "v", self.variability_rule_id, "variability_log_contrast"))

            for component, prefix, rule_id, estimand in components:
                effect = native[f"{prefix}_effect"]
                fdr = native[f"{prefix}_FDR"]
                ph0 = native[f"{prefix}_pH0"]
                row, result_id, evidence_id = public_row(
                    method_id=self.method_id,
                    method_version=self.method_version,
                    analysis_id=analysis_id,
                    diagnostic_id=diagnostic_id,
                    contrast=contrast,
                    cell_type=str(native["cell_type"]),
                    effect_component=component,
                    estimate=effect,
                    effect_estimand=estimand,
                    effect_scale="logit_unconstrained" if component == "composition" else "log_variability",
                    direction_basis=f"signed_{component}_contrast:null=0",
                    decision_rule_id=rule_id,
                    result_interpretation=(
                        "Bayesian posterior contrast on sccomp's native unconstrained scale; "
                        "not an absolute abundance difference."
                    ),
                    is_benchmark_eligible=component == "composition",
                )
                public_rows.append(row)
                evidence_rows.append(bayesian_evidence(
                    evidence_id=evidence_id,
                    result_id=result_id,
                    native_decision=bool(float(fdr) < self.nominal_alpha),
                    native_metric=f"{prefix}_FDR",
                    native_value=fdr,
                    native_rule_id=f"sccomp-{component}-native-fdr-{self.nominal_alpha}-v1",
                    posterior_probability=ph0,
                    posterior_probability_type=f"sccomp_{prefix}_pH0_native",
                    native_discovery_metric_name=f"{prefix}_FDR",
                    native_discovery_metric_value=fdr,
                    credible_interval_lower=native[f"{prefix}_lower"],
                    credible_interval_upper=native[f"{prefix}_upper"],
                    extra={
                        "c_FDR": fdr if prefix == "c" else pd.NA,
                        "v_FDR": fdr if prefix == "v" else pd.NA,
                        "parameter": native.get("parameter", pd.NA),
                        "factor": native.get("factor", pd.NA),
                        "effective_sample_size": native.get(f"{prefix}_n_eff", pd.NA),
                        "r_k_hat": native.get(f"{prefix}_R_k_hat", pd.NA),
                        "model_rhat_max": native.get("model_rhat_max", pd.NA),
                        "model_ess_bulk_min": native.get("model_ess_bulk_min", pd.NA),
                        "model_ess_tail_min": native.get("model_ess_tail_min", pd.NA),
                        "model_divergences": native.get("model_divergences", pd.NA),
                        "test_effect_threshold": self.effect_threshold,
                        "outlier_probability": native.get("outlier_probability", pd.NA),
                    },
                ))
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)
