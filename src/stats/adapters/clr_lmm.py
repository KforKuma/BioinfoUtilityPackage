from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.stats.adapters._shared import (
    NativeInput,
    frequentist_evidence,
    prepare_pairwise_input,
    public_row,
    require_columns,
)
from src.stats.adapters.base import BaseDifferentialAbundanceAdapter
from src.stats.engine.clr import run_CLR_LMM
from src.stats.multiple_testing import apply_bh_by_family
from src.stats.schemas import CanonicalDAInput, DecisionRule, load_default_decision_rules


class CLRLMMAdapter(BaseDifferentialAbundanceAdapter):
    """Canonical cell-type CLR mixed-model adapter.

    The native global full-vs-reduced LRT is retained as supporting evidence only.
    Primary discoveries use the requested binary coefficient p-value with one BH
    adjustment across the fixed cell-type family.
    """

    method_id = "clr_lmm"
    primary_decision_rule_id = "clr-lmm-celltype-bh-0.05-v1"

    def __init__(
        self,
        *,
        method_version: str = "statsmodels",
        nominal_alpha: float = 0.05,
        pseudocount: float = 1.0,
        group_label: str = "donor_id",
        covariates: tuple[str, ...] = ("tissue",),
    ) -> None:
        super().__init__(method_version=method_version)
        if nominal_alpha != 0.05:
            raise ValueError("The registered CLR_LMM rule is fixed at alpha=0.05.")
        if pseudocount <= 0:
            raise ValueError("CLR_LMM pseudocount must be positive.")
        self.nominal_alpha = nominal_alpha
        self.pseudocount = pseudocount
        self.group_label = group_label
        self.covariates = tuple(covariates)

    @classmethod
    def decision_rules(cls, alpha: float = 0.05) -> list[DecisionRule]:
        if alpha != 0.05:
            raise ValueError("The registered CLR_LMM rule is fixed at alpha=0.05.")
        return load_default_decision_rules().for_method(cls.method_id)

    def prepare_native_input(
        self, canonical_input: CanonicalDAInput, contrast: pd.Series
    ) -> NativeInput:
        return prepare_pairwise_input(canonical_input, contrast, options={
            "nominal_alpha": self.nominal_alpha,
            "pseudocount": self.pseudocount,
            "group_label": self.group_label,
            "covariates": list(self.covariates),
        })

    def execute_native(self, native_input: NativeInput, contrast: pd.Series) -> pd.DataFrame:
        factor = str(native_input.contrast["factor"])
        group_1 = str(native_input.contrast["group_1"])
        group_2 = str(native_input.contrast["group_2"])
        metadata_columns = ["sample_id", factor, self.group_label]
        active_covariates = [
            column for column in self.covariates
            if column in native_input.sample_manifest.columns and column != factor
        ]
        metadata_columns.extend(active_covariates)
        missing = set(metadata_columns) - set(native_input.sample_manifest.columns)
        if missing:
            raise ValueError(f"CLR_LMM sample metadata is missing: {sorted(missing)}")
        frame = native_input.abundance.merge(
            native_input.sample_manifest[metadata_columns].drop_duplicates("sample_id"),
            on="sample_id",
            how="left",
            validate="many_to_one",
        )
        formula = " + ".join([factor, *active_covariates])
        rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        for cell_type in native_input.cell_type_manifest["cell_type"].astype(str):
            result = run_CLR_LMM(
                frame,
                cell_type,
                formula=formula,
                main_variable=factor,
                ref_label=group_2,
                group_label=self.group_label,
                alpha=self.nominal_alpha,
                use_reml=False,
                pseudocount=self.pseudocount,
            )
            extra = result.get("extra", {}) or {}
            table = result.get("contrast_table")
            native_status = str(result.get("contrast_status", "unavailable"))
            coefficient = None
            if isinstance(table, pd.DataFrame) and group_1 in table.index:
                coefficient = table.loc[group_1]
                if isinstance(coefficient, pd.DataFrame):
                    coefficient = coefficient.iloc[0]
            coefficient_pvalue = (
                pd.to_numeric(pd.Series([coefficient.get("P>|z|", np.nan)]), errors="coerce").iloc[0]
                if coefficient is not None else np.nan
            )
            coefficient_estimate = (
                pd.to_numeric(pd.Series([coefficient.get("Coef.", np.nan)]), errors="coerce").iloc[0]
                if coefficient is not None else np.nan
            )
            if (
                native_status == "success"
                and coefficient is not None
                and np.isfinite(coefficient_pvalue)
                and np.isfinite(coefficient_estimate)
            ):
                rows.append({
                    "cell_type": cell_type,
                    "estimate": coefficient_estimate,
                    "standard_error": coefficient.get("Std.Err.", np.nan),
                    "statistic": coefficient.get("z", np.nan),
                    "pvalue_raw": coefficient_pvalue,
                    "global_lrt_pvalue": extra.get("global_pvalue", np.nan),
                    "global_lrt_statistic": extra.get("global_statistic", np.nan),
                    "global_lrt_df": extra.get("global_df", np.nan),
                    "native_status": "success",
                    "failure_reason": pd.NA,
                    "singular_fit": bool(extra.get("singular_fit", False)),
                })
            else:
                reason = str(
                    result.get("failure_reason")
                    or (
                        "nonfinite_celltype_coefficient_test"
                        if native_status == "success" and coefficient is not None
                        else "celltype_contrast_unavailable"
                    )
                )
                rows.append({
                    "cell_type": cell_type,
                    "estimate": np.nan,
                    "standard_error": np.nan,
                    "statistic": np.nan,
                    "pvalue_raw": np.nan,
                    "global_lrt_pvalue": result.get("p_val", np.nan),
                    "global_lrt_statistic": extra.get("global_statistic", np.nan),
                    "global_lrt_df": extra.get("global_df", np.nan),
                    "native_status": "unavailable",
                    "failure_reason": reason,
                    "singular_fit": bool(extra.get("singular_fit", False)),
                })
                warnings.append(f"{cell_type}:{reason}")
            warnings.extend(str(value) for value in extra.get("warnings", []))
        output = pd.DataFrame(rows)
        output["adjustment_family"] = (
            f"clr_lmm:{native_input.contrast['contrast_id']}:composition"
        )
        output = apply_bh_by_family(
            output,
            pvalue_col="pvalue_raw",
            family_cols=["adjustment_family"],
        )
        successful = int(output["native_status"].eq("success").sum())
        output.attrs["diagnostics"] = {
            "converged": successful > 0,
            "number_celltypes": int(len(output)),
            "number_successful_celltypes": successful,
            "number_unavailable_celltypes": int(len(output) - successful),
            "number_singular_fits": int(output["singular_fit"].fillna(False).sum()),
            "global_test": "full_vs_reduced_likelihood_ratio_supporting_only",
            "primary_test": "binary_fixed_effect_coefficient_BH",
            "warnings": warnings,
        }
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
            {
                "cell_type", "estimate", "pvalue_raw", "pvalue_adjusted",
                "native_status", "global_lrt_pvalue",
            },
            self.method_id,
        )
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
                effect_estimand="sample_celltype_centered_log_ratio_fixed_effect",
                effect_scale="clr_log_ratio",
                direction_basis=(
                    "signed disease fixed-effect coefficient on sample CLR; null=0; "
                    "random intercept=donor"
                ),
                decision_rule_id=self.primary_decision_rule_id,
                effect_estimate_source="method_native",
                result_interpretation=(
                    "Cell-type CLR location contrast relative to the geometric mean of the "
                    "complete modeled cell-type composition; not absolute abundance."
                ),
            )
            if str(native["native_status"]) != "success":
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
                    "is_benchmark_eligible": False,
                    "estimand_compatibility": "unavailable",
                    "is_available": False,
                    "is_valid": False,
                    "contrast_status": "unavailable",
                    "failure_reason": native.get("failure_reason", "celltype_contrast_unavailable"),
                })
            else:
                evidence_rows.append(frequentist_evidence(
                    evidence_id=evidence_id,
                    result_id=result_id,
                    pvalue_raw=native["pvalue_raw"],
                    pvalue_adjusted=native["pvalue_adjusted"],
                    native_rule_id="clr-lmm-celltype-bh-0.05-v1",
                    adjustment_family=str(native["adjustment_family"]),
                    alpha=self.nominal_alpha,
                    test_name="clr_lmm_binary_fixed_effect_wald",
                    statistic=native.get("statistic", pd.NA),
                    statistic_type="Wald_z",
                    standard_error=native.get("standard_error", pd.NA),
                    extra={
                        "global_lrt_pvalue": native.get("global_lrt_pvalue", pd.NA),
                        "global_lrt_statistic": native.get("global_lrt_statistic", pd.NA),
                        "global_lrt_df": native.get("global_lrt_df", pd.NA),
                        "global_test_role": "supporting_only_not_primary_decision",
                        "pseudocount": self.pseudocount,
                        "random_effect": self.group_label,
                        "singular_fit": native.get("singular_fit", False),
                    },
                ))
            public_rows.append(row)
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)
