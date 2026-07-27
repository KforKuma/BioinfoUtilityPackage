from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from src.stats.engine import (
    run_ANOVA_naive,
    run_ANOVA_transformed,
    run_DKD,
    run_Dirichlet_Multinomial_Wald,
    run_Dirichlet_Wald,
    run_Perm_Mixed,
    run_PyDESeq2,
    run_pCLR_LMM,
    run_pCLR_OLS,
)
from src.stats.multiple_testing import apply_bh_by_family
from src.stats.schemas import CanonicalDAInput


@dataclass(frozen=True)
class _MethodSpec:
    estimand: str
    scale: str
    decision_rule_id: str
    test_name: str
    statistic_type: str
    interpretation: str


_SPECS = {
    "dirichlet_multinomial_wald": _MethodSpec(
        "dirichlet_multinomial_log_alpha_contrast", "log_ratio",
        "dirichlet-multinomial-wald-bh-0.05-v1", "dirichlet_multinomial_wald",
        "Wald_z", "Compositional log-alpha contrast; not an absolute abundance difference.",
    ),
    "dirichlet_wald": _MethodSpec(
        "dirichlet_log_alpha_contrast", "log_ratio", "dirichlet-wald-bh-0.05-v1",
        "dirichlet_wald", "Wald_z",
        "Compositional log-alpha contrast; not an absolute abundance difference.",
    ),
    "dkd": _MethodSpec(
        "sample_mean_raw_proportion_difference_with_deconfounded_rank_test",
        "proportion_difference", "dkd-bh-0.05-v1", "deconfounded_kruskal_binary",
        "Kruskal_Wallis_H",
        "Raw-proportion direction paired with a deconfounded rank-test decision; not absolute abundance.",
    ),
    "pydeseq2": _MethodSpec(
        "size_factor_normalized_count_log_fold_change", "natural_log_fold_change",
        "pydeseq2-native-bh-0.05-v1", "pydeseq2_wald", "Wald_z",
        "DESeq2 size-factor-normalized count fold change; not a raw proportion difference.",
    ),
    "permutation_mixed": _MethodSpec(
        "sample_mean_raw_proportion_difference_with_block_permutation_test",
        "proportion_difference", "permutation-mixed-bh-0.05-v1",
        "donor_block_permutation_mann_whitney", "permutation_rank_statistic",
        "Raw-proportion direction paired with a donor-block permutation decision; not absolute abundance.",
    ),
    "pclr_ols": _MethodSpec(
        "probabilistic_clr_ols_location_contrast", "clr_log_ratio",
        "pclr-ols-bh-0.05-v1", "probabilistic_clr_ols", "median_t",
        "Monte-Carlo CLR location contrast relative to the modeled composition geometric mean.",
    ),
    "pclr_lmm": _MethodSpec(
        "probabilistic_clr_mixed_location_contrast", "clr_log_ratio",
        "pclr-lmm-bh-0.05-v1", "probabilistic_clr_lmm", "median_Wald_z",
        "Monte-Carlo CLR mixed-model contrast relative to the modeled composition geometric mean.",
    ),
    "anova_naive": _MethodSpec(
        "sample_mean_raw_proportion_difference", "proportion_difference",
        "anova-naive-bh-0.05-v1", "anova_raw_proportion", "ANOVA_F",
        "Global sanity check on raw proportions; intentionally ignores compositional and donor structure.",
    ),
    "anova_transformed": _MethodSpec(
        "sample_mean_arcsine_sqrt_proportion_difference", "arcsine_sqrt_difference",
        "anova-transformed-bh-0.05-v1", "anova_arcsine_sqrt_proportion", "ANOVA_F",
        "Global sanity check after arcsine-square-root transformation; intentionally ignores donor structure.",
    ),
}


class EngineMethodAdapter(BaseDifferentialAbundanceAdapter):
    """Canonical adapter shared by the audited historical Python engines.

    Native p-values stay in the evidence layer. Except for PyDESeq2's native
    feature-level adjusted p-values, the primary family is one BH correction
    across the fixed cell-type universe for the requested contrast.
    """

    def __init__(
        self,
        method_id: str,
        *,
        method_version: str = "BioinfoUtilityPackage-engine",
        nominal_alpha: float = 0.05,
        max_workers: int = 1,
        random_seed: int = 0,
        n_permutations: int = 499,
        pclr_samples: int = 4,
        dirichlet_maxiter: int = 300,
    ) -> None:
        if method_id not in _SPECS:
            raise ValueError(f"Unsupported engine adapter method: {method_id!r}")
        self.method_id = method_id
        self.primary_decision_rule_id = _SPECS[method_id].decision_rule_id
        super().__init__(method_version=method_version)
        if nominal_alpha != 0.05:
            raise ValueError("Registered engine decision rules are fixed at alpha=0.05.")
        if n_permutations < 19:
            raise ValueError("Permutation budget must be at least 19.")
        if pclr_samples < 1:
            raise ValueError("pCLR Monte-Carlo sample count must be positive.")
        self.nominal_alpha = nominal_alpha
        self.max_workers = max(1, int(max_workers))
        self.random_seed = int(random_seed)
        self.n_permutations = int(n_permutations)
        self.pclr_samples = int(pclr_samples)
        self.dirichlet_maxiter = int(dirichlet_maxiter)

    def prepare_native_input(
        self, canonical_input: CanonicalDAInput, contrast: pd.Series
    ) -> NativeInput:
        return prepare_pairwise_input(canonical_input, contrast, options={
            "nominal_alpha": self.nominal_alpha,
            "max_workers": self.max_workers,
            "random_seed": self.random_seed,
            "n_permutations": self.n_permutations,
            "pclr_samples": self.pclr_samples,
            "dirichlet_maxiter": self.dirichlet_maxiter,
            "multiplicity_family": "fixed_cell_types_within_contrast",
        })

    @staticmethod
    def _engine_frame(native_input: NativeInput) -> pd.DataFrame:
        factor = str(native_input.contrast["factor"])
        metadata = ["sample_id", factor]
        metadata.extend(
            column for column in ("donor_id", "tissue")
            if column in native_input.sample_manifest.columns and column not in metadata
        )
        frame = native_input.abundance.rename(columns={"proportion": "prop"}).merge(
            native_input.sample_manifest[metadata].drop_duplicates("sample_id"),
            on="sample_id", how="inner", validate="many_to_one",
        )
        if factor != "disease":
            frame["disease"] = frame[factor]
        if "donor_id" not in frame:
            frame["donor_id"] = frame["sample_id"]
        if "tissue" not in frame:
            frame["tissue"] = "single_tissue"
        return frame

    def _run_one(
        self, frame: pd.DataFrame, cell_type: str, contrast: dict[str, Any], index: int
    ) -> dict[str, Any]:
        group_1 = str(contrast["group_1"])
        group_2 = str(contrast["group_2"])
        factor = str(contrast["factor"])
        formula = "disease + tissue" if frame["tissue"].nunique(dropna=True) > 1 else "disease"
        seed = self.random_seed + index * 1009
        common = dict(df_all=frame, cell_type=cell_type, alpha=self.nominal_alpha)
        if self.method_id == "dirichlet_wald":
            result = run_Dirichlet_Wald(
                **common, formula=formula, ref_label=group_2,
                group_label="sample_id", maxiter=self.dirichlet_maxiter,
            )
        elif self.method_id == "dirichlet_multinomial_wald":
            result = run_Dirichlet_Multinomial_Wald(
                **common, formula=formula, ref_label=group_2,
                group_label="sample_id", maxiter=self.dirichlet_maxiter,
            )
        elif self.method_id == "dkd":
            result = run_DKD(
                **common, formula=formula, main_variable="disease", ref_label=group_2,
                group_label="donor_id", use_reml=True,
            )
        elif self.method_id == "pydeseq2":
            result = run_PyDESeq2(
                **common, formula=formula, main_variable="disease", ref_label=group_2,
                group_label="sample_id",
            )
        elif self.method_id == "permutation_mixed":
            result = run_Perm_Mixed(
                **common, formula=formula, main_variable="disease",
                n_perm=self.n_permutations, ref_label=group_2,
                group_label="donor_id", pairwise_level="donor_id", seed=seed,
            )
        elif self.method_id == "pclr_ols":
            result = run_pCLR_OLS(
                **common, formula=formula, n_samples=self.pclr_samples,
                random_state=seed, disease_ref=group_2,
                tissue_ref="nif" if "nif" in set(frame["tissue"].astype(str)) else str(frame["tissue"].iloc[0]),
            )
        elif self.method_id == "pclr_lmm":
            result = run_pCLR_LMM(
                **common, formula=formula, random_effect="1 | donor_id",
                n_samples=self.pclr_samples, random_state=seed,
                disease_ref=group_2,
                tissue_ref="nif" if "nif" in set(frame["tissue"].astype(str)) else str(frame["tissue"].iloc[0]),
            )
        elif self.method_id == "anova_naive":
            result = run_ANOVA_naive(**common, formula="disease", ref_label=group_2)
        else:
            result = run_ANOVA_transformed(**common, formula="disease", ref_label=group_2)

        extra = result.get("extra", {}) or {}
        table = result.get("contrast_table")
        native = None
        if isinstance(table, pd.DataFrame) and group_1 in table.index:
            native = table.loc[group_1]
            if isinstance(native, pd.DataFrame):
                native = native.iloc[0]
        subset = frame.loc[frame["cell_type"].astype(str).eq(str(cell_type))]
        group_values = subset["disease"].astype(str)
        raw = pd.to_numeric(subset["prop"], errors="coerce")
        raw_difference = raw.loc[group_values.eq(group_1)].mean() - raw.loc[group_values.eq(group_2)].mean()

        estimate = np.nan
        pvalue = np.nan
        adjusted_native = np.nan
        standard_error = np.nan
        statistic = np.nan
        if self.method_id in {"dirichlet_wald", "dirichlet_multinomial_wald", "pydeseq2"} and native is not None:
            estimate = pd.to_numeric(pd.Series([native.get("Coef.", np.nan)]), errors="coerce").iloc[0]
            pvalue = pd.to_numeric(pd.Series([native.get("P>|z|", np.nan)]), errors="coerce").iloc[0]
            adjusted_native = pd.to_numeric(pd.Series([native.get("p_adj", np.nan)]), errors="coerce").iloc[0]
            standard_error = pd.to_numeric(
                pd.Series([native.get("Std.Err", native.get("Std.Err.", np.nan))]), errors="coerce"
            ).iloc[0]
            statistic = pd.to_numeric(pd.Series([native.get("z", np.nan)]), errors="coerce").iloc[0]
        elif self.method_id in {"pclr_ols", "pclr_lmm"} and native is not None:
            estimate = pd.to_numeric(pd.Series([native.get("Coef.", np.nan)]), errors="coerce").iloc[0]
            pvalue = pd.to_numeric(pd.Series([native.get("pval", np.nan)]), errors="coerce").iloc[0]
            standard_error = pd.to_numeric(pd.Series([native.get("Std.Err", np.nan)]), errors="coerce").iloc[0]
            statistic = pd.to_numeric(pd.Series([native.get("z", np.nan)]), errors="coerce").iloc[0]
        elif self.method_id == "permutation_mixed" and native is not None:
            estimate = raw_difference
            pvalue = pd.to_numeric(pd.Series([native.get("pval", np.nan)]), errors="coerce").iloc[0]
            statistic = pd.to_numeric(pd.Series([native.get("H stats", np.nan)]), errors="coerce").iloc[0]
        elif self.method_id == "dkd":
            estimate = raw_difference
            pvalue = pd.to_numeric(pd.Series([result.get("p_val", np.nan)]), errors="coerce").iloc[0]
        elif self.method_id in {"anova_naive", "anova_transformed"}:
            if self.method_id == "anova_naive":
                estimate = raw_difference
            else:
                transformed = np.arcsin(np.sqrt(raw.clip(0, 1)))
                estimate = (
                    transformed.loc[group_values.eq(group_1)].mean()
                    - transformed.loc[group_values.eq(group_2)].mean()
                )
            pvalue = pd.to_numeric(pd.Series([result.get("p_val", np.nan)]), errors="coerce").iloc[0]

        engine_status = str(result.get("contrast_status", "unavailable"))
        error = extra.get("error")
        nonconverged = (
            self.method_id == "dirichlet_wald" and extra.get("success") is False
        ) or (
            self.method_id == "dirichlet_multinomial_wald"
            and not bool(extra.get("convergence_accepted", False))
        )
        success = (
            engine_status == "success" and not error and not nonconverged
            and np.isfinite(estimate) and np.isfinite(pvalue)
        )
        return {
            "cell_type": str(cell_type), "estimate": estimate,
            "standard_error": standard_error, "statistic": statistic,
            "pvalue_raw": pvalue, "pvalue_adjusted_native": adjusted_native,
            "native_status": "success" if success else "unavailable",
            "failure_reason": pd.NA if success else str(
                error or ("optimizer_nonconvergence" if nonconverged else "native_contrast_unavailable")
            ),
            "native_p_type": result.get("p_type", pd.NA),
            "native_global_pvalue": result.get("p_val", np.nan),
            "native_model_type": extra.get("model_type", pd.NA),
            "factor": factor,
        }

    def _run_dirichlet_all(
        self, frame: pd.DataFrame, native_input: NativeInput
    ) -> pd.DataFrame:
        contrast = native_input.contrast
        group_1 = str(contrast["group_1"])
        group_2 = str(contrast["group_2"])
        reference = contrast.get("reference_cell_type")
        cell_types = native_input.cell_type_manifest["cell_type"].astype(str).tolist()
        if reference not in cell_types:
            raise ValueError("Dirichlet common compositional reference is absent from the input.")
        target = next(value for value in cell_types if value != reference)
        formula = "disease + tissue" if frame["tissue"].nunique(dropna=True) > 1 else "disease"
        function = (
            run_Dirichlet_Wald
            if self.method_id == "dirichlet_wald"
            else run_Dirichlet_Multinomial_Wald
        )
        result = function(
            frame, target, formula=formula, ref_label=group_2,
            group_label="sample_id", maxiter=self.dirichlet_maxiter,
            alpha=self.nominal_alpha,
            composition_reference_cell_type=str(reference),
            return_all_celltypes=True,
        )
        extra = result.get("extra", {}) or {}
        records = pd.DataFrame(extra.get("all_celltype_contrasts", []))
        nonconverged = (
            self.method_id == "dirichlet_wald" and extra.get("success") is False
        ) or (
            self.method_id == "dirichlet_multinomial_wald"
            and not bool(extra.get("convergence_accepted", False))
        )
        rows: list[dict[str, Any]] = []
        for cell_type in cell_types:
            native = records.loc[
                records.get("cell_type", pd.Series(dtype=str)).astype(str).eq(cell_type)
                & records.get("other", pd.Series(dtype=str)).astype(str).eq(group_1)
            ] if not records.empty else pd.DataFrame()
            if cell_type == str(reference):
                failure = "common_compositional_reference_excluded"
            elif nonconverged:
                failure = "optimizer_nonconvergence"
            elif extra.get("error"):
                failure = str(extra["error"])
            elif native.empty:
                failure = "native_contrast_unavailable"
            else:
                failure = None
            item = native.iloc[0] if not native.empty else pd.Series(dtype=object)
            rows.append({
                "cell_type": cell_type,
                "estimate": item.get("Coef.", np.nan) if failure is None else np.nan,
                "standard_error": item.get("Std.Err", np.nan),
                "statistic": item.get("z", np.nan),
                "pvalue_raw": item.get("P>|z|", np.nan),
                "pvalue_adjusted_native": np.nan,
                "native_status": "success" if failure is None else "unavailable",
                "failure_reason": pd.NA if failure is None else failure,
                "native_p_type": "cell_type_specific_Wald",
                "native_global_pvalue": result.get("p_val", np.nan),
                "native_model_type": "common_reference_compositional_regression",
                "optimizer_success": extra.get("optimizer_success", extra.get("success", pd.NA)),
                "optimizer_gradient_inf_norm": extra.get("optimizer_gradient_inf_norm", pd.NA),
                "factor": str(contrast["factor"]),
            })
        return pd.DataFrame(rows)

    def execute_native(self, native_input: NativeInput, contrast: pd.Series) -> pd.DataFrame:
        frame = self._engine_frame(native_input)
        cell_types = native_input.cell_type_manifest["cell_type"].astype(str).tolist()
        if self.method_id in {"dirichlet_wald", "dirichlet_multinomial_wald"}:
            output = self._run_dirichlet_all(frame, native_input)
        else:
            args = [
                (frame, cell_type, native_input.contrast, index)
                for index, cell_type in enumerate(cell_types)
            ]
            workers = 1 if self.method_id == "pydeseq2" else min(self.max_workers, len(args))
            if workers == 1:
                rows = [self._run_one(*item) for item in args]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    rows = list(pool.map(lambda item: self._run_one(*item), args))
            output = pd.DataFrame(rows)
        output["adjustment_family"] = f"{self.method_id}:{contrast['contrast_id']}:composition"
        if self.method_id == "pydeseq2" and output["pvalue_adjusted_native"].notna().any():
            output["pvalue_adjusted"] = output["pvalue_adjusted_native"]
        else:
            output = apply_bh_by_family(
                output, pvalue_col="pvalue_raw", family_cols=["adjustment_family"]
            )
        successful = int(output["native_status"].eq("success").sum())
        failed = output.loc[
            output["native_status"].ne("success"), ["cell_type", "failure_reason"]
        ]
        failure_warnings = (
            failed.astype(str).agg(":".join, axis=1).tolist() if not failed.empty else []
        )
        output.attrs["diagnostics"] = {
            "converged": successful > 0,
            "number_celltypes": int(len(output)),
            "number_successful_celltypes": successful,
            "number_unavailable_celltypes": int(len(output) - successful),
            "native_engine": self.method_id,
            "multiplicity": (
                "PyDESeq2 native feature-level BH with independent filtering"
                if self.method_id == "pydeseq2"
                else "BH across fixed cell types within contrast"
            ),
            "n_permutations": self.n_permutations if self.method_id == "permutation_mixed" else None,
            "pclr_samples": self.pclr_samples if self.method_id.startswith("pclr_") else None,
            "warnings": failure_warnings,
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
            {"cell_type", "estimate", "pvalue_raw", "pvalue_adjusted", "native_status"},
            self.method_id,
        )
        spec = _SPECS[self.method_id]
        public_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        for _, native in native_output.iterrows():
            is_dirichlet = self.method_id in {
                "dirichlet_wald", "dirichlet_multinomial_wald"
            }
            reference_cell_type = (
                contrast.get("reference_cell_type", pd.NA) if is_dirichlet else pd.NA
            )
            reference_is_fixed = bool(
                contrast.get("reference_is_fixed", False)
            ) if is_dirichlet else False
            direction_basis = (
                f"signed group_1({contrast['group_1']}) minus group_2({contrast['group_2']}) "
                f"log-alpha contrast; reference_cell_type={reference_cell_type}; null=0"
                if is_dirichlet else
                f"signed group_1({contrast['group_1']}) minus group_2({contrast['group_2']}); null=0"
            )
            row, result_id, evidence_id = public_row(
                method_id=self.method_id, method_version=self.method_version,
                analysis_id=analysis_id, diagnostic_id=diagnostic_id, contrast=contrast,
                cell_type=str(native["cell_type"]), effect_component="composition",
                estimate=native["estimate"], effect_estimand=spec.estimand,
                effect_scale=spec.scale,
                direction_basis=direction_basis,
                decision_rule_id=spec.decision_rule_id,
                result_interpretation=spec.interpretation,
                reference_cell_type=reference_cell_type,
                reference_strategy=(
                    contrast.get("reference_strategy", "common_exclusion")
                    if is_dirichlet else "not_applicable"
                ),
                reference_selection_reason=(
                    contrast.get("reference_selection_reason", pd.NA)
                    if is_dirichlet else pd.NA
                ),
                reference_is_fixed=reference_is_fixed,
            )
            if str(native["native_status"]) != "success" or not np.isfinite(native["pvalue_adjusted"]):
                row.update({
                    "evidence_id": pd.NA, "estimate": np.nan,
                    "effect_direction": "not_applicable", "primary_decision": pd.NA,
                    "decision_metric": pd.NA, "decision_value": pd.NA,
                    "decision_operator": pd.NA, "decision_threshold": pd.NA,
                    "decision_rule_id": pd.NA, "decision_rule_description": pd.NA,
                    "is_benchmark_eligible": False, "estimand_compatibility": "unavailable",
                    "is_available": False, "is_valid": False,
                    "contrast_status": "unavailable",
                    "failure_reason": native.get("failure_reason", "native_contrast_unavailable"),
                })
            else:
                evidence_rows.append(frequentist_evidence(
                    evidence_id=evidence_id, result_id=result_id,
                    pvalue_raw=native["pvalue_raw"],
                    pvalue_adjusted=native["pvalue_adjusted"],
                    native_rule_id=spec.decision_rule_id,
                    adjustment_family=str(native["adjustment_family"]),
                    alpha=self.nominal_alpha, test_name=spec.test_name,
                    statistic=native.get("statistic", pd.NA),
                    statistic_type=spec.statistic_type,
                    standard_error=native.get("standard_error", pd.NA),
                    extra={
                        "native_p_type": native.get("native_p_type", pd.NA),
                        "native_global_pvalue": native.get("native_global_pvalue", pd.NA),
                        "native_model_type": native.get("native_model_type", pd.NA),
                        "optimizer_success": native.get("optimizer_success", pd.NA),
                        "optimizer_gradient_inf_norm": native.get("optimizer_gradient_inf_norm", pd.NA),
                        "monte_carlo_samples": self.pclr_samples if self.method_id.startswith("pclr_") else pd.NA,
                        "permutation_count": self.n_permutations if self.method_id == "permutation_mixed" else pd.NA,
                    },
                ))
            public_rows.append(row)
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)
