from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from src.stats.adapters._shared import (
    NativeInput,
    frequentist_evidence,
    prepare_pairwise_input,
    public_row,
    require_columns,
)
from src.stats.adapters.base import BaseDifferentialAbundanceAdapter
from src.stats.multiple_testing import apply_bh_by_family
from src.stats.schemas import CanonicalDAInput


class NaiveWelchProportionAdapter(BaseDifferentialAbundanceAdapter):
    """Intentional sanity check that ignores composition and donor structure."""

    method_id = "naive_welch_proportion"
    primary_decision_rule_id = "naive-welch-proportion-bh-0.05-v1"

    def __init__(self, *, method_version: str = "scipy") -> None:
        super().__init__(method_version=method_version)

    def prepare_native_input(
        self, canonical_input: CanonicalDAInput, contrast: pd.Series
    ) -> NativeInput:
        return prepare_pairwise_input(canonical_input, contrast, options={
            "nominal_alpha": 0.05,
            "intentional_limitation": (
                "Welch test on raw proportions; ignores simplex dependence, donor pairing, "
                "batch structure, and count uncertainty."
            ),
        })

    def execute_native(self, native_input: NativeInput, contrast: pd.Series) -> pd.DataFrame:
        factor = str(native_input.contrast["factor"])
        group_1 = str(native_input.contrast["group_1"])
        group_2 = str(native_input.contrast["group_2"])
        merged = native_input.abundance.merge(
            native_input.sample_manifest[["sample_id", factor]],
            on="sample_id",
            how="inner",
            validate="many_to_one",
        )
        rows: list[dict[str, Any]] = []
        for cell_type, group in merged.groupby("cell_type", sort=False):
            first = pd.to_numeric(
                group.loc[group[factor].astype(str).eq(group_1), "proportion"],
                errors="coerce",
            ).dropna()
            second = pd.to_numeric(
                group.loc[group[factor].astype(str).eq(group_2), "proportion"],
                errors="coerce",
            ).dropna()
            statistic, pvalue = ttest_ind(first, second, equal_var=False, nan_policy="omit")
            rows.append({
                "cell_type": str(cell_type),
                "estimate": float(first.mean() - second.mean()),
                "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "pvalue_raw": float(pvalue) if np.isfinite(pvalue) else np.nan,
                "adjustment_family": str(contrast["contrast_id"]),
            })
        result = apply_bh_by_family(
            pd.DataFrame(rows),
            pvalue_col="pvalue_raw",
            family_cols=["adjustment_family"],
        )
        result.attrs["diagnostics"] = {
            "converged": True,
            "warnings": [
                "Intentional sanity check: raw-proportion Welch tests ignore compositional "
                "and repeated-measures structure."
            ],
            "scientific_status": "intentionally_misspecified",
        }
        return result

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
                effect_estimand="sample_mean_raw_proportion_difference",
                effect_scale="proportion_difference",
                direction_basis="signed_raw_proportion_difference:null=0",
                decision_rule_id=self.primary_decision_rule_id,
                result_interpretation=(
                    "Intentional misspecification for benchmark sanity checking; not an "
                    "absolute abundance effect and not a formal candidate method."
                ),
            )
            public_rows.append(row)
            evidence_rows.append(frequentist_evidence(
                evidence_id=evidence_id,
                result_id=result_id,
                pvalue_raw=native["pvalue_raw"],
                pvalue_adjusted=native["pvalue_adjusted"],
                native_rule_id="naive-welch-proportion-native-bh-v1",
                adjustment_family=family,
                alpha=0.05,
                test_name="welch_t_raw_proportion_intentionally_misspecified",
                statistic=native.get("statistic", pd.NA),
                statistic_type="welch_t",
                extra={
                    "scientific_status": "intentionally_misspecified",
                    "ignored_structure": "composition;donor;batch;count_uncertainty",
                },
            ))
        return pd.DataFrame(public_rows), pd.DataFrame(evidence_rows)
