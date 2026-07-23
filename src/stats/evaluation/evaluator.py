from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd

from src.stats.schemas import validate_contrast_public_view, validate_truth_table
from src.stats.validation import parse_boolean_series


@dataclass(frozen=True)
class EvaluationSpec:
    truth_source: str = "population"
    required_effect_component: str = "composition"
    eligible_estimand_levels: tuple[str, ...] = ("compatible", "direction_only")
    method_universe_policy: str = "common_complete_universe"
    missing_result_policy: str = "exclude_and_report"
    invalid_result_policy: str = "exclude_and_report"
    reference_policy: str = "common_exclusion"
    replicate_group_columns: tuple[str, ...] = ("scenario_id", "replicate_id")
    methods: tuple[str, ...] | None = None

    def validate(self) -> "EvaluationSpec":
        if self.truth_source not in {"injected", "population", "observed"}:
            raise ValueError("EvaluationSpec.truth_source must be explicit and supported.")
        if self.method_universe_policy != "common_complete_universe":
            raise ValueError("Only method_universe_policy='common_complete_universe' is formal.")
        if self.reference_policy != "common_exclusion":
            raise ValueError("Only reference_policy='common_exclusion' is formal.")
        if self.missing_result_policy != "exclude_and_report":
            raise ValueError("Missing results must use exclude_and_report.")
        if self.invalid_result_policy != "exclude_and_report":
            raise ValueError("Invalid results must use exclude_and_report.")
        if not self.eligible_estimand_levels:
            raise ValueError("At least one eligible estimand compatibility level is required.")
        if self.methods is not None and len(set(self.methods)) != len(self.methods):
            raise ValueError("EvaluationSpec.methods contains duplicates.")
        return self


@dataclass(frozen=True)
class EvaluationResult:
    aligned: pd.DataFrame
    method_completion: pd.DataFrame
    replicate_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    evaluation_spec: EvaluationSpec


_RESULT_KEY = ["run_id", "analysis_id", "contrast_id", "cell_type", "effect_component"]
_REPLICATE_CONTEXT = [
    "method", "run_id", "analysis_id", "scenario_id", "replicate_id",
    "simulation_seed", "contrast_id", "effect_component",
]


def _safe_ratio(numerator: int, denominator: int, metric: str) -> tuple[float, str | None]:
    if denominator == 0:
        return np.nan, f"{metric}_denominator_zero"
    return numerator / denominator, None


def _confusion_metrics(truth: pd.Series, decision: pd.Series) -> dict[str, Any]:
    truth_bool = parse_boolean_series(truth)
    decision_bool = parse_boolean_series(decision)
    if truth_bool.isna().any() or decision_bool.isna().any():
        raise ValueError("Included evaluation rows require complete truth and primary decisions.")
    truth_values = truth_bool.astype(bool).to_numpy()
    decision_values = decision_bool.astype(bool).to_numpy()
    tp = int(np.sum(truth_values & decision_values))
    fp = int(np.sum(~truth_values & decision_values))
    tn = int(np.sum(~truth_values & ~decision_values))
    fn = int(np.sum(truth_values & ~decision_values))
    power, power_reason = _safe_ratio(tp, tp + fn, "power")
    fpr, fpr_reason = _safe_ratio(fp, fp + tn, "fpr")
    specificity, specificity_reason = _safe_ratio(tn, tn + fp, "specificity")
    precision, precision_reason = _safe_ratio(tp, tp + fp, "precision")
    fdp, fdp_reason = _safe_ratio(fp, tp + fp, "fdp")
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Power": power,
        "TPR": power,
        "FPR": fpr,
        "Specificity": specificity,
        "Precision": precision,
        "FDP_descriptive": fdp,
        "FDP_for_FDR": 0.0 if tp + fp == 0 else fdp,
        "Power_reason": power_reason,
        "FPR_reason": fpr_reason,
        "Specificity_reason": specificity_reason,
        "Precision_reason": precision_reason,
        "FDP_descriptive_reason": fdp_reason,
        "FDP_for_FDR_reason": (
            "zero_discoveries_defined_as_zero_for_fdr" if tp + fp == 0 else None
        ),
    }


def _cross_truth_with_methods(truth: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    method_frame = pd.DataFrame({"method": list(methods), "_cross": 1})
    return (
        truth.assign(_cross=1)
        .merge(method_frame, on="_cross", how="inner", validate="many_to_many")
        .drop(columns="_cross")
    )


def _classify_alignment(expected: pd.DataFrame, spec: EvaluationSpec) -> pd.DataFrame:
    aligned = expected.copy()
    aligned["exclusion_reason"] = pd.Series(pd.NA, index=aligned.index, dtype="object")
    status = aligned["contrast_status"].astype("string")
    available = parse_boolean_series(aligned["is_available"])
    valid = parse_boolean_series(aligned["is_valid"])
    benchmark = parse_boolean_series(aligned["is_benchmark_eligible"])
    present = aligned["result_id"].notna()
    reference = (
        aligned["cell_type"].astype(str).eq(aligned["truth_reference_cell_type"].astype(str))
        | status.eq("reference")
    )
    aligned.loc[reference, "exclusion_reason"] = "reference_excluded"
    unclassified = aligned["exclusion_reason"].isna()
    aligned.loc[unclassified & ~present, "exclusion_reason"] = "missing_result"
    unclassified = aligned["exclusion_reason"].isna()
    unavailable = present & (~available.fillna(False) | status.isin(
        ["unavailable", "failed", "not_tested"]
    ))
    aligned.loc[unclassified & unavailable, "exclusion_reason"] = "unavailable_result"
    unclassified = aligned["exclusion_reason"].isna()
    invalid = present & available.fillna(False) & (~valid.fillna(False) | status.eq("invalid"))
    aligned.loc[unclassified & invalid, "exclusion_reason"] = "invalid_result"
    unclassified = aligned["exclusion_reason"].isna()
    compatible = aligned["estimand_compatibility"].isin(spec.eligible_estimand_levels)
    aligned.loc[unclassified & ~compatible, "exclusion_reason"] = "estimand_incompatible"
    unclassified = aligned["exclusion_reason"].isna()
    aligned.loc[unclassified & ~benchmark.fillna(False), "exclusion_reason"] = "not_benchmark_eligible"
    unclassified = aligned["exclusion_reason"].isna()
    aligned.loc[unclassified & aligned["primary_decision"].isna(), "exclusion_reason"] = (
        "decision_unavailable"
    )
    aligned["method_eligible"] = aligned["exclusion_reason"].isna()
    return aligned


def _apply_common_complete_universe(aligned: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    result = aligned.copy()
    universe_group = ["run_id", "analysis_id", "contrast_id", "effect_component"]
    result["in_common_complete_universe"] = False
    for _, group in result.groupby(universe_group, dropna=False, sort=False):
        eligible = group.loc[group["method_eligible"]]
        method_cells = {
            method: set(eligible.loc[eligible["method"].eq(method), "cell_type"].astype(str))
            for method in methods
        }
        common = set.intersection(*(method_cells[method] for method in methods)) if methods else set()
        common_mask = group["cell_type"].astype(str).isin(common)
        result.loc[group.index[common_mask], "in_common_complete_universe"] = True
        outside = group["method_eligible"] & ~common_mask
        result.loc[group.index[outside], "exclusion_reason"] = "not_in_common_complete_universe"
    result["included_for_metrics"] = result["method_eligible"] & result["in_common_complete_universe"]
    return result


def _completion_summary(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in aligned.groupby(_REPLICATE_CONTEXT, dropna=False, sort=False):
        row = dict(zip(_REPLICATE_CONTEXT, key if isinstance(key, tuple) else (key,)))
        nonreference = group.loc[~group["exclusion_reason"].eq("reference_excluded")]
        reasons = nonreference["exclusion_reason"]
        expected = len(nonreference)
        missing = int(reasons.eq("missing_result").sum())
        row.update({
            "number_expected": expected,
            "number_tested": int(nonreference["method_eligible"].sum()),
            "number_common_tested": int(nonreference["included_for_metrics"].sum()),
            "number_missing": missing,
            "number_unavailable": int(reasons.eq("unavailable_result").sum()),
            "number_invalid": int(reasons.eq("invalid_result").sum()),
            "number_reference_excluded": int(group["exclusion_reason"].eq("reference_excluded").sum()),
            "number_estimand_incompatible": int(reasons.eq("estimand_incompatible").sum()),
            "completion_rate": (expected - missing) / expected if expected else np.nan,
            "validity_rate": int(nonreference["method_eligible"].sum()) / expected if expected else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _replicate_metrics(aligned: pd.DataFrame, completion: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    completion_index = completion.set_index(_REPLICATE_CONTEXT)
    for key, group in aligned.groupby(_REPLICATE_CONTEXT, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(_REPLICATE_CONTEXT, key_tuple))
        tested = group.loc[group["included_for_metrics"]]
        row["replicate_included"] = bool(len(tested))
        row.update(completion_index.loc[key_tuple].to_dict())
        if tested.empty:
            row.update({
                "number_true_effects": 0,
                "number_null_effects": 0,
                "number_closure_induced_truth": 0,
                "TP": 0, "FP": 0, "TN": 0, "FN": 0,
                "Power": np.nan, "TPR": np.nan, "FPR": np.nan,
                "Specificity": np.nan, "Precision": np.nan,
                "FDP_descriptive": np.nan, "FDP_for_FDR": np.nan,
                "Power_reason": "no_tested_contrasts",
                "FPR_reason": "no_tested_contrasts",
                "Specificity_reason": "no_tested_contrasts",
                "Precision_reason": "no_tested_contrasts",
                "FDP_descriptive_reason": "no_tested_contrasts",
                "FDP_for_FDR_reason": "no_tested_contrasts",
            })
        else:
            truth_values = parse_boolean_series(tested["is_true_effect"])
            row["number_true_effects"] = int(truth_values.eq(True).sum())
            row["number_null_effects"] = int(truth_values.eq(False).sum())
            row["number_closure_induced_truth"] = (
                int(parse_boolean_series(tested["is_closure_induced"]).eq(True).sum())
                if "is_closure_induced" in tested.columns else 0
            )
            row.update(_confusion_metrics(tested["is_true_effect"], tested["primary_decision"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_metrics(replicates: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["method", "run_id", "scenario_id", "contrast_id", "effect_component"]
    rows: list[dict[str, Any]] = []
    for key, group in replicates.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        expected = int(group["replicate_id"].nunique())
        included = int(group["replicate_included"].sum())
        complete = expected == included
        row.update({
            "number_expected_replicates": expected,
            "number_included_replicates": included,
            "number_missing_replicates": expected - included,
            "empirical_FDR": (
                float(group["FDP_for_FDR"].mean()) if complete and expected else np.nan
            ),
            "empirical_FDR_reason": None if complete and expected else "incomplete_replicates",
            "TP": int(group.loc[group["replicate_included"], "TP"].sum()),
            "FP": int(group.loc[group["replicate_included"], "FP"].sum()),
            "TN": int(group.loc[group["replicate_included"], "TN"].sum()),
            "FN": int(group.loc[group["replicate_included"], "FN"].sum()),
            "number_true_effects": int(group["number_true_effects"].sum()),
            "number_null_effects": int(group["number_null_effects"].sum()),
            "number_closure_induced_truth": int(group["number_closure_induced_truth"].sum()),
        })
        row["binary_benchmark_informative"] = bool(
            row["number_true_effects"] > 0 and row["number_null_effects"] > 0
        )
        row["benchmark_limitation_reason"] = (
            None if row["binary_benchmark_informative"] else
            "no_true_effect_truth" if row["number_true_effects"] == 0 else
            "no_null_truth"
        )
        for metric in ("Power", "TPR", "FPR", "Specificity", "Precision"):
            values = pd.to_numeric(
                group.loc[group["replicate_included"], metric], errors="coerce"
            )
            row[f"mean_{metric}"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"number_finite_{metric}"] = int(values.notna().sum())
        for count in (
            "number_expected", "number_tested", "number_common_tested", "number_missing",
            "number_unavailable", "number_invalid", "number_reference_excluded",
            "number_estimand_incompatible",
        ):
            row[count] = int(group[count].sum())
        row["completion_rate"] = (
            (row["number_expected"] - row["number_missing"]) / row["number_expected"]
            if row["number_expected"] else np.nan
        )
        row["validity_rate"] = (
            row["number_tested"] / row["number_expected"] if row["number_expected"] else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_contrasts(
    contrast_view: pd.DataFrame,
    truth_table: pd.DataFrame,
    evaluation_spec: EvaluationSpec | None = None,
) -> EvaluationResult:
    """Evaluate canonical contrast decisions against an explicitly selected truth source."""
    spec = (evaluation_spec or EvaluationSpec()).validate()
    public = validate_contrast_public_view(contrast_view)
    truth = validate_truth_table(truth_table)
    selected_truth = truth.loc[
        truth["truth_source"].eq(spec.truth_source)
        & truth["effect_component"].eq(spec.required_effect_component)
    ].copy()
    if selected_truth.empty:
        raise ValueError(
            f"No truth rows for truth_source={spec.truth_source!r} and "
            f"effect_component={spec.required_effect_component!r}."
        )
    if selected_truth["truth_status"].ne("available").any():
        raise ValueError("Selected truth source contains unavailable rows; evaluation cannot guess replacements.")
    if selected_truth.duplicated(_RESULT_KEY).any():
        raise ValueError("Selected truth contains duplicate canonical result keys.")

    methods = spec.methods or tuple(sorted(public["method"].astype(str).unique()))
    if not methods:
        raise ValueError("No methods are available for evaluation.")
    public = public.loc[
        public["method"].astype(str).isin(methods)
        & public["effect_component"].eq(spec.required_effect_component)
    ].copy()
    result_key = ["method", *_RESULT_KEY]
    if public.duplicated(result_key).any():
        raise ValueError("Duplicate canonical result keys detected within a method.")

    truth_for_merge = selected_truth.rename(columns={
        "effect_value": "truth_effect_value",
        "effect_estimand": "truth_effect_estimand",
        "effect_scale": "truth_effect_scale",
        "reference_cell_type": "truth_reference_cell_type",
    })
    expected = _cross_truth_with_methods(truth_for_merge, tuple(methods))
    public_columns = [
        *result_key, "result_id", "primary_decision", "is_available", "is_valid",
        "is_benchmark_eligible", "estimand_compatibility", "contrast_status",
        "failure_reason", "reference_cell_type",
    ]
    aligned = expected.merge(
        public[public_columns],
        on=result_key,
        how="left",
        validate="one_to_one",
    )
    aligned = _classify_alignment(aligned, spec)
    aligned = _apply_common_complete_universe(aligned, tuple(methods))
    completion = _completion_summary(aligned)
    replicate_metrics = _replicate_metrics(aligned, completion)
    aggregate_metrics = _aggregate_metrics(replicate_metrics)
    return EvaluationResult(aligned, completion, replicate_metrics, aggregate_metrics, spec)


# Temporary compatibility wrappers. These are simulation-calibration utilities,
# not a second contrast-evaluation path.
def get_refined_noise_estimates(*args, **kwargs):
    warnings.warn(
        "get_refined_noise_estimates moved to src.stats.simulation.calibration",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.stats.simulation.calibration import get_refined_noise_estimates as implementation
    return implementation(*args, **kwargs)


def get_all_simulation_params(*args, **kwargs):
    warnings.warn(
        "get_all_simulation_params moved to src.stats.simulation.calibration",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.stats.simulation.calibration import get_all_simulation_params as implementation
    return implementation(*args, **kwargs)
