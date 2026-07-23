from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .validation import parse_boolean_series


def _warn_legacy_evaluation_api(name: str) -> None:
    warnings.warn(
        f"{name} is a legacy metric primitive; the sole formal evaluation entry is "
        "src.stats.evaluation.evaluate_contrasts.",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int
    n_evaluated: int
    n_excluded: int


def _safe_ratio(numerator: int, denominator: int, metric: str) -> tuple[float, str | None]:
    if denominator == 0:
        return np.nan, f"{metric}_denominator_zero"
    return numerator / denominator, None


def confusion_counts(truth: pd.Series, decision: pd.Series) -> ConfusionCounts:
    """Count outcomes after excluding missing truth or unavailable decisions."""
    if len(truth) != len(decision):
        raise ValueError("`truth` and `decision` must have the same length.")

    truth_bool = parse_boolean_series(truth)
    decision_bool = parse_boolean_series(decision)
    valid = truth_bool.notna() & decision_bool.notna()
    truth_values = truth_bool[valid].astype(bool).to_numpy()
    decision_values = decision_bool[valid].astype(bool).to_numpy()

    return ConfusionCounts(
        tp=int(np.sum(truth_values & decision_values)),
        fp=int(np.sum(~truth_values & decision_values)),
        tn=int(np.sum(~truth_values & ~decision_values)),
        fn=int(np.sum(truth_values & ~decision_values)),
        n_evaluated=int(valid.sum()),
        n_excluded=int((~valid).sum()),
    )


def calculate_binary_metrics(truth: pd.Series, decision: pd.Series) -> dict[str, Any]:
    """Return confusion counts and correctly named benchmark metrics."""
    _warn_legacy_evaluation_api("calculate_binary_metrics")
    counts = confusion_counts(truth, decision)
    power, power_reason = _safe_ratio(counts.tp, counts.tp + counts.fn, "power")
    fpr, fpr_reason = _safe_ratio(counts.fp, counts.fp + counts.tn, "fpr")
    fdp, fdp_reason = _safe_ratio(counts.fp, counts.tp + counts.fp, "fdp")
    precision, precision_reason = _safe_ratio(counts.tp, counts.tp + counts.fp, "precision")
    specificity, specificity_reason = _safe_ratio(counts.tn, counts.tn + counts.fp, "specificity")

    result: dict[str, Any] = {
        "TP": counts.tp,
        "FP": counts.fp,
        "TN": counts.tn,
        "FN": counts.fn,
        "N_Evaluated": counts.n_evaluated,
        "N_Excluded": counts.n_excluded,
        "Power": power,
        "TPR": power,
        "FPR": fpr,
        "FDP": fdp,
        "FDP_descriptive": fdp,
        "FDP_for_FDR": (
            np.nan if counts.n_evaluated == 0 else
            0.0 if counts.tp + counts.fp == 0 else fdp
        ),
        "Precision": precision,
        "Specificity": specificity,
        "Power_Reason": power_reason,
        "FPR_Reason": fpr_reason,
        "FDP_Reason": fdp_reason,
        "FDP_for_FDR_Reason": (
            "no_evaluated_contrasts" if counts.n_evaluated == 0 else
            "zero_discoveries_defined_as_zero_for_fdr" if counts.tp + counts.fp == 0 else None
        ),
        "Precision_Reason": precision_reason,
        "Specificity_Reason": specificity_reason,
    }
    return result


def estimate_fdr_from_replicates(
    frame: pd.DataFrame,
    *,
    truth_col: str,
    decision_col: str,
    replicate_col: str,
    group_cols: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Compatibility metric primitive using all replicate-level FDP_for_FDR values."""
    _warn_legacy_evaluation_api("estimate_fdr_from_replicates")
    required = {truth_col, decision_col, replicate_col, *group_cols}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    grouping = [*group_cols, replicate_col]
    replicate_rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(grouping, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(grouping, key_tuple))
        row.update(calculate_binary_metrics(group[truth_col], group[decision_col]))
        replicate_rows.append(row)

    replicate_metrics = pd.DataFrame(replicate_rows)
    output_columns = [
        *group_cols, "FDR_Estimate", "N_FDP_Replicates",
        "N_Expected_Replicates", "N_Missing_Replicates", "FDR_Reason",
    ]
    if replicate_metrics.empty:
        return pd.DataFrame(columns=output_columns)

    if group_cols:
        rows = []
        for key, group in replicate_metrics.groupby(list(group_cols), dropna=False, sort=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = dict(zip(group_cols, key_tuple))
            expected = len(group)
            included = int(group["FDP_for_FDR"].notna().sum())
            row.update({
                "FDR_Estimate": float(group["FDP_for_FDR"].mean()) if included == expected else np.nan,
                "N_FDP_Replicates": included,
                "N_Expected_Replicates": expected,
                "N_Missing_Replicates": expected - included,
                "FDR_Reason": None if included == expected else "incomplete_replicates",
            })
            rows.append(row)
        summary = pd.DataFrame(rows, columns=output_columns)
    else:
        expected = len(replicate_metrics)
        included = int(replicate_metrics["FDP_for_FDR"].notna().sum())
        summary = pd.DataFrame(
            {
                "FDR_Estimate": [
                    float(replicate_metrics["FDP_for_FDR"].mean())
                    if included == expected else np.nan
                ],
                "N_FDP_Replicates": [included],
                "N_Expected_Replicates": [expected],
                "N_Missing_Replicates": [expected - included],
                "FDR_Reason": [None if included == expected else "incomplete_replicates"],
            }
        )
    return summary


def build_common_reference_benchmark_view(
    public_view: pd.DataFrame,
    reference_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Return a head-to-head composition view with one shared cell-type universe."""
    _warn_legacy_evaluation_api("build_common_reference_benchmark_view")
    required_public = {
        "analysis_id", "contrast_id", "method", "cell_type", "effect_component",
        "is_benchmark_eligible", "primary_decision",
    }
    required_manifest = {"analysis_id", "contrast_id", "reference_cell_type", "reference_strategy"}
    if missing := required_public - set(public_view.columns):
        raise ValueError(f"Public view is missing benchmark columns: {sorted(missing)}")
    if missing := required_manifest - set(reference_manifest.columns):
        raise ValueError(f"Reference manifest is missing columns: {sorted(missing)}")

    manifest = reference_manifest.copy()
    if manifest.duplicated(["analysis_id", "contrast_id"]).any():
        raise ValueError("Reference manifest must have one row per analysis_id x contrast_id.")
    if not manifest["reference_strategy"].eq("common_exclusion").all():
        raise ValueError("Phase 1.5 benchmark requires reference_strategy='common_exclusion'.")

    composition = public_view.loc[public_view["effect_component"].eq("composition")].copy()
    composition = composition.merge(
        manifest,
        on=["analysis_id", "contrast_id"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_manifest"),
    )
    selected_frames: list[pd.DataFrame] = []
    for _, group in composition.groupby(["analysis_id", "contrast_id"], sort=False):
        reference = str(group["reference_cell_type_manifest"].iloc[0])
        candidate = group.loc[group["cell_type"].astype(str).ne(reference)].copy()
        methods = sorted(candidate["method"].astype(str).unique())
        if len(methods) < 2:
            continue
        counts = candidate.groupby("cell_type", sort=False)["method"].nunique()
        complete_cells = set(counts[counts.eq(len(methods))].index.astype(str))
        eligible = candidate["is_benchmark_eligible"].astype("boolean").fillna(False)
        eligibility = candidate.assign(_eligible=eligible).groupby("cell_type")["_eligible"].all()
        common_cells = complete_cells & set(eligibility[eligibility].index.astype(str))
        selected = candidate.loc[candidate["cell_type"].astype(str).isin(common_cells)].copy()
        expected = set(common_cells)
        for method, method_rows in selected.groupby("method", sort=False):
            if set(method_rows["cell_type"].astype(str)) != expected:
                raise ValueError(f"Benchmark universe differs for method {method!r}.")
        selected["benchmark_reference_cell_type"] = reference
        selected["benchmark_universe_size"] = len(common_cells)
        selected_frames.append(selected)
    if not selected_frames:
        return composition.iloc[0:0].assign(
            benchmark_reference_cell_type=pd.Series(dtype="object"),
            benchmark_universe_size=pd.Series(dtype="int64"),
        )
    return pd.concat(selected_frames, ignore_index=True)
