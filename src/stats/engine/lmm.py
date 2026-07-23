from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy.stats import chi2

from src.stats.support import make_result, split_C_terms
from src.utils.hier_logger import logged

logger = logging.getLogger(__name__)


def _formula_pair(formula: str, main_variable: str, ref_label: str) -> tuple[str, str]:
    rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()
    if not rhs:
        raise ValueError("`formula` must contain at least one fixed-effect term.")

    terms = [term.strip() for term in rhs.split("+") if term.strip()]
    main_pattern = re.compile(rf"(^|\W){re.escape(main_variable)}($|\W)")
    full_terms: list[str] = []
    reduced_terms: list[str] = []
    main_replaced = False
    for term in terms:
        is_main_term = bool(main_pattern.search(term))
        is_plain_main = term == main_variable or bool(
            re.match(rf"^C\(\s*{re.escape(main_variable)}(?:\s*,.*)?\)$", term)
        )
        if is_plain_main:
            if not main_replaced:
                full_terms.append(
                    f'C({main_variable}, Treatment(reference="{ref_label}"))'
                )
                main_replaced = True
            continue
        full_terms.append(term)
        if not is_main_term:
            reduced_terms.append(term)

    if not main_replaced:
        raise ValueError(f"Main variable {main_variable!r} is not present as a model term.")
    full_formula = "prop ~ " + " + ".join(full_terms)
    reduced_formula = "prop ~ " + (" + ".join(reduced_terms) if reduced_terms else "1")
    return full_formula, reduced_formula


def _failure(
    cell_type: str,
    alpha: float,
    reason: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    details = {} if extra is None else dict(extra)
    details.update({"error": message, "failure_reason": reason})
    return make_result(
        method="LMM",
        cell_type=cell_type,
        p_val=np.nan,
        p_type="unavailable",
        contrast_table=None,
        extra=details,
        alpha=alpha,
        contrast_status="unavailable",
        failure_reason=reason,
    )


def _fit_mixedlm(formula: str, frame: pd.DataFrame, group_label: str):
    model = smf.mixedlm(formula, frame, groups=frame[group_label])
    last_error: Exception | None = None
    captured_messages: list[str] = []
    for method in ("lbfgs", "nm"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(method=method, maxiter=300, reml=False, disp=False)
            captured_messages.extend(str(item.message) for item in caught)
            if getattr(result, "converged", True):
                return result, captured_messages
        except Exception as exc:  # pragma: no cover - optimizer-specific fallback
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("MixedLM did not converge with the configured optimizers.")


@logged
def run_LMM(
    df_all: pd.DataFrame,
    cell_type: str,
    formula: str = "disease",
    main_variable: str | None = None,
    ref_label: str = "HC",
    group_label: str = "sample_id",
    use_reml: bool = False,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Fit an LMM and test the main variable with a full-vs-reduced ML LRT.

    ``p_val`` is the global likelihood-ratio p-value for all terms involving
    ``main_variable``. Coefficient p-values remain in ``contrast_table`` and
    are never collapsed with ``min(pvalues)``. ``use_reml`` remains in the
    signature for compatibility but the global comparison always uses ML.
    """
    required_cols = {"cell_type", "prop", group_label}
    missing_cols = required_cols - set(df_all.columns)
    if missing_cols:
        return _failure(
            cell_type,
            alpha,
            "missing_required_columns",
            f"Missing required columns: {sorted(missing_cols)}",
        )

    frame = df_all[df_all["cell_type"] == cell_type].copy()
    if frame.empty:
        return _failure(
            cell_type,
            alpha,
            "no_rows_for_cell_type",
            f"No rows for cell_type: {cell_type!r}",
        )

    rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()
    if main_variable is None:
        simple_terms = [term.strip() for term in rhs.split("+") if term.strip()]
        if len(simple_terms) != 1:
            raise KeyError(
                "Main explanatory variable must be specified when `formula` contains more than one variable."
            )
        main_variable = re.sub(r"^C\(([^,\)]+).*$", r"\1", simple_terms[0]).strip()

    if main_variable not in frame.columns:
        return _failure(
            cell_type,
            alpha,
            "missing_main_variable",
            f"Missing main variable column: {main_variable!r}",
        )
    observed_levels = frame[main_variable].dropna().astype(str).unique()
    if len(observed_levels) < 2:
        return _failure(
            cell_type,
            alpha,
            "insufficient_groups",
            f"Main variable {main_variable!r} has fewer than two observed levels.",
        )
    if str(ref_label) not in set(observed_levels):
        return _failure(
            cell_type,
            alpha,
            "reference_group_missing",
            f"Reference level {ref_label!r} is not present in {main_variable!r}.",
        )

    try:
        full_formula, reduced_formula = _formula_pair(formula, main_variable, ref_label)
        _, full_design = dmatrices(full_formula, frame, return_type="dataframe", NA_action="drop")
        complete_index = full_design.index.intersection(frame.index[frame[group_label].notna()])
        model_frame = frame.loc[complete_index].copy()
        if model_frame.empty:
            return _failure(
                cell_type,
                alpha,
                "no_complete_cases",
                "No complete cases remain for the full model and grouping variable.",
            )

        full_result, full_warnings = _fit_mixedlm(full_formula, model_frame, group_label)
        reduced_result, reduced_warnings = _fit_mixedlm(reduced_formula, model_frame, group_label)
        df_diff = int(len(full_result.fe_params) - len(reduced_result.fe_params))
        if df_diff <= 0:
            return _failure(
                cell_type,
                alpha,
                "invalid_lrt_degrees_of_freedom",
                "The full model did not add fixed-effect parameters.",
            )

        lrt_statistic = max(0.0, float(2.0 * (full_result.llf - reduced_result.llf)))
        global_pvalue = float(chi2.sf(lrt_statistic, df_diff))
        if not np.isfinite(global_pvalue):
            return _failure(
                cell_type,
                alpha,
                "non_finite_global_test",
                "The likelihood-ratio test returned a non-finite p-value.",
            )

        main_prefix = f"C({main_variable}, Treatment(reference=\"{ref_label}\"))"
        coefficient_names = [
            name for name in full_result.fe_params.index if name.startswith(main_prefix)
        ]
        if not coefficient_names:
            return _failure(
                cell_type,
                alpha,
                "main_coefficients_unavailable",
                "No fixed-effect coefficients were generated for the main variable.",
            )

        intervals = full_result.conf_int().loc[coefficient_names]
        rows = []
        for name in coefficient_names:
            parsed = split_C_terms(pd.Series([name])).iloc[0]
            estimate = float(full_result.fe_params[name])
            coefficient_pvalue = float(full_result.pvalues[name])
            rows.append(
                {
                    "ref": parsed["baseline"] or ref_label,
                    "other": parsed["category"],
                    "Coef.": estimate,
                    "Std.Err.": float(full_result.bse_fe[name]),
                    "z": float(full_result.tvalues[name]),
                    "P>|z|": coefficient_pvalue,
                    "[0.025": float(intervals.loc[name, 0]),
                    "0.975]": float(intervals.loc[name, 1]),
                    "significant": bool(coefficient_pvalue < alpha),
                    "direction": "other_greater" if estimate > 0 else (
                        "ref_greater" if estimate < 0 else "None"
                    ),
                    "contrast_status": "success",
                    "failure_reason": None,
                }
            )
        contrast_table = pd.DataFrame(rows).set_index("other")

        extra: dict[str, Any] = {
            "mixedlm_summary": full_result.summary().tables[0],
            "global_test": "full_vs_reduced_likelihood_ratio",
            "global_pvalue": global_pvalue,
            "global_statistic": lrt_statistic,
            "global_df": df_diff,
            "full_formula": full_formula,
            "reduced_formula": reduced_formula,
            "n_complete_cases": int(len(model_frame)),
            "fit_reml": False,
            "requested_use_reml": bool(use_reml),
            "warnings": [*full_warnings, *reduced_warnings],
            "singular_fit": any("singular" in warning.lower() for warning in [*full_warnings, *reduced_warnings]),
        }
        return make_result(
            method="LMM",
            cell_type=cell_type,
            p_val=global_pvalue,
            p_type="Global",
            contrast_table=contrast_table,
            extra=extra,
            alpha=alpha,
            contrast_status="success",
        )
    except Exception as exc:
        logger.exception("LMM global test failed for cell type %s", cell_type)
        return _failure(
            cell_type,
            alpha,
            "model_fit_failed",
            str(exc),
            {"error_type": type(exc).__name__},
        )
