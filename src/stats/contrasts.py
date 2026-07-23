from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class WaldContrastResult:
    estimate: float
    standard_error: float
    statistic: float
    pvalue: float
    contrast_status: str
    failure_reason: str | None = None


def _unavailable(reason: str) -> WaldContrastResult:
    return WaldContrastResult(
        estimate=np.nan,
        standard_error=np.nan,
        statistic=np.nan,
        pvalue=np.nan,
        contrast_status="unavailable",
        failure_reason=reason,
    )


def wald_linear_contrast(
    params: pd.Series | Sequence[float] | np.ndarray,
    covariance: pd.DataFrame | Sequence[Sequence[float]] | np.ndarray | None,
    contrast: pd.Series | Mapping[str, float] | Sequence[float] | np.ndarray,
) -> WaldContrastResult:
    """Evaluate ``L beta`` using the full coefficient covariance matrix.

    Unsupported or incomplete inputs return an explicit unavailable result;
    the function never substitutes a coefficient-wise minimum p-value.
    """
    if covariance is None:
        return _unavailable("covariance_unavailable")

    if isinstance(params, pd.Series):
        beta = params.astype(float)
        names = beta.index
        if isinstance(contrast, Mapping):
            unknown = set(contrast) - set(names)
            if unknown:
                return _unavailable("term_missing")
            vector = pd.Series(0.0, index=names)
            for term, weight in contrast.items():
                vector.loc[term] = float(weight)
        elif isinstance(contrast, pd.Series):
            unknown = set(contrast.index) - set(names)
            if unknown:
                return _unavailable("term_missing")
            vector = contrast.reindex(names, fill_value=0.0).astype(float)
        else:
            vector = pd.Series(np.asarray(contrast, dtype=float), index=names)

        if isinstance(covariance, pd.DataFrame):
            if not set(names).issubset(covariance.index) or not set(names).issubset(covariance.columns):
                return _unavailable("covariance_unavailable")
            cov = covariance.loc[names, names].to_numpy(dtype=float)
        else:
            cov = np.asarray(covariance, dtype=float)
        beta_values = beta.to_numpy(dtype=float)
        vector_values = vector.to_numpy(dtype=float)
    else:
        beta_values = np.asarray(params, dtype=float)
        vector_values = np.asarray(contrast, dtype=float)
        cov = np.asarray(covariance, dtype=float)

    if beta_values.ndim != 1 or vector_values.ndim != 1 or beta_values.shape != vector_values.shape:
        return _unavailable("contrast_dimension_mismatch")
    if cov.shape != (beta_values.size, beta_values.size):
        return _unavailable("covariance_dimension_mismatch")
    if not (np.all(np.isfinite(beta_values)) and np.all(np.isfinite(vector_values)) and np.all(np.isfinite(cov))):
        return _unavailable("non_finite_contrast_input")

    estimate = float(vector_values @ beta_values)
    variance = float(vector_values @ cov @ vector_values)
    if variance < -1e-12:
        return _unavailable("invalid_covariance")
    variance = max(variance, 0.0)
    standard_error = float(np.sqrt(variance))
    if standard_error == 0.0:
        return _unavailable("zero_contrast_standard_error")

    statistic = estimate / standard_error
    pvalue = float(2.0 * norm.sf(abs(statistic)))
    return WaldContrastResult(
        estimate=estimate,
        standard_error=standard_error,
        statistic=statistic,
        pvalue=pvalue,
        contrast_status="success",
    )

