from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def apply_bh_by_family(
    frame: pd.DataFrame,
    *,
    pvalue_col: str,
    family_cols: Sequence[str],
    output_col: str = "pvalue_adjusted",
) -> pd.DataFrame:
    """Apply BH only within an explicitly supplied, predeclared family."""
    if not family_cols:
        raise ValueError("`family_cols` must explicitly define the adjustment family.")
    required = {pvalue_col, *family_cols}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    result[output_col] = np.nan
    grouping = family_cols[0] if len(family_cols) == 1 else list(family_cols)
    for _, group in result.groupby(grouping, dropna=False, sort=False):
        numeric = pd.to_numeric(group[pvalue_col], errors="coerce")
        finite = numeric.notna() & np.isfinite(numeric) & numeric.between(0.0, 1.0)
        if finite.any():
            adjusted = multipletests(numeric[finite].to_numpy(), method="fdr_bh")[1]
            result.loc[numeric[finite].index, output_col] = adjusted
    return result

