from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd


_TRUE_STRINGS = frozenset({"true", "1", "yes"})
_FALSE_STRINGS = frozenset({"false", "0", "no"})


def parse_boolean_value(
    value: Any,
    *,
    errors: Literal["raise", "coerce"] = "raise",
) -> Any:
    """Parse one value without relying on Python truthiness.

    Empty values become ``pd.NA``. Only booleans, numeric 0/1 and the
    case-insensitive strings true/false, yes/no and 0/1 are accepted.
    """
    if errors not in {"raise", "coerce"}:
        raise ValueError("`errors` must be either 'raise' or 'coerce'.")

    if value is None or value is pd.NA:
        return pd.NA
    try:
        if bool(pd.isna(value)):
            return pd.NA
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(value)
    elif isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value) in (0.0, 1.0):
            return bool(value)
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "":
            return pd.NA
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False

    if errors == "coerce":
        return pd.NA
    raise ValueError(f"Unknown boolean value: {value!r}")


def parse_boolean_series(
    values: Iterable[Any] | pd.Series,
    *,
    errors: Literal["raise", "coerce"] = "raise",
    name: str | None = None,
) -> pd.Series:
    """Strictly parse a sequence into pandas' nullable boolean dtype."""
    if isinstance(values, pd.Series):
        index = values.index
        output_name = values.name if name is None else name
        raw_values = values.tolist()
    else:
        index = None
        output_name = name
        raw_values = list(values)

    parsed = [parse_boolean_value(value, errors=errors) for value in raw_values]
    return pd.Series(parsed, index=index, name=output_name, dtype="boolean")
