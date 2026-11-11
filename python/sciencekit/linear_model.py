"""Linear model estimators exposed over `sciencekit_ffi`."""

from __future__ import annotations

from typing import Any

from sciencekit._native import LinearRegression, LinearRegressionModel

LinearRegressionBuilder = LinearRegression


def linear_regression(
    dtype: Any = None,
    *,
    include_intercept: bool = True,
    backend: str = "faer",
) -> LinearRegression:
    """Create a chainable ordinary least-squares `LinearRegression` builder."""
    return LinearRegression(dtype=dtype, include_intercept=include_intercept, backend=backend)


__all__ = [
    "LinearRegression",
    "LinearRegressionBuilder",
    "LinearRegressionModel",
    "linear_regression",
]
