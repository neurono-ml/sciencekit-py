"""Preprocessing transformers exposed over `sciencekit_ffi`."""

from __future__ import annotations

from typing import Any

from sciencekit._native import StandardScaler, StandardScalerModel

StandardScalerBuilder = StandardScaler


def standard_scaler(
    dtype: Any = None,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    backend: str = "faer",
) -> StandardScaler:
    """Create a chainable column-wise `StandardScaler` builder."""
    return StandardScaler(dtype=dtype, with_mean=with_mean, with_std=with_std, backend=backend)


__all__ = [
    "StandardScaler",
    "StandardScalerBuilder",
    "StandardScalerModel",
    "standard_scaler",
]
