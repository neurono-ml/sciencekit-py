from __future__ import annotations

import os
from types import TracebackType
from typing import Any
import numpy as np
import numpy.typing as npt

class StandardScalerModel:
    @property
    def family(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def closed(self) -> bool: ...
    def transform(
        self,
        features: npt.NDArray[np.floating[Any]],
        *,
        copy: bool = False,
    ) -> npt.NDArray[np.floating[Any]]: ...
    def predict(
        self,
        features: npt.NDArray[np.floating[Any]],
        *,
        copy: bool = False,
    ) -> npt.NDArray[np.floating[Any]]: ...
    def score(
        self,
        features: npt.NDArray[np.floating[Any]],
        targets: npt.NDArray[np.floating[Any]],
        *,
        copy: bool = False,
    ) -> float: ...
    def save(self, path: str | os.PathLike[str]) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> StandardScalerModel: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool: ...

class StandardScaler:
    def __init__(
        self,
        dtype: str | npt.DTypeLike | None = None,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        backend: str = "faer",
    ) -> None: ...
    @property
    def family(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def closed(self) -> bool: ...
    def with_mean(self, enabled: bool) -> StandardScaler: ...
    def with_std(self, enabled: bool) -> StandardScaler: ...
    def backend(self, backend: str) -> StandardScaler: ...
    def fit(
        self,
        features: npt.NDArray[np.floating[Any]] | Any,
        targets: npt.NDArray[np.floating[Any]] | None = None,
        *,
        copy: bool = False,
    ) -> StandardScalerModel: ...
    def fit_transform(
        self,
        features: npt.NDArray[np.floating[Any]],
        *,
        copy: bool = False,
    ) -> npt.NDArray[np.floating[Any]]: ...
    def close(self) -> None: ...
    def __enter__(self) -> StandardScaler: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool: ...

StandardScalerBuilder = StandardScaler

def standard_scaler(
    dtype: str | npt.DTypeLike | None = None,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    backend: str = "faer",
) -> StandardScaler: ...

__all__: list[str]
