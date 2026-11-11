from __future__ import annotations

import os
from types import TracebackType
from typing import Any
import numpy as np
import numpy.typing as npt

class LinearRegressionModel:
    @property
    def family(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def closed(self) -> bool: ...
    @staticmethod
    def load(path: str | os.PathLike[str]) -> LinearRegressionModel: ...
    def predict(
        self,
        features: npt.NDArray[np.floating[Any]],
        *,
        copy: bool = False,
    ) -> npt.NDArray[np.floating[Any]]: ...
    def transform(
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
    def __enter__(self) -> LinearRegressionModel: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool: ...

class LinearRegression:
    def __init__(
        self,
        dtype: str | npt.DTypeLike | None = None,
        *,
        include_intercept: bool = True,
        backend: str = "faer",
    ) -> None: ...
    @property
    def family(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def closed(self) -> bool: ...
    def include_intercept(self, enabled: bool) -> LinearRegression: ...
    def backend(self, backend: str) -> LinearRegression: ...
    def fit(
        self,
        features: npt.NDArray[np.floating[Any]] | Any,
        targets: npt.NDArray[np.floating[Any]],
        *,
        copy: bool = False,
    ) -> LinearRegressionModel: ...
    def close(self) -> None: ...
    def __enter__(self) -> LinearRegression: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool: ...

LinearRegressionBuilder = LinearRegression

def linear_regression(
    dtype: str | npt.DTypeLike | None = None,
    *,
    include_intercept: bool = True,
    backend: str = "faer",
) -> LinearRegression: ...

__all__: list[str]
