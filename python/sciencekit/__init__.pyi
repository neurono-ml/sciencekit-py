from __future__ import annotations

import os
from typing import Any
import numpy as np
import numpy.typing as npt

from sciencekit import datasets as datasets, linear_model as linear_model, preprocessing as preprocessing
from sciencekit.linear_model import (
    LinearRegression as LinearRegression,
    LinearRegressionBuilder as LinearRegressionBuilder,
    LinearRegressionModel as LinearRegressionModel,
    linear_regression as linear_regression,
)
from sciencekit.datasets import (
    ConversionReport as ConversionReport,
    describe as describe,
    supported_representations as supported_representations,
)
from sciencekit.preprocessing import (
    StandardScaler as StandardScaler,
    StandardScalerBuilder as StandardScalerBuilder,
    StandardScalerModel as StandardScalerModel,
    standard_scaler as standard_scaler,
)

__version__: str

STATUS_OK: int
STATUS_INVALID_HANDLE: int
STATUS_SHAPE: int
STATUS_INVALID_ARGUMENT: int
STATUS_UNSUPPORTED: int
STATUS_NOT_FITTED: int
STATUS_IO: int
STATUS_INTERNAL: int

FEATURE_LINEAR_REGRESSION: int
FEATURE_STANDARD_SCALER: int
FEATURE_MODEL_PERSISTENCE: int
FEATURE_F32: int
FEATURE_F64: int

BACKEND_FAER: str
BACKEND_MATRIX_MULTIPLY: str
BACKEND_NDARRAY_LINALG: str

class ScienceKitError(Exception):
    status_code: int
    status_name: str
    message: str
    def __init__(self, status_code: int, status_name: str, message: str) -> None: ...

class InvalidHandleError(ScienceKitError, RuntimeError): ...
class ShapeError(ScienceKitError, ValueError): ...
class InvalidArgumentError(ScienceKitError, ValueError): ...
class UnknownColumnError(InvalidArgumentError, KeyError): ...
class UnsupportedColumnTypeError(InvalidArgumentError, TypeError): ...
class UnsupportedError(ScienceKitError, NotImplementedError): ...
class NotFittedError(ScienceKitError, RuntimeError): ...
class IoError(ScienceKitError, OSError): ...
class InternalError(ScienceKitError, RuntimeError): ...

class DenseDescriptor:
    def __init__(self, array: npt.NDArray[Any], *, copy: bool = False) -> None: ...
    @property
    def dtype(self) -> str: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def strides(self) -> tuple[int, ...]: ...
    @property
    def length(self) -> int: ...
    @property
    def data_pointer(self) -> int: ...
    @property
    def copied(self) -> bool: ...

class CsrDescriptor:
    def __init__(
        self,
        data: npt.NDArray[np.floating[Any]],
        indices: npt.NDArray[np.integer[Any]],
        indptr: npt.NDArray[np.integer[Any]],
        shape: tuple[int, int],
    ) -> None: ...
    @staticmethod
    def from_csr(matrix: Any) -> CsrDescriptor: ...
    @property
    def dtype(self) -> str: ...
    @property
    def index_dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...
    @property
    def data_pointer(self) -> int: ...
    def to_dense(self) -> npt.NDArray[np.floating[Any]]: ...

def abi_version() -> dict[str, int]: ...
def features() -> int: ...
def status_name(status_code: int) -> str: ...
def live_handles() -> int: ...
def load(path: str | os.PathLike[str]) -> LinearRegressionModel: ...

__all__: list[str]
