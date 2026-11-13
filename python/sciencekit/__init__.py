"""Python bindings for the `sciencekit` machine-learning library."""

from __future__ import annotations

from sciencekit import datasets, linear_model, preprocessing
from sciencekit._exceptions import (
    InternalError,
    InvalidArgumentError,
    InvalidHandleError,
    IoError,
    NotFittedError,
    ScienceKitError,
    ShapeError,
    UnknownColumnError,
    UnsupportedColumnTypeError,
    UnsupportedError,
)
from sciencekit._native import (
    FEATURE_F32,
    FEATURE_F64,
    FEATURE_LINEAR_REGRESSION,
    FEATURE_MODEL_PERSISTENCE,
    FEATURE_STANDARD_SCALER,
    STATUS_INTERNAL,
    STATUS_INVALID_ARGUMENT,
    STATUS_INVALID_HANDLE,
    STATUS_IO,
    STATUS_NOT_FITTED,
    STATUS_OK,
    STATUS_SHAPE,
    STATUS_UNSUPPORTED,
    CsrDescriptor,
    DenseDescriptor,
    __version__,
    abi_version,
    features,
    live_handles,
    load,
    status_name,
)
from sciencekit.datasets import ConversionReport, describe, supported_representations
from sciencekit.linear_model import (
    LinearRegression,
    LinearRegressionBuilder,
    LinearRegressionModel,
    linear_regression,
)
from sciencekit.preprocessing import (
    StandardScaler,
    StandardScalerBuilder,
    StandardScalerModel,
    standard_scaler,
)

BACKEND_FAER = "faer"
BACKEND_MATRIX_MULTIPLY = "matrixmultiply"
BACKEND_NDARRAY_LINALG = "ndarray-linalg"

__all__ = [
    "BACKEND_FAER",
    "BACKEND_MATRIX_MULTIPLY",
    "BACKEND_NDARRAY_LINALG",
    "ConversionReport",
    "CsrDescriptor",
    "DenseDescriptor",
    "FEATURE_F32",
    "FEATURE_F64",
    "FEATURE_LINEAR_REGRESSION",
    "FEATURE_MODEL_PERSISTENCE",
    "FEATURE_STANDARD_SCALER",
    "InternalError",
    "InvalidArgumentError",
    "InvalidHandleError",
    "IoError",
    "LinearRegression",
    "LinearRegressionBuilder",
    "LinearRegressionModel",
    "NotFittedError",
    "STATUS_INTERNAL",
    "STATUS_INVALID_ARGUMENT",
    "STATUS_INVALID_HANDLE",
    "STATUS_IO",
    "STATUS_NOT_FITTED",
    "STATUS_OK",
    "STATUS_SHAPE",
    "STATUS_UNSUPPORTED",
    "ScienceKitError",
    "ShapeError",
    "StandardScaler",
    "StandardScalerBuilder",
    "StandardScalerModel",
    "UnknownColumnError",
    "UnsupportedColumnTypeError",
    "UnsupportedError",
    "__version__",
    "abi_version",
    "datasets",
    "describe",
    "features",
    "linear_model",
    "linear_regression",
    "live_handles",
    "load",
    "preprocessing",
    "standard_scaler",
    "status_name",
    "supported_representations",
]
