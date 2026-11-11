//! Estimator family and math backend code resolution.

use sciencekit_ffi::SKFfiEstimatorFamily;

use crate::error::SKPyError;

/// Supported estimator families carried by `sciencekit_ffi` ABI 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SKPyEstimatorFamily {
    /// Ordinary least squares (`SK_FFI_ESTIMATOR_LINEAR_REGRESSION = 0`).
    LinearRegression,
    /// Column standardization (`SK_FFI_ESTIMATOR_STANDARD_SCALER = 1`).
    StandardScaler,
}

impl SKPyEstimatorFamily {
    /// Stable integer code accepted by `sk_ffi_builder_create`.
    pub(crate) fn code(self) -> i32 {
        match self {
            SKPyEstimatorFamily::LinearRegression => SKFfiEstimatorFamily::LinearRegression as i32,
            SKPyEstimatorFamily::StandardScaler => SKFfiEstimatorFamily::StandardScaler as i32,
        }
    }

    /// Stable snake_case Python name for the family.
    pub(crate) fn name(self) -> &'static str {
        match self {
            SKPyEstimatorFamily::LinearRegression => "linear_regression",
            SKPyEstimatorFamily::StandardScaler => "standard_scaler",
        }
    }
}

/// Resolve a backend string (`"faer"`, `"matrixmultiply"`, `"ndarray-linalg"`)
/// into its `SKFfiBackendKind` discriminant.
pub(crate) fn sk_py_parse_backend_code(backend: &str) -> Result<i32, SKPyError> {
    match backend {
        "faer" => Ok(0),
        "matrixmultiply" | "matrix_multiply" => Ok(1),
        "ndarray-linalg" | "ndarray_linalg" => Ok(2),
        other => Err(SKPyError::invalid_argument(format!(
            "unknown backend '{other}'; expected 'faer', 'matrixmultiply', or 'ndarray-linalg'"
        ))),
    }
}
