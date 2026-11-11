//! Input validation helpers bridging Python arguments to `SKPyBorrowedArray`.

use std::path::PathBuf;

use pyo3::prelude::*;
use sciencekit_ffi::SKFfiDataType;

use crate::array::csr::{SKPyCsrDescriptor, sk_py_is_csr_like};
use crate::array::dense::SKPyBorrowedArray;
use crate::error::SKPyError;

/// Validate a 2D feature matrix argument, rejecting CSR inputs explicitly after
/// structural validation so sparse inputs are never silently densified.
pub(crate) fn sk_py_prepare_matrix<'py>(
    py: Python<'py>,
    features: &Bound<'py, PyAny>,
    expected_dtype: SKFfiDataType,
    copy: bool,
) -> Result<SKPyBorrowedArray<'py>, SKPyError> {
    if sk_py_is_csr_like(features) {
        if !features.is_instance_of::<SKPyCsrDescriptor>() {
            SKPyCsrDescriptor::validate_csr_object(py, features)?;
        }
        return Err(SKPyError::unsupported(
            "CSR sparse descriptors are not accepted by sciencekit_ffi ABI 1.0 entry points; call csr_descriptor.to_dense() explicitly when dense materialization is intended",
        ));
    }
    SKPyBorrowedArray::from_any(py, features, Some(2), Some(expected_dtype), copy)
}

/// Validate a 1D target vector argument matching `expected_dtype`.
pub(crate) fn sk_py_prepare_vector<'py>(
    py: Python<'py>,
    targets: &Bound<'py, PyAny>,
    expected_dtype: SKFfiDataType,
    copy: bool,
) -> Result<SKPyBorrowedArray<'py>, SKPyError> {
    SKPyBorrowedArray::from_any(py, targets, Some(1), Some(expected_dtype), copy)
}

/// Extract a filesystem path (`str` or `os.PathLike`) as UTF-8.
pub(crate) fn sk_py_extract_path(path: &Bound<'_, PyAny>) -> Result<String, SKPyError> {
    let extracted: PathBuf = path.extract().map_err(|_| {
        SKPyError::invalid_argument("expected path to be a str or os.PathLike[str]")
    })?;
    extracted
        .into_os_string()
        .into_string()
        .map_err(|_| SKPyError::invalid_argument("file path must be valid UTF-8"))
}
