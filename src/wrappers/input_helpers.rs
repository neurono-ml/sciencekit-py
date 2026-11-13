//! Input validation helpers bridging Python arguments to `SKPyBorrowedArray`.

use std::path::PathBuf;

use pyo3::prelude::*;

use crate::error::SKPyError;

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
