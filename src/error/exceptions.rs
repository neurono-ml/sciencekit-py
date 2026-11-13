//! Conversion from `SKPyError` into structured Python exceptions.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use sciencekit_ffi::SKFfiStatus;

use super::definition::{SKPyError, SKPyErrorKind};

/// Convert an [`SKPyError`] into the matching structured Python exception in
/// `sciencekit._exceptions`, carrying `status_code`, `status_name` and `message`.
pub(crate) fn sk_py_to_py_err(py: Python<'_>, error: SKPyError) -> PyErr {
    let class_name = match error.kind() {
        SKPyErrorKind::Key => "UnknownColumnError",
        SKPyErrorKind::Type => "UnsupportedColumnTypeError",
        SKPyErrorKind::Status => match error.status() {
            SKFfiStatus::InvalidHandle => "InvalidHandleError",
            SKFfiStatus::Shape => "ShapeError",
            SKFfiStatus::InvalidArgument => "InvalidArgumentError",
            SKFfiStatus::Unsupported => "UnsupportedError",
            SKFfiStatus::NotFitted => "NotFittedError",
            SKFfiStatus::Io => "IoError",
            SKFfiStatus::Internal | SKFfiStatus::Ok => "InternalError",
        },
    };

    let instantiate = (|| -> PyResult<Py<PyAny>> {
        let exceptions_module = py.import("sciencekit._exceptions")?;
        let exception_class = exceptions_module.getattr(class_name)?;
        let instance =
            exception_class.call1((error.status_code(), error.status_name(), error.message()))?;
        Ok(instance.unbind())
    })();

    match instantiate {
        Ok(instance) => PyErr::from_value(instance.into_bound(py)),
        Err(_) => PyRuntimeError::new_err(error.to_string()),
    }
}
