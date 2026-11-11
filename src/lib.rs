//! `sciencekit_python` — Python bindings for `sciencekit` over `sciencekit_ffi`.
//!
//! The native extension (`sciencekit._native`) links `sciencekit_ffi` in
//! process as an `rlib` and delegates every builder, fitting, inference and
//! persistence call to the stable `sk_ffi_*` entry points.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod array;
mod engine;
mod error;
mod lifecycle;
mod wrappers;

use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array::{SKPyCsrDescriptor, SKPyDenseDescriptor};
use crate::wrappers::{
    SKPyLinearRegression, SKPyLinearRegressionModel, SKPyStandardScaler, SKPyStandardScalerModel,
    load,
};

/// Return the native C ABI semantic version as `{"major", "minor", "patch"}`.
#[pyfunction]
fn abi_version(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let mut major = 0_i32;
    let mut minor = 0_i32;
    let mut patch = 0_i32;
    let status = sciencekit_ffi::sk_ffi_version(&mut major, &mut minor, &mut patch);
    if status != sciencekit_ffi::SKFfiStatus::Ok {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "failed to query sciencekit_ffi version",
        ));
    }
    let dictionary = PyDict::new(py);
    dictionary.set_item("major", major)?;
    dictionary.set_item("minor", minor)?;
    dictionary.set_item("patch", patch)?;
    Ok(dictionary)
}

/// Return the bitmask of capabilities compiled into `sciencekit_ffi`.
#[pyfunction]
fn features() -> u64 {
    sciencekit_ffi::sk_ffi_features()
}

/// Return the stable lowercase name of an ABI status discriminant.
#[pyfunction]
fn status_name(status_code: i32) -> String {
    let pointer = sciencekit_ffi::sk_ffi_status_name(status_code);
    if pointer.is_null() {
        return "unknown".to_string();
    }
    // SAFETY: `sk_ffi_status_name` always returns a static NUL-terminated string.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

/// Return the number of live opaque handles in this process.
#[pyfunction]
fn live_handles() -> usize {
    lifecycle::sk_py_live_handles()
}

/// Native extension module `sciencekit._native`.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("STATUS_OK", sciencekit_ffi::SKFfiStatus::Ok as i32)?;
    module.add(
        "STATUS_INVALID_HANDLE",
        sciencekit_ffi::SKFfiStatus::InvalidHandle as i32,
    )?;
    module.add("STATUS_SHAPE", sciencekit_ffi::SKFfiStatus::Shape as i32)?;
    module.add(
        "STATUS_INVALID_ARGUMENT",
        sciencekit_ffi::SKFfiStatus::InvalidArgument as i32,
    )?;
    module.add(
        "STATUS_UNSUPPORTED",
        sciencekit_ffi::SKFfiStatus::Unsupported as i32,
    )?;
    module.add(
        "STATUS_NOT_FITTED",
        sciencekit_ffi::SKFfiStatus::NotFitted as i32,
    )?;
    module.add("STATUS_IO", sciencekit_ffi::SKFfiStatus::Io as i32)?;
    module.add(
        "STATUS_INTERNAL",
        sciencekit_ffi::SKFfiStatus::Internal as i32,
    )?;

    module.add(
        "FEATURE_LINEAR_REGRESSION",
        sciencekit_ffi::SK_FFI_FEATURE_LINEAR_REGRESSION,
    )?;
    module.add(
        "FEATURE_STANDARD_SCALER",
        sciencekit_ffi::SK_FFI_FEATURE_STANDARD_SCALER,
    )?;
    module.add(
        "FEATURE_MODEL_PERSISTENCE",
        sciencekit_ffi::SK_FFI_FEATURE_MODEL_PERSISTENCE,
    )?;
    module.add("FEATURE_F32", sciencekit_ffi::SK_FFI_FEATURE_F32)?;
    module.add("FEATURE_F64", sciencekit_ffi::SK_FFI_FEATURE_F64)?;

    module.add_class::<SKPyDenseDescriptor>()?;
    module.add_class::<SKPyCsrDescriptor>()?;
    module.add_class::<SKPyLinearRegression>()?;
    module.add_class::<SKPyLinearRegressionModel>()?;
    module.add_class::<SKPyStandardScaler>()?;
    module.add_class::<SKPyStandardScalerModel>()?;

    module.add_function(wrap_pyfunction!(abi_version, module)?)?;
    module.add_function(wrap_pyfunction!(features, module)?)?;
    module.add_function(wrap_pyfunction!(status_name, module)?)?;
    module.add_function(wrap_pyfunction!(live_handles, module)?)?;
    module.add_function(wrap_pyfunction!(load, module)?)?;
    Ok(())
}
