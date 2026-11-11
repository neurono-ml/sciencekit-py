//! Copying `sciencekit_ffi` output descriptors into owned NumPy arrays and
//! releasing the C-owned buffer deterministically via `sk_ffi_array_free`.

use std::ptr;
use std::slice;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use sciencekit_ffi::{SKFfiArray, SKFfiDataType, SKFfiStatus};

use crate::error::{SKPyError, sk_py_to_py_err};

/// Pure-Rust owned payload extracted inside `py.detach` before `sk_ffi_array_free`.
#[derive(Debug)]
pub(crate) enum SKPyOwnedOutput {
    VectorF64(Vec<f64>),
    VectorF32(Vec<f32>),
    MatrixF64 {
        rows: usize,
        columns: usize,
        values: Vec<f64>,
    },
    MatrixF32 {
        rows: usize,
        columns: usize,
        values: Vec<f32>,
    },
}

impl SKPyOwnedOutput {
    /// Convert this owned Rust vector into a `numpy.ndarray` without a second copy.
    pub(crate) fn into_py_array(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            SKPyOwnedOutput::VectorF64(values) => {
                Ok(PyArray1::from_vec(py, values).into_any().unbind())
            }
            SKPyOwnedOutput::VectorF32(values) => {
                Ok(PyArray1::from_vec(py, values).into_any().unbind())
            }
            SKPyOwnedOutput::MatrixF64 {
                rows,
                columns,
                values,
            } => {
                let array = PyArray1::from_vec(py, values)
                    .reshape([rows, columns])
                    .map_err(|err| {
                        sk_py_to_py_err(py, SKPyError::invalid_argument(err.to_string()))
                    })?;
                Ok(array.into_any().unbind())
            }
            SKPyOwnedOutput::MatrixF32 {
                rows,
                columns,
                values,
            } => {
                let array = PyArray1::from_vec(py, values)
                    .reshape([rows, columns])
                    .map_err(|err| {
                        sk_py_to_py_err(py, SKPyError::invalid_argument(err.to_string()))
                    })?;
                Ok(array.into_any().unbind())
            }
        }
    }
}

/// Zero-initialized output descriptor ready for `sk_ffi_predict` or `sk_ffi_transform`.
pub(crate) fn sk_py_empty_output() -> SKFfiArray {
    SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 0,
        shape: ptr::null(),
        strides: ptr::null(),
        data: ptr::null(),
        len: 0,
    }
}

/// Copy a library-owned `SKFfiArray` into `SKPyOwnedOutput` and unconditionally
/// release the C-owned buffers with `sk_ffi_array_free`.
pub(crate) fn sk_py_take_output_array(
    status: SKFfiStatus,
    mut output: SKFfiArray,
) -> Result<SKPyOwnedOutput, SKPyError> {
    struct OutputGuard<'a>(&'a mut SKFfiArray);
    impl Drop for OutputGuard<'_> {
        fn drop(&mut self) {
            let _ = sciencekit_ffi::sk_ffi_array_free(self.0);
        }
    }

    let guard = OutputGuard(&mut output);
    if status != SKFfiStatus::Ok {
        return Err(SKPyError::from_last_error(status));
    }
    if guard.0.shape.is_null() || guard.0.data.is_null() || guard.0.len < 0 {
        return Err(SKPyError::new(
            SKFfiStatus::Internal,
            "sciencekit_ffi returned an incomplete output descriptor",
        ));
    }

    let ndim = guard.0.ndim as usize;
    let length = guard.0.len as usize;
    // SAFETY: `sciencekit_ffi` guarantees `ndim` readable `i64` values at `shape`
    // and `length` readable elements at `data` until `sk_ffi_array_free` runs.
    let dimensions = unsafe { slice::from_raw_parts(guard.0.shape, ndim) };
    let data_type = SKFfiDataType::from_code(guard.0.dtype).ok_or_else(|| {
        SKPyError::new(
            SKFfiStatus::Internal,
            format!(
                "sciencekit_ffi returned unknown dtype code {}",
                guard.0.dtype
            ),
        )
    })?;

    match (ndim, data_type) {
        (1, SKFfiDataType::F64) => {
            let slice = unsafe { slice::from_raw_parts(guard.0.data as *const f64, length) };
            Ok(SKPyOwnedOutput::VectorF64(slice.to_vec()))
        }
        (1, SKFfiDataType::F32) => {
            let slice = unsafe { slice::from_raw_parts(guard.0.data as *const f32, length) };
            Ok(SKPyOwnedOutput::VectorF32(slice.to_vec()))
        }
        (2, SKFfiDataType::F64) => {
            let slice = unsafe { slice::from_raw_parts(guard.0.data as *const f64, length) };
            Ok(SKPyOwnedOutput::MatrixF64 {
                rows: dimensions[0] as usize,
                columns: dimensions[1] as usize,
                values: slice.to_vec(),
            })
        }
        (2, SKFfiDataType::F32) => {
            let slice = unsafe { slice::from_raw_parts(guard.0.data as *const f32, length) };
            Ok(SKPyOwnedOutput::MatrixF32 {
                rows: dimensions[0] as usize,
                columns: dimensions[1] as usize,
                values: slice.to_vec(),
            })
        }
        _ => Err(SKPyError::new(
            SKFfiStatus::Internal,
            format!("unexpected output ndim={ndim} dtype={data_type:?}"),
        )),
    }
}
