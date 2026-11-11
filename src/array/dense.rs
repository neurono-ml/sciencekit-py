//! Zero-copy dense NumPy buffer validation and `SKFfiArray` construction.

use std::ffi::c_void;
use std::ptr;

use numpy::{PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use sciencekit_ffi::{SKFfiArray, SKFfiDataType};

use super::dtype::{sk_py_contiguous_strides, sk_py_detect_array_dtype, sk_py_dtype_label};
use crate::error::SKPyError;

/// Keep a validated NumPy `ndarray` alive while exposing its raw buffer to C.
pub(crate) struct SKPyBorrowedArray<'py> {
    _owner: Bound<'py, PyAny>,
    pub(crate) dtype: SKFfiDataType,
    pub(crate) shape: Vec<i64>,
    pub(crate) strides: Vec<i64>,
    pub(crate) data_pointer: usize,
    pub(crate) length: i64,
    pub(crate) copied: bool,
}

impl<'py> SKPyBorrowedArray<'py> {
    /// Validate a 1D or 2D `float32`/`float64` NumPy array, borrowing its
    /// buffer zero-copy when C-contiguous or copying only if `allow_copy` is true.
    pub(crate) fn from_any(
        py: Python<'py>,
        object: &Bound<'py, PyAny>,
        expected_rank: Option<usize>,
        expected_dtype: Option<SKFfiDataType>,
        allow_copy: bool,
    ) -> Result<SKPyBorrowedArray<'py>, SKPyError> {
        let untyped = object.cast::<PyUntypedArray>().map_err(|_| {
            SKPyError::invalid_argument("expected a numpy.ndarray of float32 or float64")
        })?;
        let data_type = sk_py_detect_array_dtype(py, untyped)?;
        if let Some(required) = expected_dtype {
            if data_type != required {
                return Err(SKPyError::invalid_argument(format!(
                    "array dtype mismatch: expected {}, found {}",
                    sk_py_dtype_label(required),
                    sk_py_dtype_label(data_type)
                )));
            }
        }

        let rank = untyped.ndim();
        if !(1..=2).contains(&rank) {
            return Err(SKPyError::invalid_argument(format!(
                "array rank must be 1 or 2, found {rank}"
            )));
        }
        if let Some(required_rank) = expected_rank {
            if rank != required_rank {
                return Err(SKPyError::invalid_argument(format!(
                    "expected a {required_rank}D numpy.ndarray, found {rank}D with shape {:?}",
                    untyped.shape()
                )));
            }
        }

        let mut shape = Vec::with_capacity(rank);
        let mut length = 1_i64;
        for (axis, &dimension) in untyped.shape().iter().enumerate() {
            if dimension == 0 {
                return Err(SKPyError::invalid_argument(format!(
                    "array dimension {axis} is zero; non-empty observations are required"
                )));
            }
            let dim_i64 = i64::try_from(dimension)
                .map_err(|_| SKPyError::invalid_argument("array dimension overflows i64"))?;
            length = length
                .checked_mul(dim_i64)
                .ok_or_else(|| SKPyError::invalid_argument("array element count overflows i64"))?;
            shape.push(dim_i64);
        }

        match data_type {
            SKFfiDataType::F64 => {
                sk_py_borrow_typed::<f64>(py, object, data_type, shape, length, allow_copy)
            }
            SKFfiDataType::F32 => {
                sk_py_borrow_typed::<f32>(py, object, data_type, shape, length, allow_copy)
            }
            _ => Err(SKPyError::invalid_argument(
                "only float32 and float64 arrays are supported",
            )),
        }
    }

    /// Extract plain-Rust `Send` descriptor parameters for `py.detach`.
    pub(crate) fn as_raw_parts(&self) -> SKPyRawDescriptor {
        SKPyRawDescriptor {
            dtype: self.dtype as i32,
            shape: self.shape.clone(),
            data_pointer: self.data_pointer,
            length: self.length,
        }
    }
}

/// Pure-Rust `Send` descriptor parameters safe to pass into `py.detach`.
#[derive(Debug, Clone)]
pub(crate) struct SKPyRawDescriptor {
    dtype: i32,
    shape: Vec<i64>,
    data_pointer: usize,
    length: i64,
}

impl SKPyRawDescriptor {
    /// Materialize the C-layout `SKFfiArray` pointing into the borrowed buffer.
    pub(crate) fn to_ffi_array(&self) -> SKFfiArray {
        SKFfiArray {
            dtype: self.dtype,
            ndim: self.shape.len() as i32,
            shape: self.shape.as_ptr(),
            strides: ptr::null(),
            data: self.data_pointer as *const c_void,
            len: self.length,
        }
    }
}

fn sk_py_borrow_typed<'py, T: numpy::Element>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    data_type: SKFfiDataType,
    shape: Vec<i64>,
    length: i64,
    allow_copy: bool,
) -> Result<SKPyBorrowedArray<'py>, SKPyError> {
    let typed = object
        .cast::<numpy::PyArrayDyn<T>>()
        .map_err(|_| SKPyError::invalid_argument("failed to cast typed numpy.ndarray"))?;
    if typed.is_c_contiguous() {
        let readonly = typed.readonly();
        if let Ok(slice) = readonly.as_slice() {
            let strides = sk_py_contiguous_strides(&shape);
            let data_pointer = slice.as_ptr() as usize;
            drop(readonly);
            return Ok(SKPyBorrowedArray {
                _owner: object.clone(),
                dtype: data_type,
                shape,
                strides,
                data_pointer,
                length,
                copied: false,
            });
        }
    }
    if !allow_copy {
        return Err(SKPyError::unsupported(
            "non-contiguous or misaligned numpy.ndarray is not supported for zero-copy ABI calls; pass copy=True or numpy.ascontiguousarray(array)",
        ));
    }
    let contiguous = py
        .import("numpy")
        .and_then(|np| np.call_method1("ascontiguousarray", (object,)))
        .map_err(|err| SKPyError::invalid_argument(err.to_string()))?;
    let contiguous_typed = contiguous
        .cast::<numpy::PyArrayDyn<T>>()
        .map_err(|_| SKPyError::invalid_argument("ascontiguousarray returned unexpected type"))?;
    let contiguous_readonly = contiguous_typed.readonly();
    let slice = contiguous_readonly.as_slice().map_err(|_| {
        SKPyError::unsupported("copied array is still not C-contiguous or properly aligned")
    })?;
    let data_pointer = slice.as_ptr() as usize;
    drop(contiguous_readonly);
    let strides = sk_py_contiguous_strides(&shape);
    Ok(SKPyBorrowedArray {
        _owner: contiguous,
        dtype: data_type,
        shape,
        strides,
        data_pointer,
        length,
        copied: true,
    })
}
