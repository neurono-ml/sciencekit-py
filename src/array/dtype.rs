//! Data-type parsing and contiguous stride helpers for NumPy descriptors.

use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods, dtype};
use pyo3::prelude::*;
use sciencekit_ffi::SKFfiDataType;

use crate::error::SKPyError;

/// Resolve a Python dtype name or `numpy.dtype` object to `SKFfiDataType`.
#[allow(dead_code)]
pub(crate) fn sk_py_parse_dtype(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> Result<SKFfiDataType, SKPyError> {
    if let Ok(text) = value.extract::<String>() {
        return match text.as_str() {
            "float64" | "f64" | "f8" => Ok(SKFfiDataType::F64),
            "float32" | "f32" | "f4" => Ok(SKFfiDataType::F32),
            other => Err(SKPyError::invalid_argument(format!(
                "unsupported dtype '{other}'; expected 'float64' or 'float32'"
            ))),
        };
    }
    let numpy_mod = py
        .import("numpy")
        .map_err(|err| SKPyError::invalid_argument(err.to_string()))?;
    let descr = numpy_mod.call_method1("dtype", (value,)).map_err(|_| {
        SKPyError::invalid_argument("expected 'float64', 'float32', or numpy.dtype")
    })?;
    let name: String = descr
        .getattr("name")
        .and_then(|attr| attr.extract())
        .map_err(|_| SKPyError::invalid_argument("could not inspect numpy.dtype"))?;
    match name.as_str() {
        "float64" => Ok(SKFfiDataType::F64),
        "float32" => Ok(SKFfiDataType::F32),
        other => Err(SKPyError::invalid_argument(format!(
            "unsupported numpy dtype '{other}'; expected float64 or float32"
        ))),
    }
}

/// Canonical Python dtype string for an ABI data type.
pub(crate) fn sk_py_dtype_label(data_type: SKFfiDataType) -> &'static str {
    match data_type {
        SKFfiDataType::F32 => "float32",
        SKFfiDataType::F64 => "float64",
        SKFfiDataType::I64 => "int64",
        SKFfiDataType::U8 => "uint8",
    }
}

/// Inspect a NumPy array's element type, accepting `float64` and `float32`.
pub(crate) fn sk_py_detect_array_dtype(
    py: Python<'_>,
    array: &Bound<'_, PyUntypedArray>,
) -> Result<SKFfiDataType, SKPyError> {
    let array_dtype = array.dtype();
    if array_dtype.is_equiv_to(&dtype::<f64>(py)) {
        Ok(SKFfiDataType::F64)
    } else if array_dtype.is_equiv_to(&dtype::<f32>(py)) {
        Ok(SKFfiDataType::F32)
    } else {
        Err(SKPyError::invalid_argument(format!(
            "unsupported array dtype {array_dtype}; expected float64 or float32"
        )))
    }
}

/// Compute row-major element strides for `shape`.
pub(crate) fn sk_py_contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1_i64; shape.len()];
    let mut running = 1_i64;
    for (stride, &dimension) in strides.iter_mut().zip(shape.iter()).rev() {
        *stride = running;
        running *= dimension;
    }
    strides
}
