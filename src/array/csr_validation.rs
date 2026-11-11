//! Validation helpers for CSR `data`, `indices` and `indptr` NumPy buffers.

use numpy::{
    PyArray1, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::prelude::*;

use crate::error::SKPyError;

/// Validate the 1D contiguous floating-point `data` array of a CSR matrix.
pub(crate) fn sk_py_validate_csr_data(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
) -> Result<(&'static str, usize, usize), SKPyError> {
    let untyped = data.cast::<PyUntypedArray>().map_err(|_| {
        SKPyError::invalid_argument("CSR data must be a 1D C-contiguous numpy.ndarray")
    })?;
    if untyped.ndim() != 1 {
        return Err(SKPyError::invalid_argument(format!(
            "CSR data must be 1D, found {}D",
            untyped.ndim()
        )));
    }
    let array_dtype = untyped.dtype();
    if array_dtype.is_equiv_to(&dtype::<f64>(py)) {
        let typed = data
            .cast::<PyArray1<f64>>()
            .map_err(|_| SKPyError::invalid_argument("failed to cast float64 CSR data"))?;
        let readonly = typed.readonly();
        let slice = readonly.as_slice().map_err(|_| {
            SKPyError::unsupported("CSR data array must be C-contiguous and aligned")
        })?;
        Ok(("float64", slice.len(), slice.as_ptr() as usize))
    } else if array_dtype.is_equiv_to(&dtype::<f32>(py)) {
        let typed = data
            .cast::<PyArray1<f32>>()
            .map_err(|_| SKPyError::invalid_argument("failed to cast float32 CSR data"))?;
        let readonly = typed.readonly();
        let slice = readonly.as_slice().map_err(|_| {
            SKPyError::unsupported("CSR data array must be C-contiguous and aligned")
        })?;
        Ok(("float32", slice.len(), slice.as_ptr() as usize))
    } else {
        Err(SKPyError::invalid_argument(format!(
            "CSR data dtype must be float64 or float32, found {array_dtype}"
        )))
    }
}

/// Validate a 1D contiguous `int32` or `int64` CSR index buffer with non-negative entries.
pub(crate) fn sk_py_extract_non_negative_indices(
    py: Python<'_>,
    array_obj: &Bound<'_, PyAny>,
    field_name: &str,
) -> Result<(&'static str, Vec<usize>), SKPyError> {
    let untyped = array_obj.cast::<PyUntypedArray>().map_err(|_| {
        SKPyError::invalid_argument(format!("CSR {field_name} must be a 1D numpy.ndarray"))
    })?;
    if untyped.ndim() != 1 {
        return Err(SKPyError::invalid_argument(format!(
            "CSR {field_name} must be 1D, found {}D",
            untyped.ndim()
        )));
    }
    let array_dtype = untyped.dtype();
    if array_dtype.is_equiv_to(&dtype::<i32>(py)) {
        let typed = array_obj.cast::<PyArray1<i32>>().map_err(|_| {
            SKPyError::invalid_argument(format!("failed to read int32 CSR {field_name}"))
        })?;
        let readonly = typed.readonly();
        let slice = readonly.as_slice().map_err(|_| {
            SKPyError::unsupported(format!("CSR {field_name} must be C-contiguous"))
        })?;
        let mut converted = Vec::with_capacity(slice.len());
        for &value in slice {
            if value < 0 {
                return Err(SKPyError::invalid_argument(format!(
                    "CSR {field_name} contains negative entry {value}"
                )));
            }
            converted.push(value as usize);
        }
        Ok(("int32", converted))
    } else if array_dtype.is_equiv_to(&dtype::<i64>(py)) {
        let typed = array_obj.cast::<PyArray1<i64>>().map_err(|_| {
            SKPyError::invalid_argument(format!("failed to read int64 CSR {field_name}"))
        })?;
        let readonly = typed.readonly();
        let slice = readonly.as_slice().map_err(|_| {
            SKPyError::unsupported(format!("CSR {field_name} must be C-contiguous"))
        })?;
        let mut converted = Vec::with_capacity(slice.len());
        for &value in slice {
            if value < 0 {
                return Err(SKPyError::invalid_argument(format!(
                    "CSR {field_name} contains negative entry {value}"
                )));
            }
            converted.push(value as usize);
        }
        Ok(("int64", converted))
    } else {
        Err(SKPyError::invalid_argument(format!(
            "CSR {field_name} dtype must be int32 or int64, found {array_dtype}"
        )))
    }
}

/// Expand validated CSR buffers into a row-major dense buffer.
pub(crate) fn sk_py_expand_csr<T: Copy>(
    rows: usize,
    columns: usize,
    row_pointers: &[usize],
    column_indices: &[usize],
    values: &[T],
    zero: T,
) -> Vec<T> {
    let mut dense = vec![zero; rows * columns];
    for row in 0..rows {
        let start = row_pointers[row];
        let end = row_pointers[row + 1];
        let row_offset = row * columns;
        for position in start..end {
            dense[row_offset + column_indices[position]] = values[position];
        }
    }
    dense
}
