//! NumPy `ndarray` conversion: zero-copy borrow when C-contiguous.

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use sciencekit_ffi::SKFfiDataType;

use super::metadata::SKPyConversionMetadata;
use super::options::SKPyConversionOptions;
use super::storage::{SKPyConvertedData, SKPyStorage};
use crate::array::dense::SKPyBorrowedArray;
use crate::array::dtype::{sk_py_detect_array_dtype, sk_py_dtype_label};
use crate::error::SKPyError;

/// Stable precision name for error messages and reports.
pub(crate) fn sk_py_dtype_name(dtype: SKFfiDataType) -> &'static str {
    sk_py_dtype_label(dtype)
}

/// Convert a one- or two-dimensional NumPy array of `float32`/`float64`.
pub(crate) fn sk_py_convert_numpy<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    options: &SKPyConversionOptions,
) -> Result<SKPyConvertedData<'py>, SKPyError> {
    if options.columns.is_some() || options.target.is_some() {
        return Err(SKPyError::invalid_argument(
            "column selection and target-by-name require a frame representation; a numpy.ndarray is passed positionally",
        ));
    }
    let untyped = object.cast::<PyUntypedArray>().map_err(|_| {
        SKPyError::invalid_argument("expected a numpy.ndarray of float32 or float64")
    })?;
    let rank = untyped.ndim();
    let source_dtype = sk_py_detect_array_dtype(py, untyped)?;
    let cast = source_dtype != options.expected_dtype;

    if cast {
        if !options.may_cast() {
            return Err(SKPyError::invalid_argument(format!(
                "array dtype mismatch: expected {}, found {}; pass cast=True, or convert with numpy.asarray(array, dtype={})",
                sk_py_dtype_label(options.expected_dtype),
                sk_py_dtype_label(source_dtype),
                sk_py_dtype_label(options.expected_dtype)
            )));
        }
        let casted = object
            .call_method1("astype", (sk_py_dtype_label(options.expected_dtype),))
            .map_err(|_| SKPyError::invalid_argument("numpy.ndarray.astype failed"))?;
        let borrowed = SKPyBorrowedArray::from_any(
            py,
            &casted,
            Some(rank),
            Some(options.expected_dtype),
            true,
        )?;
        let shape: Vec<i64> = borrowed.shape.clone();
        let pointer = borrowed.data_pointer;
        let metadata = sk_py_numpy_metadata(&shape, options, true, true, false, pointer);
        return Ok(SKPyConvertedData {
            dtype: options.expected_dtype,
            shape,
            storage: SKPyStorage::Borrowed {
                _owner: casted,
                pointer,
            },
            metadata,
        });
    }

    let borrowed = SKPyBorrowedArray::from_any(
        py,
        object,
        Some(rank),
        Some(options.expected_dtype),
        options.allow_copy,
    )?;
    let shape: Vec<i64> = borrowed.shape.clone();
    let pointer = borrowed.data_pointer;
    let copied = borrowed.copied;
    let metadata = sk_py_numpy_metadata(&shape, options, false, copied, !copied, pointer);
    Ok(SKPyConvertedData {
        dtype: options.expected_dtype,
        shape,
        storage: SKPyStorage::Borrowed {
            _owner: object.clone(),
            pointer,
        },
        metadata,
    })
}

fn sk_py_numpy_metadata(
    shape: &[i64],
    options: &SKPyConversionOptions,
    cast: bool,
    copied: bool,
    zero_copy: bool,
    data_pointer: usize,
) -> SKPyConversionMetadata {
    let rows = shape[0] as usize;
    let columns_count = if shape.len() == 2 {
        shape[1] as usize
    } else {
        1
    };
    let feature_names = if shape.len() == 2 {
        (0..columns_count)
            .map(|index| format!("column_{index}"))
            .collect()
    } else {
        Vec::new()
    };
    SKPyConversionMetadata {
        representation: "numpy.ndarray".to_string(),
        dtype: sk_py_dtype_label(options.expected_dtype),
        rows,
        columns_count,
        copied,
        zero_copy,
        densified: false,
        cast,
        feature_names,
        data_pointer: Some(data_pointer),
        missing_policy: options.on_missing.name().to_string(),
        ..SKPyConversionMetadata::default()
    }
}
