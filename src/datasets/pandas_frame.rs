//! pandas frames without the Arrow PyCapsule protocol (`to_numpy` fallback).

use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use sciencekit_ffi::SKFfiDataType;

use super::metadata::SKPyConversionMetadata;
use super::options::SKPyConversionOptions;
use super::storage::{SKPyConvertedData, SKPyOwnedNumbers, SKPyStorage};
use crate::error::SKPyError;

/// Convert a legacy pandas frame through `to_numpy`, reporting the copy.
pub(crate) fn sk_py_convert_legacy_pandas<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    options: &SKPyConversionOptions,
    vector_only: bool,
) -> Result<(SKPyConvertedData<'py>, Option<SKPyConvertedData<'py>>), SKPyError> {
    let frame = sk_py_as_frame(py, object)?;
    let names = sk_py_frame_columns(&frame)?;
    let (selected, target) = if vector_only {
        (sk_py_vector_selection(&names, options)?, None)
    } else {
        sk_py_frame_selection(&names, options)?
    };

    let mut values = SKPyOwnedNumbers::empty(options.expected_dtype);
    let mut rows = 0_usize;
    for name in &selected {
        let column = sk_py_column_array(&frame, name, options)?;
        if rows == 0 {
            rows = column.len();
        } else if column.len() != rows {
            return Err(SKPyError::shape(format!(
                "column '{name}' has {} rows but earlier columns have {rows}",
                column.len()
            )));
        }
        values.extend_from(&column);
    }
    let rows = if selected.is_empty() { 0 } else { rows };
    if rows == 0 {
        return Err(SKPyError::invalid_argument(
            "the pandas frame has no rows or no selected numeric columns",
        ));
    }
    // `to_numpy` returns one contiguous column per feature; transpose to row-major.
    let values = sk_py_transpose_columns(values, rows, selected.len(), options.expected_dtype);

    let features_metadata = SKPyConversionMetadata {
        representation: "pandas.dataframe".to_string(),
        dtype: super::numpy_dataset::sk_py_dtype_name(options.expected_dtype),
        rows,
        columns_count: selected.len(),
        copied: true,
        zero_copy: false,
        cast: false,
        eager_fallback: Some("pandas.to_numpy".to_string()),
        feature_names: selected.clone(),
        missing_policy: options.on_missing.name().to_string(),
        ..SKPyConversionMetadata::default()
    };
    let features = SKPyConvertedData {
        dtype: options.expected_dtype,
        shape: vec![rows as i64, selected.len() as i64],
        storage: SKPyStorage::Owned(values),
        metadata: features_metadata,
    };
    let target_data = match target {
        Some(name) => {
            let column = sk_py_column_array(&frame, &name, options)?;
            let metadata = SKPyConversionMetadata {
                representation: "pandas.dataframe".to_string(),
                dtype: super::numpy_dataset::sk_py_dtype_name(options.expected_dtype),
                rows: column.len(),
                columns_count: 1,
                copied: true,
                zero_copy: false,
                eager_fallback: Some("pandas.to_numpy".to_string()),
                feature_names: vec![name],
                missing_policy: options.on_missing.name().to_string(),
                ..SKPyConversionMetadata::default()
            };
            Some(SKPyConvertedData {
                dtype: options.expected_dtype,
                shape: vec![column.len() as i64],
                storage: SKPyStorage::Owned(column),
                metadata,
            })
        }
        None => None,
    };
    Ok((features, target_data))
}

fn sk_py_as_frame<'py>(
    _py: Python<'py>,
    object: &Bound<'py, PyAny>,
) -> Result<Bound<'py, PyAny>, SKPyError> {
    let type_name = super::representation::sk_py_type_name(object);
    if type_name == "Series" {
        let frame = object.call_method0("to_frame").map_err(|_| {
            SKPyError::invalid_argument("a pandas.Series must expose `.to_frame()`")
        })?;
        return Ok(frame);
    }
    Ok(object.clone())
}

fn sk_py_frame_columns(frame: &Bound<'_, PyAny>) -> Result<Vec<String>, SKPyError> {
    let columns = frame
        .getattr("columns")
        .map_err(|_| SKPyError::invalid_argument("the pandas frame has no `.columns`"))?;
    columns
        .try_iter()
        .map_err(|_| SKPyError::invalid_argument("pandas frame columns are not iterable"))?
        .map(|item| {
            item.and_then(|value| value.extract::<String>())
                .map_err(|_| SKPyError::invalid_argument("pandas column names must be strings"))
        })
        .collect()
}

fn sk_py_vector_selection(
    names: &[String],
    options: &SKPyConversionOptions,
) -> Result<Vec<String>, SKPyError> {
    if names.len() != 1 {
        return Err(SKPyError::invalid_argument(format!(
            "a 1D target requires a single-column frame, found {} columns {names:?}",
            names.len()
        )));
    }
    let _ = options;
    Ok(vec![names[0].clone()])
}

fn sk_py_frame_selection(
    names: &[String],
    options: &SKPyConversionOptions,
) -> Result<(Vec<String>, Option<String>), SKPyError> {
    if let Some(target) = &options.target {
        if !names.contains(target) {
            return Err(SKPyError::key(format!(
                "unknown column '{target}'; available columns are {names:?}"
            )));
        }
    }
    let selected = match &options.columns {
        Some(requested) => {
            for name in requested {
                if !names.contains(name) {
                    return Err(SKPyError::key(format!(
                        "unknown column '{name}'; available columns are {names:?}"
                    )));
                }
            }
            requested.clone()
        }
        None => names
            .iter()
            .filter(|name| options.target.as_deref() != Some(name.as_str()))
            .cloned()
            .collect(),
    };
    if selected.is_empty() {
        return Err(SKPyError::invalid_argument(
            "no feature columns selected; name at least one numeric column",
        ));
    }
    Ok((selected, options.target.clone()))
}

fn sk_py_column_array(
    frame: &Bound<'_, PyAny>,
    name: &str,
    options: &SKPyConversionOptions,
) -> Result<SKPyOwnedNumbers, SKPyError> {
    let column = frame
        .get_item(name)
        .map_err(|_| SKPyError::key(format!("unknown column '{name}'")))?;
    let py = frame.py();
    let dtype_name = super::numpy_dataset::sk_py_dtype_name(options.expected_dtype);
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs
        .set_item("dtype", dtype_name)
        .map_err(|_| SKPyError::invalid_argument("failed to build to_numpy kwargs"))?;
    kwargs
        .set_item("na_value", f64::NAN)
        .map_err(|_| SKPyError::invalid_argument("failed to build to_numpy kwargs"))?;
    let array = column
        .call_method("to_numpy", (), Some(&kwargs))
        .map_err(|_| {
            SKPyError::type_error(format!(
                "column '{name}' cannot be converted with to_numpy(dtype={dtype_name}); cast it to a numeric dtype first"
            ))
        })?;
    let numbers = sk_py_copy_numpy_column(&array, options.expected_dtype, name)?;
    sk_py_apply_missing_policy(numbers, name, options)
}

fn sk_py_copy_numpy_column(
    array: &Bound<'_, PyAny>,
    dtype: SKFfiDataType,
    name: &str,
) -> Result<SKPyOwnedNumbers, SKPyError> {
    match dtype {
        SKFfiDataType::F32 => {
            let typed = array.cast::<PyArrayDyn<f32>>().map_err(|_| {
                SKPyError::type_error(format!("column '{name}' is not a float32 array"))
            })?;
            let readonly = typed.readonly();
            let slice = readonly
                .as_slice()
                .map_err(|_| SKPyError::invalid_argument("column array must be contiguous"))?;
            Ok(SKPyOwnedNumbers::F32(slice.to_vec()))
        }
        _ => {
            let typed = array.cast::<PyArrayDyn<f64>>().map_err(|_| {
                SKPyError::type_error(format!("column '{name}' is not a float64 array"))
            })?;
            let readonly = typed.readonly();
            let slice = readonly
                .as_slice()
                .map_err(|_| SKPyError::invalid_argument("column array must be contiguous"))?;
            Ok(SKPyOwnedNumbers::F64(slice.to_vec()))
        }
    }
}

fn sk_py_apply_missing_policy(
    numbers: SKPyOwnedNumbers,
    name: &str,
    options: &SKPyConversionOptions,
) -> Result<SKPyOwnedNumbers, SKPyError> {
    let mut resolved = SKPyOwnedNumbers::empty(options.expected_dtype);
    match numbers {
        SKPyOwnedNumbers::F64(values) => {
            for (row, value) in values.into_iter().enumerate() {
                if value.is_nan()
                    && !matches!(options.on_missing, super::missing::SKPyMissingPolicy::Nan)
                {
                    resolved.push_f64(options.on_missing.resolve(name, row)?);
                } else {
                    resolved.push_f64(value);
                }
            }
        }
        SKPyOwnedNumbers::F32(values) => {
            for (row, value) in values.into_iter().enumerate() {
                if value.is_nan()
                    && !matches!(options.on_missing, super::missing::SKPyMissingPolicy::Nan)
                {
                    resolved.push_f32(options.on_missing.resolve(name, row)? as f32);
                } else {
                    resolved.push_f32(value);
                }
            }
        }
    }
    Ok(resolved)
}

fn sk_py_transpose_columns(
    column_major: SKPyOwnedNumbers,
    rows: usize,
    columns: usize,
    dtype: SKFfiDataType,
) -> SKPyOwnedNumbers {
    let mut values = SKPyOwnedNumbers::empty(dtype);
    match column_major {
        SKPyOwnedNumbers::F64(items) => {
            for row in 0..rows {
                for column in 0..columns {
                    values.push_f64(items[column * rows + row]);
                }
            }
        }
        SKPyOwnedNumbers::F32(items) => {
            for row in 0..rows {
                for column in 0..columns {
                    values.push_f32(items[column * rows + row]);
                }
            }
        }
    }
    values
}
