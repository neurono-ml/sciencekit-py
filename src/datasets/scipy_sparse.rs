//! SciPy sparse matrices consumed through their three CSR buffers.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use sciencekit_ffi::SKFfiDataType;

use super::metadata::SKPyConversionMetadata;
use super::options::SKPyConversionOptions;
use super::storage::{SKPyConvertedData, SKPyOwnedNumbers, SKPyStorage};
use crate::array::csr::SKPyCsrDescriptor;
use crate::array::csr_validation::sk_py_expand_csr;
use crate::error::SKPyError;

/// Borrowed CSR `data`/`indices`/`indptr` buffers behind a sparse view.
pub(crate) struct SKPySparseView<'py> {
    descriptor: SKPyCsrDescriptor,
    _owner: Bound<'py, PyAny>,
}

impl SKPySparseView<'_> {
    /// Materialize a row-major dense block at the `sciencekit_ffi` call boundary.
    pub(crate) fn densify(
        &self,
        py: Python<'_>,
        dtype: SKFfiDataType,
    ) -> Result<SKPyOwnedNumbers, SKPyError> {
        let data_object = self.descriptor.data_object().bind(py);
        let rows = self.descriptor.rows();
        let columns = self.descriptor.columns();
        let row_pointers = self.descriptor.row_pointers();
        let column_indices = self.descriptor.column_indices();
        let source_is_f64 = self.descriptor.dtype_name() == "float64";

        match (source_is_f64, dtype) {
            (true, SKFfiDataType::F64) => {
                let values = data_object
                    .cast::<PyArray1<f64>>()
                    .map_err(|_| SKPyError::invalid_argument("failed to read float64 CSR data"))?;
                let readonly = values.readonly();
                let slice = readonly.as_slice().map_err(|_| {
                    SKPyError::unsupported("CSR data array must be C-contiguous and aligned")
                })?;
                Ok(SKPyOwnedNumbers::F64(sk_py_expand_csr(
                    rows,
                    columns,
                    row_pointers,
                    column_indices,
                    slice,
                    0.0f64,
                )))
            }
            (true, _) => {
                let dense = self.densify(py, SKFfiDataType::F64)?;
                Ok(sk_py_cast_numbers(&dense, SKFfiDataType::F32))
            }
            (false, SKFfiDataType::F32) => {
                let values = data_object
                    .cast::<PyArray1<f32>>()
                    .map_err(|_| SKPyError::invalid_argument("failed to read float32 CSR data"))?;
                let readonly = values.readonly();
                let slice = readonly.as_slice().map_err(|_| {
                    SKPyError::unsupported("CSR data array must be C-contiguous and aligned")
                })?;
                Ok(SKPyOwnedNumbers::F32(sk_py_expand_csr(
                    rows,
                    columns,
                    row_pointers,
                    column_indices,
                    slice,
                    0.0f32,
                )))
            }
            (false, _) => {
                let dense = self.densify(py, SKFfiDataType::F32)?;
                Ok(sk_py_cast_numbers(&dense, SKFfiDataType::F64))
            }
        }
    }
}

fn sk_py_cast_numbers(source: &SKPyOwnedNumbers, target: SKFfiDataType) -> SKPyOwnedNumbers {
    match (source, target) {
        (source, SKFfiDataType::F32) => {
            let mut values = Vec::with_capacity(source.len());
            match source {
                SKPyOwnedNumbers::F64(items) => values.extend(items.iter().map(|&v| v as f32)),
                SKPyOwnedNumbers::F32(items) => values.extend_from_slice(items),
            }
            SKPyOwnedNumbers::F32(values)
        }
        (source, _) => {
            let mut values = Vec::with_capacity(source.len());
            match source {
                SKPyOwnedNumbers::F64(items) => values.extend_from_slice(items),
                SKPyOwnedNumbers::F32(items) => {
                    values.extend(items.iter().map(|&v| f64::from(v)));
                }
            }
            SKPyOwnedNumbers::F64(values)
        }
    }
}

/// Convert a SciPy CSR/CSC/COO matrix, borrowing CSR buffers without densifying.
pub(crate) fn sk_py_convert_sparse<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    options: &SKPyConversionOptions,
) -> Result<SKPyConvertedData<'py>, SKPyError> {
    let layout = object
        .getattr("format")
        .and_then(|format| format.extract::<String>())
        .unwrap_or_else(|_| "csr".to_string());
    let (csr, owner) = if layout == "csr" {
        (object.clone(), object.clone())
    } else {
        let converted = object.call_method0("tocsr").map_err(|_| {
            SKPyError::invalid_argument(format!(
                "failed to convert sparse layout '{layout}' to CSR; pass a csr_matrix or call .tocsr() explicitly"
            ))
        })?;
        (converted.clone(), converted)
    };
    let descriptor = SKPyCsrDescriptor::validate_csr_object(py, &csr)?;
    let source_dtype = if descriptor.dtype_name() == "float32" {
        SKFfiDataType::F32
    } else {
        SKFfiDataType::F64
    };
    let cast = source_dtype != options.expected_dtype;
    if cast && !options.may_cast() {
        return Err(SKPyError::invalid_argument(format!(
            "sparse data dtype mismatch: expected {}, found {}; pass cast=True or call .astype before fitting",
            super::numpy_dataset::sk_py_dtype_name(options.expected_dtype),
            descriptor.dtype_name()
        )));
    }

    let stored_zeros = sk_py_count_stored_zeros(py, &descriptor)?;
    let rows = descriptor.rows();
    let columns = descriptor.columns();
    let metadata = SKPyConversionMetadata {
        representation: format!("scipy.{layout}"),
        dtype: super::numpy_dataset::sk_py_dtype_name(options.expected_dtype),
        rows,
        columns_count: columns,
        copied: layout != "csr",
        zero_copy: layout == "csr",
        densified: false,
        cast,
        converted_from: if layout == "csr" { None } else { Some(layout) },
        eager_fallback: Some("sciencekit_ffi dense ABI boundary".to_string()),
        abi_dense_required: true,
        data_pointer: Some(descriptor.data_pointer()),
        indices_pointer: Some(descriptor.indices_pointer()),
        indptr_pointer: Some(descriptor.indptr_pointer()),
        stored_zeros,
        missing_policy: options.on_missing.name().to_string(),
        ..SKPyConversionMetadata::default()
    };
    let view = SKPySparseView {
        _owner: owner,
        descriptor,
    };
    Ok(SKPyConvertedData {
        dtype: options.expected_dtype,
        shape: vec![rows as i64, columns as i64],
        storage: SKPyStorage::Sparse(view),
        metadata,
    })
}

fn sk_py_count_stored_zeros(
    py: Python<'_>,
    descriptor: &SKPyCsrDescriptor,
) -> Result<usize, SKPyError> {
    let data_object = descriptor.data_object().bind(py);
    if descriptor.dtype_name() == "float64" {
        let values = data_object
            .cast::<PyArray1<f64>>()
            .map_err(|_| SKPyError::invalid_argument("failed to read float64 CSR data"))?;
        let readonly = values.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| SKPyError::unsupported("CSR data array must be C-contiguous"))?;
        Ok(slice.iter().filter(|value| **value == 0.0).count())
    } else {
        let values = data_object
            .cast::<PyArray1<f32>>()
            .map_err(|_| SKPyError::invalid_argument("failed to read float32 CSR data"))?;
        let readonly = values.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| SKPyError::unsupported("CSR data array must be C-contiguous"))?;
        Ok(slice.iter().filter(|value| **value == 0.0).count())
    }
}
