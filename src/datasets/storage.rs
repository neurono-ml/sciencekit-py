//! Converted storage: a borrowed NumPy buffer, an owned buffer or a sparse view.

use pyo3::prelude::*;
use sciencekit_ffi::SKFfiDataType;

use super::metadata::SKPyConversionMetadata;
use super::scipy_sparse::SKPySparseView;
use crate::array::dense::SKPyRawDescriptor;
use crate::error::SKPyError;

/// Contiguous `float64`/`float32` values owned by the binding.
#[derive(Debug)]
pub(crate) enum SKPyOwnedNumbers {
    /// Owned `float64` values.
    F64(Vec<f64>),
    /// Owned `float32` values.
    F32(Vec<f32>),
}

impl SKPyOwnedNumbers {
    /// Start an empty buffer for `dtype`.
    pub(crate) fn empty(dtype: SKFfiDataType) -> Self {
        match dtype {
            SKFfiDataType::F32 => SKPyOwnedNumbers::F32(Vec::new()),
            _ => SKPyOwnedNumbers::F64(Vec::new()),
        }
    }

    /// Number of stored elements.
    pub(crate) fn len(&self) -> usize {
        match self {
            SKPyOwnedNumbers::F64(values) => values.len(),
            SKPyOwnedNumbers::F32(values) => values.len(),
        }
    }

    /// Pointer to the first element of the heap buffer.
    pub(crate) fn pointer(&self) -> usize {
        match self {
            SKPyOwnedNumbers::F64(values) => values.as_ptr() as usize,
            SKPyOwnedNumbers::F32(values) => values.as_ptr() as usize,
        }
    }

    /// Append one `f64` value, narrowing when the buffer is `float32`.
    pub(crate) fn push_f64(&mut self, value: f64) {
        match self {
            SKPyOwnedNumbers::F64(values) => values.push(value),
            SKPyOwnedNumbers::F32(values) => values.push(value as f32),
        }
    }

    /// Append one `f32` value, widening when the buffer is `float64`.
    pub(crate) fn push_f32(&mut self, value: f32) {
        match self {
            SKPyOwnedNumbers::F64(values) => values.push(value as f64),
            SKPyOwnedNumbers::F32(values) => values.push(value),
        }
    }

    /// Extend the buffer with row-major values already in the target precision.
    pub(crate) fn extend_from(&mut self, other: &SKPyOwnedNumbers) {
        match (self, other) {
            (SKPyOwnedNumbers::F64(target), SKPyOwnedNumbers::F64(source)) => {
                target.extend_from_slice(source);
            }
            (SKPyOwnedNumbers::F32(target), SKPyOwnedNumbers::F32(source)) => {
                target.extend_from_slice(source);
            }
            (SKPyOwnedNumbers::F64(target), SKPyOwnedNumbers::F32(source)) => {
                target.extend(source.iter().map(|&value| f64::from(value)));
            }
            (SKPyOwnedNumbers::F32(target), SKPyOwnedNumbers::F64(source)) => {
                target.extend(source.iter().map(|&value| value as f32));
            }
        }
    }
}

/// Where the converted values live.
pub(crate) enum SKPyStorage<'py> {
    /// Buffer owned by a Python object, borrowed zero-copy.
    Borrowed {
        /// Owner that keeps the buffer alive.
        _owner: Bound<'py, PyAny>,
        /// Pointer to the first element.
        pointer: usize,
    },
    /// Buffer owned by the binding.
    Owned(SKPyOwnedNumbers),
    /// SciPy CSR buffers borrowed behind a sparse view.
    Sparse(SKPySparseView<'py>),
}

/// One converted dataset plus the metadata describing the conversion.
pub(crate) struct SKPyConvertedData<'py> {
    pub(crate) dtype: SKFfiDataType,
    pub(crate) shape: Vec<i64>,
    pub(crate) storage: SKPyStorage<'py>,
    pub(crate) metadata: SKPyConversionMetadata,
}

impl SKPyConvertedData<'_> {
    /// Row count of the converted data.
    pub(crate) fn rows(&self) -> usize {
        self.shape[0] as usize
    }

    /// Build an ABI-ready view, densifying a sparse input only at the boundary.
    ///
    /// Consumes the converted data so the returned view owns the values and the
    /// Python object that keeps a borrowed buffer alive for the ABI call.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_abi(self, py: Python<'_>) -> Result<SKPyAbiArray, SKPyError> {
        match self.storage {
            SKPyStorage::Borrowed {
                _owner, pointer, ..
            } => Ok(SKPyAbiArray {
                dtype: self.dtype,
                shape: self.shape,
                pointer,
                _owned: None,
                _owner: Some(_owner.unbind()),
            }),
            SKPyStorage::Owned(numbers) => {
                let pointer = numbers.pointer();
                Ok(SKPyAbiArray {
                    dtype: self.dtype,
                    shape: self.shape,
                    pointer,
                    _owned: Some(numbers),
                    _owner: None,
                })
            }
            SKPyStorage::Sparse(view) => {
                let dense = view.densify(py, self.dtype)?;
                let pointer = dense.pointer();
                Ok(SKPyAbiArray {
                    dtype: self.dtype,
                    shape: self.shape,
                    pointer,
                    _owned: Some(dense),
                    _owner: None,
                })
            }
        }
    }
}

/// ABI-ready pointer plus the owned buffer that keeps it valid.
pub(crate) struct SKPyAbiArray {
    dtype: SKFfiDataType,
    shape: Vec<i64>,
    pointer: usize,
    _owned: Option<SKPyOwnedNumbers>,
    _owner: Option<Py<PyAny>>,
}

impl SKPyAbiArray {
    /// Raw `Send` descriptor parts safe to pass into `py.detach`.
    pub(crate) fn raw_parts(&self) -> SKPyRawDescriptor {
        SKPyRawDescriptor::from_parts(self.dtype, self.shape.clone(), self.pointer)
    }
}
