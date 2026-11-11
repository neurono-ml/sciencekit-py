//! Python `DenseDescriptor` class wrapping `SKPyBorrowedArray`.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::dense::SKPyBorrowedArray;
use super::dtype::sk_py_dtype_label;
use crate::error::sk_py_to_py_err;

/// Inspectable Python wrapper over a validated dense NumPy array descriptor.
#[pyclass(name = "DenseDescriptor", module = "sciencekit", frozen)]
pub(crate) struct SKPyDenseDescriptor {
    dtype_name: &'static str,
    shape: Vec<usize>,
    strides: Vec<usize>,
    length: usize,
    data_pointer: usize,
    copied: bool,
}

#[pymethods]
impl SKPyDenseDescriptor {
    #[new]
    #[pyo3(signature = (array, *, copy = false))]
    fn new(py: Python<'_>, array: &Bound<'_, PyAny>, copy: bool) -> PyResult<Self> {
        let borrowed = SKPyBorrowedArray::from_any(py, array, None, None, copy)
            .map_err(|error| sk_py_to_py_err(py, error))?;
        Ok(SKPyDenseDescriptor {
            dtype_name: sk_py_dtype_label(borrowed.dtype),
            shape: borrowed.shape.iter().map(|&d| d as usize).collect(),
            strides: borrowed.strides.iter().map(|&s| s as usize).collect(),
            length: borrowed.length as usize,
            data_pointer: borrowed.data_pointer,
            copied: borrowed.copied,
        })
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype_name
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.strides)
    }

    #[getter]
    fn length(&self) -> usize {
        self.length
    }

    #[getter]
    fn data_pointer(&self) -> usize {
        self.data_pointer
    }

    #[getter]
    fn copied(&self) -> bool {
        self.copied
    }
}
