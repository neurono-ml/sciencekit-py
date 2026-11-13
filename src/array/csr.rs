//! Compressed Sparse Row (CSR) descriptor class and SciPy duck-typing support.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::csr_validation::{
    sk_py_expand_csr, sk_py_extract_non_negative_indices, sk_py_validate_csr_data,
};
use crate::error::{SKPyError, sk_py_to_py_err};

/// Validated CSR sparse matrix descriptor over NumPy `data`, `indices` and `indptr` buffers.
#[pyclass(name = "CsrDescriptor", module = "sciencekit", frozen)]
pub(crate) struct SKPyCsrDescriptor {
    data: Py<PyAny>,
    dtype_name: &'static str,
    index_dtype_name: &'static str,
    rows: usize,
    columns: usize,
    nnz: usize,
    data_pointer: usize,
    indices_pointer: usize,
    indptr_pointer: usize,
    column_indices: Vec<usize>,
    row_pointers: Vec<usize>,
}

#[pymethods]
impl SKPyCsrDescriptor {
    #[new]
    #[pyo3(signature = (data, indices, indptr, shape))]
    fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        indptr: &Bound<'_, PyAny>,
        shape: (isize, isize),
    ) -> PyResult<Self> {
        Self::validate_parts(py, data, indices, indptr, shape)
            .map_err(|error| sk_py_to_py_err(py, error))
    }

    #[staticmethod]
    fn from_csr(py: Python<'_>, matrix: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::validate_csr_object(py, matrix).map_err(|error| sk_py_to_py_err(py, error))
    }

    #[getter]
    pub(crate) fn dtype(&self) -> &'static str {
        self.dtype_name
    }

    #[getter]
    fn index_dtype(&self) -> &'static str {
        self.index_dtype_name
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [self.rows, self.columns])
    }

    #[getter]
    pub(crate) fn nnz(&self) -> usize {
        self.nnz
    }

    #[getter]
    pub(crate) fn data_pointer(&self) -> usize {
        self.data_pointer
    }

    #[getter]
    pub(crate) fn indices_pointer(&self) -> usize {
        self.indices_pointer
    }

    #[getter]
    pub(crate) fn indptr_pointer(&self) -> usize {
        self.indptr_pointer
    }

    fn to_dense(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let bound_data = self.data.bind(py);
        if self.dtype_name == "float64" {
            let values = bound_data.cast::<PyArray1<f64>>()?;
            let slice = values.readonly();
            let data_slice = slice
                .as_slice()
                .map_err(|err| sk_py_to_py_err(py, SKPyError::invalid_argument(err.to_string())))?;
            let dense = sk_py_expand_csr(
                self.rows,
                self.columns,
                &self.row_pointers,
                &self.column_indices,
                data_slice,
                0.0_f64,
            );
            Ok(PyArray1::from_vec(py, dense)
                .reshape([self.rows, self.columns])?
                .into_any()
                .unbind())
        } else {
            let values = bound_data.cast::<PyArray1<f32>>()?;
            let slice = values.readonly();
            let data_slice = slice
                .as_slice()
                .map_err(|err| sk_py_to_py_err(py, SKPyError::invalid_argument(err.to_string())))?;
            let dense = sk_py_expand_csr(
                self.rows,
                self.columns,
                &self.row_pointers,
                &self.column_indices,
                data_slice,
                0.0_f32,
            );
            Ok(PyArray1::from_vec(py, dense)
                .reshape([self.rows, self.columns])?
                .into_any()
                .unbind())
        }
    }
}

impl SKPyCsrDescriptor {
    /// Owned `data` buffer object.
    pub(crate) fn data_object(&self) -> &Py<PyAny> {
        &self.data
    }

    /// Scalar precision name (`float32`/`float64`).
    pub(crate) fn dtype_name(&self) -> &'static str {
        self.dtype_name
    }

    /// Row count.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Column count.
    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    /// Row pointer buffer values.
    pub(crate) fn row_pointers(&self) -> &[usize] {
        &self.row_pointers
    }

    /// Column index buffer values.
    pub(crate) fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    pub(crate) fn validate_csr_object(
        py: Python<'_>,
        matrix: &Bound<'_, PyAny>,
    ) -> Result<SKPyCsrDescriptor, SKPyError> {
        let data = matrix
            .getattr("data")
            .map_err(|_| SKPyError::invalid_argument("CSR object is missing '.data'"))?;
        let indices = matrix
            .getattr("indices")
            .map_err(|_| SKPyError::invalid_argument("CSR object is missing '.indices'"))?;
        let indptr = matrix
            .getattr("indptr")
            .map_err(|_| SKPyError::invalid_argument("CSR object is missing '.indptr'"))?;
        let shape: (isize, isize) = matrix
            .getattr("shape")
            .and_then(|attr| attr.extract())
            .map_err(|_| SKPyError::invalid_argument("CSR object '.shape' must be a 2-tuple"))?;
        Self::validate_parts(py, &data, &indices, &indptr, shape)
    }

    pub(crate) fn validate_parts(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        indptr: &Bound<'_, PyAny>,
        shape: (isize, isize),
    ) -> Result<SKPyCsrDescriptor, SKPyError> {
        if shape.0 <= 0 || shape.1 <= 0 {
            return Err(SKPyError::invalid_argument(format!(
                "CSR shape must have positive dimensions, found ({}, {})",
                shape.0, shape.1
            )));
        }
        let (rows, columns) = (shape.0 as usize, shape.1 as usize);
        let (dtype_name, nnz, data_pointer) = sk_py_validate_csr_data(py, data)?;
        let (ind_dtype, column_indices, indices_pointer) =
            sk_py_extract_non_negative_indices(py, indices, "indices")?;
        let (ptr_dtype, row_pointers, indptr_pointer) =
            sk_py_extract_non_negative_indices(py, indptr, "indptr")?;
        if ind_dtype != ptr_dtype {
            return Err(SKPyError::invalid_argument(format!(
                "CSR index dtype mismatch: indices is {ind_dtype} while indptr is {ptr_dtype}"
            )));
        }
        if column_indices.len() != nnz || row_pointers.len() != rows + 1 {
            return Err(SKPyError::invalid_argument(
                "CSR indices/indptr lengths disagree with nnz or shape",
            ));
        }
        if row_pointers[0] != 0
            || row_pointers[rows] != nnz
            || row_pointers.windows(2).any(|w| w[0] > w[1])
        {
            return Err(SKPyError::invalid_argument(
                "CSR indptr must start at 0, be monotonic, and end at nnz",
            ));
        }
        if let Some(&bad_col) = column_indices.iter().find(|&&col| col >= columns) {
            return Err(SKPyError::invalid_argument(format!(
                "CSR column index {bad_col} is out of bounds for {columns} columns"
            )));
        }
        Ok(SKPyCsrDescriptor {
            data: data.clone().unbind(),
            dtype_name,
            index_dtype_name: ind_dtype,
            rows,
            columns,
            nnz,
            data_pointer,
            indices_pointer,
            indptr_pointer,
            column_indices,
            row_pointers,
        })
    }
}
