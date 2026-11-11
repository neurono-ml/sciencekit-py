//! `LinearRegression` builder `#[pyclass]` wrapper.

use pyo3::prelude::*;

use super::input_helpers::{sk_py_prepare_matrix, sk_py_prepare_vector};
use super::linear_regression_model::SKPyLinearRegressionModel;
use crate::array::dtype::sk_py_parse_dtype;
use crate::engine::{SKPyBuilderHandle, SKPyEstimatorFamily, sk_py_parse_backend_code};
use crate::error::sk_py_to_py_err;

/// Chainable builder for ordinary least-squares linear regression.
#[pyclass(name = "LinearRegression", module = "sciencekit.linear_model")]
pub(crate) struct SKPyLinearRegression {
    handle: SKPyBuilderHandle,
}

#[pymethods]
impl SKPyLinearRegression {
    #[new]
    #[pyo3(signature = (dtype = None, *, include_intercept = true, backend = "faer"))]
    fn new(
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        include_intercept: bool,
        backend: &str,
    ) -> PyResult<Self> {
        let data_type = match dtype {
            Some(value) => sk_py_parse_dtype(py, value).map_err(|e| sk_py_to_py_err(py, e))?,
            None => sciencekit_ffi::SKFfiDataType::F64,
        };
        let backend_code = sk_py_parse_backend_code(backend).map_err(|e| sk_py_to_py_err(py, e))?;
        let handle = py
            .detach(|| {
                let builder =
                    SKPyBuilderHandle::create(SKPyEstimatorFamily::LinearRegression, data_type)?;
                builder.set_include_intercept(include_intercept)?;
                builder.set_backend(backend_code)?;
                Ok(builder)
            })
            .map_err(|e| sk_py_to_py_err(py, e))?;
        Ok(SKPyLinearRegression { handle })
    }

    #[getter]
    fn family(&self) -> &'static str {
        self.handle.family().name()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.handle.dtype_name()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.handle.is_closed()
    }

    fn include_intercept(slf: Bound<'_, Self>, enabled: bool) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let inner = &slf.borrow().handle;
        py.detach(|| inner.set_include_intercept(enabled))
            .map_err(|e| sk_py_to_py_err(py, e))?;
        Ok(slf)
    }

    fn backend<'py>(slf: Bound<'py, Self>, backend: &str) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let code = sk_py_parse_backend_code(backend).map_err(|e| sk_py_to_py_err(py, e))?;
        let inner = &slf.borrow().handle;
        py.detach(|| inner.set_backend(code))
            .map_err(|e| sk_py_to_py_err(py, e))?;
        Ok(slf)
    }

    #[pyo3(signature = (features, targets, *, copy = false))]
    fn fit(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<SKPyLinearRegressionModel> {
        let expected_dtype = self.handle.dtype();
        let borrowed_features = sk_py_prepare_matrix(py, features, expected_dtype, copy)
            .map_err(|e| sk_py_to_py_err(py, e))?;
        let borrowed_targets = sk_py_prepare_vector(py, targets, expected_dtype, copy)
            .map_err(|e| sk_py_to_py_err(py, e))?;
        let raw_features = borrowed_features.as_raw_parts();
        let raw_targets = borrowed_targets.as_raw_parts();
        let model_handle = py
            .detach(|| self.handle.fit(&raw_features, Some(&raw_targets)))
            .map_err(|e| sk_py_to_py_err(py, e))?;
        drop((borrowed_features, borrowed_targets));
        Ok(SKPyLinearRegressionModel::from_handle(model_handle))
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| self.handle.close());
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type = None, _exc_val = None, _exc_tb = None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.close(py);
        false
    }
}
