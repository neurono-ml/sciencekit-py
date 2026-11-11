//! Fitted `LinearRegressionModel` `#[pyclass]` and `load` function.

use pyo3::prelude::*;

use super::input_helpers::sk_py_extract_path;
use super::model_operations::{
    sk_py_run_predict, sk_py_run_save, sk_py_run_score, sk_py_run_transform,
};
use crate::engine::SKPyModelHandle;
use crate::error::sk_py_to_py_err;

/// Immutable, thread-safe fitted linear regression model.
#[pyclass(name = "LinearRegressionModel", module = "sciencekit.linear_model")]
pub(crate) struct SKPyLinearRegressionModel {
    handle: SKPyModelHandle,
}

impl SKPyLinearRegressionModel {
    pub(crate) fn from_handle(handle: SKPyModelHandle) -> Self {
        SKPyLinearRegressionModel { handle }
    }
}

#[pymethods]
impl SKPyLinearRegressionModel {
    #[staticmethod]
    fn load(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Self> {
        load(py, path)
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

    #[pyo3(signature = (features, *, copy = false))]
    fn predict(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<Py<PyAny>> {
        sk_py_run_predict(py, &self.handle, features, copy)
    }

    #[pyo3(signature = (features, *, copy = false))]
    fn transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<Py<PyAny>> {
        sk_py_run_transform(py, &self.handle, features, copy)
    }

    #[pyo3(signature = (features, targets, *, copy = false))]
    fn score(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<f64> {
        sk_py_run_score(py, &self.handle, features, targets, copy)
    }

    fn save(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        sk_py_run_save(py, &self.handle, path)
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

/// Load a persisted fitted `f64` linear regression model from `path`.
#[pyfunction]
pub(crate) fn load(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<SKPyLinearRegressionModel> {
    let path_string = sk_py_extract_path(path).map_err(|e| sk_py_to_py_err(py, e))?;
    let handle = py
        .detach(|| SKPyModelHandle::load(&path_string))
        .map_err(|e| sk_py_to_py_err(py, e))?;
    Ok(SKPyLinearRegressionModel::from_handle(handle))
}
