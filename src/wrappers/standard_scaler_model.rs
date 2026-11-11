//! Fitted `StandardScalerModel` `#[pyclass]` wrapper.

use pyo3::prelude::*;

use super::model_operations::{
    sk_py_run_predict, sk_py_run_save, sk_py_run_score, sk_py_run_transform,
};
use crate::engine::SKPyModelHandle;

/// Immutable, thread-safe fitted `StandardScaler` model.
#[pyclass(name = "StandardScalerModel", module = "sciencekit.preprocessing")]
pub(crate) struct SKPyStandardScalerModel {
    handle: SKPyModelHandle,
}

impl SKPyStandardScalerModel {
    pub(crate) fn from_handle(handle: SKPyModelHandle) -> Self {
        SKPyStandardScalerModel { handle }
    }
}

#[pymethods]
impl SKPyStandardScalerModel {
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
    pub(crate) fn transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<Py<PyAny>> {
        sk_py_run_transform(py, &self.handle, features, copy)
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

    pub(crate) fn close(&self, py: Python<'_>) {
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
