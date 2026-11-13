//! Fitted `LinearRegressionModel` `#[pyclass]` and `load` function.

use std::sync::Mutex;

use pyo3::prelude::*;

use super::dataset_arguments::SKPyDatasetKeywords;
use super::input_helpers::sk_py_extract_path;
use super::model_operations::{
    sk_py_run_predict, sk_py_run_save, sk_py_run_score, sk_py_run_transform,
};
use crate::datasets::SKPyConversionMetadata;
use crate::datasets::report::SKPyConversionReport;
use crate::engine::SKPyModelHandle;
use crate::error::sk_py_to_py_err;

/// Immutable, thread-safe fitted linear regression model.
#[pyclass(name = "LinearRegressionModel", module = "sciencekit.linear_model")]
pub(crate) struct SKPyLinearRegressionModel {
    handle: SKPyModelHandle,
    last_conversion: Mutex<Option<SKPyConversionMetadata>>,
    labels: Option<Vec<String>>,
}

impl SKPyLinearRegressionModel {
    pub(crate) fn from_parts(
        handle: SKPyModelHandle,
        conversion: Option<SKPyConversionMetadata>,
        labels: Option<Vec<String>>,
    ) -> Self {
        SKPyLinearRegressionModel {
            handle,
            last_conversion: Mutex::new(conversion),
            labels,
        }
    }

    fn record(&self, metadata: SKPyConversionMetadata) {
        if let Ok(mut guard) = self.last_conversion.lock() {
            *guard = Some(metadata);
        }
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

    /// Report from the most recent prediction/transform conversion.
    #[getter]
    fn last_conversion(&self) -> Option<SKPyConversionReport> {
        self.last_conversion
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
            .map(SKPyConversionReport::from_metadata)
    }

    /// Categorical label table learned from a dictionary target column.
    #[getter]
    fn label_table(&self) -> Option<Vec<String>> {
        self.labels.clone()
    }

    #[pyo3(signature = (
        features,
        *,
        columns = None,
        copy = false,
        cast = false,
        on_missing = "reject",
        fill_value = None,
        output = "numpy"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        columns: Option<Vec<String>>,
        copy: bool,
        cast: bool,
        on_missing: &str,
        fill_value: Option<f64>,
        output: &str,
    ) -> PyResult<Py<PyAny>> {
        let keywords = SKPyDatasetKeywords {
            copy,
            cast,
            on_missing,
            fill_value,
        };
        let (result, metadata) = sk_py_run_predict(
            py,
            &self.handle,
            features,
            columns,
            &keywords,
            output,
            self.labels.clone(),
        )?;
        self.record(metadata);
        Ok(result)
    }

    #[pyo3(signature = (
        features,
        *,
        columns = None,
        copy = false,
        cast = false,
        on_missing = "reject",
        fill_value = None,
        output = "numpy"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        columns: Option<Vec<String>>,
        copy: bool,
        cast: bool,
        on_missing: &str,
        fill_value: Option<f64>,
        output: &str,
    ) -> PyResult<Py<PyAny>> {
        let keywords = SKPyDatasetKeywords {
            copy,
            cast,
            on_missing,
            fill_value,
        };
        let (result, metadata) =
            sk_py_run_transform(py, &self.handle, features, columns, &keywords, output)?;
        self.record(metadata);
        Ok(result)
    }

    #[pyo3(signature = (
        features,
        targets,
        *,
        columns = None,
        copy = false,
        cast = false,
        on_missing = "reject",
        fill_value = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn score(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        columns: Option<Vec<String>>,
        copy: bool,
        cast: bool,
        on_missing: &str,
        fill_value: Option<f64>,
    ) -> PyResult<f64> {
        let keywords = SKPyDatasetKeywords {
            copy,
            cast,
            on_missing,
            fill_value,
        };
        sk_py_run_score(py, &self.handle, features, targets, columns, &keywords)
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
    Ok(SKPyLinearRegressionModel::from_parts(handle, None, None))
}
