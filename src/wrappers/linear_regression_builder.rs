//! `LinearRegression` builder `#[pyclass]` wrapper.

use pyo3::prelude::*;

use super::dataset_arguments::{SKPyDatasetKeywords, sk_py_prepare_fit_inputs};
use super::linear_regression_model::SKPyLinearRegressionModel;
use crate::array::dtype::sk_py_parse_dtype;
use crate::datasets::SKPyConversionMetadata;
use crate::datasets::report::SKPyConversionReport;
use crate::engine::{SKPyBuilderHandle, SKPyEstimatorFamily, sk_py_parse_backend_code};
use crate::error::sk_py_to_py_err;

/// Chainable builder for ordinary least-squares linear regression.
#[pyclass(name = "LinearRegression", module = "sciencekit.linear_model")]
pub(crate) struct SKPyLinearRegression {
    handle: SKPyBuilderHandle,
    last_conversion: std::sync::Mutex<Option<SKPyConversionMetadata>>,
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
        Ok(SKPyLinearRegression {
            handle,
            last_conversion: std::sync::Mutex::new(None),
        })
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

    /// Report from the most recent conversion performed by this builder.
    #[getter]
    fn last_conversion(&self) -> Option<SKPyConversionReport> {
        self.last_conversion
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
            .map(SKPyConversionReport::from_metadata)
    }

    #[pyo3(signature = (
        features,
        targets = None,
        *,
        target = None,
        columns = None,
        copy = false,
        cast = false,
        on_missing = "reject",
        fill_value = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: Option<&Bound<'_, PyAny>>,
        target: Option<String>,
        columns: Option<Vec<String>>,
        copy: bool,
        cast: bool,
        on_missing: &str,
        fill_value: Option<f64>,
    ) -> PyResult<SKPyLinearRegressionModel> {
        let keywords = SKPyDatasetKeywords {
            copy,
            cast,
            on_missing,
            fill_value,
        };
        let prepared = sk_py_prepare_fit_inputs(
            py,
            features,
            targets,
            target,
            &keywords,
            columns,
            self.handle.dtype(),
            false,
        )
        .map_err(|e| sk_py_to_py_err(py, e))?;
        let raw_features = prepared.features.raw_parts();
        let raw_targets = prepared.targets.as_ref().map(|targets| targets.raw_parts());
        let labels = prepared.labels.clone();
        let metadata = prepared.metadata.clone();
        let model_handle = py
            .detach(|| self.handle.fit(&raw_features, raw_targets.as_ref()))
            .map_err(|e| sk_py_to_py_err(py, e))?;
        drop(prepared);
        if let Ok(mut guard) = self.last_conversion.lock() {
            *guard = Some(metadata.clone());
        }
        Ok(SKPyLinearRegressionModel::from_parts(
            model_handle,
            Some(metadata),
            labels,
        ))
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
