//! `StandardScaler` builder `#[pyclass]` wrapper.

use pyo3::prelude::*;

use super::dataset_arguments::{SKPyDatasetKeywords, sk_py_prepare_fit_inputs};
use super::standard_scaler_model::SKPyStandardScalerModel;
use crate::array::dtype::sk_py_parse_dtype;
use crate::datasets::SKPyConversionMetadata;
use crate::datasets::report::SKPyConversionReport;
use crate::engine::{SKPyBuilderHandle, SKPyEstimatorFamily, sk_py_parse_backend_code};
use crate::error::sk_py_to_py_err;

/// Chainable builder for column-wise feature standardization (`StandardScaler`).
#[pyclass(name = "StandardScaler", module = "sciencekit.preprocessing")]
pub(crate) struct SKPyStandardScaler {
    handle: SKPyBuilderHandle,
    last_conversion: std::sync::Mutex<Option<SKPyConversionMetadata>>,
}

#[pymethods]
impl SKPyStandardScaler {
    #[new]
    #[pyo3(signature = (dtype = None, *, with_mean = true, with_std = true, backend = "faer"))]
    fn new(
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        with_mean: bool,
        with_std: bool,
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
                    SKPyBuilderHandle::create(SKPyEstimatorFamily::StandardScaler, data_type)?;
                builder.set_with_mean(with_mean)?;
                builder.set_with_std(with_std)?;
                builder.set_backend(backend_code)?;
                Ok(builder)
            })
            .map_err(|e| sk_py_to_py_err(py, e))?;
        Ok(SKPyStandardScaler {
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

    fn with_mean(slf: Bound<'_, Self>, enabled: bool) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let inner = &slf.borrow().handle;
        py.detach(|| inner.set_with_mean(enabled))
            .map_err(|e| sk_py_to_py_err(py, e))?;
        Ok(slf)
    }

    fn with_std(slf: Bound<'_, Self>, enabled: bool) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let inner = &slf.borrow().handle;
        py.detach(|| inner.set_with_std(enabled))
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
    ) -> PyResult<SKPyStandardScalerModel> {
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
            true,
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
        Ok(SKPyStandardScalerModel::from_parts(
            model_handle,
            Some(metadata),
            labels,
        ))
    }

    #[pyo3(signature = (
        features,
        *,
        target = None,
        columns = None,
        copy = false,
        cast = false,
        on_missing = "reject",
        fill_value = None,
        output = "numpy"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        target: Option<String>,
        columns: Option<Vec<String>>,
        copy: bool,
        cast: bool,
        on_missing: &str,
        fill_value: Option<f64>,
        output: &str,
    ) -> PyResult<Py<PyAny>> {
        let transform_columns = columns.clone();
        let fitted = self.fit(
            py, features, None, target, columns, copy, cast, on_missing, fill_value,
        )?;
        let result = fitted.transform(
            py,
            features,
            transform_columns,
            copy,
            cast,
            on_missing,
            fill_value,
            output,
        );
        fitted.close(py);
        result
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
