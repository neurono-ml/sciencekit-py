//! `StandardScaler` builder `#[pyclass]` wrapper.

use pyo3::prelude::*;

use super::input_helpers::{sk_py_prepare_matrix, sk_py_prepare_vector};
use super::standard_scaler_model::SKPyStandardScalerModel;
use crate::array::dtype::sk_py_parse_dtype;
use crate::engine::{SKPyBuilderHandle, SKPyEstimatorFamily, sk_py_parse_backend_code};
use crate::error::sk_py_to_py_err;

/// Chainable builder for column-wise feature standardization (`StandardScaler`).
#[pyclass(name = "StandardScaler", module = "sciencekit.preprocessing")]
pub(crate) struct SKPyStandardScaler {
    handle: SKPyBuilderHandle,
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
        Ok(SKPyStandardScaler { handle })
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

    #[pyo3(signature = (features, targets = None, *, copy = false))]
    fn fit(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: Option<&Bound<'_, PyAny>>,
        copy: bool,
    ) -> PyResult<SKPyStandardScalerModel> {
        let expected_dtype = self.handle.dtype();
        let borrowed_features = sk_py_prepare_matrix(py, features, expected_dtype, copy)
            .map_err(|e| sk_py_to_py_err(py, e))?;
        let borrowed_targets = match targets {
            Some(target_obj) if !target_obj.is_none() => Some(
                sk_py_prepare_vector(py, target_obj, expected_dtype, copy)
                    .map_err(|e| sk_py_to_py_err(py, e))?,
            ),
            _ => None,
        };
        let raw_features = borrowed_features.as_raw_parts();
        let raw_targets = borrowed_targets.as_ref().map(|b| b.as_raw_parts());
        let model_handle = py
            .detach(|| self.handle.fit(&raw_features, raw_targets.as_ref()))
            .map_err(|e| sk_py_to_py_err(py, e))?;
        drop((borrowed_features, borrowed_targets));
        Ok(SKPyStandardScalerModel::from_handle(model_handle))
    }

    #[pyo3(signature = (features, *, copy = false))]
    fn fit_transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<Py<PyAny>> {
        let fitted = self.fit(py, features, None, copy)?;
        let result = fitted.transform(py, features, copy);
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
