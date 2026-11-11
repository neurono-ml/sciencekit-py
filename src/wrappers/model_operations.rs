//! Shared GIL-detached execution helpers for fitted model operations.

use pyo3::prelude::*;

use super::input_helpers::{sk_py_extract_path, sk_py_prepare_matrix, sk_py_prepare_vector};
use crate::engine::SKPyModelHandle;
use crate::error::sk_py_to_py_err;

pub(crate) fn sk_py_run_predict(
    py: Python<'_>,
    handle: &SKPyModelHandle,
    features: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    let borrowed = sk_py_prepare_matrix(py, features, handle.dtype(), copy)
        .map_err(|e| sk_py_to_py_err(py, e))?;
    let raw = borrowed.as_raw_parts();
    let owned = py
        .detach(|| handle.predict(&raw))
        .map_err(|e| sk_py_to_py_err(py, e))?;
    drop(borrowed);
    owned.into_py_array(py)
}

pub(crate) fn sk_py_run_transform(
    py: Python<'_>,
    handle: &SKPyModelHandle,
    features: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    let borrowed = sk_py_prepare_matrix(py, features, handle.dtype(), copy)
        .map_err(|e| sk_py_to_py_err(py, e))?;
    let raw = borrowed.as_raw_parts();
    let owned = py
        .detach(|| handle.transform(&raw))
        .map_err(|e| sk_py_to_py_err(py, e))?;
    drop(borrowed);
    owned.into_py_array(py)
}

pub(crate) fn sk_py_run_score(
    py: Python<'_>,
    handle: &SKPyModelHandle,
    features: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<f64> {
    let borrowed_features = sk_py_prepare_matrix(py, features, handle.dtype(), copy)
        .map_err(|e| sk_py_to_py_err(py, e))?;
    let borrowed_targets = sk_py_prepare_vector(py, targets, handle.dtype(), copy)
        .map_err(|e| sk_py_to_py_err(py, e))?;
    let raw_features = borrowed_features.as_raw_parts();
    let raw_targets = borrowed_targets.as_raw_parts();
    let score = py
        .detach(|| handle.score(&raw_features, &raw_targets))
        .map_err(|e| sk_py_to_py_err(py, e))?;
    drop((borrowed_features, borrowed_targets));
    Ok(score)
}

pub(crate) fn sk_py_run_save(
    py: Python<'_>,
    handle: &SKPyModelHandle,
    path: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let path_string = sk_py_extract_path(path).map_err(|e| sk_py_to_py_err(py, e))?;
    py.detach(|| handle.save(&path_string))
        .map_err(|e| sk_py_to_py_err(py, e))
}
