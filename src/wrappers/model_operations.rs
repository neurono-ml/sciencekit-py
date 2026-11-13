//! Shared GIL-detached execution helpers for fitted model operations.

use pyo3::prelude::*;

use super::dataset_arguments::{
    SKPyDatasetKeywords, sk_py_prepare_features, sk_py_prepare_targets,
};
use super::input_helpers::sk_py_extract_path;
use crate::datasets::SKPyConversionMetadata;
use crate::datasets::results::{SKPyResultFormat, sk_py_format_result};
use crate::engine::SKPyModelHandle;
use crate::error::sk_py_to_py_err;

/// Run `predict`, returning the formatted result and the input conversion report.
pub(crate) fn sk_py_run_predict(
    py: Python<'_>,
    handle: &SKPyModelHandle,
    features: &Bound<'_, PyAny>,
    columns: Option<Vec<String>>,
    keywords: &SKPyDatasetKeywords<'_>,
    output: &str,
    labels: Option<Vec<String>>,
) -> PyResult<(Py<PyAny>, SKPyConversionMetadata)> {
    let prepared = sk_py_prepare_features(py, features, columns, keywords, handle.dtype())
        .map_err(|error| sk_py_to_py_err(py, error))?;
    let raw = prepared.features.raw_parts();
    let owned = py
        .detach(|| handle.predict(&raw))
        .map_err(|e| sk_py_to_py_err(py, e))?;
    let metadata = prepared.metadata.clone();
    drop(prepared);
    let format = SKPyResultFormat::parse(output)
        .map_err(|error| sk_py_to_py_err(py, error))?
        .resolve(&metadata.representation);
    let result = owned.into_py_array(py)?;
    Ok((sk_py_format_result(py, result, labels, format)?, metadata))
}

/// Run `transform`, returning the formatted result and the input conversion report.
pub(crate) fn sk_py_run_transform(
    py: Python<'_>,
    handle: &SKPyModelHandle,
    features: &Bound<'_, PyAny>,
    columns: Option<Vec<String>>,
    keywords: &SKPyDatasetKeywords<'_>,
    output: &str,
) -> PyResult<(Py<PyAny>, SKPyConversionMetadata)> {
    let prepared = sk_py_prepare_features(py, features, columns, keywords, handle.dtype())
        .map_err(|error| sk_py_to_py_err(py, error))?;
    let raw = prepared.features.raw_parts();
    let owned = py
        .detach(|| handle.transform(&raw))
        .map_err(|e| sk_py_to_py_err(py, e))?;
    let metadata = prepared.metadata.clone();
    drop(prepared);
    let format = SKPyResultFormat::parse(output)
        .map_err(|error| sk_py_to_py_err(py, error))?
        .resolve(&metadata.representation);
    let result = owned.into_py_array(py)?;
    Ok((sk_py_format_result(py, result, None, format)?, metadata))
}

/// Run `score` with prepared feature and target blocks.
pub(crate) fn sk_py_run_score(
    py: Python<'_>,
    handle: &SKPyModelHandle,
    features: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    columns: Option<Vec<String>>,
    keywords: &SKPyDatasetKeywords<'_>,
) -> PyResult<f64> {
    let prepared_features = sk_py_prepare_features(py, features, columns, keywords, handle.dtype())
        .map_err(|error| sk_py_to_py_err(py, error))?;
    let prepared_targets = sk_py_prepare_targets(py, targets, keywords, handle.dtype())
        .map_err(|error| sk_py_to_py_err(py, error))?;
    let raw_features = prepared_features.features.raw_parts();
    let raw_targets = prepared_targets.features.raw_parts();
    let score = py
        .detach(|| handle.score(&raw_features, &raw_targets))
        .map_err(|e| sk_py_to_py_err(py, e))?;
    drop((prepared_features, prepared_targets));
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
