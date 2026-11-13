//! Polars `LazyFrame` collected eagerly before Arrow consumption.

use pyo3::prelude::*;

use super::SKPyFrameImport;
use super::options::SKPyConversionOptions;
use super::storage::SKPyConvertedData;
use crate::error::SKPyError;

/// Collect a Polars `LazyFrame` and consume the resulting Arrow stream.
pub(crate) fn sk_py_convert_polars_lazy<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    options: &SKPyConversionOptions,
    vector_only: bool,
) -> Result<SKPyFrameImport<'py>, SKPyError> {
    let collected = object.call_method0("collect").map_err(|error| {
        SKPyError::invalid_argument(format!(
            "failed to collect the Polars LazyFrame: {error}; pass an already collected polars.DataFrame for streaming consumption"
        ))
    })?;
    let mut import = super::sk_py_convert_arrow_object(
        py,
        &collected,
        options,
        "polars.lazyframe",
        Some("polars.lazyframe.collect"),
        vector_only,
    )?;
    sk_py_mark_eager(&mut import.features);
    if let Some(target) = import.target.as_mut() {
        sk_py_mark_eager(target);
    }
    Ok(import)
}

fn sk_py_mark_eager(data: &mut SKPyConvertedData<'_>) {
    data.metadata.eager_fallback = Some("polars.lazyframe.collect".to_string());
}
