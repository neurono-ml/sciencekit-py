//! PySpark DataFrames collected through Arrow on the driver.

use pyo3::prelude::*;

use super::SKPyFrameImport;
use super::options::SKPyConversionOptions;
use crate::error::SKPyError;

/// Convert a PySpark DataFrame through `toArrow()`, falling back to
/// `_collect_as_arrow()` and finally to `toPandas()`.
pub(crate) fn sk_py_convert_pyspark<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    options: &SKPyConversionOptions,
    vector_only: bool,
) -> Result<SKPyFrameImport<'py>, SKPyError> {
    if let Some(table) = sk_py_call_optional(object, "toArrow")? {
        return super::sk_py_convert_arrow_object(
            py,
            &table,
            options,
            "pyspark.dataframe",
            Some("pyspark.toArrow (driver collect)"),
            vector_only,
        );
    }
    if let Some(table) = sk_py_call_optional(object, "_collect_as_arrow")? {
        return super::sk_py_convert_arrow_object(
            py,
            &table,
            options,
            "pyspark.dataframe",
            Some("pyspark._collect_as_arrow (driver collect)"),
            vector_only,
        );
    }
    let frame = object.call_method0("toPandas").map_err(|error| {
        SKPyError::invalid_argument(format!(
            "PySpark DataFrame exposes neither toArrow(), _collect_as_arrow() nor toPandas(): {error}"
        ))
    })?;
    let mut import = super::sk_py_convert_frame(py, &frame, options)?;
    import.features.metadata.representation = "pyspark.dataframe".to_string();
    import.features.metadata.eager_fallback = Some("pyspark.toPandas".to_string());
    if let Some(target) = import.target.as_mut() {
        target.metadata.representation = "pyspark.dataframe".to_string();
        target.metadata.eager_fallback = Some("pyspark.toPandas".to_string());
    }
    Ok(import)
}

fn sk_py_call_optional<'py>(
    object: &Bound<'py, PyAny>,
    method: &str,
) -> Result<Option<Bound<'py, PyAny>>, SKPyError> {
    if !object.hasattr(method).unwrap_or(false) {
        return Ok(None);
    }
    let result = object.call_method0(method).map_err(|error| {
        SKPyError::invalid_argument(format!("PySpark `{method}()` failed: {error}"))
    })?;
    Ok(Some(result))
}
