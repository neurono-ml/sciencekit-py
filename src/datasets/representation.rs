//! Python representation detection and the dispatcher's public type catalogue.

use pyo3::prelude::*;

/// Every representation accepted by the dispatcher, in detection order.
pub(crate) const SK_PY_SUPPORTED_REPRESENTATIONS: [&str; 10] = [
    "numpy.ndarray",
    "scipy.sparse.csr_matrix",
    "scipy.sparse.csc_matrix",
    "scipy.sparse.coo_matrix",
    "pandas.DataFrame",
    "pandas.Series",
    "pyarrow.Table/RecordBatch/RecordBatchReader",
    "polars.DataFrame/LazyFrame",
    "dask.dataframe.DataFrame",
    "pyspark.sql.DataFrame (driver collect)",
];

/// Recognized input families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SKPyRepresentationFamily {
    /// NumPy `ndarray`.
    Numpy,
    /// SciPy sparse matrix in any supported layout.
    ScipySparse,
    /// pandas frame or series that lacks the Arrow PyCapsule protocol.
    LegacyPandas,
    /// Dask DataFrame.
    Dask,
    /// PySpark DataFrame.
    PySpark,
    /// Polars `LazyFrame` that must be collected before conversion.
    PolarsLazy,
    /// Any object exporting the Arrow PyCapsule protocol.
    ArrowProtocol,
}

/// Human-readable list of accepted representations for error messages.
pub(crate) fn sk_py_supported_representations() -> Vec<String> {
    SK_PY_SUPPORTED_REPRESENTATIONS
        .iter()
        .map(|name| (*name).to_string())
        .collect()
}

/// Classify a Python object into a representation family.
pub(crate) fn sk_py_classify(object: &Bound<'_, PyAny>) -> Option<SKPyRepresentationFamily> {
    if object.is_instance_of::<numpy::PyUntypedArray>() {
        return Some(SKPyRepresentationFamily::Numpy);
    }
    if sk_py_is_scipy_sparse(object) {
        return Some(SKPyRepresentationFamily::ScipySparse);
    }
    let module = sk_py_module_name(object);
    if module.starts_with("dask.") && sk_py_has(object, "npartitions") {
        return Some(SKPyRepresentationFamily::Dask);
    }
    if module.starts_with("pyspark.") && sk_py_has(object, "toArrow") {
        return Some(SKPyRepresentationFamily::PySpark);
    }
    if (module == "polars" || module.starts_with("polars."))
        && sk_py_type_name(object) == "LazyFrame"
    {
        return Some(SKPyRepresentationFamily::PolarsLazy);
    }
    if sk_py_has(object, "__arrow_c_stream__") || sk_py_has(object, "__arrow_c_array__") {
        return Some(SKPyRepresentationFamily::ArrowProtocol);
    }
    if (module == "pandas" || module.starts_with("pandas.")) && sk_py_has(object, "to_numpy") {
        return Some(SKPyRepresentationFamily::LegacyPandas);
    }
    None
}

/// Whether the object advertises a SciPy-compatible sparse layout.
pub(crate) fn sk_py_is_scipy_sparse(object: &Bound<'_, PyAny>) -> bool {
    let module = sk_py_module_name(object);
    module.starts_with("scipy.sparse")
        || (sk_py_has(object, "format")
            && sk_py_has(object, "indices")
            && sk_py_has(object, "indptr")
            && sk_py_has(object, "tocsr"))
}

/// Best-effort module name of an object's type.
pub(crate) fn sk_py_module_name(object: &Bound<'_, PyAny>) -> String {
    object
        .get_type()
        .getattr("__module__")
        .and_then(|module| module.extract::<String>())
        .unwrap_or_default()
}

/// Best-effort class name of an object's type.
pub(crate) fn sk_py_type_name(object: &Bound<'_, PyAny>) -> String {
    object
        .get_type()
        .getattr("__name__")
        .and_then(|name| name.extract::<String>())
        .unwrap_or_default()
}

fn sk_py_has(object: &Bound<'_, PyAny>, attribute: &str) -> bool {
    object.hasattr(attribute).unwrap_or(false)
}
