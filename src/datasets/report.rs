//! `ConversionReport` `#[pyclass]` exposing conversion metadata to Python.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::metadata::SKPyConversionMetadata;

/// Immutable description of how one Python dataset was converted.
#[pyclass(name = "ConversionReport", module = "sciencekit.datasets", frozen)]
pub(crate) struct SKPyConversionReport {
    metadata: SKPyConversionMetadata,
}

impl SKPyConversionReport {
    /// Build a report from recorded metadata.
    pub(crate) fn from_metadata(metadata: SKPyConversionMetadata) -> Self {
        SKPyConversionReport { metadata }
    }
}

#[pymethods]
impl SKPyConversionReport {
    /// Input representation name, e.g. `pandas.dataframe`.
    #[getter]
    fn representation(&self) -> &str {
        &self.metadata.representation
    }

    /// Effective scalar precision.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.metadata.dtype
    }

    /// Converted shape `(rows, columns)`.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [self.metadata.rows, self.metadata.columns_count])
    }

    /// Whether any element copy happened.
    #[getter]
    fn copied(&self) -> bool {
        self.metadata.copied
    }

    /// Whether the buffer is borrowed from the caller without a copy.
    #[getter]
    fn zero_copy(&self) -> bool {
        self.metadata.zero_copy
    }

    /// Whether a dense block was materialized from a sparse input.
    #[getter]
    fn densified(&self) -> bool {
        self.metadata.densified
    }

    /// Whether Arrow buffers were imported through the C Data Interface without a copy.
    #[getter]
    fn arrow_zero_copy(&self) -> bool {
        self.metadata.arrow_zero_copy
    }

    /// Whether the precision changed during conversion.
    #[getter]
    fn cast(&self) -> bool {
        self.metadata.cast
    }

    /// Source layout when the input had to be converted (`csc`, `coo`).
    #[getter]
    fn converted_from(&self) -> Option<&str> {
        self.metadata.converted_from.as_deref()
    }

    /// Documented eager fallback used for a lazy or distributed container.
    #[getter]
    fn eager_fallback(&self) -> Option<&str> {
        self.metadata.eager_fallback.as_deref()
    }

    /// Number of Arrow batches consumed.
    #[getter]
    fn batches(&self) -> usize {
        self.metadata.batches
    }

    /// Number of partitions consumed.
    #[getter]
    fn partitions(&self) -> usize {
        self.metadata.partitions
    }

    /// Row count of each consumed partition, in order.
    #[getter]
    fn partition_sizes(&self) -> Vec<usize> {
        self.metadata.partition_sizes.clone()
    }

    /// Feature column names; empty for unnamed array input.
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.metadata.feature_names.clone()
    }

    /// Pointer to the borrowed data buffer, when applicable.
    #[getter]
    fn data_pointer(&self) -> Option<usize> {
        self.metadata.data_pointer
    }

    /// Pointer to the borrowed CSR `indices` buffer.
    #[getter]
    fn indices_pointer(&self) -> Option<usize> {
        self.metadata.indices_pointer
    }

    /// Pointer to the borrowed CSR `indptr` buffer.
    #[getter]
    fn indptr_pointer(&self) -> Option<usize> {
        self.metadata.indptr_pointer
    }

    /// Number of explicitly stored zero entries in a CSR input.
    #[getter]
    fn stored_zeros(&self) -> usize {
        self.metadata.stored_zeros
    }

    /// Categorical label table, when the target was a dictionary column.
    #[getter]
    fn label_table(&self) -> Option<Vec<String>> {
        self.metadata.label_table.clone()
    }

    /// Whether the ABI requires a dense block at the call boundary.
    #[getter]
    fn abi_dense_required(&self) -> bool {
        self.metadata.abi_dense_required
    }

    /// Applied missing-value policy name.
    #[getter]
    fn missing_policy(&self) -> &str {
        &self.metadata.missing_policy
    }

    fn __repr__(&self) -> String {
        format!(
            "ConversionReport(representation='{}', dtype='{}', shape=({}, {}), zero_copy={}, copied={}, densified={})",
            self.metadata.representation,
            self.metadata.dtype,
            self.metadata.rows,
            self.metadata.columns_count,
            self.metadata.zero_copy,
            self.metadata.copied,
            self.metadata.densified,
        )
    }
}
