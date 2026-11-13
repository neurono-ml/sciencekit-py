//! Conversion metadata recorded for one dataset import.

/// Observable facts about a single conversion from a Python representation into
/// the dense ABI seam.
#[derive(Debug, Clone, Default)]
pub(crate) struct SKPyConversionMetadata {
    /// Input representation, e.g. `numpy`, `pandas.dataframe`, `scipy.csr`.
    pub(crate) representation: String,
    /// Effective scalar precision of the converted matrix (`float32`/`float64`).
    pub(crate) dtype: &'static str,
    /// Row count.
    pub(crate) rows: usize,
    /// Column count (`1` for a target vector).
    pub(crate) columns_count: usize,
    /// Whether any element was copied during conversion.
    pub(crate) copied: bool,
    /// Whether the backing buffer is borrowed from the Python object.
    pub(crate) zero_copy: bool,
    /// Whether the conversion layer materialized a dense block from a sparse input.
    pub(crate) densified: bool,
    /// Whether Arrow buffers were imported through the C Data Interface without a copy.
    pub(crate) arrow_zero_copy: bool,
    /// Whether the scalar precision changed during conversion.
    pub(crate) cast: bool,
    /// Source sparse layout when a conversion happened (`csc`/`coo`).
    pub(crate) converted_from: Option<String>,
    /// Documented eager fallback used for a lazy or distributed container.
    pub(crate) eager_fallback: Option<String>,
    /// Number of Arrow record batches consumed.
    pub(crate) batches: usize,
    /// Number of partitions consumed from a distributed container.
    pub(crate) partitions: usize,
    /// Row counts of the consumed partitions, in consumption order.
    pub(crate) partition_sizes: Vec<usize>,
    /// Feature column names (empty for unnamed arrays).
    pub(crate) feature_names: Vec<String>,
    /// Pointer to the borrowed `float64`/`float32` buffer, when applicable.
    pub(crate) data_pointer: Option<usize>,
    /// Pointer to the borrowed CSR `indices` buffer.
    pub(crate) indices_pointer: Option<usize>,
    /// Pointer to the borrowed CSR `indptr` buffer.
    pub(crate) indptr_pointer: Option<usize>,
    /// Number of explicitly stored zero entries in a CSR input.
    pub(crate) stored_zeros: usize,
    /// Categorical label table when the target column was a dictionary column.
    pub(crate) label_table: Option<Vec<String>>,
    /// Whether the `sciencekit_ffi` ABI requires a dense block at the call boundary.
    pub(crate) abi_dense_required: bool,
    /// Missing-value policy applied to null elements.
    pub(crate) missing_policy: String,
}
