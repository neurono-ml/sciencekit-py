//! Options controlling one Python dataset conversion.

use sciencekit_ffi::SKFfiDataType;

use super::missing::SKPyMissingPolicy;

/// Conversion options shared by every representation adapter.
#[derive(Debug, Clone)]
pub(crate) struct SKPyConversionOptions {
    /// Scalar precision required by the estimator.
    pub(crate) expected_dtype: SKFfiDataType,
    /// Whether a non-contiguous NumPy array may be copied.
    pub(crate) allow_copy: bool,
    /// Whether a mismatched scalar precision may be cast.
    pub(crate) allow_cast: bool,
    /// Policy applied to null elements.
    pub(crate) on_missing: SKPyMissingPolicy,
    /// Feature column selection; `None` selects every non-target column.
    pub(crate) columns: Option<Vec<String>>,
    /// Target column name for a single-frame `fit`.
    pub(crate) target: Option<String>,
}

impl SKPyConversionOptions {
    /// Options for a matrix conversion that must keep the estimator precision.
    pub(crate) fn new(expected_dtype: SKFfiDataType) -> Self {
        SKPyConversionOptions {
            expected_dtype,
            allow_copy: false,
            allow_cast: false,
            on_missing: SKPyMissingPolicy::Reject,
            columns: None,
            target: None,
        }
    }

    /// Whether the conversion may produce values in a different precision.
    pub(crate) fn may_cast(&self) -> bool {
        self.allow_cast || self.allow_copy
    }
}
