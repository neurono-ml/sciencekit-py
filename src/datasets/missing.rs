//! Missing-value policy applied while a column is converted to `float32`/`float64`.

use crate::error::SKPyError;

/// Policy applied to a null or NaN element of an incoming column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SKPyMissingPolicy {
    /// Raise a `ValueError` naming the column and row.
    Reject,
    /// Represent the missing element as `NaN`.
    Nan,
    /// Replace the missing element with a constant.
    Fill(f64),
}

impl SKPyMissingPolicy {
    /// Parse the public `on_missing` keyword, validating `fill_value`.
    pub(crate) fn parse(name: &str, fill_value: Option<f64>) -> Result<Self, SKPyError> {
        match name {
            "reject" => Ok(SKPyMissingPolicy::Reject),
            "nan" => Ok(SKPyMissingPolicy::Nan),
            "fill" => match fill_value {
                Some(value) => Ok(SKPyMissingPolicy::Fill(value)),
                None => Err(SKPyError::invalid_argument(
                    "on_missing='fill' requires fill_value to be set",
                )),
            },
            other => Err(SKPyError::invalid_argument(format!(
                "unknown on_missing policy '{other}'; expected 'reject', 'nan' or 'fill'"
            ))),
        }
    }

    /// Resolve one missing element for `column` at `row`.
    pub(crate) fn resolve(self, column: &str, row: usize) -> Result<f64, SKPyError> {
        match self {
            SKPyMissingPolicy::Reject => Err(SKPyError::invalid_argument(format!(
                "column '{column}' contains a missing value at row {row}; pass on_missing='nan' or on_missing='fill' with fill_value to replace it"
            ))),
            SKPyMissingPolicy::Nan => Ok(f64::NAN),
            SKPyMissingPolicy::Fill(value) => Ok(value),
        }
    }

    /// Stable public name of the policy.
    pub(crate) fn name(self) -> &'static str {
        match self {
            SKPyMissingPolicy::Reject => "reject",
            SKPyMissingPolicy::Nan => "nan",
            SKPyMissingPolicy::Fill(_) => "fill",
        }
    }
}

#[cfg(test)]
#[path = "missing_tests.rs"]
mod missing_tests;
