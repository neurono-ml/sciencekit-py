//! Structured error value carrying the C ABI status code and native message.

use std::fmt;

use sciencekit_ffi::SKFfiStatus;

/// Python exception family selected for one failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SKPyErrorKind {
    /// Map the ABI status onto the `ScienceKitError` hierarchy.
    Status,
    /// Raise a `KeyError` subclass naming the offending column.
    Key,
    /// Raise a `TypeError` subclass naming the offending column and dtype.
    Type,
}

/// A failure from descriptor validation or from a `sciencekit_ffi` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SKPyError {
    status: SKFfiStatus,
    kind: SKPyErrorKind,
    message: String,
}

impl SKPyError {
    /// Construct a structured error from an explicit status and message.
    pub(crate) fn new(status: SKFfiStatus, message: impl Into<String>) -> SKPyError {
        SKPyError {
            status,
            kind: SKPyErrorKind::Status,
            message: message.into(),
        }
    }

    /// An unknown column or feature name (`KeyError` in Python).
    pub(crate) fn key(message: impl Into<String>) -> SKPyError {
        SKPyError {
            status: SKFfiStatus::InvalidArgument,
            kind: SKPyErrorKind::Key,
            message: message.into(),
        }
    }

    /// A column dtype that cannot become a numeric feature (`TypeError` in Python).
    pub(crate) fn type_error(message: impl Into<String>) -> SKPyError {
        SKPyError {
            status: SKFfiStatus::InvalidArgument,
            kind: SKPyErrorKind::Type,
            message: message.into(),
        }
    }

    /// Python exception family selected for this failure.
    pub(crate) fn kind(&self) -> SKPyErrorKind {
        self.kind
    }

    /// Construct an error from a non-OK `SKFfiStatus`, reading the thread-local
    /// native message recorded by `sciencekit_ffi` on the calling OS thread.
    pub(crate) fn from_last_error(status: SKFfiStatus) -> SKPyError {
        let recorded = sciencekit_ffi::error::sk_ffi_last_error_message();
        let message = if recorded.is_empty() {
            status.name().to_string()
        } else {
            recorded
        };
        SKPyError {
            status,
            kind: SKPyErrorKind::Status,
            message,
        }
    }

    /// Construct an error after a pointer-returning ABI entry point returned
    /// null, classifying the recorded thread-local diagnostic onto its status.
    pub(crate) fn from_null_return(default_status: SKFfiStatus) -> SKPyError {
        let recorded = sciencekit_ffi::error::sk_ffi_last_error_message();
        let status = sk_py_infer_status_from_message(&recorded, default_status);
        let message = if recorded.is_empty() {
            status.name().to_string()
        } else {
            recorded
        };
        SKPyError {
            status,
            kind: SKPyErrorKind::Status,
            message,
        }
    }

    /// An operation attempted on a closed or unknown handle (`1`).
    pub(crate) fn invalid_handle(message: impl Into<String>) -> SKPyError {
        SKPyError::new(SKFfiStatus::InvalidHandle, message)
    }

    /// Incompatible shapes or dimensions (`2`).
    #[allow(dead_code)]
    pub(crate) fn shape(message: impl Into<String>) -> SKPyError {
        SKPyError::new(SKFfiStatus::Shape, message)
    }

    /// Malformed argument, dtype, rank, or path (`3`).
    pub(crate) fn invalid_argument(message: impl Into<String>) -> SKPyError {
        SKPyError::new(SKFfiStatus::InvalidArgument, message)
    }

    /// Unsupported representation, layout, or operation (`4`).
    pub(crate) fn unsupported(message: impl Into<String>) -> SKPyError {
        SKPyError::new(SKFfiStatus::Unsupported, message)
    }

    /// The ABI status enum.
    pub(crate) fn status(&self) -> SKFfiStatus {
        self.status
    }

    /// Numeric ABI status discriminant.
    pub(crate) fn status_code(&self) -> i32 {
        self.status as i32
    }

    /// Stable lowercase ABI status name.
    pub(crate) fn status_name(&self) -> &'static str {
        self.status.name()
    }

    /// Human-readable diagnostic without prefix.
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SKPyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "sciencekit {} ({}): {}",
            self.status_name(),
            self.status_code(),
            self.message
        )
    }
}

impl std::error::Error for SKPyError {}

/// Classify a diagnostic recorded by `sk_ffi_builder_create` or `sk_ffi_fit`
/// (which return null pointers instead of status integers) onto `SKFfiStatus`.
fn sk_py_infer_status_from_message(message: &str, fallback: SKFfiStatus) -> SKFfiStatus {
    let lower = message.to_ascii_lowercase();
    if lower.contains("invalid handle") || lower.contains("already closed") {
        SKFfiStatus::InvalidHandle
    } else if lower.contains("shape mismatch") {
        SKFfiStatus::Shape
    } else if lower.contains("not supported") || lower.contains("unsupported") {
        SKFfiStatus::Unsupported
    } else if lower.contains("not fitted") {
        SKFfiStatus::NotFitted
    } else if lower.contains("dtype")
        || lower.contains("dimension")
        || lower.contains("rank")
        || lower.contains("requires a targets")
        || lower.contains("unknown")
        || lower.contains("empty")
        || lower.contains("non-finite")
        || lower.contains("nan")
        || lower.contains("infinite")
    {
        SKFfiStatus::InvalidArgument
    } else {
        fallback
    }
}
