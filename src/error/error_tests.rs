//! Unit tests for structured error formatting and status inference.

use sciencekit_ffi::SKFfiStatus;

use super::SKPyError;

#[test]
fn error_display_includes_status_name_code_and_message() {
    let error = SKPyError::new(SKFfiStatus::Shape, "expected 4 rows, found 3");
    assert_eq!(error.status_code(), 2);
    assert_eq!(error.status_name(), "shape");
    assert_eq!(
        error.to_string(),
        "sciencekit shape (2): expected 4 rows, found 3"
    );
}

#[test]
fn null_return_infers_shape_and_invalid_argument_statuses() {
    sciencekit_ffi::sk_ffi_set_last_error("shape mismatch: expected (4, 2), got (3, 2)");
    let shape_error = SKPyError::from_null_return(SKFfiStatus::Internal);
    assert_eq!(shape_error.status(), SKFfiStatus::Shape);

    sciencekit_ffi::sk_ffi_set_last_error("linear regression requires a targets descriptor");
    let argument_error = SKPyError::from_null_return(SKFfiStatus::Internal);
    assert_eq!(argument_error.status(), SKFfiStatus::InvalidArgument);
}
