//! Unit tests for output descriptor initialization and error cleanup.

use sciencekit_ffi::SKFfiStatus;

use super::output::{sk_py_empty_output, sk_py_take_output_array};

#[test]
fn failed_output_status_propagates_recorded_error_and_frees_safely() {
    sciencekit_ffi::sk_ffi_set_last_error("shape mismatch on predict");
    let empty = sk_py_empty_output();
    let error = sk_py_take_output_array(SKFfiStatus::Shape, empty).unwrap_err();
    assert_eq!(error.status_code(), 2);
    assert_eq!(error.message(), "shape mismatch on predict");
}
