//! Rust unit tests for builder/model handle lifecycle and backend parsing.

use sciencekit_ffi::SKFfiDataType;

use super::{SKPyBuilderHandle, SKPyEstimatorFamily, sk_py_parse_backend_code};
use crate::lifecycle::sk_py_live_handles;

#[test]
fn builder_close_is_idempotent_and_rejects_subsequent_mutations() {
    let baseline = sk_py_live_handles();
    let builder =
        SKPyBuilderHandle::create(SKPyEstimatorFamily::LinearRegression, SKFfiDataType::F64)
            .expect("builder creation must succeed");
    assert_eq!(sk_py_live_handles(), baseline + 1);
    assert!(!builder.is_closed());

    builder.close();
    builder.close();
    assert!(builder.is_closed());
    assert_eq!(sk_py_live_handles(), baseline);

    let err = builder.set_include_intercept(false).unwrap_err();
    assert_eq!(err.status_code(), 1);
}

#[test]
fn backend_parser_accepts_known_names_and_rejects_unknown() {
    assert_eq!(sk_py_parse_backend_code("faer").unwrap(), 0);
    assert_eq!(sk_py_parse_backend_code("matrixmultiply").unwrap(), 1);
    assert_eq!(sk_py_parse_backend_code("ndarray-linalg").unwrap(), 2);
    assert_eq!(
        sk_py_parse_backend_code("bogus").unwrap_err().status_code(),
        3
    );
}
