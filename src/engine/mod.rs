//! Safe, thread-synchronized wrappers around opaque `sciencekit_ffi` handles.

pub(crate) mod builder;
pub(crate) mod family;
pub(crate) mod model;

pub(crate) use builder::SKPyBuilderHandle;
pub(crate) use family::{SKPyEstimatorFamily, sk_py_parse_backend_code};
pub(crate) use model::SKPyModelHandle;

#[cfg(test)]
#[path = "engine_tests.rs"]
mod engine_tests;
