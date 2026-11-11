//! Structured errors and Python exception mapping.

pub(crate) mod definition;
pub(crate) mod exceptions;

pub(crate) use definition::SKPyError;
pub(crate) use exceptions::sk_py_to_py_err;

#[cfg(test)]
#[path = "error_tests.rs"]
mod error_tests;
