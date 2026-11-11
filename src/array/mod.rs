//! NumPy dense and CSR descriptor validation and output conversion.

pub(crate) mod csr;
pub(crate) mod csr_validation;
pub(crate) mod dense;
pub(crate) mod dense_class;
pub(crate) mod dtype;
pub(crate) mod output;

pub(crate) use csr::SKPyCsrDescriptor;
pub(crate) use dense_class::SKPyDenseDescriptor;

#[cfg(test)]
#[path = "array_tests.rs"]
mod array_tests;
