//! Safe `Mutex`-synchronized wrapper over an opaque `SKFfiBuilder` handle.

use std::ptr;
use std::sync::Mutex;

use sciencekit_ffi::{SKFfiDataType, SKFfiStatus};

use super::family::SKPyEstimatorFamily;
use super::model::SKPyModelHandle;
use crate::array::dense::SKPyRawDescriptor;
use crate::array::dtype::sk_py_dtype_label;
use crate::error::SKPyError;
use crate::lifecycle::{sk_py_handle_created, sk_py_handle_freed};

/// Owns one `SKFfiBuilder` handle and serializes mutations across threads.
#[derive(Debug)]
pub(crate) struct SKPyBuilderHandle {
    address: Mutex<Option<usize>>,
    family: SKPyEstimatorFamily,
    dtype: SKFfiDataType,
}

impl SKPyBuilderHandle {
    /// Allocate a native builder for `family` and `dtype`.
    pub(crate) fn create(
        family: SKPyEstimatorFamily,
        dtype: SKFfiDataType,
    ) -> Result<SKPyBuilderHandle, SKPyError> {
        let pointer = sciencekit_ffi::sk_ffi_builder_create(family.code(), dtype as i32);
        if pointer.is_null() {
            return Err(SKPyError::from_null_return(SKFfiStatus::InvalidArgument));
        }
        sk_py_handle_created();
        Ok(SKPyBuilderHandle {
            address: Mutex::new(Some(pointer as usize)),
            family,
            dtype,
        })
    }

    pub(crate) fn family(&self) -> SKPyEstimatorFamily {
        self.family
    }

    pub(crate) fn dtype(&self) -> SKFfiDataType {
        self.dtype
    }

    pub(crate) fn dtype_name(&self) -> &'static str {
        sk_py_dtype_label(self.dtype)
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.address
            .lock()
            .map(|guard| guard.is_none())
            .unwrap_or(true)
    }

    pub(crate) fn set_include_intercept(&self, enabled: bool) -> Result<(), SKPyError> {
        self.with_pointer(|pointer| {
            sk_py_check(sciencekit_ffi::sk_ffi_builder_set_include_intercept(
                pointer, enabled,
            ))
        })
    }

    pub(crate) fn set_with_mean(&self, enabled: bool) -> Result<(), SKPyError> {
        self.with_pointer(|pointer| {
            sk_py_check(sciencekit_ffi::sk_ffi_builder_set_with_mean(
                pointer, enabled,
            ))
        })
    }

    pub(crate) fn set_with_std(&self, enabled: bool) -> Result<(), SKPyError> {
        self.with_pointer(|pointer| {
            sk_py_check(sciencekit_ffi::sk_ffi_builder_set_with_std(
                pointer, enabled,
            ))
        })
    }

    pub(crate) fn set_backend(&self, code: i32) -> Result<(), SKPyError> {
        self.with_pointer(|pointer| {
            sk_py_check(sciencekit_ffi::sk_ffi_builder_set_backend(pointer, code))
        })
    }

    pub(crate) fn fit(
        &self,
        features: &SKPyRawDescriptor,
        targets: Option<&SKPyRawDescriptor>,
    ) -> Result<SKPyModelHandle, SKPyError> {
        self.with_pointer(|pointer| {
            let features_array = features.to_ffi_array();
            let targets_array = targets.map(SKPyRawDescriptor::to_ffi_array);
            let targets_pointer = match &targets_array {
                Some(array) => array as *const _,
                None => ptr::null(),
            };
            let fitted_pointer =
                sciencekit_ffi::sk_ffi_fit(pointer, &features_array, targets_pointer);
            SKPyModelHandle::from_fitted_pointer(fitted_pointer, self.family, self.dtype)
        })
    }

    /// Release the handle idempotently.
    pub(crate) fn close(&self) {
        if let Ok(mut guard) = self.address.lock() {
            if let Some(raw) = guard.take() {
                let pointer = raw as *mut sciencekit_ffi::builder::SKFfiBuilder;
                let _ = sciencekit_ffi::sk_ffi_builder_free(pointer);
                sk_py_handle_freed();
            }
        }
    }

    fn with_pointer<T>(
        &self,
        operation: impl FnOnce(*mut sciencekit_ffi::builder::SKFfiBuilder) -> Result<T, SKPyError>,
    ) -> Result<T, SKPyError> {
        let guard = self.address.lock().map_err(|_| {
            SKPyError::new(
                SKFfiStatus::Internal,
                "builder synchronization lock poisoned",
            )
        })?;
        let raw =
            guard.ok_or_else(|| SKPyError::invalid_handle("the builder was already closed"))?;
        operation(raw as *mut sciencekit_ffi::builder::SKFfiBuilder)
    }
}

impl Drop for SKPyBuilderHandle {
    fn drop(&mut self) {
        self.close();
    }
}

/// Convert a non-OK `SKFfiStatus` into a structured `SKPyError`.
pub(crate) fn sk_py_check(status: SKFfiStatus) -> Result<(), SKPyError> {
    if status == SKFfiStatus::Ok {
        Ok(())
    } else {
        Err(SKPyError::from_last_error(status))
    }
}
