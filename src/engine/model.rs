//! Safe `RwLock`-synchronized wrapper over an immutable `SKFfiModel` handle.

use std::ffi::CString;
use std::ptr;
use std::sync::RwLock;

use sciencekit_ffi::{SKFfiDataType, SKFfiStatus};

use super::builder::sk_py_check;
use super::family::SKPyEstimatorFamily;
use crate::array::dense::SKPyRawDescriptor;
use crate::array::dtype::sk_py_dtype_label;
use crate::array::output::{SKPyOwnedOutput, sk_py_empty_output, sk_py_take_output_array};
use crate::error::SKPyError;
use crate::lifecycle::{sk_py_handle_created, sk_py_handle_freed};

/// Owns one `SKFfiModel` handle, permitting concurrent read operations while
/// serializing `close` so no reader races with `sk_ffi_model_free`.
#[derive(Debug)]
pub(crate) struct SKPyModelHandle {
    address: RwLock<Option<usize>>,
    family: SKPyEstimatorFamily,
    dtype: SKFfiDataType,
}

impl SKPyModelHandle {
    pub(crate) fn from_fitted_pointer(
        pointer: *mut sciencekit_ffi::model::SKFfiModel,
        family: SKPyEstimatorFamily,
        dtype: SKFfiDataType,
    ) -> Result<SKPyModelHandle, SKPyError> {
        if pointer.is_null() {
            return Err(SKPyError::from_null_return(SKFfiStatus::Internal));
        }
        sk_py_handle_created();
        Ok(SKPyModelHandle {
            address: RwLock::new(Some(pointer as usize)),
            family,
            dtype,
        })
    }

    pub(crate) fn load(path: &str) -> Result<SKPyModelHandle, SKPyError> {
        let c_path = sk_py_c_path(path)?;
        let mut pointer: *mut sciencekit_ffi::model::SKFfiModel = ptr::null_mut();
        let status = sciencekit_ffi::sk_ffi_model_load(c_path.as_ptr(), &mut pointer);
        if status != SKFfiStatus::Ok {
            return Err(SKPyError::from_last_error(status));
        }
        sk_py_handle_created();
        Ok(SKPyModelHandle {
            address: RwLock::new(Some(pointer as usize)),
            family: SKPyEstimatorFamily::LinearRegression,
            dtype: SKFfiDataType::F64,
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
            .read()
            .map(|guard| guard.is_none())
            .unwrap_or(true)
    }

    pub(crate) fn predict(
        &self,
        features: &SKPyRawDescriptor,
    ) -> Result<SKPyOwnedOutput, SKPyError> {
        self.with_shared_pointer(|pointer| {
            let features_array = features.to_ffi_array();
            let mut output = sk_py_empty_output();
            let status = sciencekit_ffi::sk_ffi_predict(pointer, &features_array, &mut output);
            sk_py_take_output_array(status, output)
        })
    }

    pub(crate) fn transform(
        &self,
        features: &SKPyRawDescriptor,
    ) -> Result<SKPyOwnedOutput, SKPyError> {
        self.with_shared_pointer(|pointer| {
            let features_array = features.to_ffi_array();
            let mut output = sk_py_empty_output();
            let status = sciencekit_ffi::sk_ffi_transform(pointer, &features_array, &mut output);
            sk_py_take_output_array(status, output)
        })
    }

    pub(crate) fn score(
        &self,
        features: &SKPyRawDescriptor,
        targets: &SKPyRawDescriptor,
    ) -> Result<f64, SKPyError> {
        self.with_shared_pointer(|pointer| {
            let features_array = features.to_ffi_array();
            let targets_array = targets.to_ffi_array();
            let mut score_value = 0.0_f64;
            let status = sciencekit_ffi::sk_ffi_score(
                pointer,
                &features_array,
                &targets_array,
                &mut score_value,
            );
            sk_py_check(status)?;
            Ok(score_value)
        })
    }

    pub(crate) fn save(&self, path: &str) -> Result<(), SKPyError> {
        let c_path = sk_py_c_path(path)?;
        self.with_shared_pointer(|pointer| {
            let status = sciencekit_ffi::sk_ffi_model_save(pointer, c_path.as_ptr());
            sk_py_check(status)
        })
    }

    /// Release the handle idempotently, waiting for active readers to finish.
    pub(crate) fn close(&self) {
        if let Ok(mut guard) = self.address.write() {
            if let Some(raw) = guard.take() {
                let pointer = raw as *mut sciencekit_ffi::model::SKFfiModel;
                let _ = sciencekit_ffi::sk_ffi_model_free(pointer);
                sk_py_handle_freed();
            }
        }
    }

    fn with_shared_pointer<T>(
        &self,
        operation: impl FnOnce(*mut sciencekit_ffi::model::SKFfiModel) -> Result<T, SKPyError>,
    ) -> Result<T, SKPyError> {
        let guard = self.address.read().map_err(|_| {
            SKPyError::new(SKFfiStatus::Internal, "model synchronization lock poisoned")
        })?;
        let raw = guard.ok_or_else(|| SKPyError::invalid_handle("the model was already closed"))?;
        operation(raw as *mut sciencekit_ffi::model::SKFfiModel)
    }
}

impl Drop for SKPyModelHandle {
    fn drop(&mut self) {
        self.close();
    }
}

fn sk_py_c_path(path: &str) -> Result<CString, SKPyError> {
    CString::new(path)
        .map_err(|_| SKPyError::invalid_argument("the file path contains an interior NUL byte"))
}
