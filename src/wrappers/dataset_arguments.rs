//! Bridge from estimator keyword arguments to dataset conversion.

use pyo3::prelude::*;
use sciencekit_ffi::SKFfiDataType;

use crate::datasets::metadata::SKPyConversionMetadata;
use crate::datasets::storage::SKPyAbiArray;
use crate::datasets::{
    SKPyConversionOptions, SKPyConvertedData, SKPyFrameImport, SKPyMissingPolicy,
    sk_py_convert_frame, sk_py_convert_matrix, sk_py_convert_vector,
};
use crate::error::SKPyError;

/// Keyword arguments shared by every dataset-accepting estimator operation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKPyDatasetKeywords<'a> {
    /// Allow copying a non-contiguous NumPy array.
    pub(crate) copy: bool,
    /// Allow casting a mismatched scalar precision.
    pub(crate) cast: bool,
    /// Missing-value policy name.
    pub(crate) on_missing: &'a str,
    /// Replacement value for `on_missing='fill'`.
    pub(crate) fill_value: Option<f64>,
}

impl SKPyDatasetKeywords<'_> {
    /// Build conversion options for `dtype`.
    pub(crate) fn options(
        &self,
        dtype: SKFfiDataType,
        columns: Option<Vec<String>>,
        target: Option<String>,
    ) -> Result<SKPyConversionOptions, SKPyError> {
        let mut options = SKPyConversionOptions::new(dtype);
        options.allow_copy = self.copy;
        options.allow_cast = self.cast;
        options.on_missing = SKPyMissingPolicy::parse(self.on_missing, self.fill_value)?;
        options.columns = columns;
        options.target = target;
        Ok(options)
    }
}

/// Inputs prepared for one detached ABI call.
pub(crate) struct SKPyPreparedInputs {
    /// Feature block consumed by the ABI call.
    pub(crate) features: SKPyAbiArray,
    /// Optional target block consumed by the ABI call.
    pub(crate) targets: Option<SKPyAbiArray>,
    /// Metadata of the feature conversion.
    pub(crate) metadata: SKPyConversionMetadata,
    /// Categorical label table of the target, when present.
    pub(crate) labels: Option<Vec<String>>,
}

/// Prepare the feature/target pair of a `fit` call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_py_prepare_fit_inputs(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    targets: Option<&Bound<'_, PyAny>>,
    target_name: Option<String>,
    keywords: &SKPyDatasetKeywords<'_>,
    columns: Option<Vec<String>>,
    dtype: SKFfiDataType,
    targets_optional: bool,
) -> Result<SKPyPreparedInputs, SKPyError> {
    if targets.is_none() && target_name.is_none() && targets_optional {
        let options = keywords.options(dtype, columns, None)?;
        let data = sk_py_convert_matrix(py, features, &options)?;
        return sk_py_finish_inputs(py, data, None);
    }
    match targets {
        Some(targets_object) if !targets_object.is_none() => {
            let options = keywords.options(dtype, columns, None)?;
            let features_data = sk_py_convert_matrix(py, features, &options)?;
            let vector_options = keywords.options(dtype, None, None)?;
            let targets_data = sk_py_convert_vector(py, targets_object, &vector_options)?;
            Ok(sk_py_finish_inputs(py, features_data, Some(targets_data))?)
        }
        _ => {
            if target_name.is_none() {
                return Err(SKPyError::invalid_argument(
                    "fit requires either a targets argument or the target= keyword naming a column in the frame",
                ));
            }
            let options = keywords.options(dtype, columns, target_name)?;
            let SKPyFrameImport { features, target } = sk_py_convert_frame(py, features, &options)?;
            let target = target.ok_or_else(|| {
                SKPyError::invalid_argument(
                    "the target= keyword requires a frame with named columns",
                )
            })?;
            Ok(sk_py_finish_inputs(py, features, Some(target))?)
        }
    }
}

fn sk_py_finish_inputs(
    py: Python<'_>,
    features: SKPyConvertedData<'_>,
    targets: Option<SKPyConvertedData<'_>>,
) -> Result<SKPyPreparedInputs, SKPyError> {
    let labels = targets
        .as_ref()
        .and_then(|target| target.metadata.label_table.clone());
    let metadata = features.metadata.clone();
    let features_abi = features.to_abi(py)?;
    let targets_abi = match targets {
        Some(target) => Some(target.to_abi(py)?),
        None => None,
    };
    Ok(SKPyPreparedInputs {
        features: features_abi,
        targets: targets_abi,
        metadata,
        labels,
    })
}

/// Prepare a feature block for `predict`/`transform`/`score`.
pub(crate) fn sk_py_prepare_features(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    columns: Option<Vec<String>>,
    keywords: &SKPyDatasetKeywords<'_>,
    dtype: SKFfiDataType,
) -> Result<SKPyPreparedInputs, SKPyError> {
    let options = keywords.options(dtype, columns, None)?;
    let data = sk_py_convert_matrix(py, features, &options)?;
    sk_py_finish_inputs(py, data, None)
}

/// Prepare a target vector for `score`.
pub(crate) fn sk_py_prepare_targets(
    py: Python<'_>,
    targets: &Bound<'_, PyAny>,
    keywords: &SKPyDatasetKeywords<'_>,
    dtype: SKFfiDataType,
) -> Result<SKPyPreparedInputs, SKPyError> {
    let options = keywords.options(dtype, None, None)?;
    let data = sk_py_convert_vector(py, targets, &options)?;
    sk_py_finish_inputs(py, data, None)
}
