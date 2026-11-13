//! Dask DataFrame consumed partition by partition with a documented eager join.

use pyo3::prelude::*;

use super::SKPyFrameImport;
use super::metadata::SKPyConversionMetadata;
use super::options::SKPyConversionOptions;
use super::storage::{SKPyConvertedData, SKPyOwnedNumbers, SKPyStorage};
use crate::error::SKPyError;

/// Convert a Dask DataFrame by consuming each partition in order.
pub(crate) fn sk_py_convert_dask<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    options: &SKPyConversionOptions,
    vector_only: bool,
) -> Result<SKPyFrameImport<'py>, SKPyError> {
    let npartitions: usize = object
        .getattr("npartitions")
        .and_then(|value| value.extract())
        .map_err(|_| SKPyError::invalid_argument("dask DataFrame has no integer `npartitions`"))?;
    if npartitions == 0 {
        return Err(SKPyError::invalid_argument(
            "dask DataFrame has no partitions",
        ));
    }

    let mut values = SKPyOwnedNumbers::empty(options.expected_dtype);
    let mut target_values = SKPyOwnedNumbers::empty(options.expected_dtype);
    let mut has_target = false;
    let mut rows = 0_usize;
    let mut partition_sizes = Vec::with_capacity(npartitions);
    let mut feature_names: Vec<String> = Vec::new();
    let mut label_table: Option<Vec<String>> = None;
    let mut batches = 0_usize;

    for index in 0..npartitions {
        let partition = object
            .call_method1("get_partition", (index,))
            .map_err(|error| {
                SKPyError::invalid_argument(format!(
                    "failed to materialize dask partition {index}: {error}"
                ))
            })?;
        // `get_partition` returns a lazy collection; materialize it before
        // conversion so the adapter never recurses into the dispatcher.
        let partition = if super::representation::sk_py_module_name(&partition).starts_with("dask.")
        {
            partition.call_method0("compute").map_err(|error| {
                SKPyError::invalid_argument(format!(
                    "failed to compute dask partition {index}: {error}"
                ))
            })?
        } else {
            partition
        };
        let import = super::sk_py_convert_frame(py, &partition, options)?;
        let partition_rows = import.features.rows();
        partition_sizes.push(partition_rows);
        rows += partition_rows;
        batches += import.features.metadata.batches;
        if feature_names.is_empty() {
            feature_names = import.features.metadata.feature_names.clone();
        }
        values.extend_from(&sk_py_into_owned(py, import.features)?);
        if let Some(target) = import.target {
            if let Some(labels) = target.metadata.label_table.clone() {
                label_table = Some(labels);
            }
            has_target = true;
            target_values.extend_from(&sk_py_into_owned(py, target)?);
        }
    }

    let columns_count = if vector_only {
        1
    } else {
        feature_names.len().max(1)
    };
    let metadata = SKPyConversionMetadata {
        representation: "dask.dataframe".to_string(),
        dtype: super::numpy_dataset::sk_py_dtype_name(options.expected_dtype),
        rows,
        columns_count,
        copied: true,
        zero_copy: false,
        eager_fallback: Some("dask.partitions".to_string()),
        partitions: npartitions,
        partition_sizes,
        batches,
        feature_names,
        abi_dense_required: true,
        missing_policy: options.on_missing.name().to_string(),
        ..SKPyConversionMetadata::default()
    };
    let features = SKPyConvertedData {
        dtype: options.expected_dtype,
        shape: if vector_only {
            vec![rows as i64]
        } else {
            vec![rows as i64, columns_count as i64]
        },
        storage: SKPyStorage::Owned(values),
        metadata,
    };
    let target = if has_target {
        Some(SKPyConvertedData {
            dtype: options.expected_dtype,
            shape: vec![rows as i64],
            storage: SKPyStorage::Owned(target_values),
            metadata: SKPyConversionMetadata {
                representation: "dask.dataframe".to_string(),
                dtype: super::numpy_dataset::sk_py_dtype_name(options.expected_dtype),
                rows,
                columns_count: 1,
                copied: true,
                zero_copy: false,
                eager_fallback: Some("dask.partitions".to_string()),
                partitions: npartitions,
                label_table,
                missing_policy: options.on_missing.name().to_string(),
                ..SKPyConversionMetadata::default()
            },
        })
    } else {
        None
    };
    Ok(SKPyFrameImport { features, target })
}

fn sk_py_into_owned<'py>(
    py: Python<'py>,
    data: SKPyConvertedData<'py>,
) -> Result<SKPyOwnedNumbers, SKPyError> {
    match data.storage {
        SKPyStorage::Owned(numbers) => Ok(numbers),
        SKPyStorage::Sparse(view) => view.densify(py, data.dtype),
        SKPyStorage::Borrowed { .. } => Err(SKPyError::invalid_argument(
            "a borrowed buffer cannot be joined across partitions",
        )),
    }
}
