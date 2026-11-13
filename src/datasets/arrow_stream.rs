//! Arrow C Data Interface consumption of `__arrow_c_stream__`/`__arrow_c_array__`.

use std::ffi::{CStr, c_void};
use std::ptr;
use std::sync::Arc;

use arrow::array::make_array;
use arrow::array::{
    Array, ArrayRef, DictionaryArray, Float64Array, RecordBatch, RecordBatchReader, StructArray,
    types::{Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type},
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use pyo3::prelude::*;

use super::missing::SKPyMissingPolicy;
use super::options::SKPyConversionOptions;
use crate::error::SKPyError;

const STREAM_CAPSULE_NAME: &CStr = c"arrow_array_stream";
const SCHEMA_CAPSULE_NAME: &CStr = c"arrow_schema";
const ARRAY_CAPSULE_NAME: &CStr = c"arrow_array";

/// One imported Arrow source: a C stream or a single C array.
enum SKPyArrowSource {
    /// Owning reader over an imported `ArrowArrayStream`.
    Stream(Box<ArrowArrayStreamReader>),
    /// One imported `RecordBatch` from `__arrow_c_array__`.
    Single(Option<RecordBatch>),
}

impl SKPyArrowSource {
    fn schema(&self) -> SchemaRef {
        match self {
            SKPyArrowSource::Stream(reader) => reader.schema(),
            SKPyArrowSource::Single(batch) => batch
                .as_ref()
                .map(|batch| batch.schema())
                .unwrap_or_else(|| Arc::new(Schema::empty())),
        }
    }

    fn next_batch(&mut self) -> Option<Result<RecordBatch, String>> {
        match self {
            SKPyArrowSource::Stream(reader) => reader
                .next()
                .map(|item| item.map_err(|error| error.to_string())),
            SKPyArrowSource::Single(batch) => batch.take().map(Ok),
        }
    }
}

/// Row-major dense values imported from Arrow batches.
pub(crate) struct SKPyArrowImport {
    /// Total rows consumed.
    pub(crate) rows: usize,
    /// Row-major feature values.
    pub(crate) values: super::storage::SKPyOwnedNumbers,
    /// Optional one-dimensional target column.
    pub(crate) target: Option<super::storage::SKPyOwnedNumbers>,
    /// Feature column names.
    pub(crate) feature_names: Vec<String>,
    /// Categorical label table when the target is a dictionary column.
    pub(crate) label_table: Option<Vec<String>>,
    /// Number of record batches consumed.
    pub(crate) batches: usize,
    /// Whether the Arrow precision differs from the estimator precision.
    pub(crate) cast: bool,
}

/// Consume every Arrow batch of `object`, converting the selected columns.
pub(crate) fn sk_py_read_arrow_records(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    options: &SKPyConversionOptions,
    vector_only: bool,
) -> Result<SKPyArrowImport, SKPyError> {
    let mut source = sk_py_arrow_source(py, object)?;
    let schema = source.schema();
    if schema.fields().is_empty() {
        return Err(SKPyError::invalid_argument(
            "Arrow input has no columns; at least one numeric feature column is required",
        ));
    }
    let plan = sk_py_resolve_plan(&schema, options, vector_only)?;
    let cast = plan.feature_indices.iter().any(|&index| {
        matches!(
            (schema.field(index).data_type(), options.expected_dtype),
            (DataType::Float32, sciencekit_ffi::SKFfiDataType::F64)
                | (DataType::Float64, sciencekit_ffi::SKFfiDataType::F32)
        )
    });
    let mut accumulator = SKPyArrowAccumulator::new(&plan, options);
    while let Some(batch) = source.next_batch() {
        let batch = batch.map_err(|error| {
            SKPyError::invalid_argument(format!("failed to consume an Arrow batch: {error}"))
        })?;
        accumulator.push_batch(&batch, &plan)?;
    }
    Ok(accumulator.finish(&plan, cast))
}

struct SKPyArrowPlan {
    feature_indices: Vec<usize>,
    feature_names: Vec<String>,
    target_index: Option<usize>,
    target_name: Option<String>,
}

fn sk_py_resolve_plan(
    schema: &SchemaRef,
    options: &SKPyConversionOptions,
    vector_only: bool,
) -> Result<SKPyArrowPlan, SKPyError> {
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    if vector_only {
        if names.len() != 1 {
            return Err(SKPyError::invalid_argument(format!(
                "a 1D target requires a single-column Arrow source, found {} columns {names:?}",
                names.len()
            )));
        }
        return Ok(SKPyArrowPlan {
            feature_indices: vec![0],
            feature_names: vec![names[0].clone()],
            target_index: None,
            target_name: None,
        });
    }
    let selected = match &options.columns {
        Some(requested) => requested
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| sk_py_unknown_column(name, &names))
            })
            .collect::<Result<Vec<usize>, SKPyError>>()?,
        None => names
            .iter()
            .enumerate()
            .filter(|(_, name)| options.target.as_deref() != Some(name.as_str()))
            .filter(|(_, name)| !sk_py_is_pandas_index_column(name))
            .map(|(index, _)| index)
            .collect(),
    };
    if selected.is_empty() {
        return Err(SKPyError::invalid_argument(
            "no feature columns selected; name at least one numeric column",
        ));
    }
    let target_index = match &options.target {
        Some(name) => Some(
            names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| sk_py_unknown_column(name, &names))?,
        ),
        None => None,
    };
    if let Some(index) = target_index {
        if selected.contains(&index) {
            return Err(SKPyError::invalid_argument(format!(
                "target column '{}' also appears in the feature selection",
                names[index]
            )));
        }
    }
    Ok(SKPyArrowPlan {
        feature_names: selected.iter().map(|&index| names[index].clone()).collect(),
        feature_indices: selected,
        target_name: target_index.map(|index| names[index].clone()),
        target_index,
    })
}

/// pandas exports a non-default index as an `__index_level_N__` helper column.
fn sk_py_is_pandas_index_column(name: &str) -> bool {
    name.starts_with("__index_level_") && name.ends_with("__")
}

fn sk_py_unknown_column(name: &str, available: &[String]) -> SKPyError {
    SKPyError::key(format!(
        "unknown column '{name}'; available columns are {available:?}"
    ))
}

struct SKPyArrowAccumulator {
    dtype: sciencekit_ffi::SKFfiDataType,
    policy: SKPyMissingPolicy,
    rows: usize,
    values: super::storage::SKPyOwnedNumbers,
    target: Option<super::storage::SKPyOwnedNumbers>,
    label_table: Option<Vec<String>>,
    batches: usize,
}

impl SKPyArrowAccumulator {
    fn new(plan: &SKPyArrowPlan, options: &SKPyConversionOptions) -> Self {
        SKPyArrowAccumulator {
            dtype: options.expected_dtype,
            policy: options.on_missing,
            rows: 0,
            values: super::storage::SKPyOwnedNumbers::empty(options.expected_dtype),
            target: plan
                .target_index
                .map(|_| super::storage::SKPyOwnedNumbers::empty(options.expected_dtype)),
            label_table: None,
            batches: 0,
        }
    }

    fn push_batch(&mut self, batch: &RecordBatch, plan: &SKPyArrowPlan) -> Result<(), SKPyError> {
        let height = batch.num_rows();
        let mut blocks = Vec::with_capacity(plan.feature_indices.len());
        for (&index, name) in plan.feature_indices.iter().zip(&plan.feature_names) {
            let block = sk_py_column_block(
                batch.column(index),
                name,
                self.rows,
                self.policy,
                self.dtype,
            )?;
            if block.0.len() != height {
                return Err(SKPyError::shape(format!(
                    "column '{name}' has {} values but the batch has {height} rows",
                    block.0.len()
                )));
            }
            blocks.push(block.0);
        }
        for row in 0..height {
            for block in &blocks {
                self.values.push_f64(block[row]);
            }
        }
        if let (Some(index), Some(name)) = (plan.target_index, &plan.target_name) {
            let (values, labels) = sk_py_column_block(
                batch.column(index),
                name,
                self.rows,
                self.policy,
                self.dtype,
            )?;
            if values.len() != height {
                return Err(SKPyError::shape(format!(
                    "target column '{name}' has {} values but the batch has {height} rows",
                    values.len()
                )));
            }
            if labels.is_some() {
                self.label_table = labels;
            }
            let target = self
                .target
                .as_mut()
                .ok_or_else(|| SKPyError::invalid_argument("target accumulator missing"))?;
            for value in values {
                target.push_f64(value);
            }
        }
        self.rows += height;
        self.batches += 1;
        Ok(())
    }

    fn finish(self, plan: &SKPyArrowPlan, cast: bool) -> SKPyArrowImport {
        SKPyArrowImport {
            rows: self.rows,
            values: self.values,
            target: self.target,
            feature_names: plan.feature_names.clone(),
            label_table: self.label_table,
            batches: self.batches,
            cast,
        }
    }
}

/// Extract one Arrow column as `f64` values, honoring the missing-value policy.
fn sk_py_column_block(
    array: &ArrayRef,
    name: &str,
    base_row: usize,
    policy: SKPyMissingPolicy,
    dtype: sciencekit_ffi::SKFfiDataType,
) -> Result<(Vec<f64>, Option<Vec<String>>), SKPyError> {
    if let Some((codes, labels)) = sk_py_dictionary_codes(array) {
        let mut values = Vec::with_capacity(codes.len());
        for (row, code) in codes.iter().enumerate() {
            let value = match code {
                Some(code) => *code as f64,
                None => policy.resolve(name, base_row + row)?,
            };
            values.push(value);
        }
        return Ok((values, Some(labels)));
    }
    if !sk_py_is_numeric_type(array.data_type()) {
        return Err(SKPyError::type_error(format!(
            "column '{name}' has Arrow dtype {} which is not numeric; cast it to float64 before fitting (for example with pyarrow.compute.cast)",
            array.data_type()
        )));
    }
    let casted = cast(array, &DataType::Float64).map_err(|_| {
        SKPyError::type_error(format!(
            "column '{name}' has Arrow dtype {} which is not numeric; cast it to float64 before fitting (for example with pyarrow.compute.cast)",
            array.data_type()
        ))
    })?;
    let numeric = casted
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            SKPyError::type_error(format!(
                "column '{name}' could not be read as float64 (found {dtype:?})"
            ))
        })?;
    let mut values = Vec::with_capacity(numeric.len());
    for row in 0..numeric.len() {
        if numeric.is_null(row) {
            values.push(policy.resolve(name, base_row + row)?);
        } else {
            values.push(numeric.value(row));
        }
    }
    Ok((values, None))
}

/// Whether an Arrow dtype can be used as a numeric feature without ambiguity.
fn sk_py_is_numeric_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}

fn sk_py_dictionary_codes(array: &ArrayRef) -> Option<(Vec<Option<i64>>, Vec<String>)> {
    // `ArrayRef` is an `Arc<dyn Array>`; call through the trait object so
    // `as_any()` returns the concrete Arrow array rather than the `Arc` itself.
    let concrete = array.as_ref().as_any();
    if let Some(dictionary) = concrete.downcast_ref::<DictionaryArray<Int8Type>>() {
        return Some(sk_py_dictionary_values(
            dictionary.keys(),
            dictionary.values(),
        ));
    }
    if let Some(dictionary) = concrete.downcast_ref::<DictionaryArray<Int16Type>>() {
        return Some(sk_py_dictionary_values(
            dictionary.keys(),
            dictionary.values(),
        ));
    }
    if let Some(dictionary) = concrete.downcast_ref::<DictionaryArray<UInt8Type>>() {
        return Some(sk_py_dictionary_values(
            dictionary.keys(),
            dictionary.values(),
        ));
    }
    if let Some(dictionary) = concrete.downcast_ref::<DictionaryArray<UInt16Type>>() {
        return Some(sk_py_dictionary_values(
            dictionary.keys(),
            dictionary.values(),
        ));
    }
    if let Some(dictionary) = concrete.downcast_ref::<DictionaryArray<Int32Type>>() {
        return Some(sk_py_dictionary_values(
            dictionary.keys(),
            dictionary.values(),
        ));
    }
    if let Some(dictionary) = concrete.downcast_ref::<DictionaryArray<Int64Type>>() {
        return Some(sk_py_dictionary_values(
            dictionary.keys(),
            dictionary.values(),
        ));
    }
    if let Some(dictionary) = concrete.downcast_ref::<DictionaryArray<UInt32Type>>() {
        return Some(sk_py_dictionary_values(
            dictionary.keys(),
            dictionary.values(),
        ));
    }
    None
}

fn sk_py_dictionary_values<K>(
    keys: &arrow::array::PrimitiveArray<K>,
    values: &ArrayRef,
) -> (Vec<Option<i64>>, Vec<String>)
where
    K: arrow::array::ArrowPrimitiveType,
    K::Native: Into<i64> + Copy,
{
    let mut codes = Vec::with_capacity(keys.len());
    for row in 0..keys.len() {
        if keys.is_null(row) {
            codes.push(None);
        } else {
            codes.push(Some(keys.value(row).into()));
        }
    }
    let mut labels = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let label = arrow::util::display::array_value_to_string(values, index)
            .unwrap_or_else(|_| index.to_string());
        labels.push(label);
    }
    (codes, labels)
}

fn sk_py_arrow_source<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
) -> Result<SKPyArrowSource, SKPyError> {
    if object.hasattr("__arrow_c_stream__").unwrap_or(false) {
        let capsule = object.call_method0("__arrow_c_stream__").map_err(|error| {
            SKPyError::invalid_argument(format!("`__arrow_c_stream__` failed: {error}"))
        })?;
        let stream = unsafe { sk_py_take_stream(py, &capsule)? };
        let reader = ArrowArrayStreamReader::try_new(stream).map_err(|error| {
            SKPyError::invalid_argument(format!("failed to import the Arrow stream: {error}"))
        })?;
        return Ok(SKPyArrowSource::Stream(Box::new(reader)));
    }
    let capsules = object.call_method0("__arrow_c_array__").map_err(|error| {
        SKPyError::invalid_argument(format!("`__arrow_c_array__` failed: {error}"))
    })?;
    let schema_capsule = capsules.get_item(0).map_err(|_| {
        SKPyError::invalid_argument(
            "`__arrow_c_array__` must return (schema_capsule, array_capsule)",
        )
    })?;
    let array_capsule = capsules.get_item(1).map_err(|_| {
        SKPyError::invalid_argument(
            "`__arrow_c_array__` must return (schema_capsule, array_capsule)",
        )
    })?;
    let schema = unsafe { sk_py_take_schema(py, &schema_capsule)? };
    let array = unsafe { sk_py_take_array(py, &array_capsule)? };
    let data = unsafe { from_ffi(array, &schema) }.map_err(|error| {
        SKPyError::invalid_argument(format!("failed to import the Arrow array: {error}"))
    })?;
    let batch = if matches!(data.data_type(), DataType::Struct(_)) {
        let structure = StructArray::from(data);
        RecordBatch::from(&structure)
    } else {
        let field = Field::new("value", data.data_type().clone(), true);
        let schema = Arc::new(Schema::new(vec![field]));
        RecordBatch::try_new(schema, vec![make_array(data)])
            .map_err(|error| SKPyError::invalid_argument(error.to_string()))?
    };
    Ok(SKPyArrowSource::Single(Some(batch)))
}

unsafe fn sk_py_capsule_pointer(
    _py: Python<'_>,
    capsule: &Bound<'_, PyAny>,
    name: &CStr,
) -> Result<*mut c_void, SKPyError> {
    let raw = capsule.as_ptr();
    if unsafe { pyo3::ffi::PyCapsule_IsValid(raw, name.as_ptr()) } == 0 {
        return Err(SKPyError::invalid_argument(format!(
            "expected a PyCapsule named '{}', got a different capsule name",
            name.to_string_lossy()
        )));
    }
    let pointer = unsafe { pyo3::ffi::PyCapsule_GetPointer(raw, name.as_ptr()) };
    if pointer.is_null() {
        return Err(SKPyError::invalid_argument(format!(
            "PyCapsule '{}' carries a null pointer",
            name.to_string_lossy()
        )));
    }
    Ok(pointer)
}

unsafe fn sk_py_take_stream(
    py: Python<'_>,
    capsule: &Bound<'_, PyAny>,
) -> Result<FFI_ArrowArrayStream, SKPyError> {
    let pointer = unsafe { sk_py_capsule_pointer(py, capsule, STREAM_CAPSULE_NAME)? };
    let stream_pointer = pointer as *mut FFI_ArrowArrayStream;
    // SAFETY: the capsule owns a valid `FFI_ArrowArrayStream`; moving it out and
    // zeroing the capsule storage transfers ownership exactly once.
    let stream = unsafe { ptr::read(stream_pointer) };
    unsafe { ptr::write(stream_pointer, FFI_ArrowArrayStream::empty()) };
    Ok(stream)
}

unsafe fn sk_py_take_schema(
    py: Python<'_>,
    capsule: &Bound<'_, PyAny>,
) -> Result<FFI_ArrowSchema, SKPyError> {
    let pointer = unsafe { sk_py_capsule_pointer(py, capsule, SCHEMA_CAPSULE_NAME)? };
    let schema_pointer = pointer as *mut FFI_ArrowSchema;
    let schema = unsafe { ptr::read(schema_pointer) };
    unsafe { ptr::write(schema_pointer, FFI_ArrowSchema::empty()) };
    Ok(schema)
}

unsafe fn sk_py_take_array(
    py: Python<'_>,
    capsule: &Bound<'_, PyAny>,
) -> Result<FFI_ArrowArray, SKPyError> {
    let pointer = unsafe { sk_py_capsule_pointer(py, capsule, ARRAY_CAPSULE_NAME)? };
    let array_pointer = pointer as *mut FFI_ArrowArray;
    let array = unsafe { ptr::read(array_pointer) };
    unsafe { ptr::write(array_pointer, FFI_ArrowArray::empty()) };
    Ok(array)
}
