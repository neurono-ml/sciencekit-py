# Arrow tables, batches and capsules

Any object exporting the Arrow PyCapsule protocol is accepted: `pyarrow.Table`,
`pyarrow.RecordBatch`, `pyarrow.RecordBatchReader`, pandas frames, Polars
frames, DuckDB relations and third-party producers.

```python
import pyarrow as pa
from sciencekit import LinearRegression, describe

table = pa.table(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 1.0, 4.0, 3.0, 6.0],
        "y": [3.0, 5.0, 9.0, 11.0, 15.0],
    }
)

report = describe(table, target="y")
assert report.representation == "pyarrow.table"
assert report.arrow_zero_copy is True     # Arrow buffers imported in place
assert report.feature_names == ["a", "b"]

with LinearRegression() as builder:
    with builder.fit(table, target="y") as model:
        model.predict(table, columns=["a", "b"])
```

## Batches are consumed incrementally

A `RecordBatchReader` is pulled batch by batch; the whole table is never
materialized as one Arrow table:

```python
reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=2))
report = describe(reader, target="y")
assert report.batches == 3
```

Each batch contributes its rows to the dense design matrix in order, and the
conversion report counts the batches.

## Generic producers

The adapter does not import pandas or Polars to recognize them. A minimal class
is enough:

```python
class CapsuleProducer:
    def __init__(self, table):
        self._table = table

    def __arrow_c_stream__(self, requested_schema=None):
        return self._table.__arrow_c_stream__(requested_schema)

with LinearRegression() as builder:
    with builder.fit(CapsuleProducer(table), target="y") as model:
        assert model.score(table, targets, columns=["a", "b"]) == 1.0
```

This is the extension point for DuckDB, datafusion-python or any future library
that implements the protocol.

## dtypes and precision

Integer columns are promoted to the estimator precision without ceremony.
Floating-point precision is only changed when `cast=True` is passed, and the
cast is reported:

```python
float32_table = pa.table({"a": pa.array([1.0, 2.0], dtype=pa.float32())})
report = describe(float32_table, dtype="float64")
assert report.cast is True
```

Temporal, string and nested columns are not numeric features and raise
`UnsupportedColumnTypeError` naming the column and suggesting an explicit cast:

```python
try:
    builder.fit(pa.table({"when": pa.array([...], type=pa.timestamp("ns"))}), y)
except TypeError as error:
    assert "when" in str(error)
```

## The import contract

`sk_py_arrow_source` calls `__arrow_c_stream__`, takes ownership of the
capsule's `FFI_ArrowArrayStream`, and zeroes the capsule storage so its
destructor cannot release the struct twice. The same rule applies to the
`__arrow_c_array__` schema/array capsules. `arrow_zero_copy=True` records that
the Arrow buffers crossed the interface without an intermediate table copy; the
values are then assembled into the row-major design matrix the C ABI consumes.
