# Datasets and representations

`sciencekit` accepts the table and array representations the Python scientific
ecosystem already uses: NumPy arrays, pandas frames, SciPy sparse matrices,
pyarrow tables and readers, Polars frames, Dask DataFrames and PySpark
DataFrames. The same estimator fits all of them and produces the same numbers.

```python
import numpy as np
from sciencekit.linear_model import LinearRegression

features = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]])
targets = np.array([3.0, 5.0, 9.0, 11.0, 15.0])

with LinearRegression() as builder:
    with builder.fit(features, targets) as model:
        predictions = model.predict(features)
```

Every operation returns the conversion it performed through
`builder.last_conversion` and `model.last_conversion`, so a zero-copy claim is
observable rather than assumed:

```python
report = builder.last_conversion
assert report.zero_copy is True
assert report.data_pointer == features.__array_interface__["data"][0]
```

## Dispatch order

One dispatcher recognizes the input, in this order:

1. `numpy.ndarray` — borrowed zero-copy when C-contiguous.
2. `scipy.sparse` matrices — CSR buffers borrowed, CSC/COO converted.
3. `dask.dataframe.DataFrame` — partitions consumed in order.
4. `pyspark.sql.DataFrame` — driver `toArrow()` collect.
5. `polars.LazyFrame` — collected eagerly.
6. any object with `__arrow_c_stream__` / `__arrow_c_array__` — pyarrow,
   pandas ≥ 2.2, Polars `DataFrame`, DuckDB relations and future producers.
7. pandas < 2.2 — documented `to_numpy` fallback.
8. anything else — `UnsupportedError` listing the accepted types.

`sciencekit.supported_representations()` returns that catalogue, and
`sciencekit.describe(dataset)` reports how a dataset *would* be converted
without fitting.

## The neutral seam and the ABI boundary

The stable `sciencekit_ffi` C ABI (1.0) accepts dense `float32`/`float64`
blocks. The dataset layer therefore keeps a sparse input sparse — it borrows
`data`, `indices` and `indptr` and reports `densified=False` — and materializes
a dense block only at the C call boundary, which `ConversionReport` reports as
`abi_dense_required=True`. No conversion is silent.

| Representation | Zero-copy | Copy | Documented fallback |
|---|---|---|---|
| NumPy C-contiguous `f32`/`f64` | yes | no | `copy=True` for strided views |
| SciPy CSR | yes (3 buffers) | no | dense block at the ABI boundary |
| SciPy CSC / COO | no | yes | explicit `tocsr()` |
| pyarrow / generic PyCapsule | Arrow buffers | design matrix | none |
| pandas DataFrame | Arrow buffers | design matrix | `to_numpy` for pandas < 2.2 |
| Polars DataFrame | Arrow buffers | design matrix | none |
| Polars LazyFrame | Arrow buffers | design matrix | `collect()` |
| Dask DataFrame | no | yes | partitions joined eagerly |
| PySpark DataFrame | no | yes | `toArrow()` driver collect |

## Missing values

`on_missing` selects the policy for null elements:

- `"reject"` (default) raises `InvalidArgumentError` naming the column and row;
- `"nan"` keeps the missing element as `NaN`;
- `"fill"` replaces it with `fill_value`.

## Result formats

`predict` and `transform` return NumPy arrays by default. Pass `output=` to ask
for another representation: `"pandas"`, `"arrow"`, `"polars"`, `"sparse"`, or
`"auto"` to mirror the input representation. See
[Result formats](./datasets/results.md).

## Errors

Unknown columns raise `UnknownColumnError`, a `KeyError`; non-numeric columns
raise `UnsupportedColumnTypeError`, a `TypeError` naming the column and the
suggested cast; shape disagreements raise `ShapeError`, a `ValueError`. All of
them remain `ScienceKitError` subclasses with `.status_code`.
