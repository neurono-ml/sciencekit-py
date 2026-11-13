# Result formats

`predict` and `transform` return a `numpy.ndarray` by default. The `output`
keyword asks for another representation without changing the numbers.

```python
with LinearRegression() as builder:
    with builder.fit(features, targets) as model:
        numpy_result = model.predict(features)              # default
        pandas_result = model.predict(features, output="pandas")
        arrow_result = model.predict(features, output="arrow")
        polars_result = model.predict(features, output="polars")
```

## The options

| `output=` | Prediction (1D) | Transform (2D) |
|---|---|---|
| `"numpy"` (default) | `numpy.ndarray` | `numpy.ndarray` |
| `"pandas"` | `pandas.Series` | `pandas.DataFrame` |
| `"arrow"` | `pyarrow.Table` | `pyarrow.Table` |
| `"polars"` | `polars.Series` | `polars.DataFrame` |
| `"sparse"` | `scipy.sparse.csr_matrix` (n×1) | `scipy.sparse.csr_matrix` |
| `"auto"` | mirrors the input representation | mirrors the input |

`"auto"` resolves from the input conversion: pandas → pandas, Polars → Polars,
pyarrow/pandas-Arrow-less/Dask/PySpark → Arrow, SciPy sparse → sparse, NumPy →
NumPy.

## Arrow results implement the PyCapsule protocol

An Arrow-backed result is a `pyarrow.Table`, so other libraries consume it
through `__arrow_c_stream__` without a copy:

```python
import pyarrow as pa

table = model.predict(features, output="arrow")
assert hasattr(table, "__arrow_c_stream__")
consumed = pa.RecordBatchReader.from_stream(table).read_all()
np.testing.assert_allclose(consumed.column("prediction").to_numpy(), targets)
```

Polars consumes the same object with `polars.from_arrow(table)`.

## Sparse results

`output="sparse"` wraps the dense block in a CSR matrix. The values round-trip
exactly:

```python
with StandardScaler() as scaler:
    with scaler.fit(matrix) as model:
        sparse_result = model.transform(matrix, output="sparse")
        dense_result = model.transform(matrix)

np.testing.assert_allclose(sparse_result.toarray(), dense_result)
```

A one-dimensional prediction is reshaped to a single column so it can be a valid
sparse matrix.

## Categorical predictions

When the model learned a label table from a dictionary target column, labels are
applied before the result is formatted. `output="pandas"` therefore returns a
`Series` of labels and `output="arrow"` a table whose `prediction` column is a
string array.

## Errors

An unknown `output=` value raises `InvalidArgumentError` listing every accepted
name.
