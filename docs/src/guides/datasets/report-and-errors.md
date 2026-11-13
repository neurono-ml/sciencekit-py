# Conversion reports, missing values and errors

## ConversionReport

Every conversion returns a `ConversionReport` through
`builder.last_conversion` / `model.last_conversion`, and `describe(dataset)`
produces one without fitting.

| Property | Meaning |
|---|---|
| `representation` | input family, e.g. `pandas.dataframe`, `scipy.csc` |
| `dtype` | effective precision (`float32`/`float64`) |
| `shape` | `(rows, columns)` of the converted block |
| `zero_copy` | the values are borrowed from the Python object |
| `copied` | any element copy happened |
| `densified` | the conversion layer materialized a dense block from sparse |
| `arrow_zero_copy` | Arrow buffers crossed the C Data Interface in place |
| `cast` | the precision changed |
| `converted_from` | `csc`/`coo` when a sparse layout was converted |
| `eager_fallback` | documented fallback for a lazy/distributed container |
| `batches` | Arrow record batches consumed |
| `partitions`, `partition_sizes` | Dask partitions consumed, in order |
| `feature_names` | feature columns |
| `data_pointer`, `indices_pointer`, `indptr_pointer` | borrowed buffers |
| `stored_zeros` | explicitly stored zeros in a CSR input |
| `label_table` | categorical categories |
| `abi_dense_required` | the C ABI needs a dense block at the call boundary |
| `missing_policy` | applied `on_missing` policy |

The pointers are the zero-copy evidence. A conversion that copies reports a
different pointer (or none) and `zero_copy=False`.

## Missing-value policy

```python
# reject (default): raises, naming column and row
try:
    builder.fit(frame, targets)
except sciencekit.InvalidArgumentError as error:
    assert "'a'" in error.message

# nan: keeps NaN
with StandardScaler() as scaler:
    with scaler.fit(frame, on_missing="nan") as model:
        transformed = model.transform(frame, on_missing="nan")
        assert np.isnan(transformed[1]).any()

# fill: constant replacement
with builder.fit(frame, targets, on_missing="fill", fill_value=0.0) as model:
    model.predict(frame, on_missing="fill", fill_value=0.0)
```

`on_missing="fill"` without `fill_value` is an argument error. An unknown policy
name lists the accepted values.

## Error mapping

| Failure | Exception | Standard base |
|---|---|---|
| unknown column / feature name | `UnknownColumnError` | `KeyError` |
| non-numeric column dtype | `UnsupportedColumnTypeError` | `TypeError` |
| shape disagreement | `ShapeError` | `ValueError` |
| unsupported representation | `UnsupportedError` | `NotImplementedError` |
| malformed argument, dtype, rank | `InvalidArgumentError` | `ValueError` |

Every one of them is also a `ScienceKitError` with `.status_code`,
`.status_name` and `.message`, so generic handling keeps working.

```python
with LinearRegression() as builder:
    try:
        builder.fit(datetime_frame, targets)
    except TypeError as error:
        assert "when" in str(error)
        assert "cast" in str(error)
```

Datetime, duration, string, binary and nested Arrow dtypes are rejected because
a timestamp is not a numeric feature; the message names the column and suggests
the cast.

## Concurrency

Conversion happens while holding the GIL and takes ownership of the values; the
fit/predict call runs with `py.detach`, releasing the GIL. Two Python threads
can fit independent models concurrently:

```python
threads = [threading.Thread(target=worker, args=(scale,)) for scale in (1.0, 3.0)]
```

No deadlock, no cross-talk: each builder owns its own opaque ABI handle and each
fitted model is immutable.
