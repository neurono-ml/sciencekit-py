# NumPy arrays

A C-contiguous `float32` or `float64` array is borrowed, never copied. The
Pointer recorded in the conversion report is the caller's buffer:

```python
import numpy as np
from sciencekit import LinearRegression, describe

features = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], dtype=np.float64)
targets = np.array([3.0, 5.0, 9.0], dtype=np.float64)

report = describe(features, dtype="float64")
assert report.zero_copy is True
assert report.copied is False
assert report.data_pointer == features.__array_interface__["data"][0]

with LinearRegression() as builder:
    with builder.fit(features, targets) as model:
        assert builder.last_conversion.data_pointer == (
            features.__array_interface__["data"][0]
        )
        predictions = model.predict(features)
```

The `float32` path is identical: `LinearRegression(dtype="float32").fit(x32, y32)`
borrows the `float32` buffer.

## Strided views require an explicit copy

A transposed or strided view is rejected instead of being silently copied:

```python
strided = np.arange(20.0).reshape(4, 5)[:, ::2]
with LinearRegression() as builder:
    try:
        builder.fit(strided, np.array([1.0, 2.0, 3.0, 4.0]))
    except sciencekit.UnsupportedError as error:
        assert "copy=True" in error.message

    with builder.fit(strided, np.array([1.0, 2.0, 3.0, 4.0]), copy=True) as model:
        assert builder.last_conversion.copied is True
        assert builder.last_conversion.zero_copy is False
```

`copy=True` also works for `predict`, `transform` and `score`.

## Precision mismatches require `cast=True`

```python
x32 = features.astype(np.float32)
with LinearRegression() as builder:
    try:
        builder.fit(x32, targets)
    except sciencekit.InvalidArgumentError as error:
        assert "cast=True" in error.message

    with builder.fit(x32, targets.astype(np.float32), cast=True) as model:
        assert builder.last_conversion.cast is True
        assert model.predict(x32, cast=True).shape == targets.shape
```

A cast is reported as `cast=True` and is never zero-copy, because a new buffer
holds the promoted values.

## Column names

NumPy arrays have no column names. Passing `columns=` raises an
`InvalidArgumentError`; use a pandas, Arrow or Polars frame to select features
by name.
