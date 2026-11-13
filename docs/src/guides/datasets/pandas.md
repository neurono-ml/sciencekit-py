# pandas DataFrames and Series

pandas ≥ 2.2 exports frames through the Arrow PyCapsule protocol, so a
`DataFrame` is consumed by the same code path as pyarrow — no intermediate
`to_numpy` copy.

```python
import numpy as np
import pandas as pd
from sciencekit import LinearRegression, describe

frame = pd.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 1.0, 4.0, 3.0, 6.0],
        "y": [3.0, 5.0, 9.0, 11.0, 15.0],
    }
)

report = describe(frame, target="y")
assert report.representation == "pandas.dataframe"
assert report.feature_names == ["a", "b"]

with LinearRegression() as builder:
    with builder.fit(frame, target="y") as model:
        predictions = model.predict(frame, columns=["a", "b"])
```

`fit(frame, target="y")` takes the target from the same frame; every other
column becomes a feature unless `columns=` selects a subset. A `Series` can be
passed as the positional `targets` argument:

```python
with LinearRegression() as builder:
    with builder.fit(frame, frame["y"]) as model:
        assert model.score(frame, frame["y"]) == pytest.approx(1.0)
```

## The index is metadata, not a feature

pandas exports a non-default index as a helper column named
`__index_level_0__`. The adapter drops those helper columns: only real columns
become features.

```python
indexed = frame.set_index(pd.Index([100, 200, 300, 400, 500]))
assert describe(indexed).feature_names == ["a", "b", "y"]
```

Naming an index column explicitly in `columns=` keeps it, because the caller
asked for it.

## Nullable dtypes

pandas nullable columns (`Float64`, `Int64`, …) arrive as Arrow validity
bitmaps, so the missing-value policy applies uniformly:

```python
frame = pd.DataFrame(
    {
        "a": pd.array([1.0, None, 3.0, 4.0, 5.0], dtype="Float64"),
        "b": pd.array([2.0, 1.0, 4.0, 3.0, 6.0], dtype="Float64"),
    }
)
with LinearRegression() as builder:
    try:
        builder.fit(frame, targets)          # on_missing="reject"
    except sciencekit.InvalidArgumentError as error:
        assert "'a'" in error.message

    with builder.fit(frame, targets, on_missing="fill", fill_value=0.0) as model:
        model.predict(frame, on_missing="fill", fill_value=0.0)
```

`on_missing="nan"` keeps `NaN`, which propagates through standardization.

## Categorical targets

`CategoricalDtype` arrives as an Arrow dictionary. The category order becomes
the label table, the numeric codes are fitted, and predictions map back to the
labels:

```python
frame = pd.DataFrame(
    {
        "x": [0.0, 1.0, 2.0, 3.0],
        "label": pd.Categorical(["low", "low", "high", "high"]),
    }
)
with LinearRegression() as builder:
    with builder.fit(frame, target="label") as model:
        assert model.label_table in (["high", "low"], ["low", "high"])
        assert list(model.predict(frame, columns=["x"])) == [
            "low", "low", "high", "high"
        ]
```

The label table is also reported by `describe(frame, target="label").label_table`.

## Older pandas

pandas < 2.2 has no `__arrow_c_stream__`. The adapter then converts each
selected column with `Series.to_numpy(dtype=…)` and records the copy:

```python
report = describe(legacy_frame, target="y")
assert report.eager_fallback == "pandas.to_numpy"
assert report.copied is True
```

The fallback is best-effort: object columns that are not numeric raise
`UnsupportedColumnTypeError` with the suggested cast.

## Unknown columns

```python
with LinearRegression() as builder:
    try:
        builder.fit(frame, target="missing")
    except KeyError as error:            # UnknownColumnError
        assert "missing" in str(error)
```
