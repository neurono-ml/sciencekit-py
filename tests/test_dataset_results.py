"""Result formats, PyCapsule results, threading and error mapping."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import sciencekit
from sciencekit import (
    InvalidArgumentError,
    LinearRegression,
    NotFittedError,
    ShapeError,
    StandardScaler,
    UnsupportedColumnTypeError,
)

FEATURES = np.array(
    [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]], dtype=np.float64
)
TARGETS = np.array([3.0, 5.0, 9.0, 11.0, 15.0], dtype=np.float64)


def _fitted():
    builder = LinearRegression()
    model = builder.fit(FEATURES, TARGETS)
    return builder, model


def test_default_prediction_is_a_numpy_array() -> None:
    builder, model = _fitted()
    try:
        predictions = model.predict(FEATURES)
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == np.float64
        np.testing.assert_allclose(predictions, TARGETS, rtol=1e-10)
    finally:
        model.close()
        builder.close()


def test_pandas_result_round_trips_values() -> None:
    pd = pytest.importorskip("pandas")
    builder, model = _fitted()
    try:
        series = model.predict(FEATURES, output="pandas")
        assert isinstance(series, pd.Series)
        np.testing.assert_allclose(series.to_numpy(), TARGETS, rtol=1e-10)
    finally:
        model.close()
        builder.close()


def test_arrow_result_supports_the_pycapsule_protocol() -> None:
    pa = pytest.importorskip("pyarrow")
    builder, model = _fitted()
    try:
        table = model.predict(FEATURES, output="arrow")
        assert isinstance(table, pa.Table)
        assert hasattr(table, "__arrow_c_stream__")
        consumed = pa.RecordBatchReader.from_stream(table).read_all()
        np.testing.assert_allclose(
            consumed.column("prediction").to_numpy(), TARGETS, rtol=1e-10
        )
    finally:
        model.close()
        builder.close()


def test_polars_result_round_trips_values() -> None:
    pl = pytest.importorskip("polars")
    builder, model = _fitted()
    try:
        series = model.predict(FEATURES, output="polars")
        assert isinstance(series, pl.Series)
        np.testing.assert_allclose(series.to_numpy(), TARGETS, rtol=1e-10)
    finally:
        model.close()
        builder.close()


def test_sparse_result_for_transform() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    with StandardScaler() as scaler:
        with scaler.fit(FEATURES) as model:
            result = model.transform(FEATURES, output="sparse")
            dense = model.transform(FEATURES)
    assert sparse.issparse(result)
    np.testing.assert_allclose(result.toarray(), dense, rtol=1e-12)


def test_auto_output_mirrors_the_input_representation() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(FEATURES, columns=["a", "b"])
    with LinearRegression() as builder:
        with builder.fit(frame, TARGETS) as model:
            result = model.predict(frame, output="auto")
    assert isinstance(result, pd.Series)


def test_unknown_output_format_lists_the_options() -> None:
    builder, model = _fitted()
    try:
        with pytest.raises(InvalidArgumentError) as error:
            model.predict(FEATURES, output="json")
        assert "'numpy', 'pandas', 'arrow'" in error.value.message
    finally:
        model.close()
        builder.close()


def test_concurrent_fits_release_the_gil_and_stay_independent() -> None:
    results: dict[float, np.ndarray] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker(scale: float) -> None:
        try:
            features = FEATURES * scale
            targets = TARGETS * scale
            barrier.wait(timeout=5)
            with LinearRegression() as builder:
                with builder.fit(features, targets) as model:
                    results[scale] = model.predict(features)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(scale,)) for scale in (1.0, 3.0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert set(results) == {1.0, 3.0}
    np.testing.assert_allclose(results[1.0], TARGETS, rtol=1e-10)
    np.testing.assert_allclose(results[3.0], TARGETS * 3.0, rtol=1e-10)


def test_shape_mismatch_raises_value_error() -> None:
    with LinearRegression() as builder:
        with pytest.raises(ShapeError) as error:
            builder.fit(FEATURES, TARGETS[:3])
        assert isinstance(error.value, ValueError)


def test_datetime_column_raises_type_error() -> None:
    pa = pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame(
        {
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "b": [1.0, 2.0, 3.0],
        }
    )
    with LinearRegression() as builder:
        with pytest.raises(TypeError) as error:
            builder.fit(frame, np.array([1.0, 2.0, 3.0]))
        assert isinstance(error.value, UnsupportedColumnTypeError)
        assert "when" in str(error.value)
        assert "cast" in str(error.value)

    table = pa.table({"when": pa.array(frame["when"]), "b": pa.array([1.0, 2.0, 3.0])})
    with LinearRegression() as builder:
        with pytest.raises(UnsupportedColumnTypeError):
            builder.fit(table, np.array([1.0, 2.0, 3.0]))


def test_large_and_small_inputs_are_supported() -> None:
    generator = np.random.default_rng(7)
    large = generator.normal(size=(20_000, 4))
    weights = np.array([1.5, -2.0, 0.5, 3.0])
    large_targets = large @ weights + 0.25
    with LinearRegression() as builder:
        with builder.fit(large, large_targets) as model:
            assert model.score(large, large_targets) > 0.99

    small = np.array([[1.0]], dtype=np.float64)
    with LinearRegression() as builder:
        with builder.fit(small, np.array([2.0], dtype=np.float64)) as model:
            np.testing.assert_allclose(model.predict(small), [2.0])

    frame = sciencekit.datasets.describe(
        np.ascontiguousarray(large.astype(np.float32)), dtype="float32"
    )
    assert frame.zero_copy is True
