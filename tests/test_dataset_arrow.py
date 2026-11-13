"""Arrow PyCapsule producers: pyarrow objects and generic capsules (tasks 2.1, 2.6)."""

from __future__ import annotations

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

import sciencekit
from sciencekit import LinearRegression, describe

FEATURES = np.array(
    [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]], dtype=np.float64
)
TARGETS = np.array([3.0, 5.0, 9.0, 11.0, 15.0], dtype=np.float64)


def _table() -> "pa.Table":
    return pa.table(
        {
            "a": pa.array(FEATURES[:, 0]),
            "b": pa.array(FEATURES[:, 1]),
            "y": pa.array(TARGETS),
        }
    )


def _reference_predictions() -> np.ndarray:
    with LinearRegression() as builder:
        with builder.fit(FEATURES, TARGETS) as model:
            return model.predict(FEATURES)


def test_pyarrow_table_is_consumed_batch_by_batch() -> None:
    table = _table()
    report = describe(table, target="y")
    assert report.representation == "pyarrow.table"
    assert report.arrow_zero_copy is True
    assert report.feature_names == ["a", "b"]
    assert report.batches >= 1

    with LinearRegression() as builder:
        with builder.fit(table, target="y") as model:
            np.testing.assert_allclose(model.predict(table, columns=["a", "b"]), _reference_predictions())


def test_pyarrow_record_batch_uses_the_c_array_interface() -> None:
    batch = pa.RecordBatch.from_arrays(
        [pa.array(FEATURES[:, 0]), pa.array(FEATURES[:, 1]), pa.array(TARGETS)],
        names=["a", "b", "y"],
    )
    report = describe(batch, target="y")
    assert report.shape == (5, 2)
    assert report.arrow_zero_copy is True
    with LinearRegression() as builder:
        with builder.fit(batch, target="y") as model:
            assert model.score(batch, TARGETS, columns=["a", "b"]) == pytest.approx(1.0)


def test_record_batch_reader_batches_are_counted() -> None:
    table = _table()
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=2))
    report = describe(reader, target="y")
    assert report.representation in {"pyarrow.record_batch_reader", "pyarrow.table"}
    assert report.batches == 3
    assert report.shape == (5, 2)

    fresh = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=2))
    with LinearRegression() as builder:
        with builder.fit(fresh, target="y") as model:
            np.testing.assert_allclose(
                model.predict(_table(), columns=["a", "b"]), _reference_predictions()
            )


class _CapsuleProducer:
    """Minimal third-party Arrow PyCapsule producer with no pandas involved."""

    def __init__(self, table: "pa.Table") -> None:
        self._table = table

    def __arrow_c_stream__(self, requested_schema=None):  # noqa: ANN001
        return self._table.__arrow_c_stream__(requested_schema)


def test_generic_capsule_producer_is_accepted() -> None:
    producer = _CapsuleProducer(_table())
    report = describe(producer, target="y")
    assert report.representation in {"arrow.capsule", "pyarrow.table"}
    assert report.arrow_zero_copy is True
    with LinearRegression() as builder:
        with builder.fit(producer, target="y") as model:
            assert model.score(_table(), TARGETS, columns=["a", "b"]) == pytest.approx(1.0)


def test_float32_arrow_table_is_cast_explicitly() -> None:
    table = pa.table({"a": pa.array(FEATURES[:, 0].astype(np.float32))})
    report = describe(table, dtype="float64")
    assert report.cast is True
    with LinearRegression() as builder:
        with builder.fit(table, TARGETS, cast=True) as model:
            assert np.isfinite(model.predict(table, cast=True)).all()
