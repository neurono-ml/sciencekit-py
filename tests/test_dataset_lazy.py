"""Polars, Dask and PySpark inputs with documented eager fallbacks."""

from __future__ import annotations

import numpy as np
import pytest

import sciencekit
from sciencekit import LinearRegression, describe

FEATURES = np.array(
    [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]], dtype=np.float64
)
TARGETS = np.array([3.0, 5.0, 9.0, 11.0, 15.0], dtype=np.float64)


def _reference() -> np.ndarray:
    with LinearRegression() as builder:
        with builder.fit(FEATURES, TARGETS) as model:
            return model.predict(FEATURES)


def test_polars_dataframe_uses_the_arrow_protocol_without_pandas() -> None:
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame(
        {"a": FEATURES[:, 0], "b": FEATURES[:, 1], "y": TARGETS}
    )
    report = describe(frame, target="y")
    assert report.representation == "polars.dataframe"
    assert report.feature_names == ["a", "b"]
    assert report.arrow_zero_copy is True
    with LinearRegression() as builder:
        with builder.fit(frame, target="y") as model:
            np.testing.assert_allclose(model.predict(frame, columns=["a", "b"]), _reference())


def test_polars_lazyframe_is_collected_in_batches() -> None:
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"a": FEATURES[:, 0], "b": FEATURES[:, 1], "y": TARGETS})
    lazy = frame.lazy()
    report = describe(lazy, target="y")
    assert report.representation == "polars.lazyframe"
    assert report.eager_fallback == "polars.lazyframe.collect"
    assert report.batches >= 1
    with LinearRegression() as builder:
        with builder.fit(lazy, target="y") as model:
            assert builder.last_conversion.eager_fallback == "polars.lazyframe.collect"
            np.testing.assert_allclose(model.predict(frame, columns=["a", "b"]), _reference())


def test_dask_partitions_are_consumed_in_order() -> None:
    dd = pytest.importorskip("dask.dataframe")
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"a": FEATURES[:, 0], "b": FEATURES[:, 1], "y": TARGETS},
        index=np.arange(5),
    )
    dask_frame = dd.from_pandas(frame, npartitions=3)
    report = describe(dask_frame, target="y")
    assert report.representation == "dask.dataframe"
    assert report.partitions == 3
    assert report.partition_sizes == [2, 2, 1]
    assert report.eager_fallback == "dask.partitions"

    with LinearRegression() as builder:
        with builder.fit(dask_frame, target="y") as model:
            dask_predictions = model.predict(dask_frame, columns=["a", "b"])
        with builder.fit(frame, target="y") as pandas_model:
            pandas_predictions = pandas_model.predict(frame, columns=["a", "b"])
    np.testing.assert_allclose(dask_predictions, pandas_predictions, rtol=1e-12)


def test_dask_standard_scaler_matches_concatenated_fit() -> None:
    dd = pytest.importorskip("dask.dataframe")
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(FEATURES, columns=["a", "b"])
    dask_frame = dd.from_pandas(frame, npartitions=2)
    with sciencekit.StandardScaler() as scaler:
        streamed = scaler.fit_transform(dask_frame)
        with scaler.fit(frame) as model:
            concatenated = model.transform(frame)
    np.testing.assert_allclose(streamed, concatenated, rtol=1e-12, atol=1e-12)


class _FakeArrowFrame:
    """Duck-typed stand-in for a Spark 4 DataFrame exposing `toArrow()`."""

    __module__ = "pyspark.sql.dataframe"

    def __init__(self, table) -> None:  # noqa: ANN001
        self._table = table

    def toArrow(self):  # noqa: ANN201
        return self._table


def test_pyspark_driver_path_via_to_arrow() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table(
        {"a": FEATURES[:, 0], "b": FEATURES[:, 1], "y": TARGETS}
    )
    frame = _FakeArrowFrame(table)
    report = describe(frame, target="y")
    assert report.representation == "pyspark.dataframe"
    assert report.eager_fallback is not None
    assert report.eager_fallback.startswith("pyspark.toArrow")
    with LinearRegression() as builder:
        with builder.fit(frame, target="y") as model:
            np.testing.assert_allclose(model.predict(table, columns=["a", "b"]), _reference())


def test_map_in_arrow_requires_a_spark_frame() -> None:
    from sciencekit.datasets import map_in_arrow

    with pytest.raises(TypeError):
        map_in_arrow(object(), lambda batch: batch, schema=None)


def test_pyspark_local_frame_matches_pandas() -> None:
    pyspark = pytest.importorskip("pyspark")
    pd = pytest.importorskip("pandas")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("sciencekit-dataset-test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        frame = pd.DataFrame({"a": FEATURES[:, 0], "b": FEATURES[:, 1], "y": TARGETS})
        spark_frame = spark.createDataFrame(frame)
        report = describe(spark_frame, target="y")
        assert report.representation == "pyspark.dataframe"
        with LinearRegression() as builder:
            with builder.fit(spark_frame, target="y") as model:
                spark_predictions = model.predict(spark_frame, columns=["a", "b"])
        with LinearRegression() as builder:
            with builder.fit(frame, target="y") as model:
                pandas_predictions = model.predict(frame, columns=["a", "b"])
        np.testing.assert_allclose(spark_predictions, pandas_predictions, rtol=1e-9)
    finally:
        spark.stop()
