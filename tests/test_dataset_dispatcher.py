"""Dispatcher catalogue and unsupported-representation errors (tasks 1.1, 1.4)."""

from __future__ import annotations

import numpy as np
import pytest

import sciencekit
from sciencekit import ConversionReport, UnsupportedError, describe, supported_representations


def test_supported_representations_are_documented() -> None:
    representations = supported_representations()
    assert isinstance(representations, list)
    joined = ", ".join(representations)
    for name in (
        "numpy.ndarray",
        "pandas.DataFrame",
        "scipy.sparse.csr_matrix",
        "pyarrow.Table",
        "polars.DataFrame",
        "dask.dataframe.DataFrame",
        "pyspark.sql.DataFrame",
    ):
        assert name in joined


def test_unsupported_object_lists_supported_representations() -> None:
    with pytest.raises(UnsupportedError) as error:
        describe({"a": [1.0, 2.0]})
    message = error.value.message
    assert "numpy.ndarray" in message
    assert "pandas.DataFrame" in message
    assert "scipy.sparse.csr_matrix" in message

    with pytest.raises(UnsupportedError):
        with sciencekit.LinearRegression() as builder:
            builder.fit([[1.0, 2.0], [3.0, 4.0]], np.array([1.0, 2.0]))


def test_describe_reports_shape_and_representation() -> None:
    report = describe(np.zeros((3, 4), dtype=np.float64))
    assert isinstance(report, ConversionReport)
    assert report.shape == (3, 4)
    assert report.dtype == "float64"
    assert report.representation == "numpy.ndarray"
    assert "ConversionReport" in repr(report)
