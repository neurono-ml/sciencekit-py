"""NumPy datasets: zero-copy borrow, explicit copy/cast (tasks 1.2, 1.3)."""

from __future__ import annotations

import numpy as np
import pytest

import sciencekit
from sciencekit import (
    ConversionReport,
    InvalidArgumentError,
    LinearRegression,
    StandardScaler,
    UnsupportedError,
    describe,
)

FEATURES = np.array(
    [
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 6.0],
    ],
    dtype=np.float64,
    order="C",
)
TARGETS = np.array([3.0, 5.0, 9.0, 11.0, 15.0], dtype=np.float64)


def _pointer(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0])


def test_contiguous_float64_array_is_borrowed_zero_copy() -> None:
    report = describe(FEATURES, dtype="float64")
    assert isinstance(report, ConversionReport)
    assert report.representation == "numpy.ndarray"
    assert report.zero_copy is True
    assert report.copied is False
    assert report.data_pointer == _pointer(FEATURES)
    assert report.shape == (5, 2)

    with LinearRegression() as builder:
        with builder.fit(FEATURES, TARGETS) as model:
            # The pointer the ABI call consumed is the caller's buffer.
            fit_report = builder.last_conversion
            assert fit_report is not None
            assert fit_report.zero_copy is True
            assert fit_report.data_pointer == _pointer(FEATURES)

            predictions = model.predict(FEATURES)
            predict_report = model.last_conversion
            assert predict_report is not None
            assert predict_report.data_pointer == _pointer(FEATURES)
            assert predict_report.zero_copy is True
            np.testing.assert_allclose(predictions, TARGETS, rtol=1e-10, atol=1e-10)


def test_contiguous_float32_array_is_borrowed_zero_copy() -> None:
    features = np.ascontiguousarray(FEATURES, dtype=np.float32)
    targets = np.ascontiguousarray(TARGETS, dtype=np.float32)
    report = describe(features, dtype="float32")
    assert report.zero_copy is True
    assert report.data_pointer == _pointer(features)
    with LinearRegression(dtype="float32") as builder:
        with builder.fit(features, targets) as model:
            assert builder.last_conversion.data_pointer == _pointer(features)
            np.testing.assert_allclose(model.predict(features), targets, rtol=1e-5, atol=1e-5)


def test_readonly_contiguous_array_is_still_borrowed() -> None:
    features = FEATURES.copy()
    features.setflags(write=False)
    with LinearRegression() as builder:
        with builder.fit(features, TARGETS) as model:
            report = builder.last_conversion
            assert report is not None
            assert report.zero_copy is True
            assert report.data_pointer == _pointer(features)
            assert model.last_conversion is not None

def test_non_contiguous_array_is_not_silently_copied() -> None:
    strided = np.arange(20.0, dtype=np.float64).reshape(4, 5)[:, ::2]
    assert not strided.flags["C_CONTIGUOUS"]
    targets = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    with LinearRegression() as builder:
        with pytest.raises(UnsupportedError) as error:
            builder.fit(strided, targets)
        assert "copy=True" in error.value.message
        assert "ascontiguousarray" in error.value.message

        with builder.fit(strided, targets, copy=True) as model:
            report = builder.last_conversion
            assert report is not None
            assert report.copied is True
            assert report.zero_copy is False
            assert report.data_pointer != _pointer(strided)
            assert np.isfinite(model.predict(strided, copy=True)).all()


def test_wrong_dtype_requires_an_explicit_cast() -> None:
    float32_features = np.ascontiguousarray(FEATURES, dtype=np.float32)
    with LinearRegression() as builder:
        with pytest.raises(InvalidArgumentError) as error:
            builder.fit(float32_features, TARGETS)
        assert "cast=True" in error.value.message
        assert "float32" in error.value.message

        with builder.fit(
            float32_features, TARGETS.astype(np.float32), cast=True
        ) as model:
            report = builder.last_conversion
            assert report is not None
            assert report.cast is True
            assert report.zero_copy is False
            np.testing.assert_allclose(model.predict(float32_features, cast=True), TARGETS, rtol=1e-5)


def test_column_selection_is_rejected_for_unnamed_arrays() -> None:
    with pytest.raises(InvalidArgumentError) as error:
        describe(FEATURES, columns=["a"])
    assert "frame representation" in error.value.message


def test_standard_scaler_borrows_and_accepts_frames_by_name() -> None:
    with StandardScaler() as scaler:
        transformed = scaler.fit_transform(FEATURES)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
        report = scaler.last_conversion
        assert report is not None
        assert report.zero_copy is True
