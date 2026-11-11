"""Linear regression fit, predict, score, backend, and f64/f32 tests."""

from __future__ import annotations

import numpy as np
import pytest

import sciencekit
from sciencekit.linear_model import LinearRegression, LinearRegressionBuilder, LinearRegressionModel


def test_linear_regression_f64_known_fixture_and_backends() -> None:
    # Exact plane: y = 1.0 + 2.0 * x0 + 3.0 * x1
    features = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
        dtype=np.float64,
    )
    targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

    for backend in ("faer", "matrixmultiply", "ndarray-linalg"):
        with LinearRegression(dtype="float64") as builder:
            assert isinstance(builder, LinearRegressionBuilder)
            assert builder.family == "linear_regression"
            assert builder.dtype == "float64"
            assert builder.closed is False

            returned = builder.include_intercept(True).backend(backend)
            assert returned is builder

            with builder.fit(features, targets) as model:
                assert isinstance(model, LinearRegressionModel)
                assert model.family == "linear_regression"
                assert model.dtype == "float64"
                assert model.closed is False

                predictions = model.predict(features)
                assert predictions.dtype == np.float64
                assert predictions.shape == (4,)
                np.testing.assert_allclose(predictions, targets, rtol=1e-10, atol=1e-10)

                r_squared = model.score(features, targets)
                assert isinstance(r_squared, float)
                assert r_squared == pytest.approx(1.0, abs=1e-12)


def test_linear_regression_without_intercept_and_f32_precision() -> None:
    # Exact plane through origin: y = 2.0 * x0 - 4.0 * x1
    features = np.array(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [3.0, -1.0]],
        dtype=np.float32,
    )
    targets = np.array([2.0, -4.0, 0.0, 10.0], dtype=np.float32)

    builder = sciencekit.linear_regression(dtype=np.float32, include_intercept=False)
    assert builder.dtype == "float32"
    model = builder.fit(features, targets)
    try:
        assert model.dtype == "float32"
        predictions = model.predict(features)
        assert predictions.dtype == np.float32
        assert predictions.shape == (4,)
        np.testing.assert_allclose(predictions, targets, rtol=1e-5, atol=1e-5)
        assert model.score(features, targets) == pytest.approx(1.0, abs=1e-5)
    finally:
        model.close()
        builder.close()


def test_linear_regression_supports_explicit_copy_for_strided_views() -> None:
    raw = np.array(
        [
            [0.0, 99.0, 0.0],
            [1.0, 99.0, 0.0],
            [0.0, 99.0, 1.0],
            [2.0, 99.0, 3.0],
        ],
        dtype=np.float64,
    )
    strided_features = raw[:, ::2]
    targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

    with pytest.raises(sciencekit.UnsupportedError):
        with LinearRegression() as builder:
            builder.fit(strided_features, targets)

    with LinearRegression() as builder:
        with builder.fit(strided_features, targets, copy=True) as model:
            predictions = model.predict(strided_features, copy=True)
            np.testing.assert_allclose(predictions, targets, rtol=1e-10, atol=1e-10)
