"""Model persistence round-trip and ABI persistence boundary tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sciencekit
from sciencekit import IoError, InvalidArgumentError, UnsupportedError
from sciencekit.linear_model import LinearRegression, LinearRegressionModel
from sciencekit.preprocessing import StandardScaler


def test_linear_regression_persistence_round_trip(tmp_path: Path) -> None:
    features = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
        dtype=np.float64,
    )
    targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)
    probe = np.array([[4.0, -1.0], [-2.0, 5.0]], dtype=np.float64)
    state_file = tmp_path / "linear_regression.state"

    with LinearRegression() as builder:
        with builder.fit(features, targets) as fitted:
            expected_predictions = fitted.predict(probe)
            expected_score = fitted.score(features, targets)
            fitted.save(state_file)

    assert state_file.is_file()

    with sciencekit.load(state_file) as loaded:
        assert isinstance(loaded, LinearRegressionModel)
        assert loaded.family == "linear_regression"
        assert loaded.dtype == "float64"
        np.testing.assert_allclose(loaded.predict(probe), expected_predictions, rtol=1e-12, atol=1e-12)
        assert loaded.score(features, targets) == pytest.approx(expected_score, abs=1e-12)

    with LinearRegressionModel.load(str(state_file)) as loaded_via_cls:
        np.testing.assert_allclose(loaded_via_cls.predict(probe), expected_predictions, rtol=1e-12, atol=1e-12)


def test_persistence_rejects_f32_scaler_missing_and_corrupt_files(tmp_path: Path) -> None:
    features_f32 = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    targets_f32 = np.array([1.0, 2.0, 7.0], dtype=np.float32)

    with LinearRegression(dtype="float32") as builder_f32:
        with builder_f32.fit(features_f32, targets_f32) as model_f32:
            with pytest.raises(UnsupportedError) as exc_f32:
                model_f32.save(tmp_path / "f32.state")
            assert exc_f32.value.status_code == sciencekit.STATUS_UNSUPPORTED

    features_f64 = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]], dtype=np.float64)
    with StandardScaler() as scaler:
        with scaler.fit(features_f64) as scaler_model:
            with pytest.raises(UnsupportedError) as exc_scaler:
                scaler_model.save(tmp_path / "scaler.state")
            assert exc_scaler.value.status_code == sciencekit.STATUS_UNSUPPORTED

    with pytest.raises(IoError) as exc_missing:
        sciencekit.load(tmp_path / "does_not_exist.state")
    assert exc_missing.value.status_code == sciencekit.STATUS_IO
    assert isinstance(exc_missing.value, OSError)

    corrupt_path = tmp_path / "corrupt.state"
    corrupt_path.write_bytes(b"not-a-sciencekit-state")
    with pytest.raises(InvalidArgumentError) as exc_corrupt:
        sciencekit.load(corrupt_path)
    assert exc_corrupt.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT
