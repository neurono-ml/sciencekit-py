"""Handle lifecycle, close idempotency, leak-free failure loops, concurrency, and large arrays."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path

import numpy as np
import pytest

import sciencekit
from sciencekit import (
    InvalidArgumentError,
    InvalidHandleError,
    IoError,
    LinearRegression,
    ScienceKitError,
    ShapeError,
    StandardScaler,
    UnsupportedError,
)


def test_close_is_idempotent_and_closed_handles_raise_invalid_handle() -> None:
    gc.collect()
    baseline = sciencekit.live_handles()

    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

    builder = LinearRegression()
    assert sciencekit.live_handles() == baseline + 1
    model = builder.fit(features, targets)
    assert sciencekit.live_handles() == baseline + 2

    builder.close()
    builder.close()
    builder.close()
    assert builder.closed is True
    assert sciencekit.live_handles() == baseline + 1

    with pytest.raises(InvalidHandleError) as exc_builder:
        builder.fit(features, targets)
    assert exc_builder.value.status_code == sciencekit.STATUS_INVALID_HANDLE

    with pytest.raises(InvalidHandleError):
        builder.include_intercept(False)

    model.close()
    model.close()
    model.close()
    assert model.closed is True
    assert sciencekit.live_handles() == baseline

    with pytest.raises(InvalidHandleError) as exc_model:
        model.predict(features)
    assert exc_model.value.status_code == sciencekit.STATUS_INVALID_HANDLE


def test_finalizer_releases_unclosed_handles_on_garbage_collection() -> None:
    gc.collect()
    baseline = sciencekit.live_handles()

    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

    def _allocate_and_drop() -> None:
        builder = LinearRegression()
        _model = builder.fit(features, targets)
        scaler = StandardScaler()
        _scaler_model = scaler.fit(features)
        assert sciencekit.live_handles() == baseline + 4

    _allocate_and_drop()
    gc.collect()
    assert sciencekit.live_handles() == baseline


def test_no_handle_leaks_across_repeated_failures(tmp_path: Path) -> None:
    gc.collect()
    baseline = sciencekit.live_handles()

    good_features = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]], dtype=np.float64)
    good_targets = np.array([1.0, 2.0, 7.0], dtype=np.float64)
    bad_targets = np.array([1.0, 2.0], dtype=np.float64)
    bad_columns = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    non_contiguous = np.asfortranarray(good_features)

    for _ in range(75):
        with pytest.raises(InvalidArgumentError):
            LinearRegression(backend="invalid")

        with LinearRegression() as builder:
            with pytest.raises(ShapeError):
                builder.fit(good_features, bad_targets)
            with pytest.raises(UnsupportedError):
                builder.fit(non_contiguous, good_targets)
            with pytest.raises(InvalidArgumentError):
                builder.fit(np.array([1, 2, 3], dtype=np.int64), good_targets)

            with builder.fit(good_features, good_targets) as model:
                with pytest.raises(ShapeError):
                    model.predict(bad_columns)
                with pytest.raises(UnsupportedError):
                    model.transform(good_features)
                with pytest.raises(IoError):
                    model.save(tmp_path / "missing" / "out.state")

        with pytest.raises(ScienceKitError):
            sciencekit.load(tmp_path / "nonexistent.state")

    gc.collect()
    assert sciencekit.live_handles() == baseline


def test_large_array_fit_predict_score_and_scaler_transform() -> None:
    rng = np.random.default_rng(42)
    rows, columns = 12_000, 12
    features = rng.standard_normal((rows, columns), dtype=np.float64)
    true_weights = np.linspace(0.5, 6.0, columns, dtype=np.float64)
    intercept = -3.25
    targets = intercept + features @ true_weights

    with StandardScaler() as scaler_builder:
        with scaler_builder.fit(features) as scaler_model:
            transformed = scaler_model.transform(features)
            assert transformed.shape == (rows, columns)
            assert transformed.dtype == np.float64
            np.testing.assert_allclose(transformed.mean(axis=0), np.zeros(columns), atol=1e-10)
            np.testing.assert_allclose(transformed.std(axis=0, ddof=0), np.ones(columns), atol=1e-10)

    with LinearRegression() as lr_builder:
        with lr_builder.fit(features, targets) as lr_model:
            predictions = lr_model.predict(features)
            assert predictions.shape == (rows,)
            np.testing.assert_allclose(predictions, targets, rtol=1e-9, atol=1e-9)
            assert lr_model.score(features, targets) == pytest.approx(1.0, abs=1e-11)


def test_concurrent_multithreaded_predictions_and_transforms() -> None:
    gc.collect()
    baseline = sciencekit.live_handles()

    rng = np.random.default_rng(7)
    features = rng.standard_normal((2_048, 8), dtype=np.float64)
    weights = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.25, 4.0], dtype=np.float64)
    targets = 2.0 + features @ weights

    with LinearRegression() as lr_builder, StandardScaler() as sc_builder:
        with lr_builder.fit(features, targets) as lr_model, sc_builder.fit(features) as sc_model:
            expected_pred = lr_model.predict(features)
            expected_trans = sc_model.transform(features)

            def _worker(worker_id: int) -> bool:
                batch = np.ascontiguousarray(features[worker_id::8])
                batch_targets = np.ascontiguousarray(targets[worker_id::8])
                for _ in range(25):
                    pred = lr_model.predict(batch)
                    trans = sc_model.transform(batch)
                    r2 = lr_model.score(batch, batch_targets)
                    np.testing.assert_allclose(pred, expected_pred[worker_id::8], rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(trans, expected_trans[worker_id::8], rtol=1e-10, atol=1e-10)
                    assert r2 == pytest.approx(1.0, abs=1e-11)
                return True

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(_worker, range(8)))
            assert all(results)

    gc.collect()
    assert sciencekit.live_handles() == baseline
