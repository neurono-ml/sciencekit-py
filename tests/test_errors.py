"""Structured exception mapping and status code tests across every failure mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sciencekit
from sciencekit import (
    CsrDescriptor,
    InternalError,
    InvalidArgumentError,
    InvalidHandleError,
    IoError,
    LinearRegression,
    NotFittedError,
    ScienceKitError,
    ShapeError,
    StandardScaler,
    UnsupportedError,
)


def test_exception_hierarchy_preserves_status_code_name_and_stdlib_bases() -> None:
    err_shape = ShapeError(2, "shape", "expected 4 rows, found 3")
    assert isinstance(err_shape, ScienceKitError)
    assert isinstance(err_shape, ValueError)
    assert err_shape.status_code == 2
    assert err_shape.status_name == "shape"
    assert err_shape.message == "expected 4 rows, found 3"
    assert str(err_shape) == "sciencekit shape (2): expected 4 rows, found 3"

    assert issubclass(InvalidHandleError, (ScienceKitError, RuntimeError))
    assert issubclass(InvalidArgumentError, (ScienceKitError, ValueError))
    assert issubclass(UnsupportedError, (ScienceKitError, NotImplementedError))
    assert issubclass(NotFittedError, (ScienceKitError, RuntimeError))
    assert issubclass(IoError, (ScienceKitError, OSError))
    assert issubclass(InternalError, (ScienceKitError, RuntimeError))


def test_shape_mismatch_carries_shape_status_code() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    wrong_targets = np.array([1.0, 2.0], dtype=np.float64)

    with LinearRegression() as builder:
        with pytest.raises(ShapeError) as exc_fit:
            builder.fit(features, wrong_targets)
        assert exc_fit.value.status_code == sciencekit.STATUS_SHAPE
        assert exc_fit.value.status_name == "shape"

        targets = np.array([5.0, 11.0, 17.0], dtype=np.float64)
        with builder.fit(features, targets) as model:
            wrong_columns = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
            with pytest.raises(ShapeError) as exc_pred:
                model.predict(wrong_columns)
            assert exc_pred.value.status_code == sciencekit.STATUS_SHAPE

            with pytest.raises(ShapeError) as exc_score:
                model.score(features, wrong_targets)
            assert exc_score.value.status_code == sciencekit.STATUS_SHAPE


def test_invalid_argument_carries_status_code_for_dtype_rank_backend_and_path() -> None:
    with pytest.raises(InvalidArgumentError) as exc_backend:
        LinearRegression(backend="unknown-backend")
    assert exc_backend.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT

    with pytest.raises(InvalidArgumentError) as exc_dtype:
        LinearRegression(dtype="int32")
    assert exc_dtype.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT

    with LinearRegression(dtype="float64") as builder:
        # Wrong dtype on input array (float32 passed to float64 builder)
        with pytest.raises(InvalidArgumentError) as exc_arr_dtype:
            builder.fit(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([1.0, 2.0], dtype=np.float32),
            )
        assert exc_arr_dtype.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT

        # Wrong rank (1D passed for 2D features)
        with pytest.raises(InvalidArgumentError) as exc_rank:
            builder.fit(
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([1.0, 2.0], dtype=np.float64),
            )
        assert exc_rank.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT

        # Interior NUL in path
        with builder.fit(
            np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]], dtype=np.float64),
            np.array([2.0, 3.0, 12.0], dtype=np.float64),
        ) as model:
            with pytest.raises(InvalidArgumentError) as exc_nul:
                model.save("bad\x00path.state")
            assert exc_nul.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT


def test_unsupported_operations_and_csr_boundary_carry_unsupported_status_code() -> None:
    features = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]], dtype=np.float64)
    targets = np.array([2.0, 3.0, 12.0], dtype=np.float64)

    with LinearRegression() as lr_builder:
        with lr_builder.fit(features, targets) as lr_model:
            with pytest.raises(UnsupportedError) as exc_transform:
                lr_model.transform(features)
            assert exc_transform.value.status_code == sciencekit.STATUS_UNSUPPORTED

    with StandardScaler() as sc_builder:
        with sc_builder.fit(features) as sc_model:
            with pytest.raises(UnsupportedError) as exc_predict:
                sc_model.predict(features)
            assert exc_predict.value.status_code == sciencekit.STATUS_UNSUPPORTED

            with pytest.raises(UnsupportedError) as exc_score:
                sc_model.score(features, targets)
            assert exc_score.value.status_code == sciencekit.STATUS_UNSUPPORTED

    csr = CsrDescriptor(
        np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1, 0, 0, 1], dtype=np.int32),
        np.array([0, 1, 2, 4], dtype=np.int32),
        shape=(3, 2),
    )
    with LinearRegression() as builder:
        with pytest.raises(UnsupportedError) as exc_csr:
            builder.fit(csr, targets)
        assert exc_csr.value.status_code == sciencekit.STATUS_UNSUPPORTED
        assert "to_dense()" in exc_csr.value.message

        # Explicit densification succeeds cleanly
        with builder.fit(csr.to_dense(), targets) as model:
            assert model.score(csr.to_dense(), targets) == pytest.approx(1.0, abs=1e-12)


def test_io_error_carries_io_status_code(tmp_path: Path) -> None:
    features = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]], dtype=np.float64)
    targets = np.array([2.0, 3.0, 12.0], dtype=np.float64)

    with LinearRegression() as builder:
        with builder.fit(features, targets) as model:
            missing_dir = tmp_path / "missing_subdir" / "model.state"
            with pytest.raises(IoError) as exc_save:
                model.save(missing_dir)
            assert exc_save.value.status_code == sciencekit.STATUS_IO
