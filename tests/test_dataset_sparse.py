"""SciPy sparse matrices: borrowed CSR buffers and explicit CSC/COO conversion."""

from __future__ import annotations

import numpy as np
import pytest

sparse = pytest.importorskip("scipy.sparse")

import sciencekit
from sciencekit import LinearRegression, StandardScaler, describe

# Explicitly stored zero at (0, 2): a CSR-aware conversion must preserve it.
DATA = np.array([1.0, 0.0, 2.0, 3.0, 5.0], dtype=np.float64)
INDICES = np.array([0, 2, 1, 0, 2], dtype=np.int32)
INDPTR = np.array([0, 2, 4, 5], dtype=np.int32)
SHAPE = (3, 3)


def _csr(index_dtype=np.int32) -> "sparse.csr_matrix":
    return sparse.csr_matrix(
        (DATA.copy(), INDICES.astype(index_dtype), INDPTR.astype(index_dtype)), shape=SHAPE
    )


def _pointer(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0])


def test_csr_buffers_are_borrowed_without_densifying() -> None:
    matrix = _csr()
    report = describe(matrix)
    assert report.representation == "scipy.csr"
    assert report.zero_copy is True
    assert report.densified is False
    assert report.copied is False
    assert report.data_pointer == _pointer(matrix.data)
    assert report.indices_pointer == _pointer(matrix.indices)
    assert report.indptr_pointer == _pointer(matrix.indptr)
    assert report.stored_zeros == 1
    assert report.abi_dense_required is True


def test_csr_fit_matches_the_dense_equivalent() -> None:
    matrix = _csr()
    dense = matrix.toarray()
    targets = np.array([1.0, 4.0, 7.0], dtype=np.float64)
    with LinearRegression() as builder:
        with builder.fit(matrix, targets) as model:
            assert builder.last_conversion.data_pointer == _pointer(matrix.data)
            sparse_predictions = model.predict(matrix)
        with builder.fit(dense, targets) as model:
            dense_predictions = model.predict(dense)
    np.testing.assert_allclose(sparse_predictions, dense_predictions, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_int32_and_int64_index_dtypes_are_accepted(index_dtype) -> None:
    matrix = _csr(index_dtype)
    report = describe(matrix)
    assert report.converted_from is None
    assert report.data_pointer == _pointer(matrix.data)
    targets = np.array([1.0, 4.0, 7.0], dtype=np.float64)
    with LinearRegression() as builder:
        with builder.fit(matrix, targets) as model:
            assert np.isfinite(model.predict(matrix)).all()


@pytest.mark.parametrize("layout", ["csc", "coo"])
def test_csc_and_coo_convert_explicitly(layout) -> None:
    dense = _csr().toarray()
    matrix = _csr().tocsc() if layout == "csc" else _csr().tocoo()
    report = describe(matrix)
    assert report.converted_from == layout
    assert report.copied is True
    assert report.zero_copy is False
    assert report.stored_zeros == 1
    targets = np.array([1.0, 4.0, 7.0], dtype=np.float64)
    with LinearRegression() as builder:
        with builder.fit(matrix, targets) as model:
            with builder.fit(dense, targets) as dense_model:
                np.testing.assert_allclose(
                    model.predict(matrix), dense_model.predict(dense), rtol=1e-12
                )


def test_transform_round_trip_and_sparse_result() -> None:
    matrix = _csr()
    with StandardScaler() as scaler:
        with scaler.fit(matrix) as model:
            as_sparse = model.transform(matrix, output="sparse")
            as_numpy = model.transform(matrix)
    assert sparse.issparse(as_sparse)
    np.testing.assert_allclose(as_sparse.toarray(), as_numpy, rtol=1e-12, atol=1e-12)

    with StandardScaler() as scaler:
        with scaler.fit(matrix.toarray()) as dense_model:
            np.testing.assert_allclose(
                dense_model.transform(matrix.toarray()), as_numpy, rtol=1e-12
            )
