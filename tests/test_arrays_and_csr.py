"""Zero-copy dense NumPy descriptors and CSR sparse descriptors."""

from __future__ import annotations

import numpy as np
import pytest

import sciencekit
from sciencekit import (
    CsrDescriptor,
    DenseDescriptor,
    InvalidArgumentError,
    ScienceKitError,
    UnsupportedError,
)


class _DuckCsrMatrix:
    """Minimal duck-typed CSR object mimicking `scipy.sparse.csr_matrix`."""

    format = "csr"

    def __init__(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
    ) -> None:
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape


def test_dense_descriptor_borrows_c_contiguous_f64_and_f32_zero_copy() -> None:
    matrix_f64 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    descriptor_f64 = DenseDescriptor(matrix_f64)
    assert descriptor_f64.dtype == "float64"
    assert descriptor_f64.ndim == 2
    assert descriptor_f64.shape == (2, 3)
    assert descriptor_f64.strides == (3, 1)
    assert descriptor_f64.length == 6
    assert descriptor_f64.copied is False
    assert descriptor_f64.data_pointer == matrix_f64.__array_interface__["data"][0]

    vector_f32 = np.array([1.5, -2.5, 3.5], dtype=np.float32)
    descriptor_f32 = DenseDescriptor(vector_f32)
    assert descriptor_f32.dtype == "float32"
    assert descriptor_f32.ndim == 1
    assert descriptor_f32.shape == (3,)
    assert descriptor_f32.strides == (1,)
    assert descriptor_f32.length == 3
    assert descriptor_f32.copied is False
    assert descriptor_f32.data_pointer == vector_f32.__array_interface__["data"][0]


def test_dense_descriptor_rejects_non_contiguous_unless_copy_enabled() -> None:
    base = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    fortran = np.asfortranarray(base)
    strided = base[:, ::2]

    for candidate in (fortran, strided):
        with pytest.raises(UnsupportedError) as exc_info:
            DenseDescriptor(candidate)
        assert exc_info.value.status_code == sciencekit.STATUS_UNSUPPORTED
        assert exc_info.value.status_name == "unsupported"

    copied = DenseDescriptor(strided, copy=True)
    assert copied.copied is True
    assert copied.shape == (3, 2)
    assert copied.strides == (2, 1)


def test_dense_descriptor_rejects_bad_dtype_rank_and_empty_shape() -> None:
    with pytest.raises(InvalidArgumentError) as exc_dtype:
        DenseDescriptor(np.array([[1, 2], [3, 4]], dtype=np.int64))
    assert exc_dtype.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT
    assert isinstance(exc_dtype.value, ValueError)
    assert isinstance(exc_dtype.value, ScienceKitError)

    with pytest.raises(InvalidArgumentError) as exc_rank:
        DenseDescriptor(np.zeros((2, 2, 2), dtype=np.float64))
    assert exc_rank.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT

    with pytest.raises(InvalidArgumentError) as exc_empty:
        DenseDescriptor(np.zeros((0, 3), dtype=np.float64))
    assert exc_empty.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT


def test_csr_descriptor_validates_and_materializes_explicit_dense() -> None:
    data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    indices = np.array([0, 2, 1], dtype=np.int32)
    indptr = np.array([0, 2, 3], dtype=np.int32)

    csr = CsrDescriptor(data, indices, indptr, shape=(2, 3))
    assert csr.dtype == "float64"
    assert csr.index_dtype == "int32"
    assert csr.shape == (2, 3)
    assert csr.nnz == 3
    assert csr.data_pointer == data.__array_interface__["data"][0]

    dense = csr.to_dense()
    assert dense.dtype == np.float64
    np.testing.assert_allclose(dense, np.array([[10.0, 0.0, 20.0], [0.0, 30.0, 0.0]]))

    duck = _DuckCsrMatrix(
        np.array([1.5, -4.0], dtype=np.float32),
        np.array([1, 0], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        (2, 2),
    )
    csr_f32 = CsrDescriptor.from_csr(duck)
    assert csr_f32.dtype == "float32"
    assert csr_f32.index_dtype == "int64"
    dense_f32 = csr_f32.to_dense()
    assert dense_f32.dtype == np.float32
    np.testing.assert_allclose(dense_f32, np.array([[0.0, 1.5], [-4.0, 0.0]], dtype=np.float32))


def test_csr_descriptor_rejects_malformed_structure() -> None:
    data = np.array([1.0, 2.0], dtype=np.float64)

    # Column index out of bounds
    with pytest.raises(InvalidArgumentError) as exc_col:
        CsrDescriptor(
            data,
            np.array([0, 5], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
            shape=(2, 3),
        )
    assert exc_col.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT

    # Non-monotonic indptr
    with pytest.raises(InvalidArgumentError) as exc_indptr:
        CsrDescriptor(
            data,
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2, 1], dtype=np.int32),
            shape=(2, 3),
        )
    assert exc_indptr.value.status_code == sciencekit.STATUS_INVALID_ARGUMENT

    # Wrong indptr length
    with pytest.raises(InvalidArgumentError):
        CsrDescriptor(
            data,
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
            shape=(2, 3),
        )

    # Unsupported data dtype
    with pytest.raises(InvalidArgumentError):
        CsrDescriptor(
            np.array([1, 2], dtype=np.int64),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
            shape=(2, 3),
        )
