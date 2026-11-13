# SciPy sparse matrices

A `scipy.sparse.csr_matrix` is consumed through its three buffers — `data`,
`indices` and `indptr` — without densifying in the conversion layer:

```python
import numpy as np
import scipy.sparse as sp
from sciencekit import LinearRegression, StandardScaler, describe

data = np.array([1.0, 0.0, 2.0, 3.0, 5.0])      # one explicitly stored zero
indices = np.array([0, 2, 1, 0, 2], dtype=np.int32)
indptr = np.array([0, 2, 4, 5], dtype=np.int32)
matrix = sp.csr_matrix((data, indices, indptr), shape=(3, 3))

report = describe(matrix)
assert report.representation == "scipy.csr"
assert report.zero_copy is True
assert report.densified is False
assert report.data_pointer == matrix.data.__array_interface__["data"][0]
assert report.indices_pointer == matrix.indices.__array_interface__["data"][0]
assert report.indptr_pointer == matrix.indptr.__array_interface__["data"][0]
assert report.stored_zeros == 1
assert report.abi_dense_required is True
```

The pointer assertions prove the adapter aliases the caller's buffers; the
`stored_zeros` count proves it reads `data` directly instead of rebuilding the
matrix from a dense image (a dense round trip would lose the explicit zero).

## Why a dense block still exists

`sciencekit_ffi` ABI 1.0 accepts dense blocks only. The sparse view is therefore
densified once, at the C call boundary, inside Rust. `ConversionReport` says so
explicitly: `densified=False` (the conversion layer stayed sparse) and
`abi_dense_required=True` (the ABI needs a dense block). Nothing is silent.

## Fitting and transforming

```python
targets = np.array([1.0, 4.0, 7.0])
with LinearRegression() as builder:
    with builder.fit(matrix, targets) as model:
        assert builder.last_conversion.data_pointer == (
            matrix.data.__array_interface__["data"][0]
        )
        predictions = model.predict(matrix)
```

`predict` and `transform` accept the sparse matrix directly. Ask for a sparse
result with `output="sparse"`:

```python
with StandardScaler() as scaler:
    with scaler.fit(matrix) as model:
        sparse_result = model.transform(matrix, output="sparse")
        dense_result = model.transform(matrix)

assert sp.issparse(sparse_result)
np.testing.assert_allclose(sparse_result.toarray(), dense_result)
```

## Index dtypes

`indices` and `indptr` may be `int32` or `int64`; both are accepted and reported
as the input layout. `int64` indices are converted to the `usize` values the
sparse view needs, and that conversion is part of the reported work, never a
change of values.

## CSC and COO

`csc_matrix` and `coo_matrix` are converted explicitly with SciPy's `.tocsr()`
and reported:

```python
report = describe(matrix.tocsc())
assert report.converted_from == "csc"
assert report.copied is True
assert report.stored_zeros == 1
```

Fitting a CSC matrix and fitting the equivalent dense array produce the same
predictions, which is the parity contract for every representation.
