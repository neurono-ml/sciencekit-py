# NumPy Dense & CSR Arrays

`sciencekit` validates shape, rank, alignment, contiguity, and scalar dtype before crossing into C ABI compute kernels.

<figure class="sk-diagram reveal">
  <svg viewBox="0 0 860 215" role="img" aria-label="Diagram comparing three NumPy input paths: C-contiguous float64/float32 borrowed zero-copy, strided/Fortran arrays rejected unless copy=True, and CSR sparse descriptors validated and requiring explicit to_dense()">
    <title>Dense zero-copy, explicit copy, and CSR validation paths</title>
    <rect x="18" y="20" width="258" height="172" rx="14" fill="#f0fdf4" stroke="#16a34a" stroke-width="2" />
    <text x="147" y="50" text-anchor="middle" font-size="13.5" font-weight="700" fill="#14532d">1. C-Contiguous ndarray</text>
    <text x="147" y="74" text-anchor="middle" font-size="11.5" fill="#166534">float64 or float32 · 1D / 2D</text>
    <text x="147" y="102" text-anchor="middle" font-size="12" font-weight="600" fill="#14532d">Zero-Copy SKFfiArray Borrow</text>
    <text x="147" y="125" text-anchor="middle" font-size="11" fill="#166534">copied == False</text>
    <text x="147" y="146" text-anchor="middle" font-size="11" fill="#166534">Same buffer address passed</text>
    <text x="147" y="166" text-anchor="middle" font-size="11" fill="#166534">directly to sciencekit_ffi</text>

    <rect x="301" y="20" width="258" height="172" rx="14" fill="#fef3c7" stroke="#d97706" stroke-width="2" />
    <text x="430" y="50" text-anchor="middle" font-size="13.5" font-weight="700" fill="#78350f">2. Strided / Fortran View</text>
    <text x="430" y="74" text-anchor="middle" font-size="11.5" fill="#92400e">X[:, ::2] or np.asfortranarray(X)</text>
    <text x="430" y="102" text-anchor="middle" font-size="12" font-weight="600" fill="#78350f">Default: UnsupportedError (4)</text>
    <text x="430" y="125" text-anchor="middle" font-size="11" fill="#92400e">Opt-in copy=True materializes</text>
    <text x="430" y="146" text-anchor="middle" font-size="11" fill="#92400e">a row-major buffer explicitly</text>
    <text x="430" y="166" text-anchor="middle" font-size="11" fill="#92400e">(copied == True)</text>

    <rect x="584" y="20" width="258" height="172" rx="14" fill="#ede9fe" stroke="#7c3aed" stroke-width="2" />
    <text x="713" y="50" text-anchor="middle" font-size="13.5" font-weight="700" fill="#3b0764">3. CsrDescriptor / SciPy CSR</text>
    <text x="713" y="74" text-anchor="middle" font-size="11.5" fill="#4c1d95">(data, indices, indptr, shape)</text>
    <text x="713" y="102" text-anchor="middle" font-size="12" font-weight="600" fill="#3b0764">Never Silently Densified</text>
    <text x="713" y="125" text-anchor="middle" font-size="11" fill="#4c1d95">Validates CSR invariants first,</text>
    <text x="713" y="146" text-anchor="middle" font-size="11" fill="#4c1d95">raises UnsupportedError (4);</text>
    <text x="713" y="166" text-anchor="middle" font-size="11" fill="#4c1d95">use csr.to_dense() explicitly</text>
  </svg>
  <figcaption>Figure 1 — Array validation guarantees zero silent copies and zero transposed reads.</figcaption>
</figure>

## Zero-Copy Dense Descriptors (`DenseDescriptor`)

When you pass a C-contiguous, aligned `float64` or `float32` `numpy.ndarray`, `sciencekit` borrows the exact NumPy buffer pointer without copying (`tests/test_arrays_and_csr.py`):

```python
import numpy as np
from sciencekit import DenseDescriptor, UnsupportedError

matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
descriptor = DenseDescriptor(matrix)

assert descriptor.dtype == "float64"
assert descriptor.shape == (2, 3)
assert descriptor.strides == (3, 1)
assert descriptor.copied is False
assert descriptor.data_pointer == matrix.__array_interface__["data"][0]
```

Fortran-ordered arrays (`np.asfortranarray`) and strided slices (`matrix[:, ::2]`) are rejected by default with `UnsupportedError` (`status_code == 4`) so column-major memory is never misread as row-major. Pass `copy=True` to any descriptor or estimator operation (`fit`, `predict`, `transform`, `score`) to opt into an explicit C-contiguous copy:

```python
strided = np.arange(12.0, dtype=np.float64).reshape(3, 4)[:, ::2]
copied = DenseDescriptor(strided, copy=True)
assert copied.copied is True
assert copied.shape == (3, 2)
```

## CSR Sparse Descriptors (`CsrDescriptor`)

`CsrDescriptor(data, indices, indptr, shape)` and `CsrDescriptor.from_csr(matrix)` validate Compressed Sparse Row structure in Rust (`float64`/`float32` `data`, `int32`/`int64` `indices` and `indptr`, monotonic `indptr` from `0` to `nnz`, in-range column indices, and positive 2D `shape`):

```python
import numpy as np
from sciencekit import CsrDescriptor, LinearRegression, UnsupportedError

csr = CsrDescriptor(
    np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float64),
    np.array([1, 0, 0, 1], dtype=np.int32),
    np.array([0, 1, 2, 4], dtype=np.int32),
    shape=(3, 2),
)
assert csr.nnz == 4
assert csr.dtype == "float64"

# ABI 1.0 accepts dense SKFfiArray buffers only — never silently densifies:
targets = np.array([2.0, 3.0, 12.0], dtype=np.float64)
with LinearRegression() as builder:
    with builder.fit(csr.to_dense(), targets) as model:
        assert abs(model.score(csr.to_dense(), targets) - 1.0) < 1e-12
```
