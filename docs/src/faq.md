# Frequently Asked Questions

## Why does `sciencekit-py` live in its own repository?

The main [`neurono-ml/sciencekit`](https://github.com/neurono-ml/sciencekit) repository is a pure-Rust Cargo workspace published to crates.io. Hosting each language binding in its own dedicated repository (`sciencekit-py`, `sciencekit-go`, `sciencekit-lua`) lets each ecosystem use its native build tool (`maturin` and `pyproject.toml` for Python) and release cadence without coupling Python wheel matrices to Rust crate publishing.

## How does `sciencekit-py` consume `sciencekit_ffi` if it is not on crates.io?

`Cargo.toml` declares a git dependency on `sciencekit_ffi = { git = "https://github.com/neurono-ml/sciencekit", branch = "main" }` and links it as an `rlib` directly into `sciencekit._native`. You get a single self-contained Python wheel with no external `.so`/`.dylib`/`.dll` path setup, while still exercising the exact `sk_ffi_*` C ABI entry points.

## Why are Fortran-ordered arrays and strided slices rejected by default?

`SKFfiArray` in `sciencekit_ffi` ABI 1.0 requires C-contiguous (row-major) memory. Silently copying non-contiguous arrays hides $O(N \cdot D)$ allocations in tight loops, while reading column-major buffers without checking `is_c_contiguous()` would transpose values. Passing `copy=True` makes any contiguous copy explicit at the call site.

## Why does passing a `CsrDescriptor` to `fit()` raise `UnsupportedError`?

`sciencekit` enforces a strict no-silent-densification rule: materializing a sparse matrix can blow up memory by orders of magnitude. Because `sciencekit_ffi` ABI 1.0 exposes dense `SKFfiArray` descriptors only, `fit`/`predict`/`transform` first validate the CSR structure and then raise `UnsupportedError` (`status_code == 4`), prompting callers to invoke `csr.to_dense()` only when dense expansion is intentional.

## Are fitted models safe to share across `threading.Thread` workers?

Yes. Fitted models in Rust are immutable, `sciencekit._native` detaches the Python GIL via `py.detach` around compute, and `SKPyModelHandle` coordinates readers and `.close()` with an `RwLock` so concurrent predictions run in parallel with zero data races.
