# sciencekit-py

[![CI](https://github.com/neurono-ml/sciencekit-py/actions/workflows/ci.yml/badge.svg)](https://github.com/neurono-ml/sciencekit-py/actions/workflows/ci.yml)
[![Documentation](https://github.com/neurono-ml/sciencekit-py/actions/workflows/deploy-documentation.yml/badge.svg)](https://neurono-ml.github.io/sciencekit-py/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](Cargo.toml)

Python bindings for the [sciencekit](https://github.com/neurono-ml/sciencekit) machine-learning library, built with [PyO3](https://pyo3.rs/) `0.30` and [maturin](https://www.maturin.rs/) over the stable `sciencekit_ffi` C ABI.

- **Documentation:** <https://neurono-ml.github.io/sciencekit-py/>
- **Main Rust Repository:** <https://github.com/neurono-ml/sciencekit>

## Native C ABI Dependency Choice

`sciencekit_ffi` is not published separately on crates.io. Rather than vendoring a loose C header and requiring users to manage `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` at runtime, `sciencekit-py` depends directly on the git repository in `Cargo.toml`:

```toml
sciencekit_ffi = { git = "https://github.com/neurono-ml/sciencekit", branch = "main" }
```

`sciencekit_ffi` is linked in-process as an `rlib` inside the `sciencekit._native` PyO3 wheel, calling the exact `sk_ffi_*` entry points, opaque handle registry, thread-local error channel, and panic guards defined by `include/sciencekit.h` while shipping a self-contained Python wheel. To pin a fixed commit for reproducible releases, replace `branch = "main"` with `rev = "<commit>"`.

## Quickstart

```python
import numpy as np
import sciencekit
from sciencekit.linear_model import LinearRegression
from sciencekit.preprocessing import StandardScaler

features = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
    dtype=np.float64,
)
targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

with LinearRegression(dtype="float64") as builder:
    with builder.include_intercept(True).backend("faer").fit(features, targets) as model:
        predictions = model.predict(features)
        r_squared = model.score(features, targets)
        model.save("linear.state")

with StandardScaler(dtype="float64") as scaler:
    transformed = scaler.with_mean(True).with_std(True).fit_transform(features)
```

## Scope & Capabilities

- **Estimator families (`sciencekit_ffi` ABI 1.0):**
  - `sciencekit.linear_model.LinearRegression` / `LinearRegressionModel` (`fit`, `predict`, `score`, `save`, `load`, `float64` and `float32`).
  - `sciencekit.preprocessing.StandardScaler` / `StandardScalerModel` (`fit`, `transform`, `fit_transform`, `float64` and `float32`).
- **NumPy array & CSR descriptors:**
  - C-contiguous `float64` and `float32` `numpy.ndarray` inputs are borrowed **zero-copy** for the duration of the native call (`DenseDescriptor`). Non-contiguous views raise `UnsupportedError` unless `copy=True` is passed explicitly.
  - `CsrDescriptor` validates Compressed Sparse Row (`data`, `indices`, `indptr`, `shape`) buffers and SciPy-compatible CSR objects. Because `sciencekit_ffi` ABI 1.0 accepts dense `SKFfiArray` buffers only, sparse inputs are never silently densified — call `csr.to_dense()` explicitly.
- **Structured exceptions:** Every failure raises a `ScienceKitError` subclass (`InvalidHandleError`, `ShapeError`, `InvalidArgumentError`, `UnsupportedError`, `NotFittedError`, `IoError`, `InternalError`) carrying `.status_code`, `.status_name`, and `.message`.
- **Lifecycle & concurrency:** Compute calls detach the Python GIL via `py.detach`. Fitted models are immutable and safe across concurrent Python threads. Handles are released deterministically via idempotent `.close()`, `with` context managers, or garbage-collector finalization.

## Building & Testing

```bash
maturin build --release --out dist
pip install --force-reinstall dist/sciencekit-*.whl
pytest -q
```

## License

Apache-2.0. See [LICENSE](LICENSE).
