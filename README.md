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
- **Dataset interoperability (`sciencekit.datasets`):**
  - `numpy.ndarray` (`float32`/`float64`) is borrowed **zero-copy** when C-contiguous; strided views raise `UnsupportedError` unless `copy=True`, and precision changes require `cast=True`.
  - `pandas.DataFrame`/`Series` (Arrow PyCapsule protocol, `to_numpy` fallback for pandas < 2.2), `pyarrow.Table`/`RecordBatch`/`RecordBatchReader`, any `__arrow_c_stream__` producer, `polars.DataFrame`/`LazyFrame`, `dask.dataframe.DataFrame` and `pyspark.sql.DataFrame` (driver `toArrow()`) are accepted by `fit`, `predict`, `transform` and `score`.
  - `scipy.sparse.csr_matrix` is consumed through its `data`/`indices`/`indptr` buffers without densifying in the conversion layer; `csc_matrix`/`coo_matrix` convert explicitly. The dense block required by `sciencekit_ffi` ABI 1.0 is materialized only at the call boundary and is reported as `abi_dense_required=True`.
  - `builder.last_conversion` / `model.last_conversion` return a `ConversionReport` with the borrowed buffer pointers, copy/cast/densification flags, Arrow batch counts, Dask partition sizes and categorical label tables. `sciencekit.describe(dataset)` reports the same facts without fitting.
  - Results are NumPy arrays by default; `output="pandas" | "arrow" | "polars" | "sparse" | "auto"` selects another representation, and Arrow results implement `__arrow_c_stream__`.
- **Dense & sparse descriptors:** `DenseDescriptor` and `CsrDescriptor` remain available for explicit buffer validation; a manually validated `CsrDescriptor` still requires `.to_dense()` before it reaches the ABI.
- **Structured exceptions:** Every failure raises a `ScienceKitError` subclass (`InvalidHandleError`, `ShapeError`, `InvalidArgumentError`, `UnknownColumnError`, `UnsupportedColumnTypeError`, `UnsupportedError`, `NotFittedError`, `IoError`, `InternalError`) carrying `.status_code`, `.status_name`, and `.message`. `UnknownColumnError` is a `KeyError`, `UnsupportedColumnTypeError` a `TypeError`, and shape failures a `ValueError`.
- **Lifecycle & concurrency:** Compute calls detach the Python GIL via `py.detach`. Fitted models are immutable and safe across concurrent Python threads. Handles are released deterministically via idempotent `.close()`, `with` context managers, or garbage-collector finalization.

## Optional extras

The dataset adapters import their library only when the corresponding object is
passed, so none of them is a hard dependency:

```bash
pip install "sciencekit[datasets]"   # pandas, pyarrow, polars, scipy, dask
pip install "sciencekit[spark]"      # pyspark 4
```

## Building & Testing

```bash
maturin build --release --out dist
pip install --force-reinstall dist/sciencekit-*.whl
pytest -q
mdbook build docs
```

## License

Apache-2.0. See [LICENSE](LICENSE).
