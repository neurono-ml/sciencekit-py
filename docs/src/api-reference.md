# API Reference

## Top-Level Module (`sciencekit`)

### Metadata & Diagnostics

- `sciencekit.__version__ -> str` — Package version (`"0.1.0"`).
- `sciencekit.abi_version() -> dict[str, int]` — Returns `{"major": 1, "minor": 0, "patch": 0}` from `sk_ffi_version`.
- `sciencekit.features() -> int` — Bitmask of compiled ABI capabilities (`FEATURE_LINEAR_REGRESSION`, `FEATURE_STANDARD_SCALER`, `FEATURE_MODEL_PERSISTENCE`, `FEATURE_F32`, `FEATURE_F64`).
- `sciencekit.status_name(status_code: int) -> str` — Stable lowercase name from `sk_ffi_status_name`.
- `sciencekit.live_handles() -> int` — Process-wide atomic count of live native builder and model handles.
- `sciencekit.load(path: str | os.PathLike[str]) -> LinearRegressionModel` — Load a persisted `float64` `LinearRegressionModel` via `sk_ffi_model_load`.

### Constants

- Status codes: `STATUS_OK` (`0`), `STATUS_INVALID_HANDLE` (`1`), `STATUS_SHAPE` (`2`), `STATUS_INVALID_ARGUMENT` (`3`), `STATUS_UNSUPPORTED` (`4`), `STATUS_NOT_FITTED` (`5`), `STATUS_IO` (`6`), `STATUS_INTERNAL` (`7`).
- Backends: `BACKEND_FAER` (`"faer"`), `BACKEND_MATRIX_MULTIPLY` (`"matrixmultiply"`), `BACKEND_NDARRAY_LINALG` (`"ndarray-linalg"`).

---

## `sciencekit.linear_model`

### `LinearRegression(dtype="float64", *, include_intercept=True, backend="faer")`

Also exported as `LinearRegressionBuilder` and factory `linear_regression(...)`.

- **Properties:** `.family -> str` (`"linear_regression"`), `.dtype -> str` (`"float64"` | `"float32"`), `.closed -> bool`.
- **Methods:**
  - `.include_intercept(enabled: bool) -> LinearRegression`
  - `.backend(backend: str) -> LinearRegression`
  - `.fit(features, targets, *, copy: bool = False) -> LinearRegressionModel`
  - `.close() -> None` (idempotent; also invoked by `with` block exit and finalizer)

### `LinearRegressionModel`

- **Properties:** `.family -> str`, `.dtype -> str`, `.closed -> bool`.
- **Methods:**
  - `LinearRegressionModel.load(path: str | os.PathLike[str]) -> LinearRegressionModel`
  - `.predict(features, *, copy: bool = False) -> np.ndarray` (1D `float64` or `float32` array)
  - `.score(features, targets, *, copy: bool = False) -> float` ($R^2$ coefficient of determination)
  - `.save(path: str | os.PathLike[str]) -> None` (supported for `float64`)
  - `.transform(features, *, copy: bool = False)` (raises `UnsupportedError` with `status_code == 4`)
  - `.close() -> None` (idempotent; supports `with` context manager)

---

## `sciencekit.preprocessing`

### `StandardScaler(dtype="float64", *, with_mean=True, with_std=True, backend="faer")`

Also exported as `StandardScalerBuilder` and factory `standard_scaler(...)`.

- **Properties:** `.family -> str` (`"standard_scaler"`), `.dtype -> str` (`"float64"` | `"float32"`), `.closed -> bool`.
- **Methods:**
  - `.with_mean(enabled: bool) -> StandardScaler`
  - `.with_std(enabled: bool) -> StandardScaler`
  - `.backend(backend: str) -> StandardScaler`
  - `.fit(features, targets=None, *, copy: bool = False) -> StandardScalerModel`
  - `.fit_transform(features, *, copy: bool = False) -> np.ndarray`
  - `.close() -> None` (idempotent; supports `with` context manager)

### `StandardScalerModel`

- **Properties:** `.family -> str`, `.dtype -> str`, `.closed -> bool`.
- **Methods:**
  - `.transform(features, *, copy: bool = False) -> np.ndarray` (2D `float64` or `float32` array)
  - `.predict(...)` / `.score(...)` / `.save(...)` (raise `UnsupportedError` with `status_code == 4`)
  - `.close() -> None` (idempotent; supports `with` context manager)

---

## Descriptors & Exceptions

- `DenseDescriptor(array: np.ndarray, *, copy: bool = False)` — `.dtype`, `.ndim`, `.shape`, `.strides`, `.length`, `.data_pointer`, `.copied`.
- `CsrDescriptor(data, indices, indptr, shape)` & `CsrDescriptor.from_csr(matrix)` — `.dtype`, `.index_dtype`, `.shape`, `.nnz`, `.data_pointer`, `.to_dense() -> np.ndarray`.
- `ScienceKitError(status_code: int, status_name: str, message: str)` and subclasses: `InvalidHandleError`, `ShapeError`, `InvalidArgumentError`, `UnsupportedError`, `NotFittedError`, `IoError`, `InternalError`.
