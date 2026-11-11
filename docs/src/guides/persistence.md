# Model Persistence

`model.save(path)` and `sciencekit.load(path)` (also available as `LinearRegressionModel.load(path)`) delegate to `sk_ffi_model_save` and `sk_ffi_model_load` in `sciencekit_ffi`. Both accept `str` and `os.PathLike` (`pathlib.Path`).

## Round-Trip Example

Derived from `tests/test_persistence.py::test_linear_regression_persistence_round_trip`:

```python
from pathlib import Path
import numpy as np
import sciencekit
from sciencekit.linear_model import LinearRegression, LinearRegressionModel

features = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
    dtype=np.float64,
)
targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)
probe = np.array([[4.0, -1.0], [-2.0, 5.0]], dtype=np.float64)

state_file = Path("linear_regression.state")

with LinearRegression() as builder:
    with builder.fit(features, targets) as fitted:
        expected_predictions = fitted.predict(probe)
        fitted.save(state_file)

with sciencekit.load(state_file) as loaded:
    assert isinstance(loaded, LinearRegressionModel)
    np.testing.assert_allclose(loaded.predict(probe), expected_predictions, rtol=1e-12, atol=1e-12)
```

## Current C ABI Persistence Scope

In `sciencekit_ffi` ABI `1.0.0`, binary persistence covers **fitted `float64` `LinearRegressionModel`** instances. Other combinations return structured exceptions with deterministic status codes (`tests/test_persistence.py::test_persistence_rejects_f32_scaler_missing_and_corrupt_files`):

| Scenario | Exception Raised | `status_code` |
| --- | --- | --- |
| Saving a `float32` `LinearRegressionModel` | `UnsupportedError` | `4` (`STATUS_UNSUPPORTED`) |
| Saving a `StandardScalerModel` | `UnsupportedError` | `4` (`STATUS_UNSUPPORTED`) |
| Loading a nonexistent file or saving to a missing directory | `IoError` (`OSError`) | `6` (`STATUS_IO`) |
| Loading a corrupt or truncated file | `InvalidArgumentError` (`ValueError`) | `3` (`STATUS_INVALID_ARGUMENT`) |
| Passing a path containing an interior NUL byte (`"\x00"`) | `InvalidArgumentError` (`ValueError`) | `3` (`STATUS_INVALID_ARGUMENT`) |
