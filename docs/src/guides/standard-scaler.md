# Standard Scaler

`sciencekit.preprocessing.StandardScaler` (aliased as `StandardScalerBuilder` and exposed as `sciencekit.standard_scaler(...)`) wraps `SK_FFI_ESTIMATOR_STANDARD_SCALER` (`1`) in `sciencekit_ffi`.

## Configuration Options

| Parameter / Setter | Default | Accepted Values | Description |
| --- | --- | --- | --- |
| `dtype` | `"float64"` | `"float64"`, `"float32"`, `np.float64`, `np.float32` | Scalar precision preserved by `fit` and `transform`. |
| `.with_mean(enabled)` | `True` | `bool` | Subtract each column's sample mean during `transform`. |
| `.with_std(enabled)` | `True` | `bool` | Divide each column by its population standard deviation (`ddof=0`). |
| `.backend(name)` | `"faer"` | `"faer"`, `"matrixmultiply"`, `"ndarray-linalg"` | Uniform backend selection transferred into observability spans. |

## Centering and Scaling in `float64`

Derived from `tests/test_standard_scaler.py::test_standard_scaler_f64_centers_and_scales_columns`:

```python
import numpy as np
from sciencekit.preprocessing import StandardScaler

features = np.array(
    [[0.0, 10.0], [0.0, 20.0], [1.0, 30.0], [1.0, 40.0]],
    dtype=np.float64,
)
expected = (features - features.mean(axis=0)) / features.std(axis=0, ddof=0)

with StandardScaler() as builder:
    builder.with_mean(True).with_std(True).backend("faer")
    with builder.fit(features) as model:
        transformed = model.transform(features)
        assert transformed.dtype == np.float64
        assert transformed.shape == (4, 2)
        np.testing.assert_allclose(transformed, expected, rtol=1e-12, atol=1e-12)
```

## One-Step `fit_transform` and Selective Toggles in `float32`

`StandardScaler.fit_transform(features)` fits a temporary native model, transforms `features`, releases the temporary model handle immediately, and returns the transformed 2D NumPy array (`tests/test_standard_scaler.py::test_standard_scaler_with_mean_or_std_disabled_and_f32`):

```python
import numpy as np
import sciencekit
from sciencekit.preprocessing import StandardScaler

features = np.array(
    [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0], [8.0, 16.0]],
    dtype=np.float32,
)

with sciencekit.standard_scaler(dtype="float32", with_mean=True, with_std=False) as mean_only:
    centered = mean_only.fit_transform(features)
    assert centered.dtype == np.float32
    np.testing.assert_allclose(centered, features - features.mean(axis=0), rtol=1e-5, atol=1e-5)

with StandardScaler(dtype=np.float32, with_mean=False, with_std=True) as std_only:
    scaled = std_only.fit_transform(features)
    np.testing.assert_allclose(scaled, features / features.std(axis=0, ddof=0), rtol=1e-5, atol=1e-5)
```
