# Linear Regression

`sciencekit.linear_model.LinearRegression` (aliased as `LinearRegressionBuilder` and exposed as the factory `sciencekit.linear_regression(...)`) wraps `SK_FFI_ESTIMATOR_LINEAR_REGRESSION` (`0`) in `sciencekit_ffi`.

## Configuration Options

| Parameter / Setter | Default | Accepted Values | Description |
| --- | --- | --- | --- |
| `dtype` | `"float64"` | `"float64"`, `"float32"`, `np.float64`, `np.float32` | Scalar precision of expected input and output arrays. |
| `.include_intercept(enabled)` | `True` | `bool` | Whether ordinary least squares solves for an additive intercept term. |
| `.backend(name)` | `"faer"` | `"faer"`, `"matrixmultiply"`, `"ndarray-linalg"` | Underlying Rust linear-algebra backend transferred to the fitted model. |

## Double Precision (`float64`) Across All Backends

Derived from `tests/test_linear_regression.py::test_linear_regression_f64_known_fixture_and_backends`:

```python
import numpy as np
from sciencekit.linear_model import LinearRegression

features = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
    dtype=np.float64,
)
targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

for backend in ("faer", "matrixmultiply", "ndarray-linalg"):
    with LinearRegression(dtype="float64") as builder:
        builder.include_intercept(True).backend(backend)
        with builder.fit(features, targets) as model:
            predictions = model.predict(features)
            assert predictions.dtype == np.float64
            np.testing.assert_allclose(predictions, targets, rtol=1e-10, atol=1e-10)
            assert abs(model.score(features, targets) - 1.0) < 1e-12
```

## Single Precision (`float32`) Without Intercept

When `dtype=np.float32` (or `"float32"`) is selected, both `features` and `targets` are borrowed as 32-bit floats and `model.predict(features)` returns a 1D `np.float32` array (`tests/test_linear_regression.py::test_linear_regression_without_intercept_and_f32_precision`):

```python
import numpy as np
import sciencekit

features = np.array(
    [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [3.0, -1.0]],
    dtype=np.float32,
)
targets = np.array([2.0, -4.0, 0.0, 10.0], dtype=np.float32)

with sciencekit.linear_regression(dtype=np.float32, include_intercept=False) as builder:
    with builder.fit(features, targets) as model:
        predictions = model.predict(features)
        assert predictions.dtype == np.float32
        np.testing.assert_allclose(predictions, targets, rtol=1e-5, atol=1e-5)
```

<div class="sk-box sk-box--warning">
  <p><strong>Strict dtype matching:</strong> Passing a <code>float32</code> array to a <code>float64</code> builder or model (or vice versa) raises <code>InvalidArgumentError</code> (<code>status_code == 3</code>) rather than silently casting.</p>
</div>
