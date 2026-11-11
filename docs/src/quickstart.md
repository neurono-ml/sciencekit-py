# Quickstart

Every example below is taken from the repository's `pytest` suite (`tests/test_linear_regression.py`, `tests/test_standard_scaler.py`, and `tests/test_persistence.py`).

## 1. Fit, Predict, and Score Linear Regression

Fit the exact plane $y = 1 + 2x_0 + 3x_1$ in double precision (`float64`):

```python
import numpy as np
from sciencekit.linear_model import LinearRegression

features = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
    dtype=np.float64,
)
targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

with LinearRegression(dtype="float64") as builder:
    builder.include_intercept(True).backend("faer")

    with builder.fit(features, targets) as model:
        predictions = model.predict(features)
        r_squared = model.score(features, targets)

        np.testing.assert_allclose(predictions, targets, rtol=1e-10, atol=1e-10)
        assert abs(r_squared - 1.0) < 1e-12
```

## 2. Standardize Features with `StandardScaler`

Center columns to zero mean and scale to unit population standard deviation:

```python
import numpy as np
from sciencekit.preprocessing import StandardScaler

features = np.array(
    [[0.0, 10.0], [0.0, 20.0], [1.0, 30.0], [1.0, 40.0]],
    dtype=np.float64,
)

with StandardScaler() as builder:
    with builder.with_mean(True).with_std(True).fit(features) as scaler:
        transformed = scaler.transform(features)
        np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(transformed.std(axis=0, ddof=0), [1.0, 1.0], atol=1e-12)
```

## 3. Save and Reload a Fitted Model

Persist a fitted `float64` `LinearRegressionModel` through the C ABI interop state contract and reload it in another scope:

```python
import tempfile
from pathlib import Path
import numpy as np
import sciencekit
from sciencekit.linear_model import LinearRegression

features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)
probe = np.array([[4.0, -1.0], [-2.0, 5.0]], dtype=np.float64)

with tempfile.TemporaryDirectory() as tmp:
    state_path = Path(tmp) / "linear_regression.state"

    with LinearRegression() as builder:
        with builder.fit(features, targets) as fitted:
            expected = fitted.predict(probe)
            fitted.save(state_path)

    with sciencekit.load(state_path) as restored:
        np.testing.assert_allclose(restored.predict(probe), expected, rtol=1e-12, atol=1e-12)
```
