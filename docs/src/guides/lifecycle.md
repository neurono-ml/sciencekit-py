# Handle Lifecycle & Concurrency

Builders (`SKFfiBuilder`) and fitted models (`SKFfiModel`) wrap opaque native handles registered inside `sciencekit_ffi`.

## Three Ways Handles Are Released

1. **Context manager (`with` statement — recommended):** `__exit__` calls `.close()` deterministically when exiting the block.
2. **Explicit `.close()`:** Releases the underlying native handle immediately. Calling `.close()` two or more times is a safe no-op (`tests/test_lifecycle_and_concurrency.py::test_close_is_idempotent_and_closed_handles_raise_invalid_handle`).
3. **Garbage-collector finalizer (`Drop`):** If an object becomes unreachable without `.close()`, PyO3 runs the Rust `Drop` implementation when Python collects the object, releasing the handle once.

```python
import gc
import numpy as np
import sciencekit
from sciencekit import InvalidHandleError, LinearRegression

gc.collect()
baseline = sciencekit.live_handles()

features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

builder = LinearRegression()
model = builder.fit(features, targets)
assert sciencekit.live_handles() == baseline + 2

builder.close()
builder.close()  # idempotent no-op
assert builder.closed is True

model.close()
assert sciencekit.live_handles() == baseline
```

Calling any operation on a closed builder or model raises `InvalidHandleError` (`status_code == 1`) without dereferencing freed memory.

## GIL Release & Concurrent Multi-Threaded Inference

Every native call (`fit`, `predict`, `transform`, `score`, `save`, `load`, `close`) detaches the Python GIL via PyO3's `py.detach(...)`:

- **Builder handles** synchronize mutations and `fit` with a Rust `std::sync::Mutex`.
- **Model handles** synchronize readers and `close()` with a Rust `std::sync::RwLock`: multiple Python threads execute `predict`, `transform`, and `score` **simultaneously** without holding the GIL, while any concurrent `close()` waits for active readers to finish before calling `sk_ffi_model_free`.
- Locks are always acquired *inside* `py.detach(...)`, eliminating lock-order inversions with the Python GIL.

Derived from `tests/test_lifecycle_and_concurrency.py::test_concurrent_multithreaded_predictions_and_transforms`:

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sciencekit import LinearRegression, StandardScaler

rng = np.random.default_rng(7)
features = rng.standard_normal((2_048, 8), dtype=np.float64)
weights = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.25, 4.0], dtype=np.float64)
targets = 2.0 + features @ weights

with LinearRegression() as lr_builder, StandardScaler() as sc_builder:
    with lr_builder.fit(features, targets) as lr_model, sc_builder.fit(features) as sc_model:
        def worker(worker_id: int) -> bool:
            batch = np.ascontiguousarray(features[worker_id::8])
            pred = lr_model.predict(batch)
            trans = sc_model.transform(batch)
            return pred.shape == (256,) and trans.shape == (256, 8)

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert all(pool.map(worker, range(8)))
```
