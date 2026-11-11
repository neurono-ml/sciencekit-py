# Structured Errors

Every error raised by `sciencekit` inherits from `sciencekit.ScienceKitError` **and** from the corresponding standard Python exception base, so callers can catch either structured ABI failures or idiomatic Python exceptions (`ValueError`, `OSError`, `RuntimeError`, `NotImplementedError`).

## Status Code Mapping

| C ABI Constant | `status_code` | `status_name` | Python Exception Class | Stdlib Base |
| --- | --- | --- | --- | --- |
| `SK_FFI_STATUS_INVALID_HANDLE` | `1` | `"invalid-handle"` | `InvalidHandleError` | `RuntimeError` |
| `SK_FFI_STATUS_SHAPE` | `2` | `"shape"` | `ShapeError` | `ValueError` |
| `SK_FFI_STATUS_INVALID_ARGUMENT` | `3` | `"invalid-argument"` | `InvalidArgumentError` | `ValueError` |
| `SK_FFI_STATUS_UNSUPPORTED` | `4` | `"unsupported"` | `UnsupportedError` | `NotImplementedError` |
| `SK_FFI_STATUS_NOT_FITTED` | `5` | `"not-fitted"` | `NotFittedError` | `RuntimeError` |
| `SK_FFI_STATUS_IO` | `6` | `"io"` | `IoError` | `OSError` |
| `SK_FFI_STATUS_INTERNAL` | `7` | `"internal"` | `InternalError` | `RuntimeError` |

## Inspecting Structured Attributes

Every `ScienceKitError` instance exposes `.status_code`, `.status_name`, and `.message`, and formats as `"sciencekit <status_name> (<status_code>): <message>"` (`tests/test_errors.py`):

```python
import numpy as np
import sciencekit
from sciencekit import LinearRegression, ShapeError

features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
wrong_targets = np.array([1.0, 2.0], dtype=np.float64)

with LinearRegression() as builder:
    try:
        builder.fit(features, wrong_targets)
    except ShapeError as exc:
        assert exc.status_code == sciencekit.STATUS_SHAPE  # 2
        assert exc.status_name == "shape"
        assert isinstance(exc, ValueError)
```

## Panic Containment & Thread-Local Diagnostics

Inside `sciencekit_ffi`, every `extern "C"` entry point wraps execution in `catch_unwind` and records human-readable diagnostics into a thread-local slot (`sk_ffi_last_error`). When `sciencekit._native` detaches the GIL with `py.detach`, it reads and copies `sk_ffi_last_error` on the **same OS thread** before re-acquiring the GIL, guaranteeing thread-accurate diagnostics even under heavy concurrent multi-threading.
