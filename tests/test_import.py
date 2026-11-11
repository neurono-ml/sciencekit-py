"""Import, ABI version, and capability bitmask tests."""

from __future__ import annotations

import sciencekit
import sciencekit.linear_model
import sciencekit.preprocessing


def test_package_imports_and_reports_abi_metadata() -> None:
    assert isinstance(sciencekit.__version__, str)
    assert sciencekit.__version__ == "0.1.0"

    abi = sciencekit.abi_version()
    assert abi == {"major": 1, "minor": 0, "patch": 0}

    feature_mask = sciencekit.features()
    assert feature_mask & sciencekit.FEATURE_LINEAR_REGRESSION
    assert feature_mask & sciencekit.FEATURE_STANDARD_SCALER
    assert feature_mask & sciencekit.FEATURE_MODEL_PERSISTENCE
    assert feature_mask & sciencekit.FEATURE_F32
    assert feature_mask & sciencekit.FEATURE_F64

    assert sciencekit.status_name(sciencekit.STATUS_OK) == "ok"
    assert sciencekit.status_name(sciencekit.STATUS_INVALID_HANDLE) == "invalid-handle"
    assert sciencekit.status_name(sciencekit.STATUS_SHAPE) == "shape"
    assert sciencekit.status_name(sciencekit.STATUS_INVALID_ARGUMENT) == "invalid-argument"
    assert sciencekit.status_name(sciencekit.STATUS_UNSUPPORTED) == "unsupported"
    assert sciencekit.status_name(sciencekit.STATUS_NOT_FITTED) == "not-fitted"
    assert sciencekit.status_name(sciencekit.STATUS_IO) == "io"
    assert sciencekit.status_name(sciencekit.STATUS_INTERNAL) == "internal"

    assert sciencekit.live_handles() >= 0
    assert hasattr(sciencekit, "linear_model")
    assert hasattr(sciencekit, "preprocessing")
