"""Standard scaler fit, transform, fit_transform, and f64/f32 tests."""

from __future__ import annotations

import numpy as np

import sciencekit
from sciencekit.preprocessing import StandardScaler, StandardScalerBuilder, StandardScalerModel


def test_standard_scaler_f64_centers_and_scales_columns() -> None:
    features = np.array(
        [[0.0, 10.0], [0.0, 20.0], [1.0, 30.0], [1.0, 40.0]],
        dtype=np.float64,
    )
    expected_mean = features.mean(axis=0)
    expected_std = features.std(axis=0, ddof=0)
    expected = (features - expected_mean) / expected_std

    with StandardScaler() as builder:
        assert isinstance(builder, StandardScalerBuilder)
        assert builder.family == "standard_scaler"
        assert builder.dtype == "float64"
        chained = builder.with_mean(True).with_std(True).backend("faer")
        assert chained is builder

        with builder.fit(features) as model:
            assert isinstance(model, StandardScalerModel)
            assert model.family == "standard_scaler"
            assert model.dtype == "float64"
            transformed = model.transform(features)
            assert transformed.dtype == np.float64
            assert transformed.shape == (4, 2)
            np.testing.assert_allclose(transformed, expected, rtol=1e-12, atol=1e-12)


def test_standard_scaler_with_mean_or_std_disabled_and_f32() -> None:
    features = np.array(
        [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0], [8.0, 16.0]],
        dtype=np.float32,
    )

    with sciencekit.standard_scaler(dtype="float32", with_mean=True, with_std=False) as mean_only:
        centered = mean_only.fit_transform(features)
        assert centered.dtype == np.float32
        assert centered.shape == (4, 2)
        np.testing.assert_allclose(
            centered,
            features - features.mean(axis=0),
            rtol=1e-5,
            atol=1e-5,
        )

    with StandardScaler(dtype=np.float32, with_mean=False, with_std=True) as std_only:
        scaled = std_only.fit_transform(features)
        assert scaled.dtype == np.float32
        np.testing.assert_allclose(
            scaled,
            features / features.std(axis=0, ddof=0),
            rtol=1e-5,
            atol=1e-5,
        )
