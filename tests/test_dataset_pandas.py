"""pandas frames: Arrow protocol, nullable dtypes, categoricals, legacy fallback."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import sciencekit
from sciencekit import (
    InvalidArgumentError,
    LinearRegression,
    StandardScaler,
    UnknownColumnError,
    describe,
)

FEATURES = np.array(
    [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0]], dtype=np.float64
)
TARGETS = np.array([3.0, 5.0, 9.0, 11.0, 15.0], dtype=np.float64)


def test_dataframe_is_converted_through_the_arrow_protocol() -> None:
    frame = pd.DataFrame(FEATURES, columns=["a", "b"])
    frame["y"] = TARGETS
    report = describe(frame, target="y")
    assert report.representation == "pandas.dataframe"
    assert report.feature_names == ["a", "b"]
    assert report.arrow_zero_copy is True

    with LinearRegression() as builder:
        with builder.fit(frame, target="y") as model:
            predictions = model.predict(frame, columns=["a", "b"])
            np.testing.assert_allclose(predictions, TARGETS, rtol=1e-10)


def test_series_is_accepted_as_a_target_vector() -> None:
    frame = pd.DataFrame(FEATURES, columns=["a", "b"])
    series = pd.Series(TARGETS, name="y")
    with LinearRegression() as builder:
        with builder.fit(frame, series) as model:
            assert model.score(frame, series) == pytest.approx(1.0)


def test_dataframe_index_is_not_a_feature() -> None:
    frame = pd.DataFrame(FEATURES, columns=["a", "b"], index=[100, 200, 300, 400, 500])
    report = describe(frame, dtype="float32")
    assert report.feature_names == ["a", "b"]

    with StandardScaler() as scaler:
        transformed = scaler.fit_transform(frame)
        assert transformed.shape == (5, 2)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)


def test_nullable_integer_column_under_each_missing_policy() -> None:
    frame = pd.DataFrame(
        {
            "a": pd.array([1.0, None, 3.0, 4.0, 5.0], dtype="Float64"),
            "b": pd.array([2.0, 1.0, 4.0, 3.0, 6.0], dtype="Float64"),
        }
    )
    with LinearRegression() as builder:
        with pytest.raises(InvalidArgumentError) as error:
            builder.fit(frame, TARGETS)
        assert "'a'" in error.value.message
        assert "on_missing" in error.value.message

        with builder.fit(frame, TARGETS, on_missing="fill", fill_value=0.0) as model:
            assert np.isfinite(model.predict(frame, on_missing="fill", fill_value=0.0)).all()

    nan_frame = pd.DataFrame(
        {
            "a": pd.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype="Float64"),
            "b": pd.array([2.0, 1.0, 4.0, 3.0, 6.0], dtype="Float64"),
        }
    )
    with StandardScaler() as scaler:
        with scaler.fit(nan_frame, on_missing="nan") as model:
            transformed = model.transform(nan_frame, on_missing="nan")
            assert np.isnan(transformed[1]).any()


def test_categorical_target_maps_back_to_labels() -> None:
    frame = pd.DataFrame(
        {
            "x": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "label": pd.Categorical(["low", "low", "high", "high"]),
        }
    )
    report = describe(frame, target="label")
    assert report.label_table == ["high", "low"] or report.label_table == ["low", "high"]

    with LinearRegression() as builder:
        with builder.fit(frame, target="label") as model:
            assert model.label_table is not None
            predictions = model.predict(frame, columns=["x"])
            assert predictions.dtype == object
            assert list(predictions) == ["low", "low", "high", "high"]


class _LegacySeries:
    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def to_numpy(self, dtype=None, na_value=None):  # noqa: ANN001, ANN201
        return self._values.astype(dtype)


class _LegacyFrame:
    """Duck-typed stand-in for pandas < 2.2, which lacks `__arrow_c_stream__`."""

    __module__ = "pandas.core.frame"

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self.columns = list(columns)
        self._columns = {name: np.asarray(values) for name, values in columns.items()}

    def __getitem__(self, name: str) -> _LegacySeries:
        return _LegacySeries(self._columns[name])

    def to_numpy(self):  # noqa: ANN201
        return np.column_stack([self._columns[name] for name in self.columns])


def test_legacy_pandas_fallback_reports_its_copy() -> None:
    frame = _LegacyFrame({"a": FEATURES[:, 0], "b": FEATURES[:, 1], "y": TARGETS})
    report = describe(frame, target="y")
    assert report.representation == "pandas.dataframe"
    assert report.eager_fallback == "pandas.to_numpy"
    assert report.copied is True
    assert report.zero_copy is False

    with LinearRegression() as builder:
        with builder.fit(frame, target="y") as model:
            np.testing.assert_allclose(
                model.predict(frame, columns=["a", "b"]), TARGETS, rtol=1e-10
            )


def test_unknown_column_raises_key_error_naming_it() -> None:
    frame = pd.DataFrame(FEATURES, columns=["a", "b"])
    frame["y"] = TARGETS
    with LinearRegression() as builder:
        with pytest.raises(KeyError) as error:
            builder.fit(frame, target="missing")
        assert "missing" in str(error.value)
        with pytest.raises(UnknownColumnError):
            builder.fit(frame, target="missing")
