# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Dataset interoperability (`sciencekit.datasets`): NumPy arrays, pandas
  frames/series, Arrow PyCapsule producers (pyarrow, Polars, DuckDB and any
  `__arrow_c_stream__` object), SciPy CSR/CSC/COO sparse matrices, Polars
  `DataFrame`/`LazyFrame`, Dask DataFrames and PySpark DataFrames are accepted
  directly by `fit`, `predict`, `transform` and `score`.
- Zero-copy NumPy borrow with explicit `copy=True`/`cast=True` opt-ins and
  pointer-level evidence in `ConversionReport`.
- CSR `data`/`indices`/`indptr` buffers borrowed without densifying in the
  conversion layer; the dense block required by `sciencekit_ffi` ABI 1.0 is
  materialized only at the call boundary and reported.
- Missing-value policy (`on_missing="reject"|"nan"|"fill"` with `fill_value`).
- Categorical targets: dictionary columns produce a label table and predictions
  map back to labels.
- Result formats (`output="numpy"|"pandas"|"arrow"|"polars"|"sparse"|"auto"`);
  Arrow results implement `__arrow_c_stream__`.
- `sciencekit.describe(dataset)` and `builder.last_conversion` /
  `model.last_conversion` conversion reports.
- `UnknownColumnError` (`KeyError`) and `UnsupportedColumnTypeError`
  (`TypeError`) exceptions.
- mdBook guides for every supported dataset representation.

### Changed

- `fit`, `predict`, `transform` and `score` accept any supported dataset
  representation and expose `target=`, `columns=`, `cast=`, `on_missing=`,
  `fill_value=` and `output=` keywords.
- Sparse SciPy matrices no longer require an explicit `to_dense()` call; the
  adapter reads the CSR buffers directly and reports the ABI boundary copy.
