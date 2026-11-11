# Contributing

## Repository Layout

- `Cargo.toml` / `pyproject.toml` — Standalone Rust + `maturin` configuration targeting Rust 1.85 (edition 2024) and Python 3.10+.
- `src/` — PyO3 extension (`sciencekit._native`) organized into modular files (`<= 200` lines each):
  - `src/array/` — Zero-copy `DenseDescriptor`, `CsrDescriptor`, and owned output conversion.
  - `src/engine/` — Thread-safe `Mutex`/`RwLock` wrappers around `SKFfiBuilder` and `SKFfiModel`.
  - `src/error/` — Status inference and mapping to `sciencekit._exceptions`.
  - `src/wrappers/` — `LinearRegression`, `LinearRegressionModel`, `StandardScaler`, `StandardScalerModel`, and `load`.
- `python/sciencekit/` — Python package surface, structured exceptions, `.pyi` type stubs, and `py.typed`.
- `tests/` — `pytest` suite covering imports, linear fixtures, scalers, persistence, descriptors, structured errors, handle leaks, concurrency, and type stubs.
- `docs/` — English `mdBook` documentation site and `docs/scripts/postbuild.mjs` sitemap generator (maintained on branch `docs/documentations`).

## Local Quality Gates

Run all Rust, Python, stub, and documentation gates before opening a pull request:

```bash
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked

maturin build --release --locked --out dist
pip install --force-reinstall dist/sciencekit-*.whl
pytest -q
mypy --strict python/sciencekit/__init__.pyi
```

When editing documentation on `docs/documentations`:

```bash
mdbook build docs
node docs/scripts/postbuild.mjs
```

## License

Contributions are licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
