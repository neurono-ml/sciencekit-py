# Installation

## Requirements

- **Python:** 3.10 or newer (`3.10`–`3.13`).
- **NumPy:** `numpy >= 1.24` at runtime.
- **Rust toolchain (when building from source):** Rust `1.85` or newer and `maturin >= 1.7, < 2.0`.

## Building and Installing a Release Wheel

Clone `neurono-ml/sciencekit-py` and build an optimized wheel with `maturin`:

```bash
git clone https://github.com/neurono-ml/sciencekit-py.git
cd sciencekit-py

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip "maturin>=1.7,<2.0" numpy pytest mypy

maturin build --release --locked --out dist
pip install dist/sciencekit-*.whl
```

<div class="sk-box sk-box--info">
  <p><strong>Self-contained wheel:</strong> Cargo fetches <code>sciencekit_ffi</code> from <code>https://github.com/neurono-ml/sciencekit</code> and links it into <code>sciencekit._native</code> at compile time. Installed wheels require no external <code>libsciencekit_ffi.so</code> or <code>LD_LIBRARY_PATH</code> configuration.</p>
</div>

## Editable Development Install

When iterating on the Rust wrapper or Python package locally, use `maturin develop`:

```bash
maturin develop
pytest -q
```

## Reproducible Container Build

You can also build and test inside `sciencekit-devcontainer:latest` (Ubuntu 24.04, Rust 1.85.1, Python 3.12, maturin 1.16):

```bash
docker run --rm -v "$PWD":/workspace -w /workspace sciencekit-devcontainer:latest \
  bash -lc 'maturin build --release --locked --out dist && pip3 install --break-system-packages --force-reinstall --no-deps dist/sciencekit-*.whl && pytest -q'
```

## Verifying the Installation

Derived directly from `tests/test_import.py`:

```python
import sciencekit
import sciencekit.linear_model
import sciencekit.preprocessing

assert sciencekit.__version__ == "0.1.0"
assert sciencekit.abi_version() == {"major": 1, "minor": 0, "patch": 0}
assert sciencekit.features() & sciencekit.FEATURE_LINEAR_REGRESSION
assert sciencekit.features() & sciencekit.FEATURE_STANDARD_SCALER
```
