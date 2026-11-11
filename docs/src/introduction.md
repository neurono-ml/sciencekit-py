# Introduction

<div class="sk-hero reveal">
  <div class="sk-announce">
    <span class="sk-announce__tag">v0.1.0</span>
    PyO3 0.30 &amp; zero-copy NumPy over the stable <code>sciencekit_ffi</code> C ABI
  </div>
  <h1 class="sk-hero__title">Machine learning for Python, backed by Rust</h1>
  <p class="sk-hero__subtitle">
    <strong>sciencekit-py</strong> ships a self-contained Python wheel named <code>sciencekit</code> — chainable estimator builders, zero-copy C-contiguous NumPy borrows, GIL-detached compute, structured exceptions with C ABI status codes, and deterministic handle lifecycle.
  </p>
  <div class="sk-btn-row">
    <a class="sk-btn sk-btn--primary" href="./quickstart.html">Run the Quickstart →</a>
    <a class="sk-btn sk-btn--secondary" href="./api-reference.html">API Reference</a>
    <a class="sk-btn sk-btn--star" href="https://github.com/neurono-ml/sciencekit-py">GitHub Repository</a>
  </div>
</div>

<div class="sk-stats reveal">
  <div class="sk-stats__item">
    <div class="sk-stats__value">PyO3 0.30</div>
    <span class="sk-stats__label">maturin mixed wheel</span>
  </div>
  <div class="sk-stats__item">
    <div class="sk-stats__value">ABI 1.0</div>
    <span class="sk-stats__label">in-process <code>sciencekit_ffi</code></span>
  </div>
  <div class="sk-stats__item">
    <div class="sk-stats__value">f64 &amp; f32</div>
    <span class="sk-stats__label">zero-copy NumPy input</span>
  </div>
  <div class="sk-stats__item">
    <div class="sk-stats__value">Apache-2.0</div>
    <span class="sk-stats__label">permissive open-source license</span>
  </div>
</div>

## What is `sciencekit-py`?

[`sciencekit`](https://github.com/neurono-ml/sciencekit) is a Rust machine-learning library reimplementing scikit-learn's estimators and transformers with native linear algebra kernels (`faer`, `matrixmultiply`, `ndarray-linalg`). [`neurono-ml/sciencekit-py`](https://github.com/neurono-ml/sciencekit-py) is the standalone Python binding repository that packages the `sciencekit` Python distribution over the stable `sciencekit_ffi` C ABI.

```python
import numpy as np
from sciencekit.linear_model import LinearRegression

features = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
    dtype=np.float64,
)
targets = np.array([1.0, 3.0, 4.0, 14.0], dtype=np.float64)

with LinearRegression(dtype="float64") as builder:
    with builder.include_intercept(True).backend("faer").fit(features, targets) as model:
        predictions = model.predict(features)
        r_squared = model.score(features, targets)
```

## Architecture over `sciencekit_ffi`

`sciencekit_ffi` is not published on crates.io as a separate crate. Instead of vendoring a C header and forcing Python users to configure `LD_LIBRARY_PATH` at runtime, `sciencekit-py` depends on `sciencekit_ffi` directly via Cargo git dependency and links it in-process as an `rlib` inside `sciencekit._native`. Every operation traverses the exact `sk_ffi_*` entry points, opaque handle registry, panic containment, and thread-local error channel defined in `include/sciencekit.h`.

<figure class="sk-diagram reveal">
  <svg viewBox="0 0 860 220" role="img" aria-label="Architecture diagram showing Python code with NumPy arrays calling sciencekit, passing through PyO3 zero-copy SKFfiArray descriptors and py.detach GIL release into sciencekit_ffi and Rust numerical backends">
    <title>sciencekit-py architecture over the in-process C ABI</title>
    <defs>
      <marker id="arrow-py-arch" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
      </marker>
    </defs>
    <rect x="16" y="34" width="180" height="96" rx="14" fill="#e0f2fe" stroke="#0284c7" stroke-width="2" />
    <text x="106" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#0c4a6e">Python 3.10+ &amp; NumPy</text>
    <text x="106" y="88" text-anchor="middle" font-size="12" fill="#0c4a6e">import sciencekit</text>
    <text x="106" y="109" text-anchor="middle" font-size="11" fill="#0369a1">with builder.fit(X, y) as m</text>

    <line x1="196" y1="82" x2="236" y2="82" stroke="#64748b" stroke-width="2" marker-end="url(#arrow-py-arch)" />

    <rect x="240" y="22" width="210" height="120" rx="14" fill="#ede9fe" stroke="#7c3aed" stroke-width="2" />
    <text x="345" y="54" text-anchor="middle" font-size="14" font-weight="700" fill="#3b0764">sciencekit._native (PyO3)</text>
    <text x="345" y="76" text-anchor="middle" font-size="12" fill="#3b0764">Zero-copy SKFfiArray borrow</text>
    <text x="345" y="97" text-anchor="middle" font-size="11" fill="#4c1d95">py.detach() GIL release</text>
    <text x="345" y="118" text-anchor="middle" font-size="11" fill="#4c1d95">RwLock / Mutex + Drop / close()</text>

    <line x1="450" y1="82" x2="490" y2="82" stroke="#64748b" stroke-width="2" marker-end="url(#arrow-py-arch)" />

    <rect x="494" y="34" width="175" height="96" rx="14" fill="#f0fdf4" stroke="#16a34a" stroke-width="2" />
    <text x="581" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#14532d">sciencekit_ffi</text>
    <text x="581" y="88" text-anchor="middle" font-size="12" fill="#14532d">Stable C ABI (in-process)</text>
    <text x="581" y="109" text-anchor="middle" font-size="11" fill="#166534">SKFfiStatus · Panic guard</text>

    <line x1="669" y1="82" x2="707" y2="82" stroke="#64748b" stroke-width="2" marker-end="url(#arrow-py-arch)" />

    <rect x="711" y="34" width="133" height="96" rx="14" fill="#fef3c7" stroke="#d97706" stroke-width="2" />
    <text x="777" y="66" text-anchor="middle" font-size="14" font-weight="700" fill="#78350f">Rust Core</text>
    <text x="777" y="88" text-anchor="middle" font-size="12" fill="#78350f">faer · ndarray</text>
    <text x="777" y="109" text-anchor="middle" font-size="11" fill="#92400e">Fitted models</text>

    <rect x="240" y="160" width="429" height="42" rx="10" fill="#ffe4e6" stroke="#e11d48" stroke-width="1.5" />
    <text x="454" y="186" text-anchor="middle" font-size="12" font-weight="600" fill="#881337">Errors map to ScienceKitError subclasses with .status_code, .status_name &amp; .message</text>
  </svg>
  <figcaption>Figure 1 — Every Python call borrows C-contiguous NumPy buffers zero-copy, detaches the GIL, and invokes <code>sciencekit_ffi</code> in-process.</figcaption>
</figure>

<div class="sk-cards reveal">
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--indigo">1</span>
    <div class="sk-card__title">Chainable Builders &amp; Models</div>
    <p><code>LinearRegression</code> and <code>StandardScaler</code> builders configure native handles and produce immutable, thread-safe fitted models.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--sky">2</span>
    <div class="sk-card__title">Zero-Copy Dense &amp; Validated CSR</div>
    <p>C-contiguous <code>float64</code> and <code>float32</code> arrays are borrowed directly; <code>CsrDescriptor</code> validates sparse structure without silent densification.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--rose">3</span>
    <div class="sk-card__title">Structured Exceptions</div>
    <p>Every error raises a typed subclass of <code>ScienceKitError</code> (and standard Python exceptions like <code>ValueError</code> or <code>OSError</code>) carrying the ABI status code.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--emerald">4</span>
    <div class="sk-card__title">Deterministic Lifecycle</div>
    <p>Idempotent <code>.close()</code>, <code>with</code> context managers, and Rust <code>Drop</code> finalizers ensure zero handle leaks and zero double-frees.</p>
  </div>
</div>
