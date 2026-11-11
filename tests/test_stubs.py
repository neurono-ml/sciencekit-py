"""Type stub presence, AST syntax, PEP 561 resolution, and `__all__` parity checks."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import sciencekit
import sciencekit.linear_model
import sciencekit.preprocessing


def _package_dir() -> Path:
    init_file = sciencekit.__file__
    assert init_file is not None
    return Path(init_file).resolve().parent


def test_type_stubs_exist_and_cover_public_exports() -> None:
    pkg_dir = _package_dir()
    assert (pkg_dir / "py.typed").is_file()

    for stub_name, runtime_mod in (
        ("__init__.pyi", sciencekit),
        ("linear_model.pyi", sciencekit.linear_model),
        ("preprocessing.pyi", sciencekit.preprocessing),
    ):
        stub_path = pkg_dir / stub_name
        assert stub_path.is_file(), f"missing stub {stub_path}"
        tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
        defined_names: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined_names.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined_names.add(node.target.id)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name)

        missing = set(runtime_mod.__all__) - defined_names
        assert not missing, f"{stub_name} is missing exports: {sorted(missing)}"

    if importlib.util.find_spec("mypy") is not None:
        snippet = (
            "import numpy as np\n"
            "import sciencekit\n"
            "from sciencekit.linear_model import LinearRegression, LinearRegressionModel\n"
            "from sciencekit.preprocessing import StandardScaler, StandardScalerModel\n"
            "x = np.zeros((4, 2), dtype=np.float64)\n"
            "y = np.zeros(4, dtype=np.float64)\n"
            "m: LinearRegressionModel = LinearRegression().include_intercept(True).fit(x, y)\n"
            "p: np.ndarray = m.predict(x)\n"
            "s: float = m.score(x, y)\n"
            "sc: StandardScalerModel = StandardScaler().with_mean(True).fit(x)\n"
            "t: np.ndarray = sc.transform(x)\n"
        )
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", "-c", snippet],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
