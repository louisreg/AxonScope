from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
PUBLIC_EXAMPLE_DIRS = (
    EXAMPLES_ROOT / "basic",
    EXAMPLES_ROOT / "advanced",
    EXAMPLES_ROOT / "with_nrv",
)


def _example_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory in PUBLIC_EXAMPLE_DIRS:
        paths.extend(
            path
            for path in directory.rglob("*.py")
            if "__pycache__" not in path.parts
            and not path.name.startswith("_")
        )
    return tuple(sorted(paths))


def _load_script(path: Path):
    relative = path.relative_to(REPO_ROOT)
    safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", str(relative))
    module_name = f"_axonscope_example_{safe_name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_public_example_scripts_are_importable_and_have_main():
    paths = _example_paths()

    assert paths
    for path in paths:
        module = _load_script(path)
        assert callable(module.main), path


def test_public_examples_do_not_import_axonscope_internals():
    forbidden = (
        "axonscope.runtime",
        "from axonscope.solvers",
        "import axonscope.solvers",
        "CrankNicholson",
    )

    for path in _example_paths():
        text = path.read_text(encoding="utf-8")
        leaked = [term for term in forbidden if term in text]
        assert leaked == [], f"{path} imports internal API: {leaked}"


def test_basic_stimuli_example_runs(monkeypatch):
    module = _load_script(EXAMPLES_ROOT / "basic" / "02_stimuli_and_units.py")
    monkeypatch.setattr(plt, "show", lambda: None)
    module.main()


def test_basic_point_source_example_runs(monkeypatch):
    module = _load_script(EXAMPLES_ROOT / "basic" / "03_point_source_footprint.py")
    monkeypatch.setattr(plt, "show", lambda: None)
    module.main()


def test_pipeline_inspection_example_runs(monkeypatch):
    module = _load_script(
        EXAMPLES_ROOT / "advanced" / "runtime" / "03_pipeline_inspection.py"
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    module.main()


def test_benchmark_scripts_are_outside_examples_and_importable():
    benchmark_examples = sorted((EXAMPLES_ROOT / "benchmarks").glob("*.py"))
    assert benchmark_examples == []

    threshold_curves = _load_script(
        REPO_ROOT / "benchmark" / "curves" / "threshold_curves.py"
    )
    recruitment_curves = _load_script(
        REPO_ROOT / "benchmark" / "curves" / "recruitment_curves.py"
    )

    assert callable(threshold_curves.main)
    assert callable(recruitment_curves.main)


def test_examples_readme_references_existing_learning_and_benchmark_paths():
    readme = EXAMPLES_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    referenced_paths = sorted(
        set(re.findall(r"`((?:examples|benchmark)/[^`]+)`", text))
    )

    assert referenced_paths
    for reference in referenced_paths:
        path = REPO_ROOT / reference.rstrip("/")
        assert path.exists(), f"{readme.relative_to(REPO_ROOT)} references missing {reference}"
