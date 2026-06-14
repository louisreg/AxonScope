"""Architecture guardrails for pre-release API cleanup.

These tests intentionally encode project-level rules from GUIDELINES.md. They
should fail early when a refactor reintroduces compatibility aliases or hidden
coupling through the public package facade.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterable
from pathlib import Path

import pytest
import axonscope as axs
from axonscope.axons import Myelinated, Unmyelinated


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "axonscope"


def _python_sources(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _public_parameters(obj: object) -> set[str]:
    return set(inspect.signature(obj).parameters)


def _jax_import_locations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax" or alias.name.startswith("jax."):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "jax" or module.startswith("jax.")):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return offenders


def test_guidelines_is_the_root_project_philosophy_reference():
    assert (REPO_ROOT / "GUIDELINES.md").is_file()

    for path in (REPO_ROOT / "agent.md", REPO_ROOT / "todo.md"):
        text = path.read_text(encoding="utf-8")
        assert "GUIDELINES.md" in text
        assert "AXONSCOPE_PRODUCT_SOLVER_GUIDELINES.md" not in text


def test_public_facade_does_not_expose_removed_compatibility_aliases():
    removed_names = {
        "analysis",
        "visualization",
        "run_batch",
    }

    exposed = {name for name in removed_names if hasattr(axs, name)}

    assert exposed == set()
    assert removed_names.isdisjoint(set(axs.__all__))


def test_public_signatures_do_not_reintroduce_old_unit_suffix_arguments():
    checks = {
        axs.AxonSimulation: {"x_offset", "x_offset_um", "y", "y_um", "z", "z_um"},
        axs.AxonPopulation: {"x_offset", "x_offset_um", "y", "y_um", "z", "z_um"},
        axs.simulate: {"duration_ms", "dt_ms", "tsim"},
        axs.simulate_pool: {"duration_ms", "dt_ms", "tsim"},
        axs.AxonInstance: {"x_offset_um", "y_um", "z_um"},
        axs.AxonInstance.set_position: {"x_offset_um", "y_um", "z_um"},
        axs.Recording: {"positions_um", "sample_dt_ms"},
        axs.PointSourceElectrode: {
            "x_um",
            "y_um",
            "z_um",
            "x0_m",
            "y0_m",
            "z0_m",
            "min_distance_um",
        },
        axs.IntracellularCurrentClamp: {"position_um"},
    }

    reintroduced: dict[str, set[str]] = {}
    for obj, forbidden in checks.items():
        present = _public_parameters(obj) & forbidden
        if present:
            reintroduced[getattr(obj, "__qualname__", repr(obj))] = present

    assert reintroduced == {}


def test_root_axon_simulation_is_not_the_legacy_instance_alias():
    assert axs.AxonSimulation is not axs.AxonInstance
    assert not hasattr(axs.AxonSimulation, "add_current_clamp")
    assert not hasattr(axs.AxonSimulation, "add_extracellular_context")
    assert not hasattr(axs.AxonSimulation, "set_position")
    assert not hasattr(axs.AxonPopulation, "add_current_clamp")
    assert not hasattr(axs.AxonPopulation, "add_extracellular_context")
    assert not hasattr(axs.AxonPopulation, "set_position")


def test_current_raw_string_public_domains_are_tracked_for_phase2():
    """Keep remaining string-based domains visible until typed replacements land."""

    tracked_domains = {}
    todo_text = (REPO_ROOT / "todo.md").read_text(encoding="utf-8")
    missing_from_signature: dict[str, set[str]] = {}
    missing_from_todo: set[str] = set()

    for obj, parameters in tracked_domains.items():
        qualname = getattr(obj, "__qualname__", repr(obj))
        present = _public_parameters(obj)
        missing = set(parameters).difference(present)
        if missing:
            missing_from_signature[qualname] = missing
        for parameter in parameters:
            if parameter not in todo_text:
                missing_from_todo.add(f"{qualname}.{parameter}")

    assert missing_from_signature == {}
    assert missing_from_todo == set()


def test_recording_public_api_uses_typed_signals_not_raw_strings():
    forbidden_parameters = {
        axs.Recording: {"variables", "spatial_mode"},
        axs.Recording.only: {"variables"},
        axs.Recording.center: {"variables"},
        axs.Recording.probes: {"variables"},
        axs.Recording.indices: {"variables"},
    }
    reintroduced: dict[str, set[str]] = {}

    for obj, forbidden in forbidden_parameters.items():
        present = _public_parameters(obj) & forbidden
        if present:
            reintroduced[getattr(obj, "__qualname__", repr(obj))] = present

    assert reintroduced == {}

    for call in (
        lambda: axs.Recording(signals="Vm"),
        lambda: axs.Recording.center("Vm"),
        lambda: axs.Recording.probes("Vm"),
        lambda: axs.Recording.indices([0], "Vm"),
    ):
        try:
            call()
        except TypeError as exc:
            assert "signals" in str(exc)
        else:
            raise AssertionError("raw string recording signal was accepted")


def test_activation_criterion_uses_typed_position_targets():
    assert "positions" not in _public_parameters(axs.results.ActivationCriterion)
    assert "indices" not in _public_parameters(axs.results.ActivationCriterion)
    assert "target" in _public_parameters(axs.results.ActivationCriterion)

    criterion = axs.results.ActivationCriterion(target=axs.positions.DISTAL)
    assert criterion.target is axs.positions.DISTAL
    assert axs.positions.DISTAL.index_values is None


def test_axon_formulation_uses_public_enum_not_raw_strings():
    assert _public_parameters(Unmyelinated) >= {"formulation"}
    assert _public_parameters(Myelinated) >= {"formulation"}
    assert axs.axons.CableFormulation.SINGLE_CABLE.value == "single-cable"

    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(),
        diameter=1.0 * axs.um,
    )
    layout = axs.axons.Layout.single_uniform(
        section,
        length=100.0 * axs.um,
        compartments=3,
    )

    with pytest.raises(TypeError, match="CableFormulation"):
        axs.axons.Axon(layout=layout, formulation="single-cable")


def test_extracellular_public_contracts_are_exported():
    assert axs.AxonId is axs.identifiers.AxonId
    assert axs.DriveId is axs.identifiers.DriveId
    assert axs.ExtracellularFootprint is axs.stimulation.ExtracellularFootprint
    assert axs.ExtracellularDrive is axs.stimulation.ExtracellularDrive
    assert axs.ExtracellularStimulation is axs.stimulation.ExtracellularStimulation
    assert axs.ExtracellularPotential is axs.stimulation.ExtracellularPotential


def test_internal_modules_do_not_import_the_public_axonscope_facade():
    offenders: list[str] = []

    for path in _python_sources(SRC_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "axonscope":
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module == "axonscope":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_non_visualization_modules_do_not_import_visualization_helpers():
    allowed = {
        SRC_ROOT / "results" / "__init__.py",
        SRC_ROOT / "results" / "visualization.py",
    }
    offenders: list[str] = []

    for path in _python_sources(SRC_ROOT):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "axonscope.results.visualization":
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module == "axonscope.results.visualization":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
                elif node.level > 0 and module == "visualization":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_descriptive_layers_do_not_import_jax_backend_directly():
    guarded_paths = {
        SRC_ROOT / "recording.py",
        SRC_ROOT / "population.py",
        SRC_ROOT / "preparation" / "signatures.py",
    }
    guarded_dirs = {
        SRC_ROOT / "axons",
        SRC_ROOT / "membranes",
        SRC_ROOT / "results",
    }
    offenders: list[str] = []

    for path in _python_sources(SRC_ROOT):
        if path in guarded_paths or any(directory in path.parents for directory in guarded_dirs):
            offenders.extend(_jax_import_locations(path))

    assert offenders == []


def test_dispatcher_execution_does_not_import_concrete_jax_batch_kernels():
    path = SRC_ROOT / "dispatcher" / "execution.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_modules = {
        "axonscope.solvers.batch_kernels",
        "axonscope.solvers.runtime",
        "axonscope.icm.backends",
    }
    offenders = _jax_import_locations(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and module in forbidden_modules:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_dispatcher_runtime_batches_remains_host_side_only():
    path = SRC_ROOT / "dispatcher" / "runtime_batches.py"

    assert _jax_import_locations(path) == []


def test_public_simulation_orchestrator_does_not_import_jax_directly():
    path = SRC_ROOT / "simulation.py"

    assert _jax_import_locations(path) == []


def test_crank_nicholson_facade_delegates_to_backend_boundary():
    path = SRC_ROOT / "solvers" / "crank_nicholson.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_modules = {
        "axonscope.solvers.axon_runtime",
        "axonscope.solvers.kernels",
        "axonscope.solvers.runtime",
    }
    offenders = _jax_import_locations(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and module in forbidden_modules:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif node.level > 0 and module in {"axon_runtime", "kernels", "runtime"}:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []
