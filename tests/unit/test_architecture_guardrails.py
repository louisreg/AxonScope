"""Architecture guardrails for pre-release API cleanup.

These tests intentionally encode project-level rules from GUIDELINES.md. They
should fail early when a refactor reintroduces compatibility aliases or hidden
coupling through the public package facade.
"""

from __future__ import annotations

import ast
import inspect
import re
from enum import Enum
from collections.abc import Iterable
from pathlib import Path
from typing import get_args, get_type_hints

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


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_guidelines_is_the_root_project_philosophy_reference():
    assert (REPO_ROOT / "GUIDELINES.md").is_file()

    for path in (REPO_ROOT / "agent.md", REPO_ROOT / "todo.md"):
        text = path.read_text(encoding="utf-8")
        assert "GUIDELINES.md" in text
        assert "AXONSCOPE_PRODUCT_SOLVER_GUIDELINES.md" not in text


def test_public_facade_does_not_expose_removed_compatibility_aliases():
    removed_names = {
        "visualization",
        "run_batch",
    }

    exposed = {name for name in removed_names if hasattr(axs, name)}

    assert exposed == set()
    assert removed_names.isdisjoint(set(axs.__all__))


def test_analysis_namespace_is_real_package_not_results_alias():
    assert hasattr(axs, "analysis")
    assert not hasattr(axs.results, "analysis")
    assert not hasattr(axs.results, "ActivationCriterion")
    assert axs.analysis.Activation is axs.Activation
    assert axs.analysis.ActivationCriterion is not None
    assert "analysis" in axs.__all__


def test_recording_observer_strategy_excludes_superseded_generic_observer_design():
    text = (REPO_ROOT / "docs" / "recorders_observers_activation_strategy.md").read_text(
        encoding="utf-8"
    )

    forbidden_terms = {
        "CompiledObserver",
        "RasterObserver",
        "PeakVoltageObserver",
        'observations["activation"]',
        "observer peak equals",
    }

    assert all(term not in text for term in forbidden_terms)
    assert "observations[\"vm_raster\"]" in text
    assert "VmRasterResult" in text
    assert "PeakVoltage" in text
    assert "post-hoc on recorded Vm" in text


def test_public_stimulation_surface_avoids_factorized_runtime_terms():
    import axonscope.stimulation as stimulation

    public_names = set(axs.__all__) | set(stimulation.__all__)
    forbidden_fragments = {
        "Factorized",
        "Vstim",
        "VextBatch",
        "DenseVext",
    }

    leaked = sorted(
        name
        for name in public_names
        if any(fragment in name for fragment in forbidden_fragments)
    )
    assert leaked == []

    public_texts = [
        SRC_ROOT / "stimulation" / "__init__.py",
        SRC_ROOT / "stimulation" / "extracellular.py",
        SRC_ROOT / "preparation" / "signatures.py",
        REPO_ROOT / "docs" / "stimulation.md",
    ]
    forbidden_phrases = {
        "factorized API",
        "factorized Phase",
        "factorized extracellular drive",
        "factorized extracellular contribution",
    }
    for path in public_texts:
        text = path.read_text(encoding="utf-8")
        assert all(phrase not in text for phrase in forbidden_phrases)


def test_public_signatures_do_not_reintroduce_old_unit_suffix_arguments():
    checks = {
        axs.AxonSimulation: {"x_offset", "x_offset_um", "y", "y_um", "z", "z_um"},
        axs.AxonPopulation: {"x_offset", "x_offset_um", "y", "y_um", "z", "z_um"},
        axs.simulate: {"duration_ms", "dt_ms", "tsim"},
        axs.simulate_pool: {"duration_ms", "dt_ms", "tsim"},
        axs.AxonInstance: {"x_offset", "x_offset_um", "y", "y_um", "z", "z_um"},
        axs.Recording: {"positions_um", "sample_dt_ms"},
        axs.analytical.PointSourceElectrode: {
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
    assert not hasattr(axs.AxonInstance, "set_position")


def test_public_examples_and_docs_do_not_place_axon_instances_in_world_space():
    forbidden_instance_kwargs = {"x_offset", "x_offset_um", "y", "y_um", "z", "z_um"}
    offenders: list[str] = []

    for root in (
        REPO_ROOT / "examples",
        REPO_ROOT / "benchmark" / "hotpaths",
        REPO_ROOT / "benchmark" / "nrv_performance",
        REPO_ROOT / "benchmark" / "pseudo_double",
        REPO_ROOT / "benchmark" / "runtime",
        REPO_ROOT / "benchmark" / "solvers",
    ):
        for path in _python_sources(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                if name == "AxonInstance":
                    bad_kwargs = sorted(
                        kw.arg
                        for kw in node.keywords
                        if kw.arg in forbidden_instance_kwargs
                    )
                    if bad_kwargs:
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} "
                            f"AxonInstance kwargs {bad_kwargs}"
                        )
                elif name == "set_position":
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} set_position"
                    )

    markdown_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "GUIDELINES.md",
        *(REPO_ROOT / "docs").glob("*.md"),
        *(REPO_ROOT / "benchmark" / "notebooks").glob("*.ipynb"),
    ]
    placement_call = re.compile(
        r"AxonInstance\s*\([^)]*\b(?:x_offset|x_offset_um|y|y_um|z|z_um)\s*=",
        re.DOTALL,
    )
    for path in markdown_paths:
        text = path.read_text(encoding="utf-8")
        if "set_position(" in text:
            offenders.append(f"{path.relative_to(REPO_ROOT)} set_position")
        if placement_call.search(text):
            offenders.append(f"{path.relative_to(REPO_ROOT)} AxonInstance placement kwargs")

    assert offenders == []


def test_root_axon_simulation_is_not_the_legacy_instance_alias():
    assert axs.AxonSimulation is not axs.AxonInstance
    assert not hasattr(axs.AxonInstance, "set_position")
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
    assert "positions" not in _public_parameters(axs.analysis.ActivationCriterion)
    assert "indices" not in _public_parameters(axs.analysis.ActivationCriterion)
    assert "target" in _public_parameters(axs.analysis.ActivationCriterion)

    criterion = axs.analysis.ActivationCriterion(target=axs.positions.DISTAL)
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
    assert axs.ExtracellularStimulationContext is axs.stimulation.ExtracellularStimulationContext
    assert axs.ExtracellularPotential is axs.stimulation.ExtracellularPotential
    assert not hasattr(axs, "PointSourceElectrode")
    assert not hasattr(axs.stimulation, "PointSourceElectrode")
    assert "PointSourceElectrode" not in axs.__all__
    assert "PointSourceElectrode" not in axs.stimulation.__all__
    assert axs.analytical.PointSourceElectrode is not None


def test_pool_results_use_canonical_result_model_not_lists():
    hints = get_type_hints(axs.simulate_pool)

    assert hints["return"] is axs.AxonSimulationResult
    assert "list[SimResult]" not in (SRC_ROOT / "simulation.py").read_text(encoding="utf-8")
    assert "AxonSimulationResult" in axs.__all__
    assert "AxonResultView" in axs.__all__
    assert "CohortResult" not in axs.__all__
    assert "CohortResult" not in axs.results.__all__


def test_examples_and_public_docs_teach_one_result_path():
    public_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "GUIDELINES.md",
        REPO_ROOT / "docs" / "results_recording_analysis.md",
        REPO_ROOT / "examples" / "README.md",
        *sorted((REPO_ROOT / "examples" / "basic").rglob("*.py")),
        *sorted((REPO_ROOT / "examples" / "advanced").rglob("*.py")),
        *sorted((REPO_ROOT / "examples" / "with_nrv").rglob("*.py")),
    ]
    forbidden = {
        "SimResult",
        "to_sim_result",
        "results.cohorts",
        ".cohorts",
        "axs.CohortResult",
    }

    offenders: dict[str, list[str]] = {}
    for path in public_paths:
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        hits = sorted(term for term in forbidden if term in text)
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits

    assert offenders == {}


def test_benchmark_helpers_do_not_use_removed_public_result_paths():
    benchmark_paths = sorted((REPO_ROOT / "benchmark" / "pseudo_double").rglob("*.py"))
    forbidden = {
        "to_sim_result",
        "results.cohorts",
        ".cohorts",
        "axs.CohortResult",
        "from axonscope.results import AxonSimulationResult, CohortResult",
        "from axonscope.results import CohortResult",
    }

    offenders: dict[str, list[str]] = {}
    for path in benchmark_paths:
        text = path.read_text(encoding="utf-8")
        hits = sorted(term for term in forbidden if term in text)
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits

    assert offenders == {}


def test_signals_are_extensible_descriptors_not_closed_enums():
    assert not issubclass(axs.Signal, Enum)
    assert isinstance(axs.signals.MEMBRANE_VOLTAGE, axs.Signal)
    assert isinstance(axs.signals.MEMBRANE_VOLTAGE.id, axs.SignalId)
    assert axs.signals.Vm is axs.signals.MEMBRANE_VOLTAGE

    removed_aliases = {"VM", "VOLTAGE", "STATES"}
    assert removed_aliases.isdisjoint(set(axs.signals.__all__))
    assert all(not hasattr(axs.signals, name) for name in removed_aliases)
    assert all(not hasattr(axs.Signal, name) for name in removed_aliases)


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


def test_stimulation_package_stays_descriptive_without_jax_runtime_imports():
    assert not (SRC_ROOT / "stimulation" / "runtime.py").exists()
    assert (SRC_ROOT / "backends" / "jax" / "stimulation_runtime.py").is_file()

    offenders: list[str] = []
    for path in _python_sources(SRC_ROOT / "stimulation"):
        offenders.extend(_jax_import_locations(path))

    assert offenders == []


def test_recording_module_does_not_import_solver_options():
    path = SRC_ROOT / "recording.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "axonscope.solvers" or alias.name.startswith(
                    "axonscope.solvers."
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (
                module == "axonscope.solvers" or module.startswith("axonscope.solvers.")
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_protocols_do_not_import_jax_observer_runtime():
    offenders: list[str] = []

    for path in _python_sources(SRC_ROOT / "protocols"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "axonscope.backends.jax.observer_runtime":
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module == "axonscope.backends.jax.observer_runtime":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_vm_raster_result_container_lives_under_results_boundary():
    assert not (SRC_ROOT / "solvers" / "observer_runtime.py").exists()

    backend_text = (SRC_ROOT / "backends" / "jax" / "observer_runtime.py").read_text(
        encoding="utf-8"
    )
    results_text = (SRC_ROOT / "results" / "vm_raster.py").read_text(encoding="utf-8")

    assert "class VmRasterResult" not in backend_text
    assert "def unpack_vm_raster_words" not in backend_text
    assert "class VmRasterResult" in results_text
    assert "def unpack_vm_raster_words" in results_text


def test_public_planning_helpers_do_not_import_jax_numerical_helpers():
    forbidden = "axonscope.backends.jax.common"
    offenders: list[str] = []

    for path in (SRC_ROOT / "performance.py", SRC_ROOT / "inspection.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module == forbidden:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_jax_runtime_modules_live_under_backend_boundary():
    moved_modules = {
        "batch_inputs.py",
        "batch_kernels.py",
        "common.py",
        "experimental.py",
        "kernels.py",
        "observables.py",
        "observer_runtime.py",
        "runtime.py",
    }

    for filename in moved_modules:
        assert not (SRC_ROOT / "solvers" / filename).exists()
        assert (SRC_ROOT / "backends" / "jax" / filename).is_file()

    offenders: list[str] = []
    for path in _python_sources(SRC_ROOT / "solvers"):
        offenders.extend(_jax_import_locations(path))

    assert offenders == []


def test_solver_facade_exposes_only_stable_solver_surface():
    import axonscope.solvers as solver_facade

    stable_exports = {
        "Solver",
        "CrankNicholson",
        "BatchOptions",
        "BatchRecording",
        "DEFAULT_OBSERVER_TIME_CHUNK_STEPS",
        "SolverOptions",
        "resolve_double_cable_block_solver",
    }
    forbidden_exports = {
        "BatchKernelResult",
        "CableRuntime",
        "DoubleCableBatchKernel",
        "DoubleCableKernel",
        "ExtracellularRuntime",
        "KernelResult",
        "MembraneRuntime",
        "SimulationGrid",
        "SingleCableKernel",
        "SingleCableVStimBatchKernel",
        "SolverAxon",
        "SolverRuntime",
        "StimulationRuntime",
        "build_icm_backend_from_axon",
        "build_solver_axon",
        "compile_axon_membrane",
        "compile_membrane_model",
        "precompute_extracellular_potential_mV",
        "precompute_intracellular_current_density",
        "prepare_solver_runtime",
    }

    assert set(solver_facade.__all__) == stable_exports
    assert forbidden_exports.isdisjoint(set(solver_facade.__all__))
    assert forbidden_exports.isdisjoint(set(vars(solver_facade)))

    text = (SRC_ROOT / "solvers" / "__init__.py").read_text(encoding="utf-8")
    assert "axonscope.backends.jax.batch_kernels" not in text
    assert "axonscope.backends.jax.kernels" not in text
    assert "axonscope.backends.jax.runtime" not in text


def test_dispatcher_execution_does_not_import_concrete_jax_batch_kernels():
    path = SRC_ROOT / "dispatcher" / "execution.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_modules = {
        "axonscope.backends.jax.batch_kernels",
        "axonscope.backends.jax.runtime",
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


def test_preparation_runtime_batches_remains_host_side_only():
    assert not (SRC_ROOT / "dispatcher" / "runtime_batches.py").exists()
    path = SRC_ROOT / "preparation" / "runtime_batches.py"

    assert _jax_import_locations(path) == []


def test_public_simulation_orchestrator_uses_backend_execution_boundary():
    path = SRC_ROOT / "simulation.py"
    text = path.read_text(encoding="utf-8")

    assert _jax_import_locations(path) == []
    assert "axonscope.backends.jax" not in text
    assert "axonscope.backends.execution" in text


def test_crank_nicholson_facade_delegates_to_backend_boundary():
    path = SRC_ROOT / "solvers" / "crank_nicholson.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_modules = {
        "axonscope.solvers.axon_runtime",
        "axonscope.backends.jax.kernels",
        "axonscope.backends.jax.runtime",
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


def test_active_double_cable_solver_surface_excludes_archived_candidates():
    from axonscope.solvers.options import DoubleCableBlockSolver
    from benchmark.solvers.bench_double_cable_linear_solvers import (
        BENCHMARK_ONLY_SOLVER_RESOLUTIONS,
        SOLVER_CHOICES,
    )

    retained_public = {"auto", "thomas", "pcr", "pcr_soa", "pcr_adaptive"}
    archived = {
        "assoc_backward",
        "assoc_transfer_dense",
        "pallas_pcr_128",
        "pallas_thomas_4",
        "pallas_thomas_8",
        "pallas_thomas_16",
        "pallas_thomas_128",
        "split_jacobi_4",
        "split_jacobi_8",
        "split_jacobi4_gs1",
        "split_gs_2",
        "split_gs_3",
        "split_gs_4",
        "split_gs_8",
        "split_richardson_4",
    }

    assert set(get_args(DoubleCableBlockSolver)) == retained_public
    assert archived.isdisjoint(set(SOLVER_CHOICES))
    assert archived.isdisjoint(set(BENCHMARK_ONLY_SOLVER_RESOLUTIONS))

    common_text = (SRC_ROOT / "backends" / "jax" / "common.py").read_text(
        encoding="utf-8"
    )
    archived_common_functions = {
        "solve_block_tridiagonal_2x2_assoc_backward_batched",
        "solve_block_tridiagonal_2x2_assoc_transfer_dense_batched",
        "split_double_cable_block_system_soa",
        "solve_tridiagonal_batched",
        "split_initial_guess",
        "solve_double_cable_split_jacobi_batched",
        "solve_double_cable_split_gauss_seidel_batched",
        "solve_double_cable_split_jacobi_then_gauss_seidel_batched",
        "solve_double_cable_split_richardson_batched",
    }
    assert all(f"def {name}" not in common_text for name in archived_common_functions)


def test_factorized_vext_route_has_dense_equivalence_tests():
    text = (REPO_ROOT / "tests" / "unit" / "solvers" / "test_batch.py").read_text(
        encoding="utf-8"
    )

    required_tests = {
        "test_factorized_footprint_batch_matches_dense_builder_and_observer_raster",
        "test_factorized_footprint_batch_supports_row_specific_currents",
        "test_double_cable_factorized_footprint_observer_matches_dense_pcr_soa",
        "test_double_cable_factorized_row_specific_current_observer_matches_dense_pcr_soa",
    }

    missing = sorted(name for name in required_tests if f"def {name}" not in text)
    assert missing == []


def test_solver_route_map_documents_retained_runtime_paths():
    text = (REPO_ROOT / "docs" / "solver_organization.md").read_text(encoding="utf-8")

    required_terms = {
        "## Active Solver Route Map",
        "### Scalar Route",
        "### Pool, Planning, And Fallback Route",
        "### Single-Cable Batch Route",
        "### Double-Cable Batch Route",
        "### VmRaster, Dense/Factorized Vext, And Results",
        "run_jax_crank_nicholson",
        "build_dispatch_plan",
        "_run_scalar_group",
        "_run_batch_group",
        "_run_single_cable_batch_group",
        "_run_double_cable_batch_group",
        "build_sparse_intracellular_current_density_batch",
        "build_intracellular_current_density_batch",
        "build_factorized_vstim_midpoint_batch",
        "build_vstim_midpoint_batch",
        "build_vstim_midpoint_and_initial_previous_batch",
        "SingleCableVStimBatchKernel",
        "DoubleCableBatchKernel",
        "build_vm_raster_plan",
        "_dispatch_results_from_batch",
        "DispatchCohortResult",
        "AxonSimulationResult",
    }

    missing = sorted(term for term in required_terms if term not in text)
    assert missing == []

    option_section = text.split("The current exact double-cable block-solver options are:", 1)[
        1
    ].split("Example:", 1)[0]
    archived_options = {
        "assoc_backward",
        "assoc_transfer_dense",
        "pallas_pcr_128",
        "pallas_thomas_4",
        "split_jacobi_4",
        "split_gs_4",
    }
    assert archived_options.isdisjoint(option_section)
