"""Architecture guardrails for pre-release API cleanup.

These tests intentionally encode project-level rules from GUIDELINES.md. They
should fail early when a refactor reintroduces compatibility aliases or hidden
coupling through the public package facade.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from enum import Enum
from collections.abc import Callable, Iterable
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest
import axonscope as axs
from axonscope.axons import Myelinated, Unmyelinated


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "axonscope"


def _python_sources(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _public_parameters(obj: Callable[..., object]) -> set[str]:
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


def _attribute_root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _looks_like_matplotlib_axis(name: str) -> bool:
    return name == "ax" or name == "axes" or name.startswith("ax_") or name.endswith("_ax")


def test_guidelines_is_the_root_project_philosophy_reference():
    assert (REPO_ROOT / "GUIDELINES.md").is_file()
    assert (REPO_ROOT / "AGENTS.md").is_file()
    legacy_agent_guide = REPO_ROOT / ("agent" + ".md")
    assert not legacy_agent_guide.exists()

    for path in (REPO_ROOT / "AGENTS.md", REPO_ROOT / "todo.md"):
        text = path.read_text(encoding="utf-8")
        assert "GUIDELINES.md" in text
        assert "AXONSCOPE_PRODUCT_SOLVER_GUIDELINES.md" not in text


def test_public_facade_does_not_expose_removed_compatibility_aliases():
    removed_names = {
        "estimate_simulation",
        "inspect_simulation",
        "simulate",
        "simulate_pool",
        "visualization",
        "run_batch",
    }

    exposed = {name for name in removed_names if hasattr(axs, name)}

    assert exposed == set()
    assert removed_names.isdisjoint(set(axs.__all__))


def test_protocol_threshold_surface_uses_generic_find_threshold():
    assert hasattr(axs.protocols, "find_threshold")
    assert "find_threshold" in axs.protocols.__all__
    assert not hasattr(axs.protocols, "find_activation_threshold_curve")
    assert "find_activation_threshold_curve" not in axs.protocols.__all__


def test_analysis_namespace_is_real_package_not_results_alias():
    assert hasattr(axs, "analysis")
    assert not hasattr(axs.results, "analysis")
    assert not hasattr(axs.results, "ActivationCriterion")
    assert axs.analysis.Activation is axs.Activation
    assert axs.analysis.ActivationCriterion is not None
    assert "analysis" in axs.__all__


def test_peak_voltage_observer_is_not_public_surface():
    assert not hasattr(axs, "PeakVoltageObserver")
    assert not hasattr(axs.analysis, "PeakVoltageObserver")
    assert "PeakVoltageObserver" not in axs.__all__
    assert "PeakVoltageObserver" not in axs.analysis.__all__
    assert axs.analysis.PeakVoltage is axs.PeakVoltage


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


def test_public_examples_do_not_expose_model_ir_as_user_api():
    offenders: list[str] = []
    forbidden_text = {
        "axonscope.model_ir",
        "ModelIR",
        "Model IR",
        "intermediate representation",
        "Intermediate Representation",
    }

    for path in _python_sources(REPO_ROOT / "examples"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "axonscope.model_ir" or alias.name.startswith(
                        "axonscope.model_ir."
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and (
                    module == "axonscope.model_ir"
                    or module.startswith("axonscope.model_ir.")
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

        for term in forbidden_text:
            if term in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {term!r}")

    assert offenders == []


def test_builtin_membrane_source_files_are_standalone():
    model_files = sorted(
        path
        for path in (SRC_ROOT / "membranes" / "models").glob("*.py")
        if path.name != "__init__.py"
    )

    assert not (SRC_ROOT / "model_ir" / "models").exists()
    assert not (SRC_ROOT / "model_ir" / "builtins.py").exists()
    assert not (SRC_ROOT / "model_ir" / "units.py").exists()
    assert not (SRC_ROOT / "membranes" / "authoring.py").exists()
    assert not (SRC_ROOT / "membranes" / "models" / "common.py").exists()

    offenders: list[str] = []
    for path in model_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "axonscope.model_ir" or alias.name.startswith(
                        "axonscope.model_ir."
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names = {alias.name for alias in node.names}
                if "*" in imported_names:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
                if node.level > 0:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
                if (
                    node.level == 0
                    and (
                        module == "axonscope.model_ir"
                        or module.startswith("axonscope.model_ir.")
                    )
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_human_membrane_sources_do_not_use_type_checking_placeholders():
    source_root = SRC_ROOT / "membranes" / "models"
    forbidden_terms = {"TYPE_CHECKING", "cast("}
    offenders: list[str] = []

    for path in _python_sources(source_root):
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {term!r}")

    assert offenders == []


def test_human_membrane_sources_do_not_use_manifest_export_or_dynamics_fields():
    source_root = SRC_ROOT / "membranes" / "models"
    forbidden_names = {"exports", "dynamics"}
    offenders: list[str] = []

    for path in _python_sources(source_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for statement in class_node.body:
                target_name: str | None = None
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if isinstance(target, ast.Name):
                        target_name = target.id
                elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    target_name = statement.target.id
                if target_name in forbidden_names:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{statement.lineno} declares {target_name}"
                    )

    assert offenders == []


def test_public_builtin_membranes_are_model_classes():
    from axonscope.membranes.model import MembraneModel

    expected_kinds = {
        "AxNode": "axnode",
        "HodgkinHuxley": "hodgkin_huxley",
        "Passive": "passive",
        "RattayAberham": "rattay_aberham",
        "Schild94": "schild94",
        "Schild97": "schild97",
        "Sundt": "sundt",
        "Tigerholm": "tigerholm",
    }
    assert all(
        (SRC_ROOT / "membranes" / "models" / f"{kind}.py").is_file()
        for kind in expected_kinds.values()
    )

    offenders: list[str] = []
    for name, kind in expected_kinds.items():
        cls = getattr(axs.membranes, name)
        if not issubclass(cls, axs.membranes.Model):
            offenders.append(f"{name} does not inherit axs.membranes.Model")
        if issubclass(cls, MembraneModel):
            offenders.append(f"{name} still exposes internal MembraneModel inheritance")
        if not cls.__module__.startswith("axonscope.membranes.models."):
            offenders.append(f"{name} is still using a builtins.py bridge")
            continue
        if cls.kind_name() != kind:
            offenders.append(f"{name} maps to {cls.kind_name()!r}, expected {kind!r}")

    assert offenders == []


def test_retained_model_families_have_runnable_public_examples():
    examples_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _python_sources(REPO_ROOT / "examples")
    )
    public_axon_families = {
        "HodgkinHuxley",
        "RattayAberham",
        "Sundt",
        "Tigerholm",
        "Schild94",
        "Schild97",
        "MRG",
        "GainesMotor",
        "GainesSensory",
    }
    public_membrane_families = {"Passive", "AxNode"}
    public_nav_isoforms = {f"Nav1{index}" for index in range(1, 10)}

    missing = {
        name
        for name in public_axon_families
        if f"axs.axons.{name}" not in examples_text
    }
    missing.update(
        name
        for name in public_membrane_families | public_nav_isoforms
        if f"axs.membranes.{name}" not in examples_text
    )

    assert missing == set()


def test_builtin_axon_model_kwargs_are_forward_only_without_local_defaults():
    from axonscope.axons import unmyelinated

    expected_forwarded_kwargs = {
        axs.axons.HodgkinHuxley: {"gnabar", "gkbar", "gl", "el", "ena", "ek"},
        axs.axons.RattayAberham: {"gnabar", "gkbar", "gl", "el", "ena", "ek"},
        axs.axons.Sundt: {"gnabar", "gkdrbar", "ena", "ek", "Rm", "El"},
        axs.axons.Tigerholm: {
            "ena",
            "ek",
            "gbar_nav17",
            "gbar_nav18",
            "gbar_nav19",
            "gbar_ks",
            "gbar_kf",
            "gbar_kdr",
            "gbar_h",
            "gbar_kna",
            "nai_fixed",
            "pump_smalla",
            "pump_ko",
        },
    }
    offenders: list[str] = []

    for template, forwarded_kwargs in expected_forwarded_kwargs.items():
        signature = inspect.signature(template)
        for name in forwarded_kwargs:
            parameter = signature.parameters.get(name)
            if parameter is None:
                offenders.append(f"{template.__name__}.{name} missing from public API")
            elif parameter.default is not unmyelinated._UNSET:
                offenders.append(f"{template.__name__}.{name} defines local default")

    assert offenders == []


def test_model_ir_model_family_specific_layer_is_not_reintroduced():
    forbidden_references = (
        "model_ir" + ".models",
        "from " + ".models",
        "hodgkin_huxley",
        "rattay_aberham",
        "schild94",
        "schild97",
        "tigerholm",
        "axnode",
        "na_hh",
        "borg_kdr",
    )
    offenders: list[str] = []
    for path in _python_sources(SRC_ROOT / "model_ir"):
        text = path.read_text(encoding="utf-8")
        if any(reference in text for reference in forbidden_references):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_public_membrane_namespace_does_not_expose_rejected_builder_dsl():
    rejected_names = {
        "MembraneModel",
        "ensure_membrane_model",
        "membrane",
        "parameter",
        "gate",
        "current",
        "observable",
        "model",
        "q10",
    }

    assert rejected_names.isdisjoint(set(axs.membranes.__all__))
    assert all(not hasattr(axs.membranes, name) for name in rejected_names - {"model"})
    model_attribute = getattr(axs.membranes, "model", None)
    assert model_attribute is None or inspect.ismodule(model_attribute)
    for accepted in ("Model", "currents", "initials", "mechanism", "rates", "section", "state", "step"):
        assert accepted in axs.membranes.__all__
        assert hasattr(axs.membranes, accepted)


def test_plain_python_membrane_sources_do_not_import_compiler_or_backend_internals():
    source_root = SRC_ROOT / "membranes" / "models"
    forbidden_prefixes = {
        "axonscope.runtime",
        "axonscope.model_ir",
    }
    offenders: list[str] = []

    for path in _python_sources(source_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == prefix or alias.name.startswith(f"{prefix}.")
                        for prefix in forbidden_prefixes
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(
                    module == prefix or module.startswith(f"{prefix}.")
                    for prefix in forbidden_prefixes
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


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
        REPO_ROOT / "benchmark" / "curves",
        REPO_ROOT / "benchmark" / "workloads",
        REPO_ROOT / "benchmark" / "analysis",
        REPO_ROOT / "benchmark" / "baselines",
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


def test_public_examples_route_generic_result_plots_through_axonscope_views():
    allowed_axis_plots = {
        ("examples/basic/03_point_source_footprint.py", "ax_activation"),
        ("examples/advanced/axon_models/04_non_uniform_activation_function.py", "ax_activation"),
        ("examples/advanced/axon_models/04_non_uniform_activation_function.py", "ax_spacing"),
        ("examples/advanced/runtime/03_pipeline_inspection.py", "output_ax"),
    }
    forbidden_image_primitives = {"broken_barh", "imshow", "pcolormesh"}
    offenders: list[str] = []

    for path in _python_sources(REPO_ROOT / "examples"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            root_name = _attribute_root_name(node.func.value)
            if name in forbidden_image_primitives:
                offenders.append(f"{rel}:{node.lineno} {name}()")
            elif (
                name == "plot"
                and root_name is not None
                and _looks_like_matplotlib_axis(root_name)
                and (rel, root_name) not in allowed_axis_plots
            ):
                offenders.append(f"{rel}:{node.lineno} {root_name}.plot()")

    assert offenders == []


def test_public_examples_do_not_reach_into_view_modules():
    forbidden_fragments = (
        ".results.views",
        ".analysis.views",
        ".protocols.views",
        "from axonscope.results.views",
        "from axonscope.analysis.views",
        "from axonscope.protocols.views",
        "from axonscope.plotting",
        "import axonscope.plotting",
    )
    offenders: list[str] = []

    for path in _python_sources(REPO_ROOT / "examples"):
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                offenders.append(f"{rel} contains {fragment}")

    assert offenders == []


def test_public_examples_do_not_use_benchmark_or_profiling_apis():
    forbidden_fragments = (
        "axs.benchmark(",
        "axs.enable_benchmark(",
        "axs.disable_benchmark(",
        "axs.reset_benchmark(",
        "from axonscope.benchmarking",
        "import axonscope.benchmarking",
    )
    offenders: list[str] = []

    for path in _python_sources(REPO_ROOT / "examples"):
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                offenders.append(f"{rel} contains {fragment}")

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
    attachment_names = {
        "ExtracellularFootprint",
        "ExtracellularDrive",
        "ExtracellularStimulation",
    }
    diagnostic_names = {"ExtracellularPotential"}
    expected_public_names = attachment_names | diagnostic_names
    root_names = {name for name in axs.__all__ if name.startswith("Extracellular")}
    assert root_names == expected_public_names
    assert {
        name for name in axs.stimulation.__all__ if name.startswith("Extracellular")
    } == expected_public_names

    assert axs.AxonId is axs.identifiers.AxonId
    assert axs.DriveId is axs.identifiers.DriveId
    assert axs.ExtracellularFootprint is axs.stimulation.ExtracellularFootprint
    assert axs.ExtracellularDrive is axs.stimulation.ExtracellularDrive
    assert axs.ExtracellularStimulation is axs.stimulation.ExtracellularStimulation
    assert axs.ExtracellularPotential is axs.stimulation.ExtracellularPotential
    for legacy_name in (
        "Electrode",
        "AnalyticalElectrode",
        "ExtracellularContext",
        "AnalyticalExtracellularContext",
        "ExtracellularStimulationContext",
        "NRVExtracellularContext",
    ):
        assert not hasattr(axs, legacy_name)
        assert not hasattr(axs.stimulation, legacy_name)
        assert legacy_name not in axs.__all__
        assert legacy_name not in axs.stimulation.__all__
    assert not hasattr(axs, "PointSourceElectrode")
    assert not hasattr(axs.stimulation, "PointSourceElectrode")
    assert "PointSourceElectrode" not in axs.__all__
    assert "PointSourceElectrode" not in axs.stimulation.__all__
    assert axs.analytical.PointSourceElectrode is not None


def test_axon_simulation_results_use_canonical_result_model_not_lists():
    hints = get_type_hints(axs.AxonSimulation.run)

    assert hints["return"] is axs.AxonSimulationResult
    assert "list[SimResult]" not in (SRC_ROOT / "simulation.py").read_text(encoding="utf-8")
    assert "AxonSimulationResult" in axs.__all__
    assert "AxonResultView" in axs.__all__
    assert "CohortResult" not in axs.__all__
    assert "CohortResult" not in axs.results.__all__


def test_axon_simulation_uses_one_population_lifecycle():
    text = (SRC_ROOT / "simulation.py").read_text(encoding="utf-8")
    forbidden_fragments = {
        "_run_single_simulation",
        "_population_lifecycle",
        "_single_result_to_public",
        "batch_options are only valid for multi-axon",
        "progress is only valid for multi-axon",
    }

    assert forbidden_fragments.isdisjoint(text)
    assert "run_pool(" in text


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


def test_legacy_results_visualization_module_is_removed():
    assert not (SRC_ROOT / "results" / "visualization.py").exists()

    forbidden_public_names = {"visualization", "plot_raster", "rasterplot"}
    assert forbidden_public_names.isdisjoint(set(axs.results.__all__))
    assert all(not hasattr(axs.results, name) for name in forbidden_public_names)

    offenders: list[str] = []

    for path in _python_sources(SRC_ROOT):
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


def test_inspection_records_are_separate_from_user_facing_views():
    records = SRC_ROOT / "inspection_records.py"
    views = SRC_ROOT / "inspection_views.py"
    builder = SRC_ROOT / "inspection.py"

    assert records.is_file()
    assert views.is_file()

    records_text = records.read_text(encoding="utf-8")
    views_text = views.read_text(encoding="utf-8")
    builder_text = builder.read_text(encoding="utf-8")

    assert "class SimulationInspection" in records_text
    assert "format_simulation_inspection" in views_text
    assert "from axonscope.inspection_records import" in builder_text
    assert "import matplotlib" not in records_text
    assert "from rich" not in records_text
    assert "import matplotlib" not in builder_text
    assert "from rich" not in builder_text
    assert "@dataclass" not in builder_text


def test_performance_estimate_records_are_separate_from_user_facing_views():
    records = SRC_ROOT / "performance.py"
    views = SRC_ROOT / "performance_views.py"

    assert records.is_file()
    assert views.is_file()

    records_text = records.read_text(encoding="utf-8")
    views_text = views.read_text(encoding="utf-8")

    assert "class SimulationEstimate" in records_text
    assert "format_simulation_estimate" in views_text
    assert "print_simulation_estimate" in views_text
    assert "from axonscope.performance_views import format_simulation_estimate" in records_text
    assert "from axonscope.performance_views import print_simulation_estimate" in records_text
    assert "from rich" not in records_text
    assert "import matplotlib" not in records_text
    assert "from rich" in views_text


def test_summary_like_objects_share_rows_dataframe_text_surface():
    summary_classes = (
        axs.AnalysisResult,
        axs.analysis.AnalysisReport,
        axs.results.VmRasterResult,
        axs.protocols.ThresholdSearchResult,
        axs.protocols.RecruitmentCurve,
        axs.protocols.PoolSweepResult,
        axs.protocols.ThresholdCurve,
        axs.SimulationEstimate,
    )
    required = {"rows", "to_dataframe", "format", "print"}

    missing: dict[str, set[str]] = {}
    for cls in summary_classes:
        absent = {name for name in required if not hasattr(cls, name)}
        if absent:
            missing[cls.__name__] = absent

    assert missing == {}
    assert "rows" not in {field.name for field in fields(axs.protocols.ThresholdCurve)}
    assert "row_labels" in {field.name for field in fields(axs.protocols.ThresholdCurve)}


def test_public_view_modules_use_shared_plotting_helpers():
    view_paths = (
        SRC_ROOT / "results" / "views.py",
        SRC_ROOT / "analysis" / "views.py",
        SRC_ROOT / "protocols" / "views.py",
    )

    offenders: list[str] = []
    for path in view_paths:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT)
        if "from axonscope.plotting import" not in text:
            offenders.append(f"{rel} missing shared plotting import")
        for forbidden in ("import matplotlib.pyplot", "plt.subplots"):
            if forbidden in text:
                offenders.append(f"{rel} contains {forbidden}")

    plotting_text = (SRC_ROOT / "plotting.py").read_text(encoding="utf-8")
    assert "def ensure_axis" in plotting_text
    assert "def decorate_axis" in plotting_text
    assert offenders == []


def test_dispatch_progress_uses_structured_dispatch_and_backend_events():
    progress_text = (SRC_ROOT / "dispatcher" / "progress.py").read_text(encoding="utf-8")
    execution_text = (SRC_ROOT / "dispatcher" / "execution.py").read_text(encoding="utf-8")
    jax_runner_text = (SRC_ROOT / "runtime" / "jax" / "group_runner.py").read_text(
        encoding="utf-8"
    )

    assert "class ProgressEvent" in progress_text
    assert "def emit(self, event: ProgressEvent)" in progress_text
    assert "progress_reporter.start_group(" in execution_text
    assert "progress_reporter.route_group(" in execution_text
    assert "progress_reporter.finish_group(" in execution_text
    assert "from axonscope.dispatcher.progress import ProgressEvent" in jax_runner_text
    assert "_emit_progress(" in jax_runner_text
    assert "Simulation run {status}" in progress_text
    assert "Dispatch completed" not in progress_text


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
    assert not (SRC_ROOT / "runtime" / "jax" / "stimulation.py").exists()
    assert (SRC_ROOT / "runtime" / "jax" / "inputs" / "stimulus.py").is_file()

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
                    if alias.name == "axonscope.runtime.jax.recording.observer":
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module == "axonscope.runtime.jax.recording.observer":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_protocol_observer_path_uses_shared_vm_raster_decoders():
    path = SRC_ROOT / "protocols" / "observer_path.py"
    text = path.read_text(encoding="utf-8")

    forbidden_terms = {
        ".unpack(",
        "probe_mask",
        "getattr(raster",
        "raster.nt",
        "raster.dt_ms",
    }

    assert "activation_values_from_vm_raster" in text
    offenders = sorted(term for term in forbidden_terms if term in text)
    assert offenders == []


def test_vm_raster_result_container_lives_under_results_boundary():
    assert not (SRC_ROOT / "solvers" / "observer_runtime.py").exists()

    backend_text = (
        SRC_ROOT / "runtime" / "jax" / "recording" / "observer.py"
    ).read_text(encoding="utf-8")
    results_text = (SRC_ROOT / "results" / "vm_raster.py").read_text(encoding="utf-8")

    assert "class VmRasterResult" not in backend_text
    assert "def unpack_vm_raster_words" not in backend_text
    assert "class VmRasterResult" in results_text
    assert "def unpack_vm_raster_words" in results_text


def test_public_estimate_and_inspection_route_backend_details_through_facade():
    forbidden_prefix = "axonscope.runtime.jax"
    offenders: list[str] = []

    for path in (SRC_ROOT / "performance.py", SRC_ROOT / "inspection.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden_prefix or alias.name.startswith(
                        f"{forbidden_prefix}."
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and (
                    module == forbidden_prefix
                    or module.startswith(f"{forbidden_prefix}.")
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        assert "axonscope.runtime.execution" in text

    backend_benchmark = (
        SRC_ROOT / "runtime" / "jax" / "benchmarking" / "profile.py"
    )
    backend_text = backend_benchmark.read_text(encoding="utf-8")

    assert offenders == []
    assert "benchmark_plan_input_lowering" in backend_text
    assert "benchmark_lower_recording_options" not in backend_text
    assert "benchmark_membrane_output_names" in backend_text
    assert "benchmark_observer_output_label" not in backend_text
    assert "benchmark_observers_are_vm_raster_compatible" not in backend_text
    assert "benchmark_vm_raster_definitions" not in backend_text


def test_jax_runtime_modules_live_under_backend_boundary():
    moved_modules = {
        "types.py",
    }

    for filename in moved_modules:
        assert not (SRC_ROOT / "solvers" / filename).exists()
        assert (SRC_ROOT / "runtime" / "jax" / filename).is_file()
    assert not (SRC_ROOT / "runtime" / "jax" / "batch_kernels.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "batch_inputs.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "batch_results.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "benchmark.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "benchmark_metadata.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "execution_policy.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "kernel_results.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "observer_runtime.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "input_batches.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "input_lowering.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "recording_lowering.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "runtime_caches.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "runtime_preparation.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "shape_bucketing.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "stimulation_runtime.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "kernels.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "policy.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "solver_core.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "policy" / "solver_engines").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "jax_triton_double_cable.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "experimental.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "observables.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "membrane_backend.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "membrane_layout.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "membrane_program.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "membrane_stacking.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "model_ir_lowering.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "rate_tables.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "scalar_runner.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "reference_solvers.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "large_population_solver.py").exists()
    assert not (SRC_ROOT / "solvers" / "base.py").exists()
    assert not (SRC_ROOT / "solvers" / "crank_nicholson.py").exists()
    assert not (SRC_ROOT / "solvers" / "rate_tables.py").exists()
    assert not (SRC_ROOT / "solvers" / "_outputs.py").exists()
    assert not (SRC_ROOT / "solvers" / "axon_runtime.py").exists()
    assert (SRC_ROOT / "runtime" / "solver_axon.py").is_file()

    jax_membrane_dir = SRC_ROOT / "runtime" / "jax" / "membranes"
    jax_kernels_dir = SRC_ROOT / "runtime" / "jax" / "kernels"
    jax_policy_dir = SRC_ROOT / "runtime" / "jax" / "policy"
    jax_inputs_dir = SRC_ROOT / "runtime" / "jax" / "inputs"
    jax_benchmarking_dir = SRC_ROOT / "runtime" / "jax" / "benchmarking"
    jax_preparation_dir = SRC_ROOT / "runtime" / "jax" / "preparation"
    jax_recording_dir = SRC_ROOT / "runtime" / "jax" / "recording"
    for filename in {
        "__init__.py",
        "backend.py",
        "compile.py",
        "layout.py",
        "model_ir_lowering.py",
        "program.py",
        "stacking.py",
    }:
        assert (jax_membrane_dir / filename).is_file()
    for filename in {
        "__init__.py",
        "block_tridiagonal.py",
        "chunking.py",
        "double_cable.py",
        "double_cable_cpu.py",
        "double_cable_gpu.py",
        "double_cable_linear.py",
        "double_cable_step.py",
        "factorized.py",
        "inputs.py",
        "single_cable.py",
        "single_cable_scans.py",
        "triton_double_cable.py",
    }:
        assert (jax_kernels_dir / filename).is_file()
    assert (SRC_ROOT / "runtime" / "jax" / "cable_geometry.py").is_file()
    for filename in {
        "__init__.py",
        "engine.py",
        "engine_common.py",
        "engine_cpu.py",
        "engine_gpu.py",
        "engine_types.py",
        "execution.py",
    }:
        assert (jax_policy_dir / filename).is_file()
    for filename in {
        "__init__.py",
        "extracellular.py",
        "intracellular.py",
        "lowering.py",
        "payloads.py",
        "stimulus.py",
    }:
        assert (jax_inputs_dir / filename).is_file()
    for filename in {"__init__.py", "metadata.py", "profile.py"}:
        assert (jax_benchmarking_dir / filename).is_file()
    for filename in {
        "__init__.py",
        "base.py",
        "caches.py",
        "runtime.py",
        "shape_bucketing.py",
        "stacking.py",
    }:
        assert (jax_preparation_dir / filename).is_file()
    for filename in {"__init__.py", "lowering.py", "observer.py", "results.py"}:
        assert (jax_recording_dir / filename).is_file()
    assert not (jax_kernels_dir / "common.py").exists()
    assert not (jax_kernels_dir / "core.py").exists()
    assert not (jax_kernels_dir / "cable_geometry.py").exists()
    assert not (jax_kernels_dir / "results.py").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "execution").exists()
    assert not (SRC_ROOT / "runtime" / "jax" / "runtime.py").exists()

    jax_runtime_text = (
        SRC_ROOT / "runtime" / "jax" / "preparation" / "base.py"
    ).read_text(
        encoding="utf-8"
    )
    jax_kernels_text = "\n".join(
        path.read_text(encoding="utf-8") for path in jax_kernels_dir.glob("*.py")
    )
    jax_membrane_program_text = (jax_membrane_dir / "program.py").read_text(
        encoding="utf-8"
    )
    jax_membrane_backend_text = (jax_membrane_dir / "backend.py").read_text(
        encoding="utf-8"
    )
    jax_model_ir_lowering_text = (
        jax_membrane_dir / "model_ir_lowering.py"
    ).read_text(encoding="utf-8")
    assert "def membrane_observable_names" in jax_runtime_text
    assert "@dataclass" not in jax_runtime_text
    assert ".observables" not in jax_runtime_text
    assert ".observables" not in jax_kernels_text
    assert "def precompute_intracellular_current_density" not in jax_runtime_text
    assert "def gating_inf_tau" not in jax_membrane_program_text
    assert "def disable_rate_table" not in jax_membrane_program_text
    assert "rate_table" not in jax_membrane_program_text
    assert "def init_gates_for_row" not in jax_membrane_backend_text
    assert "source_observable_output_names" not in jax_model_ir_lowering_text

    offenders: list[str] = []
    for path in _python_sources(SRC_ROOT / "solvers"):
        offenders.extend(_jax_import_locations(path))

    assert offenders == []


def test_rejected_double_cable_solver_candidates_stay_out_of_jax_runtime_core():
    jax_kernels_dir = SRC_ROOT / "runtime" / "jax" / "kernels"
    kernel_text = "\n".join(
        path.read_text(encoding="utf-8") for path in jax_kernels_dir.glob("*.py")
    )
    benchmark_text = (
        REPO_ROOT
        / "benchmark"
        / "legacy"
        / "p11_solver_exploration"
        / "double_cable_solver_candidates.py"
    ).read_text(encoding="utf-8")

    benchmark_only = {
        "solve_block_tridiagonal_2x2_pcr_soa_batched_ref",
        "solve_block_tridiagonal_2x2_pcr_soa_batched_nomask",
        "solve_block_tridiagonal_2x2_pcr_soa_batched_shift",
        "solve_block_tridiagonal_2x2_pcr_soa_batched_transposed",
        "solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched",
        "solve_block_tridiagonal_2x2_pcr_soa_batched_padded",
        "double_cable_block_residual_norm",
        "double_cable_power_bucket",
        "pad_double_cable_system_to_power_bucket",
    }

    for name in benchmark_only:
        assert name not in kernel_text
        assert name in benchmark_text


def test_jax_runtime_does_not_compile_stateful_legacy_composites():
    text = (
        SRC_ROOT / "runtime" / "jax" / "preparation" / "base.py"
    ).read_text(encoding="utf-8")

    forbidden = {
        "Schild94CompositeICM",
        "Schild97CompositeICM",
        "TigerholmCompositeICM",
        "axonscope.channel_models.composite_models",
    }

    assert forbidden.isdisjoint(text.split())
    for token in forbidden:
        assert token not in text


def test_jax_runtime_uses_generated_cache_then_one_compiler_fallback():
    text = (
        SRC_ROOT / "runtime" / "jax" / "membranes" / "compile.py"
    ).read_text(encoding="utf-8")
    forbidden = {
        'model.kind == "axnode"',
        'model.kind == "hodgkin_huxley"',
        'model.kind == "passive"',
        'model.kind == "rattay_aberham"',
        'model.kind == "schild94"',
        'model.kind == "schild97"',
        'model.kind == "sundt"',
        'model.kind == "tigerholm"',
    }

    assert "load_generated_source_runtime(" in text
    assert "JaxMembraneProgram.from_generated_module(" in text
    assert "lower_membrane_model_with_sources(" in text
    assert 'load_generated_modules=("jax", "numpy")' in text
    assert sorted(term for term in forbidden if term in text) == []


def test_jax_solver_runtime_has_no_model_family_specific_fast_paths():
    text = (
        SRC_ROOT / "runtime" / "jax" / "preparation" / "base.py"
    ).read_text(encoding="utf-8")
    forbidden = {
        "AxNode",
        "HodgkinHuxley",
        "Passive",
        "Rattay",
        "Schild",
        "Sundt",
        "Tigerholm",
        "_rattay",
        "rattay_aberham",
        "is_model_ir_membrane_kind",
    }

    assert sorted(term for term in forbidden if term in text) == []


def test_jax_runtime_preparation_stacks_membranes_by_capabilities_not_model_names():
    text = (
        SRC_ROOT / "runtime" / "jax" / "membranes" / "stacking.py"
    ).read_text(
        encoding="utf-8"
    )
    forbidden = {
        "AxNodePassiveFamilyMembraneBackend",
        "axnode_initial_gates_numpy",
        "try_stack_axnode_passive_family_membrane",
        "_is_axnode",
        "is_model_ir_membrane_kind",
        'model.kind == "axnode"',
        'model.kind == "passive"',
        "node_model",
        "node_count",
    }

    assert "GatedLeakStackMembraneBackend" in text
    assert "supports_stateless_vm_only_fast_path" in text
    assert sorted(term for term in forbidden if term in text) == []


def test_double_cable_batch_membrane_specialization_is_capability_based():
    backend_text = (
        SRC_ROOT / "runtime" / "jax" / "membranes" / "backend.py"
    ).read_text(encoding="utf-8")
    step_text = (
        SRC_ROOT / "runtime" / "jax" / "kernels" / "double_cable_step.py"
    ).read_text(encoding="utf-8")
    specialized_text = backend_text + step_text

    assert "batch_cn_gate_update" in specialized_text
    assert "batch_membrane_conductance_terms" in specialized_text
    for model_name in ("MRG", "AxNode", "HodgkinHuxley", "Passive"):
        assert model_name not in step_text


def test_jax_runtime_preparation_does_not_own_membrane_stacking_details():
    preparation_text = (
        SRC_ROOT / "runtime" / "jax" / "preparation" / "stacking.py"
    ).read_text(encoding="utf-8")
    stacking_text = (
        SRC_ROOT / "runtime" / "jax" / "membranes" / "stacking.py"
    ).read_text(encoding="utf-8")

    forbidden_preparation_terms = {
        "class GatedLeakMembraneStack",
        "class _GatedLeakMember",
        "class _EncodedGatedLeakRow",
        "def try_stack_gated_leak_membrane_from_group",
        "def _gated_leak_member",
        "def _encode_gated_leak_group_row",
        "def _encode_gated_leak_runtime_members",
        "def _try_stack_gated_leak_membrane",
        "GatedLeakStackMembraneBackend",
        "HeterogeneousMembraneBackend",
        "membrane_backend_model",
    }
    required_stacking_terms = {
        "class GatedLeakMembraneStack",
        "def try_stack_gated_leak_membrane_from_group",
    }
    forbidden_stacking_terms = {
        "def try_stack_gated_leak_membrane_from_runtime_rows",
        "def _encode_gated_leak_runtime_members",
    }

    assert (
        sorted(term for term in forbidden_preparation_terms if term in preparation_text)
        == []
    )
    assert sorted(term for term in required_stacking_terms if term not in stacking_text) == []
    assert sorted(term for term in forbidden_stacking_terms if term in stacking_text) == []


def test_jax_runtime_no_longer_has_model_ir_membrane_adapter():
    adapter_path = SRC_ROOT / "runtime" / "jax" / "model_ir_membrane.py"
    compile_text = (
        SRC_ROOT / "runtime" / "jax" / "membranes" / "compile.py"
    ).read_text(encoding="utf-8")
    runtime_text = (
        SRC_ROOT / "runtime" / "jax" / "preparation" / "base.py"
    ).read_text(
        encoding="utf-8"
    )
    runtime_modules = {
        "preparation/base.py",
    }
    membrane_modules = {
        "backend.py",
        "compile.py",
        "layout.py",
        "program.py",
    }

    assert not adapter_path.exists()
    assert "JaxMembraneProgram.from_model_ir(" not in compile_text
    assert "JaxMembraneProgram.from_generated_module(" in compile_text
    assert "lowered.model" in compile_text
    assert "missing generated targets" in compile_text
    assert "ModelIRMembrane" not in runtime_text
    for filename in runtime_modules:
        text = (SRC_ROOT / "runtime" / "jax" / filename).read_text(encoding="utf-8")
        assert "CompiledMembrane" not in text
    for filename in membrane_modules:
        text = (
            SRC_ROOT / "runtime" / "jax" / "membranes" / filename
        ).read_text(encoding="utf-8")
        assert "CompiledMembrane" not in text
    backend_text = (
        SRC_ROOT / "runtime" / "jax" / "membranes" / "backend.py"
    ).read_text(encoding="utf-8")
    assert "class Gating" not in backend_text
    assert "GFunc" not in backend_text
    assert "RateFunc" not in backend_text


def test_historical_membrane_packages_are_removed_from_active_source():
    removed_package_paths = {
        SRC_ROOT / "channel_models",
        SRC_ROOT / "icm",
    }
    removed_modules = {
        "axonscope.channel_models",
        "axonscope.icm",
    }

    assert all(not path.exists() for path in removed_package_paths)
    assert all(importlib.util.find_spec(module) is None for module in removed_modules)


def test_membrane_descriptions_do_not_wrap_raw_legacy_channel_models():
    text = (SRC_ROOT / "membranes" / "model.py").read_text(encoding="utf-8")

    assert '"legacy"' not in text
    assert "'legacy'" not in text
    assert "IonChannelModelBase" not in text


def test_human_membrane_sources_do_not_import_model_ir_internals():
    offenders: list[str] = []
    source_root = SRC_ROOT / "membranes" / "models"
    for path in _python_sources(source_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "axonscope.model_ir" or alias.name.startswith(
                        "axonscope.model_ir."
                    ):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "axonscope.model_ir" or module.startswith(
                    "axonscope.model_ir."
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_model_ir_model_modules_are_source_adapters_not_equation_builders():
    offenders: list[str] = []
    forbidden_imports = {
        "axonscope.model_ir.expressions",
        "axonscope.model_ir.intrinsics",
    }
    forbidden_builders = {
        "Current",
        "Diagnostic",
        "Gate",
        "ModelIR",
        "Observable",
        "StateUpdate",
        "StepProgram",
        "literal",
        "symbol",
    }
    source_root = SRC_ROOT / "model_ir" / "models"
    for path in _python_sources(source_root):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_imports:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    name = ""
                if name in forbidden_builders:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{name}")

    assert offenders == []


def test_solver_facade_exposes_only_stable_solver_surface():
    import axonscope.solvers as solver_facade

    stable_exports = {
        "BatchOptions",
        "BatchRecording",
        "DEFAULT_OBSERVER_TIME_CHUNK_STEPS",
        "SolverOptions",
    }
    forbidden_exports = {
        "BatchKernelResult",
        "CableRuntime",
        "DoubleCableBatchKernel",
        "DoubleCableKernel",
        "ExtracellularRuntime",
        "KernelResult",
        "MembraneRuntime",
        "CrankNicholson",
        "RateTableConfig",
        "SimulationGrid",
        "SingleCableKernel",
        "SingleCableVStimBatchKernel",
        "Solver",
        "SolverAxon",
        "SolverRuntime",
        "StimulationRuntime",
        "build_icm_backend_from_axon",
        "build_solver_axon",
        "compile_axon_membrane",
        "compile_membrane_model",
        "precompute_intracellular_current_density",
        "prepare_solver_runtime",
    }

    assert set(solver_facade.__all__) == stable_exports
    assert forbidden_exports.isdisjoint(set(solver_facade.__all__))
    assert forbidden_exports.isdisjoint(set(vars(solver_facade)))

    text = (SRC_ROOT / "solvers" / "__init__.py").read_text(encoding="utf-8")
    assert "axonscope.runtime.jax.kernels.batch" not in text
    assert "axonscope.runtime.jax.kernels" not in text
    assert "axonscope.runtime.jax.runtime" not in text
    assert "axonscope.runtime.jax.preparation.base" not in text


def test_dispatcher_execution_does_not_import_concrete_jax_backend():
    path = SRC_ROOT / "dispatcher" / "execution.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_modules = {
        "axonscope.runtime.jax",
        "axonscope.runtime.jax.kernels.batch",
        "axonscope.runtime.jax.group_runner",
        "axonscope.runtime.jax.runtime",
        "axonscope.runtime.jax.preparation.base",
        "axonscope.icm",
        "axonscope.icm.backends",
    }
    offenders = _jax_import_locations(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules or alias.name.startswith(
                    "axonscope.runtime.jax."
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (
                module in forbidden_modules
                or module.startswith("axonscope.runtime.jax.")
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_batch_kernels_have_one_production_orchestration_owner():
    allowed_path = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    kernel_modules = {
        "axonscope.runtime.jax.kernels.single_cable",
        "axonscope.runtime.jax.kernels.double_cable",
    }
    offenders: list[str] = []

    for path in _python_sources(SRC_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in kernel_modules:
                continue
            imported = {alias.name for alias in node.names}
            if imported & {"SingleCableVStimBatchKernel", "DoubleCableBatchKernel"}:
                if path != allowed_path:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_jax_group_runner_exposes_only_enqueue_finalize_execution_route():
    import axonscope.runtime.jax.group_runner as group_runner

    assert set(group_runner.__all__) == {
        "PendingJaxBatchGroup",
        "enqueue_jax_batch_group",
        "finalize_jax_batch_group",
    }
    assert not hasattr(group_runner, "run_jax_batch_group")
    assert not hasattr(group_runner, "_run_single_cable_batch_group")
    assert not hasattr(group_runner, "_run_double_cable_batch_group")


def test_group_runner_routes_input_lowering_through_lowering_module():
    path = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    forbidden_builder_modules = {
        "axonscope.runtime.jax.inputs.extracellular",
        "axonscope.runtime.jax.inputs.intracellular",
    }
    forbidden_builder_calls = {
        "build_factorized_vstim_midpoint_batch",
        "build_intracellular_current_density_batch",
        "build_sparse_intracellular_current_density_batch",
        "build_vstim_midpoint_and_initial_previous_batch",
        "build_vstim_midpoint_batch",
    }
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_builder_modules:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden_builder_modules:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in forbidden_builder_calls:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{call_name}")

    assert "axonscope.runtime.jax.inputs.lowering" in text
    assert offenders == []


def test_group_runner_routes_recording_lowering_through_lowering_module():
    path = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    forbidden_call_names = {
        "build_threshold_observer_plan",
        "_observer_plan_for_cohort",
        "_observers_are_vm_raster_compatible",
        "_vm_raster_definitions",
    }
    forbidden_modules = {
        "axonscope.runtime.jax.recording.observer",
    }
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden_modules:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in forbidden_call_names:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{call_name}")

    assert "axonscope.runtime.jax.recording.lowering" in text
    assert "BatchRecording.full" not in text
    assert offenders == []


def test_group_runner_routes_batch_result_assembly_through_result_module():
    path = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    text = path.read_text(encoding="utf-8")
    forbidden_terms = {
        "DispatchCohortRecord",
        "DispatchRowRecord",
        "SimResult",
        "BatchKernelResult",
        "_dispatch_results_from_batch",
        "_trim_batch_kernel_result",
        "_posthoc_observations_for_row",
    }

    assert "axonscope.runtime.result_assembly" in text
    assert "axonscope.runtime.jax.recording.results" in text
    missing_boundary = [
        term for term in ("dispatch_results_from_batch", "trim_batch_kernel_result")
        if term not in text
    ]
    assert missing_boundary == []
    offenders = sorted(term for term in forbidden_terms if term in text)
    assert offenders == []

    jax_results_text = (
        SRC_ROOT / "runtime" / "jax" / "recording" / "results.py"
    ).read_text(encoding="utf-8")
    generic_result_terms = {
        "DispatchCohortRecord",
        "DispatchRowRecord",
        "SolverOutput",
        "dispatch_results_from_batch",
        "_posthoc_observations_for_row",
    }
    generic_offenders = sorted(
        term for term in generic_result_terms if term in jax_results_text
    )
    assert generic_offenders == []


def test_group_runner_routes_shape_bucketing_through_shape_module():
    path = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    text = path.read_text(encoding="utf-8")
    forbidden_terms = {
        "_DOUBLE_CABLE_SHAPE_BUCKETING_ENV",
        "_DOUBLE_CABLE_BATCH_BUCKETS",
        "_DOUBLE_CABLE_NX_BUCKET_MULTIPLE",
        "_double_cable_kernel_group",
        "_double_cable_shape_bucketing_enabled",
        "_bucket_batch_size",
        "_bucket_nx",
        "_record_kernel_bucket_metadata",
    }

    assert "axonscope.runtime.jax.preparation.shape_bucketing" in text
    assert "double_cable_kernel_group" in text
    assert "record_kernel_bucket_metadata" in text
    offenders = sorted(term for term in forbidden_terms if term in text)
    assert offenders == []


def test_group_runner_routes_runtime_caches_through_cache_module():
    group_runner = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    preparation = SRC_ROOT / "runtime" / "jax" / "preparation" / "runtime.py"
    group_runner_text = group_runner.read_text(encoding="utf-8")
    preparation_text = preparation.read_text(encoding="utf-8")
    forbidden_terms = {
        "OrderedDict",
        "_BATCH_RUNTIME_CACHE",
        "_BATCH_STATIC_RUNTIME_CACHE",
        "_PREPARED_COHORT_CACHE",
        "_GROUP_RUNNER_CACHE_MAX_SIZE",
        "_cache_get",
        "_cache_store",
    }

    assert "axonscope.runtime.jax.preparation.caches" not in group_runner_text
    assert "axonscope.runtime.jax.preparation.caches" in preparation_text
    offenders = sorted(term for term in forbidden_terms if term in group_runner_text)
    assert offenders == []


def test_prepared_cohort_caches_are_runtime_neutral_not_jax_runtime_state():
    group_preparation = SRC_ROOT / "runtime" / "group_preparation.py"
    runtime_caches = SRC_ROOT / "runtime" / "jax" / "preparation" / "caches.py"
    runtime_preparation = SRC_ROOT / "runtime" / "jax" / "preparation" / "runtime.py"

    group_text = group_preparation.read_text(encoding="utf-8")
    caches_text = runtime_caches.read_text(encoding="utf-8")
    preparation_text = runtime_preparation.read_text(encoding="utf-8")

    assert "PreparedCohort" in group_text
    assert "def prepared_cohort_for_group" in group_text
    assert "def prepared_cohort_for_current_group" in group_text
    assert "def representative_item" in group_text
    assert "def group_runtime_signature" in group_text
    assert "def runtime_context_cache_key" in group_text

    forbidden_jax_cache_terms = {
        "PreparedCohort",
        "_PREPARED_COHORT_CACHE",
        "_PREPARED_COHORT_IDENTITY_CACHE",
        "def get_prepared_cohort",
        "def store_prepared_cohort",
        "def clear_prepared_cohort_cache",
    }
    forbidden_jax_preparation_terms = {
        "from axonscope.preparation.cohort import PreparedCohort",
        "extracellular_stimulation_rows",
        "def prepared_cohort_for_group",
        "def prepared_cohort_for_current_group",
        "def _group_runtime_signature",
        "def _runtime_context_cache_key",
    }

    assert sorted(term for term in forbidden_jax_cache_terms if term in caches_text) == []
    assert (
        sorted(term for term in forbidden_jax_preparation_terms if term in preparation_text)
        == []
    )


def test_group_runner_routes_runtime_preparation_through_preparation_module():
    path = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    text = path.read_text(encoding="utf-8")
    forbidden_terms = {
        "CableRuntime",
        "ExtracellularRuntime",
        "MembraneRuntime",
        "compile_membrane_model",
        "prepare_membrane_runtime",
        "prepare_simulation_grid",
        "prepare_solver_runtime",
        "prepare_stimulation_runtime",
        "HeterogeneousMembraneBackend",
        "RowIndexedMembraneBackend",
        "GatedLeakStackMembraneBackend",
        "_prepare_batch_runtime",
        "_prepared_cohort_for_group",
        "_stack_cable_runtime",
        "_stack_membrane_runtime",
        "_stack_extracellular_runtime",
        "_try_stack_gated_leak_membrane",
        "try_stack_gated_leak_membrane_from_group",
        "_group_cm_uF_cm2",
    }

    assert "axonscope.runtime.jax.preparation.runtime" in text
    assert "axonscope.runtime.jax.preparation.stacking" in text
    assert "axonscope.runtime.group_preparation" in text
    for required in (
        "prepare_batch_runtime",
        "prepared_cohort_for_current_group",
        "representative_item",
        "group_cm_uF_cm2",
    ):
        assert required in text
    offenders = sorted(term for term in forbidden_terms if term in text)
    assert offenders == []


def test_group_runner_routes_benchmark_metadata_through_metadata_module():
    path = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    text = path.read_text(encoding="utf-8")
    forbidden_terms = {
        "_record_intracellular_lowering_metadata",
        "_record_extracellular_lowering_metadata",
        "_record_group_memory_estimate",
        "_record_zero_intracellular_metadata",
        "_default_device_memory_capacity_bytes",
        "device_memory_capacity_bytes",
        "memory_estimate_components_nbytes",
    }

    assert "axonscope.runtime.jax.benchmarking.metadata" in text
    assert "_record_lowered_input_progress_and_memory" in text
    assert text.count("record_group_memory_estimate(") == 1
    offenders = sorted(term for term in forbidden_terms if term in text)
    assert offenders == []


def test_jax_benchmark_metadata_uses_runtime_neutral_memory_estimates():
    jax_metadata = (
        SRC_ROOT / "runtime" / "jax" / "benchmarking" / "metadata.py"
    ).read_text(encoding="utf-8")
    runtime_estimates = (SRC_ROOT / "runtime" / "memory_estimates.py").read_text(
        encoding="utf-8"
    )

    assert "estimate_runtime_group_memory" in jax_metadata
    assert "class RuntimeGroupMemoryEstimate" in runtime_estimates
    assert "import jax" not in runtime_estimates
    assert "vstim_mid_nbytes =" not in jax_metadata
    assert "vm_output_nbytes =" not in jax_metadata


def test_group_runner_keeps_common_orchestration_helpers():
    path = SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    text = path.read_text(encoding="utf-8")

    required_helpers = {
        "_LoweredJaxBatchInputs",
        "_lower_single_cable_inputs",
        "_lower_double_cable_inputs",
        "_emit_kernel_compile_progress",
        "_record_kernel_output_metadata",
    }
    missing = sorted(name for name in required_helpers if name not in text)

    assert missing == []


def test_preparation_runtime_batches_remains_host_side_only():
    assert not (SRC_ROOT / "dispatcher" / "runtime_batches.py").exists()
    path = SRC_ROOT / "preparation" / "runtime_batches.py"

    assert _jax_import_locations(path) == []


def test_public_simulation_orchestrator_uses_backend_execution_boundary():
    path = SRC_ROOT / "simulation.py"
    text = path.read_text(encoding="utf-8")

    assert _jax_import_locations(path) == []
    assert "axonscope.runtime.jax" not in text
    assert "axonscope.runtime.execution" in text


def test_crank_nicholson_facade_is_not_reintroduced():
    path = SRC_ROOT / "solvers" / "crank_nicholson.py"
    assert not path.exists()


def test_benchmarking_public_modules_are_interfaces_not_runtime_engines():
    benchmark_text = (SRC_ROOT / "benchmarking" / "benchmark.py").read_text(
        encoding="utf-8"
    )
    profiling_text = (SRC_ROOT / "benchmarking" / "profiling.py").read_text(
        encoding="utf-8"
    )

    forbidden_benchmark_terms = {
        "tracemalloc",
        "nvidia-smi",
        "subprocess",
        "ContextVar",
        "import jax",
        "numpy as np",
    }
    forbidden_profiling_terms = {
        "import jax",
        "jax.profiler",
        "start_trace",
        "stop_trace",
        "save_device_memory_profile",
    }

    assert "axonscope.runtime.benchmarking" in benchmark_text
    assert all(term not in benchmark_text for term in forbidden_benchmark_terms)
    assert "axonscope.runtime.execution" in profiling_text
    assert all(term not in profiling_text for term in forbidden_profiling_terms)


def test_active_double_cable_solver_surface_uses_typed_execution_policy():
    retained_public = {
        "auto",
        "thomas",
        "tiled_thomas",
    }
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

    assert {
        item.value for item in axs.runtime.jax.DoubleCableSolverKind
    } == retained_public
    assert {item.value for item in axs.runtime.jax.SingleCableSolverKind} == {
        "auto",
        "jax_tridiagonal",
    }
    assert not hasattr(axs, "DoubleCableSolver")
    assert not hasattr(axs, "SingleCableSolver")
    assert not hasattr(axs, "PcrSolverOptions")
    assert not hasattr(axs, "TiledThomasSolverOptions")
    assert not hasattr(axs, "Runtime")
    assert importlib.util.find_spec("axonscope.jax") is None
    assert importlib.util.find_spec("axonscope.runtime.jax") is not None
    assert axs.runtime.jax.runtime_target.value == "jax"
    assert axs.ExecutionPolicy(runtime=axs.runtime.jax).runtime is axs.runtime.jax
    assert not hasattr(axs.runtime, "DoubleCableSolver")
    assert not hasattr(axs.runtime, "SingleCableSolver")
    assert not hasattr(axs.solvers, "resolve_double_cable_block_solver")
    assert not hasattr(axs.solvers.options, "resolve_double_cable_block_solver")
    assert not hasattr(axs.solvers.options, "DoubleCableBlockSolver")
    assert not hasattr(axs.BatchOptions.full(), "double_cable_block_solver")
    assert archived.isdisjoint(retained_public)

    runtime_policy_text = (SRC_ROOT / "runtime" / "policy.py").read_text(
        encoding="utf-8"
    )
    jax_runtime_text = (
        SRC_ROOT / "runtime" / "jax" / "policy" / "__init__.py"
    ).read_text(encoding="utf-8")
    for concrete_jax_name in (
        "SingleCableSolverKind",
        "DoubleCableSolverKind",
        "TiledThomasSolverOptions",
        "class SingleCableSolver(",
        "class DoubleCableSolver(",
    ):
        assert concrete_jax_name not in runtime_policy_text
        assert concrete_jax_name in jax_runtime_text

    jax_kernels_dir = SRC_ROOT / "runtime" / "jax" / "kernels"
    kernel_text = "\n".join(
        path.read_text(encoding="utf-8") for path in jax_kernels_dir.glob("*.py")
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
    assert all(f"def {name}" not in kernel_text for name in archived_common_functions)


def test_solver_route_reporting_contract_is_cable_agnostic():
    assert axs.CableSolverRoute is axs.runtime.CableSolverRoute
    assert axs.RuntimeSolverRoute is axs.runtime.RuntimeSolverRoute

    assert [field.name for field in fields(axs.CableSolverRoute)] == [
        "cable",
        "requested",
        "runtime_route",
        "internal",
        "options",
    ]
    assert [field.name for field in fields(axs.RuntimeSolverRoute)] == [
        "runtime",
        "platform",
        "engine_name",
        "single_cable",
        "double_cable",
    ]
    assert [field.name for field in fields(axs.KernelInspection)] == [
        "group_id",
        "route",
        "kernel",
        "cable_mode",
        "solver",
        "time_chunk_steps",
    ]

    single = axs.CableSolverRoute(
        cable="single_cable",
        requested="auto",
        runtime_route="jax_tridiagonal",
    )
    double = axs.CableSolverRoute(
        cable="double_cable",
        requested="tiled_thomas",
        runtime_route="jax_triton_loop_xb",
        internal=True,
        options=(("block_b", 64),),
    )
    route = axs.RuntimeSolverRoute(
        runtime="jax",
        platform="gpu",
        engine_name="jax_gpu_tiled_thomas",
        single_cable=single,
        double_cable=double,
    )

    assert route.for_cable("single") is single
    assert route.for_cable("double") is double
    assert route.for_cable("single_cable") is single
    assert route.for_cable("double_cable") is double


def test_runtime_input_contract_is_cable_agnostic_and_runtime_neutral():
    from axonscope.runtime.input_contract import (
        ExtracellularLoweringMode,
        IntracellularLoweringMode,
        PreparedRuntimeInputSummary,
        RuntimeInputContract,
        intracellular_mode_from_format,
        validate_prepared_runtime_input,
        normalize_cable_formulation,
    )
    from axonscope.runtime.jax.inputs.lowering import (
        JAX_DOUBLE_CABLE_INPUT_CONTRACT,
        JAX_SINGLE_CABLE_INPUT_CONTRACT,
    )

    assert [field.name for field in fields(RuntimeInputContract)] == [
        "cable",
        "intracellular_modes",
        "extracellular",
        "supports_padding",
        "supports_row_specific_parameters",
        "supports_threshold_observer",
    ]

    assert normalize_cable_formulation("single") == "single-cable"
    assert normalize_cable_formulation("single_cable") == "single-cable"
    assert normalize_cable_formulation("double") == "double-cable"
    assert normalize_cable_formulation("double_cable") == "double-cable"

    single = JAX_SINGLE_CABLE_INPUT_CONTRACT
    double = JAX_DOUBLE_CABLE_INPUT_CONTRACT
    assert isinstance(single, RuntimeInputContract)
    assert isinstance(double, RuntimeInputContract)
    assert single.cable == "single-cable"
    assert double.cable == "double-cable"
    assert single.supports_padding
    assert double.supports_padding
    assert single.supports_row_specific_parameters
    assert double.supports_row_specific_parameters
    assert single.supports_threshold_observer
    assert double.supports_threshold_observer

    assert single.supports_intracellular(
        IntracellularLoweringMode.SPARSE_CURRENT_CLAMP
    )
    assert not double.supports_intracellular(
        IntracellularLoweringMode.SPARSE_CURRENT_CLAMP
    )
    assert single.supports_extracellular(
        ExtracellularLoweringMode.SCALED_SHARED_WAVEFORM
    )
    assert double.supports_extracellular(
        ExtracellularLoweringMode.SCALED_SHARED_WAVEFORM
    )
    assert double.extracellular.requires_initial_previous
    assert not single.extracellular.requires_initial_previous
    summary = PreparedRuntimeInputSummary(
        cable="single-cable",
        batch_size=2,
        nx=11,
        nt=3,
        dtype="float32",
        has_padding=False,
        row_specific_parameters=False,
        recording_mode="full",
        output_sink="vm",
        observer_count=0,
        time_chunk_steps=None,
        solver_policy="jax_single_cable_tridiagonal",
        intracellular_format="dense",
        intracellular_mode=intracellular_mode_from_format("dense"),
        extracellular_format="factorized_footprint",
        extracellular_mode=ExtracellularLoweringMode.SCALED_SHARED_WAVEFORM,
        extracellular_requires_initial_previous=False,
        extracellular_has_initial_previous=False,
    )
    assert validate_prepared_runtime_input(summary, single) == ()
    assert "prepared_input_contract_extracellular_mode" in summary.as_metadata()
    invalid = PreparedRuntimeInputSummary(
        cable="double-cable",
        batch_size=2,
        nx=11,
        nt=3,
        dtype="float32",
        has_padding=False,
        row_specific_parameters=False,
        recording_mode="none",
        output_sink="vm_raster",
        observer_count=1,
        time_chunk_steps=None,
        solver_policy="default",
        intracellular_format="sparse_current_clamp",
        intracellular_mode=IntracellularLoweringMode.SPARSE_CURRENT_CLAMP,
        extracellular_format="factorized_footprint",
        extracellular_mode=ExtracellularLoweringMode.CURRENT_TABLE,
        extracellular_requires_initial_previous=True,
        extracellular_has_initial_previous=False,
    )
    violations = validate_prepared_runtime_input(invalid, double)
    assert "intracellular mode 'sparse_current_clamp' is unsupported" in violations
    assert "extracellular mode requires an initial-previous sample" in violations


def test_runtime_input_planning_is_independent_from_observer_output_plan():
    from axonscope.runtime.input_contract import (
        ExtracellularInputFormat,
        ExtracellularLoweringMode,
        IntracellularInputFormat,
        dense_nbytes_for_shape,
        dense_shape_for_group,
    )
    from axonscope.runtime.input_planning import (
        planned_factorized_extracellular_mode_from_rows,
    )
    from axonscope.runtime.jax.inputs.lowering import (
        PlannedInputLowering,
        plan_input_lowering,
    )

    assert "observer_plan" not in inspect.signature(plan_input_lowering).parameters
    assert "observer_plan" not in inspect.signature(
        planned_factorized_extracellular_mode_from_rows
    ).parameters
    assert get_type_hints(PlannedInputLowering)["extracellular_mode"] == (
        ExtracellularLoweringMode | None
    )
    assert get_type_hints(PlannedInputLowering)["intracellular_format"] == (
        IntracellularInputFormat
    )
    assert get_type_hints(PlannedInputLowering)["extracellular_format"] == (
        ExtracellularInputFormat
    )
    group = type("Group", (), {"size": 3, "nx": 7})()
    runtime = type("Runtime", (), {"grid": type("Grid", (), {"Nt": 11})()})()
    assert dense_shape_for_group(group=group, runtime=runtime) == (3, 11, 7)
    assert dense_nbytes_for_shape((3, 11, 7), dtype=np.dtype("float32")) == 924

    input_lowering_tree = ast.parse(
        (SRC_ROOT / "runtime" / "jax" / "inputs" / "lowering.py").read_text()
    )
    moved_planning_defs = {
        "array_content_signature",
        "can_factorize_footprint_rows",
        "extracellular_stimulation_count",
        "factorized_drive_count_from_rows",
        "planned_factorized_extracellular_mode_from_rows",
        "stimulus_scaled_waveform_signature_and_scale",
    }
    offenders = sorted(
        node.name
        for node in ast.walk(input_lowering_tree)
        if isinstance(node, ast.FunctionDef) and node.name in moved_planning_defs
    )
    assert offenders == []


def test_jax_input_batches_does_not_own_runtime_neutral_current_planning():
    jax_input_batches = SRC_ROOT / "runtime" / "jax" / "inputs" / "extracellular.py"
    input_planning = SRC_ROOT / "runtime" / "input_planning.py"

    jax_text = jax_input_batches.read_text(encoding="utf-8")
    planning_text = input_planning.read_text(encoding="utf-8")

    forbidden_jax_impl_fragments = {
        "import weakref",
        "_ARRAY_CONTENT_KEY_CACHE",
        "_STIMULUS_SCALED_WAVEFORM_CACHE",
        "class _Rank1CurrentRows",
        "class _ScaledSharedWaveformRows",
        "class _ScaledWaveformSignatureCacheEntry",
        "def _cached_stimulus_current_A",
        "def _stimulus_temporal_cache_key",
        "def _stimulus_scaled_waveform_signature_and_scale",
        "def _array_content_key",
        "def _cached_array_content_key",
    }

    offenders = sorted(
        fragment for fragment in forbidden_jax_impl_fragments if fragment in jax_text
    )
    assert offenders == []
    assert "from axonscope.runtime.input_planning import (" in jax_text

    required_planning_exports = {
        "def cached_stimulus_current_A",
        "def stimulus_temporal_cache_key",
        "def build_rank1_current_rows_from_unique_stimuli",
        "def build_scaled_shared_waveform_rows",
        "def cached_array_content_signature",
    }
    missing = sorted(
        fragment for fragment in required_planning_exports if fragment not in planning_text
    )
    assert missing == []


def test_compact_input_payload_contracts_are_runtime_neutral():
    payload_contract = SRC_ROOT / "runtime" / "input_payloads.py"
    jax_payload_materializers = SRC_ROOT / "runtime" / "jax" / "inputs" / "payloads.py"

    assert payload_contract.is_file()
    assert _jax_import_locations(payload_contract) == []

    contract_tree = ast.parse(payload_contract.read_text(encoding="utf-8"))
    assert {
        node.name
        for node in ast.walk(contract_tree)
        if isinstance(node, ast.ClassDef)
    } == {
        "FactorizedExtracellularPotentialBatch",
        "SparseIntracellularCurrentDensityBatch",
    }

    jax_tree = ast.parse(jax_payload_materializers.read_text(encoding="utf-8"))
    assert [
        node.name
        for node in ast.walk(jax_tree)
        if isinstance(node, ast.ClassDef)
    ] == []

    jax_text = jax_payload_materializers.read_text(encoding="utf-8")
    assert "from axonscope.runtime.input_payloads import (" in jax_text
    assert "materialize_factorized_extracellular_potential_batch" in jax_text
    assert "materialize_sparse_intracellular_current_density_batch" in jax_text


def test_p12_runtime_cleanup_uses_runtime_context_vocabulary():
    assert (
        REPO_ROOT / "docs" / "architecture" / "p12_runtime_contract_2026_07_12.md"
    ).is_file()
    assert (
        REPO_ROOT / "docs" / "architecture" / "p12a_jax_runtime_audit_2026_07_12.md"
    ).is_file()

    active_sources = [
        SRC_ROOT / "simulation.py",
        SRC_ROOT / "dispatcher" / "execution.py",
        SRC_ROOT / "runtime" / "execution.py",
        SRC_ROOT / "runtime" / "jax" / "group_runner.py",
        SRC_ROOT / "runtime" / "jax" / "preparation" / "runtime.py",
        SRC_ROOT / "runtime" / "jax" / "preparation" / "stacking.py",
    ]
    legacy_name = "backend" + "_context"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in active_sources
        if legacy_name in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_runtime_output_plan_contract_is_runtime_neutral():
    assert not (SRC_ROOT / "runtime" / "jax" / "output_plan.py").exists()

    from axonscope.runtime.output_contract import (
        OutputPlan,
        observer_output_label,
        observers_are_vm_raster_compatible,
        vm_raster_definitions,
    )
    from axonscope.solvers.options import BatchOptions

    execution_text = (SRC_ROOT / "runtime" / "execution.py").read_text(
        encoding="utf-8"
    )
    assert "benchmark_observers_are_vm_raster_compatible as jax" not in execution_text
    assert "benchmark_observer_output_label as jax" not in execution_text
    assert "benchmark_vm_raster_definitions as jax" not in execution_text

    assert OutputPlan.from_batch_options(
        BatchOptions.none(),
        observers=None,
    ).sink == "none"
    assert OutputPlan.from_batch_options(
        BatchOptions.none(),
        observers=(object(),),
    ).sink == "unsupported_observer_only"
    assert OutputPlan.from_batch_options(
        BatchOptions.full(),
        observers=(object(),),
    ).sink == "vm"
    assert observer_output_label(None, recording_mode="none") == "none"
    assert observers_are_vm_raster_compatible(None) is False
    assert vm_raster_definitions(None) == ()


def test_runtime_recording_conversion_is_runtime_neutral():
    assert (SRC_ROOT / "runtime" / "recording.py").is_file()
    assert not (SRC_ROOT / "runtime" / "jax" / "recording.py").exists()

    execution_text = (SRC_ROOT / "runtime" / "execution.py").read_text(
        encoding="utf-8"
    )
    group_runner_text = (
        SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    ).read_text(encoding="utf-8")
    jax_lowering_text = (
        SRC_ROOT / "runtime" / "jax" / "recording" / "lowering.py"
    ).read_text(encoding="utf-8")
    assert "axonscope.runtime.recording" in execution_text
    assert "axonscope.runtime.jax.recording" not in execution_text
    assert "benchmark_lower_recording_options as jax" not in execution_text
    assert "axonscope.runtime.recording" in group_runner_text
    assert "lower_batch_recording_options" not in jax_lowering_text
    assert "row_recording_indices_for_group" not in jax_lowering_text


def test_runtime_host_array_preparation_is_runtime_neutral():
    assert (SRC_ROOT / "runtime" / "host_preparation.py").is_file()

    from axonscope.runtime import host_preparation

    for name in (
        "compartment_area_cm2_numpy",
        "diffusion_operator_coeffs_numpy",
        "extracellular_runtime_numpy",
        "pad_edge_array_numpy",
        "pad_gate_array_numpy",
        "pad_space_array_numpy",
    ):
        assert hasattr(host_preparation, name)

    runtime_preparation_tree = ast.parse(
        (
            SRC_ROOT / "runtime" / "jax" / "preparation" / "stacking.py"
        ).read_text(encoding="utf-8")
    )
    moved_defs = {
        "compartment_area_cm2_numpy",
        "diffusion_operator_coeffs_numpy",
        "extracellular_runtime_numpy",
        "pad_edge_array_numpy",
        "pad_gate_array_numpy",
        "pad_space_array_numpy",
        "_compartment_area_cm2_numpy",
        "_diffusion_operator_coeffs_numpy",
        "_extracellular_runtime_numpy",
        "_pad_edge_array_numpy",
        "_pad_gate_array_numpy",
        "_pad_space_array_numpy",
    }
    offenders = sorted(
        node.name
        for node in ast.walk(runtime_preparation_tree)
        if isinstance(node, ast.FunctionDef) and node.name in moved_defs
    )
    assert offenders == []


def test_public_examples_do_not_use_backend_solver_route_labels():
    examples_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _python_sources(REPO_ROOT / "examples")
    )

    forbidden = {
        "jax_triton_loop_xb",
        "double_cable_block_solver",
        "allow_internal_double_cable_block_solver",
    }
    offenders = sorted(label for label in forbidden if label in examples_text)
    assert offenders == []


def test_non_thomas_double_cable_kernel_tests_are_diagnostic_or_gpu_scoped():
    """Keep CPU production tests from treating internal GPU routes as supported policy."""

    allowed_name_markers = {"diagnostic", "gpu", "benchmark"}
    diagnostic_solver_values = {"jax_triton_loop_xb"}
    offenders: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            for keyword in node.keywords:
                if keyword.arg == "double_cable_block_solver":
                    self._check_value(keyword.value, node.lineno)
            self.generic_visit(node)

        def visit_Dict(self, node: ast.Dict) -> None:
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "double_cable_block_solver":
                    self._check_value(value, node.lineno)
            self.generic_visit(node)

        def _check_value(self, value: ast.AST, lineno: int) -> None:
            if not isinstance(value, ast.Constant):
                return
            if value.value not in diagnostic_solver_values:
                return
            function_name = self.function_stack[-1] if self.function_stack else ""
            if any(marker in function_name for marker in allowed_name_markers):
                return
            offenders.append(f"{self.path.relative_to(REPO_ROOT)}:{lineno}:{function_name}")

    unit_root = REPO_ROOT / "tests" / "unit"
    for path in _python_sources(unit_root):
        if "benchmarking" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        Visitor(path).visit(tree)

    assert offenders == []


def test_factorized_vext_route_has_dense_equivalence_tests():
    solver_test_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "tests" / "unit" / "solvers" / "test_batch.py",
        )
    )

    required_tests = {
        "test_factorized_footprint_batch_matches_dense_builder_and_observer_raster",
        "test_factorized_footprint_batch_supports_scaled_shared_waveforms",
        "test_double_cable_factorized_footprint_observer_matches_dense_thomas",
    }

    missing = sorted(name for name in required_tests if f"def {name}" not in solver_test_text)
    assert missing == []


def test_solver_route_map_documents_retained_runtime_paths():
    text = (REPO_ROOT / "docs" / "solver_organization.md").read_text(encoding="utf-8")

    required_terms = {
        "## Active Solver Route Map",
        "### Single-Row Batch Route",
        "### Pool And Planning Route",
        "### Single-Cable Batch Route",
        "### Double-Cable Batch Route",
        "### Threshold Observers, Dense/Factorized Vext, And Results",
        "build_dispatch_plan",
        "_run_batch_group",
        "enqueue_jax_batch_group",
        "finalize_jax_batch_group",
        "build_sparse_intracellular_current_density_batch",
        "build_intracellular_current_density_batch",
        "build_factorized_vstim_midpoint_batch",
        "build_vstim_midpoint_batch",
        "build_vstim_midpoint_and_initial_previous_batch",
        "SingleCableVStimBatchKernel",
        "DoubleCableBatchKernel",
        "build_threshold_observer_plan",
        "runtime/result_assembly.py",
        "runtime/jax/membranes/",
        "runtime/jax/membranes/stacking.py",
        "runtime/jax/recording/lowering.py",
        "runtime/jax/recording/results.py",
        "runtime/jax/preparation/runtime.py",
        "runtime/jax/preparation/stacking.py",
        "runtime/jax/preparation/caches.py",
        "runtime/jax/preparation/shape_bucketing.py",
        "dispatch_results_from_batch",
        "compact dispatch cohort records",
        "AxonSimulationResult",
    }

    missing = sorted(term for term in required_terms if term not in text)
    assert missing == []

    option_section = text.split("The current typed public choices are:", 1)[1].split(
        "Example:",
        1,
    )[0]
    archived_options = {
        "assoc_backward",
        "assoc_transfer_dense",
        "pallas_pcr_128",
        "pallas_thomas_4",
        "split_jacobi_4",
        "split_gs_4",
    }
    assert archived_options.isdisjoint(option_section)


def test_numeric_execution_axis_is_protocol_independent():
    axis_text = (SRC_ROOT / "dispatcher" / "numeric_axis.py").read_text(
        encoding="utf-8"
    )
    plan_text = (SRC_ROOT / "dispatcher" / "plan.py").read_text(encoding="utf-8")
    simulation_text = (SRC_ROOT / "simulation.py").read_text(encoding="utf-8")
    group_runner_text = (
        SRC_ROOT / "runtime" / "jax" / "group_runner.py"
    ).read_text(encoding="utf-8")
    stimulus_text = (SRC_ROOT / "stimulation" / "stimuli.py").read_text(
        encoding="utf-8"
    )
    extracellular_text = (
        SRC_ROOT / "stimulation" / "extracellular.py"
    ).read_text(encoding="utf-8")

    assert "axonscope.protocols" not in axis_text
    assert "recruitment" not in axis_text.lower()
    assert "expand_dispatch_plan_for_numeric_axis" in plan_text
    assert "_run_numeric_axis" in simulation_text
    for obsolete in (
        "expand_dispatch_plan_for_waveform_axis",
        "_run_extracellular_waveform_axis",
        "waveform_axis: tuple",
        "waveform_source_size",
        "_lower_group_waveform_axis",
    ):
        assert obsolete not in plan_text
        assert obsolete not in simulation_text
        assert obsolete not in group_runner_text
    assert "_replace_runtime_waveform" not in stimulus_text
    assert "_bind_runtime_stimulus" not in extracellular_text
