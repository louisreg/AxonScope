from __future__ import annotations

import ast
import importlib.util
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import axonscope as axs
from axonscope import membranes
from axonscope.benchmarking import benchmark_span
from axonscope.runtime.jax.membranes.backend import (
    GatedLeakStackMembraneBackend,
    HeterogeneousMembraneBackend,
    UniformMembraneBackend,
)
from axonscope.runtime.jax.membranes.compile import compile_membrane_model
from axonscope.runtime.jax.membranes.generated_contract import (
    load_generated_jax_membrane_contract,
)
from axonscope.runtime.jax.membranes.program import JaxMembraneProgram
from axonscope.membranes.compiler import lower_membrane_model_to_ir
from axonscope.membranes.model import MembraneModel
from axonscope.model_ir import (
    Current,
    Diagnostic,
    Gate,
    Input,
    LinearizationGateSource,
    ModelIR,
    ModelValidationError,
    Observable,
    Parameter,
    QuantitySpec,
    SemanticRole,
    SourceModelCompileError,
    assert_valid_model_ir,
    call,
    compile_model_source_file,
    derive_model_step_contract,
    literal,
    membrane_program_from_model_ir,
    model_ir_from_json,
    parameterized_hash,
    State,
    StateUpdate,
    StepProgram,
    structural_hash,
    symbol,
)
import axonscope.model_ir.source as source_compiler
from axonscope.model_ir.interpreter import NumpyModelInterpreter
from axonscope.model_ir.intrinsics import exp
from axonscope.utils.units import (
    CONDUCTANCE_DENSITY_MS_CM2,
    CURRENT_DENSITY_UA_CM2,
    DIMENSIONLESS,
    RATE_PER_MS,
    RESISTANCE_AREA_OHM_CM2,
    VOLTAGE_MV,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_IR_ROOT = REPO_ROOT / "src" / "axonscope" / "model_ir"
PASSIVE_SOURCE = REPO_ROOT / "src" / "axonscope" / "membranes" / "models" / "passive.py"
HH_SOURCE = REPO_ROOT / "src" / "axonscope" / "membranes" / "models" / "hodgkin_huxley.py"
TIGERHOLM_SOURCE = REPO_ROOT / "src" / "axonscope" / "membranes" / "models" / "tigerholm.py"
SCHILD94_SOURCE = REPO_ROOT / "src" / "axonscope" / "membranes" / "models" / "schild94.py"
SCHILD97_SOURCE = REPO_ROOT / "src" / "axonscope" / "membranes" / "models" / "schild97.py"


def _source_model(name: str, params: dict[str, float] | None = None) -> ModelIR:
    return compile_model_source_file(
        REPO_ROOT / "src" / "axonscope" / "membranes" / "models" / f"{name}.py",
        parameter_defaults=params or {},
    ).model


def test_model_ir_package_has_no_jax_imports():
    offenders: list[str] = []
    for path in sorted(MODEL_IR_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "jax" or alias.name.startswith("jax."):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and (module == "jax" or module.startswith("jax.")):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_passive_model_ir_is_valid_and_exposes_fusion_terms():
    model = _source_model("passive")
    assert model.name == "passive"
    assert [current.name for current in model.currents] == ["I_l"]
    assert [parameter.name for parameter in model.parameters] == ["Rm", "EL"]
    assert model.parameters[0].quantity.unit == RESISTANCE_AREA_OHM_CM2
    assert model.metadata["source_contract"] == "plain_python_membrane.v1"
    assert len(model.metadata["source_hash"]) == 40
    provenance = model.metadata["source_provenance"]
    assert provenance["contract"] == model.metadata["source_contract"]
    assert provenance["compiler"] == model.metadata["source_compiler"]
    assert provenance["source_hash"] == model.metadata["source_hash"]
    assert provenance["function_names"] == ("leak",)
    with pytest.raises(TypeError):
        model.metadata["x"] = "mutating metadata is not allowed"

    contract = derive_model_step_contract(model, requested_observables=("g_l",))

    assert contract.total_outward_current == "sum(currents) + background_current"
    assert contract.total_conductance == "sum(current.conductance)"
    assert contract.conductance_reversal_sum == "sum(current.conductance * current.reversal)"
    assert contract.pruning.solver_output_names == ("I_l",)
    assert contract.pruning.retain_observables == ("g_l",)
    assert contract.pruning.recording_output_names == ("observables.g_l",)
    assert contract.supports_single_cable_fusion
    assert contract.supports_double_cable_fusion

    recording_contract = derive_model_step_contract(
        model,
        requested_observables=("g_l",),
        record_gates=True,
        record_currents=True,
        record_conductances=True,
    )
    assert recording_contract.pruning.retain_gates == ()
    assert recording_contract.pruning.retain_currents == ("I_l",)
    assert recording_contract.pruning.retain_conductances == ("I_l",)
    assert recording_contract.pruning.recording_output_names == (
        "currents.I_l",
        "conductances.I_l",
        "observables.g_l",
    )


def test_membrane_program_exposes_backend_neutral_runtime_contract():
    model = lower_membrane_model_to_ir(
        membranes.Composite(
            [
                membranes.RattayAberham(),
                membranes.Passive(Rm=1000.0, EL=-70.0),
            ]
        )
    )
    program = membrane_program_from_model_ir(model)

    assert program.name == "composite"
    assert program.gate_state_names == ("m", "h", "n")
    assert program.gate_names == (
        "rattay_aberham.m",
        "rattay_aberham.h",
        "rattay_aberham.n",
    )
    assert program.membrane_state_names == ()
    assert program.raw_current_names == ("I_na", "I_k", "I_l", "I_l")
    assert program.current_names == ("I_na", "I_k", "I_l")
    assert program.current_groups == ((0,), (1,), (2, 3))
    assert program.raw_conductance_names == ("g_na", "g_k", "g_l", "g_l")
    assert program.conductance_names == ("g_na", "g_k", "g_l")
    assert program.conductance_groups == ((0,), (1,), (2, 3))
    assert program.conductance_parameter_names == ("gnabar", "gkbar", "gl", "gl")
    assert program.diagnostic_names == ()
    assert program.final_gate_update_mode == "post_solve_voltage"
    assert program.source_provenance["kind"] == "composite"
    assert len(program.source_provenance["components"]) == 2
    assert program.structural_hash == structural_hash(model)
    assert program.parameterized_hash == parameterized_hash(model)

    source_program = membrane_program_from_model_ir(
        lower_membrane_model_to_ir(membranes.HodgkinHuxley())
    )
    assert source_program.source_provenance["source_hash"]
    assert source_program.codegen_cache["key"]


def test_composite_requires_labels_for_duplicate_component_kinds():
    with pytest.raises(ValueError, match="explicit component labels"):
        membranes.Composite([membranes.Passive(), membranes.Passive()])

    public_model = membranes.Composite(
        {
            "passive_weak": membranes.Passive(Rm=20_000.0, EL=-70.0),
            "passive_strong": membranes.Passive(Rm=5_000.0, EL=-65.0),
        }
    )
    model = lower_membrane_model_to_ir(public_model)
    program = membrane_program_from_model_ir(model)

    assert model.metadata["component_labels"] == ("passive_weak", "passive_strong")
    assert model.metadata["component_public_names"]["observables"] == (
        ("passive_weak__g_l", "passive_weak.g_l"),
        ("passive_strong__g_l", "passive_strong.g_l"),
    )
    assert [observable.name for observable in model.observables] == [
        "passive_weak__g_l",
        "passive_strong__g_l",
    ]
    assert program.observable_display_names == (
        "passive_weak.g_l",
        "passive_strong.g_l",
    )
    assert program.raw_current_names == ("I_l", "I_l")
    assert program.current_names == ("I_l",)
    assert program.conductance_names == ("g_l",)


def test_passive_plain_python_source_codegen_cache(tmp_path):
    first = compile_model_source_file(
        PASSIVE_SOURCE,
        parameter_defaults={"Rm": 20_000.0, "EL": -65.0},
        cache_root=tmp_path,
    )
    second = compile_model_source_file(
        PASSIVE_SOURCE,
        parameter_defaults={"Rm": 30_000.0, "EL": -60.0},
        cache_root=tmp_path,
    )

    assert first.source_hash == second.source_hash
    assert first.cache.cache_hit is False
    assert first.cache.cache_reason == "manifest_missing"
    assert second.cache.cache_hit is True
    assert second.cache.cache_reason == "manifest_match"
    assert first.cache.key == second.cache.key
    assert first.model.metadata["codegen_cache"] == second.model.metadata["codegen_cache"]
    assert first.model.metadata["codegen_cache"]["key"] == first.cache.key
    assert "cache_hit" not in first.model.metadata["codegen_cache"]
    assert structural_hash(first.model) == structural_hash(second.model)
    assert (first.cache.directory / "manifest.json").is_file()
    assert (first.cache.directory / "source_snapshot.py").is_file()
    assert (first.cache.directory / "graph.json").is_file()
    assert (first.cache.directory / "optimized_graph.json").is_file()
    jax_source = (first.cache.directory / "jax_model.py").read_text(encoding="utf-8")
    numpy_source = (first.cache.directory / "numpy_model.py").read_text(encoding="utf-8")

    assert "import jax.numpy as xp" in jax_source
    assert "import numpy as xp" in numpy_source
    assert "ARG_NAMES = ('Vm', 'Rm', 'EL')" in jax_source
    assert "OUTPUT_NAMES = ('I_l', 'g_l')" in jax_source
    assert "def model_step(Vm, Rm, EL):" in jax_source
    assert "g_l = (1000.0 / Rm)" in jax_source

    interpreter = NumpyModelInterpreter(first.model)
    V = np.asarray([-80.0, -65.0, -40.0], dtype=np.float32)
    gates = interpreter.init_gates(V)
    expected_conductance = np.full((3, 1), 0.05, dtype=np.float32)
    expected_current = 0.05 * (V - (-65.0))

    np.testing.assert_allclose(interpreter.conductances(gates), expected_conductance)
    np.testing.assert_allclose(interpreter.currents(V, gates), expected_current)


def test_source_codegen_cache_hit_loads_graph_without_ast_parse(tmp_path, monkeypatch):
    first = compile_model_source_file(
        PASSIVE_SOURCE,
        parameter_defaults={"Rm": 20_000.0, "EL": -65.0},
        cache_root=tmp_path,
    )

    def fail_parse(*args, **kwargs):
        raise AssertionError("cache hit should not parse source AST")

    monkeypatch.setattr(source_compiler.ast, "parse", fail_parse)
    second = compile_model_source_file(
        PASSIVE_SOURCE,
        parameter_defaults={"Rm": 30_000.0, "EL": -60.0},
        cache_root=tmp_path,
        load_generated_modules=("numpy",),
    )

    assert second.cache.cache_hit is True
    assert second.cache.cache_reason == "manifest_match"
    assert second.cache.loaded_modules["numpy"].CACHE_KEY == first.cache.key
    assert second.source_hash == first.source_hash
    assert second.model.parameters[0].default == 30_000.0
    assert second.model.parameters[1].default == -60.0
    assert structural_hash(second.model) == structural_hash(first.model)


def test_source_codegen_cache_keeps_canonical_defaults_across_overrides(tmp_path):
    first = compile_model_source_file(
        PASSIVE_SOURCE,
        parameter_defaults={"Rm": 30_000.0},
        cache_root=tmp_path,
        generated_targets=("jax",),
        load_generated_modules=("jax",),
    )
    second = compile_model_source_file(
        PASSIVE_SOURCE,
        cache_root=tmp_path,
        generated_targets=("jax",),
        load_generated_modules=("jax",),
    )

    assert first.model.parameters[0].default == 30_000.0
    assert second.model.parameters[0].default == 10_000.0
    assert second.cache.cache_hit is True
    assert second.cache.key == first.cache.key

    contract = load_generated_jax_membrane_contract(
        second.cache.loaded_modules["jax"]
    )
    assert contract.model_name == "passive"
    assert contract.parameter_defaults() == {"Rm": 10_000.0, "EL": -70.0}
    assert contract.current_names == ("I_l",)
    assert contract.conductance_names == ("g_l",)
    assert contract.structural_hash == membrane_program_from_model_ir(
        second.model
    ).structural_hash


def test_source_codegen_adds_runtime_targets_without_rewriting_cached_artifacts(tmp_path):
    first = compile_model_source_file(
        HH_SOURCE,
        cache_root=tmp_path,
        generated_targets=("jax",),
        load_generated_modules=("jax",),
    )
    jax_path = first.cache.directory / "jax_model.py"
    numpy_path = first.cache.directory / "numpy_model.py"
    jax_stat = jax_path.stat()
    jax_text = jax_path.read_text(encoding="utf-8")

    assert jax_path.is_file()
    assert not numpy_path.exists()
    assert first.model.metadata["codegen_cache"]["targets"] == ("jax",)

    second = compile_model_source_file(
        HH_SOURCE,
        cache_root=tmp_path,
        generated_targets=("numpy",),
        load_generated_modules=("numpy",),
    )

    assert second.cache.key == first.cache.key
    assert numpy_path.is_file()
    assert jax_path.read_text(encoding="utf-8") == jax_text
    assert jax_path.stat().st_mtime_ns == jax_stat.st_mtime_ns
    assert second.cache.loaded_modules["numpy"].TARGET == "numpy"

    third = compile_model_source_file(
        HH_SOURCE,
        cache_root=tmp_path,
        generated_targets=("jax",),
        load_generated_modules=("jax",),
    )
    assert third.cache.cache_hit is True
    assert third.cache.key == first.cache.key
    assert third.cache.loaded_modules["jax"] is first.cache.loaded_modules["jax"]


def test_model_ir_round_trips_from_codegen_graph_json(tmp_path):
    compiled = compile_model_source_file(PASSIVE_SOURCE, cache_root=tmp_path)

    restored = model_ir_from_json(
        (compiled.cache.directory / "optimized_graph.json").read_text(encoding="utf-8")
    )
    expected = replace(
        compiled.model,
        metadata={
            key: value
            for key, value in compiled.model.metadata.items()
            if key != "codegen_cache"
        },
    )

    assert restored.name == compiled.model.name
    assert restored.metadata["source_hash"] == compiled.model.metadata["source_hash"]
    assert structural_hash(restored) == structural_hash(expected)
    assert parameterized_hash(restored) == parameterized_hash(expected)


def test_jax_membrane_program_uses_generated_model_step_for_currents(tmp_path):
    compiled = compile_model_source_file(
        PASSIVE_SOURCE,
        cache_root=tmp_path,
        load_generated_modules=("jax",),
    )
    module = compiled.cache.loaded_modules["jax"]
    original_model_step = module.model_step

    def fake_model_step(Vm, Rm, EL):
        _ = Rm, EL
        return jnp.full_like(Vm, 7.0), jnp.full_like(Vm, 3.0)

    module.model_step = fake_model_step
    try:
        membrane = JaxMembraneProgram.from_model_ir(
            compiled.model,
            generated_module=module,
        )
        V = jnp.asarray([-80.0, -65.0, -40.0], dtype=jnp.float32)
        gates = jnp.zeros((3, 0), dtype=jnp.float32)

        assert membrane.uses_generated_model_step
        np.testing.assert_allclose(
            np.asarray(membrane.ionic_current_trace_matrix(V, gates)),
            np.full((3, 1), 7.0, dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(membrane.currents(V, gates)),
            np.full((3,), 7.0, dtype=np.float32),
        )
    finally:
        module.model_step = original_model_step


def test_jax_membrane_program_uses_generated_gate_and_membrane_terms(tmp_path):
    compiled = compile_model_source_file(
        HH_SOURCE,
        cache_root=tmp_path,
        load_generated_modules=("jax",),
    )
    module = compiled.cache.loaded_modules["jax"]
    original_gate_terms = module.gate_terms
    original_membrane_terms = module.membrane_terms

    def fake_gate_terms(Vm, celsius):
        _ = celsius
        return tuple(
            jnp.full_like(Vm, value)
            for value in (1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 7.0, 8.0, 3.0)
        )

    def fake_membrane_terms(m, h, n, gnabar, gkbar, gl, el, ena, ek):
        _ = m, h, n, gnabar, gkbar, gl, el, ena, ek
        return 1.0, 10.0, 2.0, 20.0, 3.0, 30.0

    module.gate_terms = fake_gate_terms
    module.membrane_terms = fake_membrane_terms
    try:
        membrane = JaxMembraneProgram.from_model_ir(
            compiled.model,
            generated_module=module,
        )
        V = jnp.asarray([-80.0, -40.0], dtype=jnp.float32)
        gates = jnp.ones((2, 3), dtype=jnp.float32)

        assert membrane.lowering.generated_gate_terms_available
        assert membrane.lowering.generated_membrane_terms_available
        alpha, beta, q10 = membrane.lowering.gate_terms(V)
        np.testing.assert_allclose(np.asarray(alpha), [[1.0, 4.0, 7.0]] * 2)
        np.testing.assert_allclose(np.asarray(beta), [[2.0, 5.0, 8.0]] * 2)
        np.testing.assert_allclose(np.asarray(q10), [[3.0, 3.0, 3.0]] * 2)
        gm, ge = membrane.membrane_conductance_terms(gates)
        np.testing.assert_allclose(np.asarray(gm), [6.0, 6.0])
        np.testing.assert_allclose(np.asarray(ge), [140.0, 140.0])
    finally:
        module.gate_terms = original_gate_terms
        module.membrane_terms = original_membrane_terms


def test_generated_hh_supports_model_agnostic_gated_leak_batch_capability(tmp_path):
    compiled = compile_model_source_file(
        HH_SOURCE,
        cache_root=tmp_path,
        generated_targets=("jax",),
        load_generated_modules=("jax",),
    )
    membrane = JaxMembraneProgram.from_model_ir(
        compiled.model,
        generated_module=compiled.cache.loaded_modules["jax"],
    )
    nx = 5
    backend = GatedLeakStackMembraneBackend(
        gated_model=membrane,
        target_nx=nx,
        dtype=jnp.float32,
        gated_gate_count=3,
        gated_channel_count=3,
    )
    voltage = jnp.asarray(
        [[-75.0, -70.0, -65.0, -60.0, -55.0]] * 2,
        dtype=jnp.float32,
    )
    gated = jax.vmap(membrane.init_gates)(voltage)
    leak_g = jnp.full((2, nx, 1), 0.1, dtype=jnp.float32)
    leak_ge = jnp.full((2, nx, 1), -6.5, dtype=jnp.float32)
    gated_mask = jnp.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    gated_mask = jnp.broadcast_to(gated_mask.reshape((1, nx, 1)), (2, nx, 1))
    gates = jnp.concatenate([gated, leak_g, leak_ge, gated_mask], axis=-1)

    expected = jax.vmap(
        lambda row, vm: backend.cn_gate_update_for_row(
            0,
            g_prev=row,
            V_mV=vm,
            dt=0.005,
        )
    )(gates, voltage)
    actual = backend.batch_cn_gate_update(
        g_prev=gates,
        V_mV=voltage,
        dt=0.005,
    )
    expected_gm, expected_ge = jax.vmap(
        lambda row: backend.membrane_conductance_terms_for_row(0, row)
    )(actual)
    actual_gm, actual_ge = backend.batch_membrane_conductance_terms(actual)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_gm, expected_gm, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_ge, expected_ge, rtol=1e-6, atol=1e-7)


def test_source_parameter_defaults_must_include_units_for_dimensioned_values(tmp_path):
    source = tmp_path / "bad_units.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import CurrentDensity, Voltage

class BadUnits(Model):
    model_kind = "bad_units"

    @currents
    def currents(self, Vm: Voltage, EL: Voltage = -70.0):
        I_l: CurrentDensity = Vm - EL
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(SourceModelCompileError, match="must specify unit 'mV'"):
        compile_model_source_file(source)


def test_source_parameter_defaults_accept_top_level_axonscope_units(tmp_path):
    source = tmp_path / "top_level_units.py"
    source.write_text(
        """
import axonscope as axs
from axonscope.membranes.types import ConductanceDensity, CurrentDensity, ResistanceArea, Voltage

class TopLevelUnits(axs.membranes.Model):
    model_kind = "top_level_units"

    Rm: ResistanceArea = 1.0e4 * axs.ohm_cm2
    EL: Voltage = -70.0 * axs.mV

    @axs.membranes.currents
    def currents(self, Vm: Voltage):
        g_l: ConductanceDensity = 1.0 / self.Rm
        I_l: CurrentDensity = g_l * (Vm - self.EL)
        return I_l, g_l
""".lstrip(),
        encoding="utf-8",
    )

    compiled = compile_model_source_file(source, cache_root=tmp_path / "cache")

    assert compiled.model.name == "top_level_units"
    assert [parameter.name for parameter in compiled.model.parameters] == ["Rm", "EL"]


def test_source_compiler_topologically_orders_equations(tmp_path):
    source = tmp_path / "out_of_order.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import ConductanceDensity, CurrentDensity, ResistanceArea, Voltage
from axonscope.utils.units import cm2, mV, ohm

class OutOfOrder(Model):
    model_kind = "out_of_order"

    @currents
    def currents(self, Vm: Voltage, Rm: ResistanceArea = 1.0e4 * ohm * cm2, EL: Voltage = -70.0 * mV):
        I_l: CurrentDensity = g_l * (Vm - EL)
        g_l: ConductanceDensity = 1.0 / Rm
        return I_l, g_l
""".lstrip(),
        encoding="utf-8",
    )

    compiled = compile_model_source_file(source, cache_root=tmp_path / "cache")
    generated = (compiled.cache.directory / "numpy_model.py").read_text(encoding="utf-8")

    assert compiled.model.name == "out_of_order"
    assert generated.index("g_l =") < generated.index("I_l =")


def test_source_compiler_reports_unknown_equation_dependencies(tmp_path):
    source = tmp_path / "unknown_dependency.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import CurrentDensity, Voltage
from axonscope.utils.units import mV

class UnknownDependency(Model):
    model_kind = "unknown_dependency"

    @currents
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        I_l: CurrentDensity = missing_current
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match=r"line \d+, column \d+: Unknown symbol\(s\).*missing_current",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_reports_equation_cycles(tmp_path):
    source = tmp_path / "cycle.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import CurrentDensity, Voltage
from axonscope.utils.units import mV

class Cycle(Model):
    model_kind = "cycle"

    @currents
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        a: CurrentDensity = b
        b: CurrentDensity = a
        I_l: CurrentDensity = a
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match="Cycle detected in equation dependencies: a -> b -> a",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_rejects_duplicate_equation_assignments(tmp_path):
    source = tmp_path / "duplicate_assignment.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import ConductanceDensity, CurrentDensity, ResistanceArea, Voltage
from axonscope.utils.units import cm2, mV, ohm

class DuplicateAssignment(Model):
    model_kind = "duplicate_assignment"

    @currents
    def currents(self, Vm: Voltage, Rm: ResistanceArea = 1.0e4 * ohm * cm2, EL: Voltage = -70.0 * mV):
        g_l: ConductanceDensity = 1.0 / Rm
        g_l: ConductanceDensity = 2.0 / Rm
        I_l: CurrentDensity = g_l * (Vm - EL)
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match="Duplicate equation assignment 'g_l'; first defined",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_rejects_duplicate_exports(tmp_path):
    source = tmp_path / "duplicate_export.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import CurrentDensity, Voltage
from axonscope.utils.units import mV

class DuplicateExport(Model):
    model_kind = "duplicate_export"

    @currents(outputs=("I_l", "I_l"))
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        I_l: CurrentDensity = Vm - EL
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match="Duplicate @currents currents name 'I_l'",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_supports_explicit_current_terms(tmp_path):
    source = tmp_path / "explicit_current_terms.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import ConductanceDensity, CurrentDensity, Voltage
from axonscope.utils.units import cm2, mS, mV, uA

class ExplicitCurrentTerms(Model):
    model_kind = "explicit_current_terms"

    @currents(
        outputs=("I_drive",),
        conductances={"I_drive": "g_drive"},
        reversals={"I_drive": "E_drive"},
    )
    def currents(self, Vm: Voltage, E_drive: Voltage = -55.0 * mV):
        g_drive: ConductanceDensity = 0.2 * mS / cm2
        offset: CurrentDensity = 0.0 * uA / cm2
        I_drive: CurrentDensity = g_drive * (Vm - E_drive) + offset
        return I_drive
""".lstrip(),
        encoding="utf-8",
    )

    compiled = compile_model_source_file(source, cache_root=tmp_path / "cache")
    model = compiled.model

    assert model.currents[0].name == "I_drive"

    interpreter = NumpyModelInterpreter(model, dtype=np.float64)
    V = np.asarray([-75.0, -55.0, -35.0], dtype=np.float64)
    gates = interpreter.init_gates(V)

    np.testing.assert_allclose(
        interpreter.conductances(gates),
        np.full((3, 1), 0.2, dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        interpreter.current_matrix(V, gates)[:, 0],
        0.2 * (V + 55.0),
        rtol=1e-12,
        atol=1e-12,
    )
    g_total, ge_total = interpreter.membrane_conductance_terms(gates)
    np.testing.assert_allclose(g_total, 0.2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ge_total, -11.0, rtol=1e-12, atol=1e-12)


def test_source_compiler_rejects_incomplete_explicit_current_terms(tmp_path):
    source = tmp_path / "incomplete_current_terms.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import ConductanceDensity, CurrentDensity, Voltage
from axonscope.utils.units import cm2, mS, mV

class IncompleteCurrentTerms(Model):
    model_kind = "incomplete_current_terms"

    @currents(outputs=("I_l",), conductances={"I_l": "g_l"})
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        g_l: ConductanceDensity = 0.1 * mS / cm2
        I_l: CurrentDensity = g_l * (Vm - EL)
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match="must define both conductance and reversal",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_rejects_unknown_explicit_current_terms(tmp_path):
    source = tmp_path / "unknown_current_terms.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import ConductanceDensity, CurrentDensity, Voltage
from axonscope.utils.units import cm2, mS, mV

class UnknownCurrentTerms(Model):
    model_kind = "unknown_current_terms"

    @currents(
        outputs=("I_l",),
        conductances={"I_missing": "g_l"},
        reversals={"I_missing": "EL"},
    )
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        g_l: ConductanceDensity = 0.1 * mS / cm2
        I_l: CurrentDensity = g_l * (Vm - EL)
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match="references unknown current output\\(s\\): I_missing",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_reports_explicit_terms_for_non_linear_current(tmp_path):
    source = tmp_path / "non_linear_current_terms.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import ConductanceDensity, CurrentDensity, Voltage
from axonscope.utils.units import cm2, mS, mV, uA

class NonLinearCurrentTerms(Model):
    model_kind = "non_linear_current_terms"

    @currents(outputs=("I_l",))
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        g_l: ConductanceDensity = 0.1 * mS / cm2
        offset: CurrentDensity = 0.0 * uA / cm2
        I_l: CurrentDensity = g_l * (Vm - EL) + offset
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match="Use the linear form `I_x = g_x \\* \\(Vm - E_x\\)` or declare",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_rejects_manifest_export_fields(tmp_path):
    source = tmp_path / "manifest_export.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import CurrentDensity, Voltage
from axonscope.utils.units import mV

class ManifestExport(Model):
    model_kind = "manifest_export"
    exports = {"currents": ("I_l",)}

    @currents
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        I_l: CurrentDensity = Vm - EL
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match=r"ManifestExport\.exports is no longer supported",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_reports_unsupported_helpers_with_source_location(tmp_path):
    source = tmp_path / "unsupported_helper.py"
    source.write_text(
        """
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import CurrentDensity, Voltage
from axonscope.utils.units import mV

class UnsupportedHelper(Model):
    model_kind = "unsupported_helper"

    @currents
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        I_l: CurrentDensity = unsupported_helper(Vm - EL)
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match=r"line \d+, column \d+: Unsupported equation helper 'unsupported_helper'",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


@pytest.mark.parametrize(
    ("body", "match"),
    (
        (
            """
I_l: CurrentDensity = Vm - EL
for _ in range(2):
    I_l = I_l
""",
            r"line \d+, column \d+: Data-dependent Python loops are not supported",
        ),
        (
            """
I_l: CurrentDensity = Vm - EL
if Vm > EL:
    I_l = I_l
""",
            r"line \d+, column \d+: Data-dependent Python if statements are not supported",
        ),
        (
            """
I_l: CurrentDensity = Vm - EL
self.cached = I_l
""",
            r"line \d+, column \d+: Mutation of attributes or indexed values is not supported",
        ),
        (
            """
I_l: CurrentDensity = Vm - EL
I_l += Vm - EL
""",
            r"line \d+, column \d+: Mutation and augmented assignments are not supported",
        ),
        (
            """
import numpy as np
I_l: CurrentDensity = Vm - EL
""",
            r"line \d+, column \d+: Imports inside membrane equation functions are not supported",
        ),
        (
            """
I_l: CurrentDensity = Vm - EL
print(I_l)
""",
            r"line \d+, column \d+: I/O and side-effecting calls like print",
        ),
    ),
)
def test_source_compiler_reports_rejected_statement_kinds(tmp_path, body, match):
    source = tmp_path / "bad_statement.py"
    indented = "\n".join(
        "        " + line if line else ""
        for line in body.strip("\n").splitlines()
    )
    source.write_text(
        f"""
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import CurrentDensity, Voltage
from axonscope.utils.units import mV

class BadStatement(Model):
    model_kind = "bad_statement"

    @currents
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
{indented}
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(SourceModelCompileError, match=match):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


@pytest.mark.parametrize(
    ("imports", "expression", "match"),
    (
        (
            "import numpy as np",
            "np.exp(Vm - EL)",
            r"line \d+, column \d+: Arbitrary NumPy/JAX calls are not supported",
        ),
        (
            "",
            "CurrentDensity(Vm - EL)",
            r"line \d+, column \d+: Object construction inside membrane equations is not supported",
        ),
        (
            "GLOBAL_OFFSET = 1.0",
            "Vm - EL + GLOBAL_OFFSET",
            r"line \d+, column \d+: Unknown symbol\(s\).*cannot read hidden globals",
        ),
    ),
)
def test_source_compiler_reports_rejected_expression_kinds(
    tmp_path,
    imports,
    expression,
    match,
):
    source = tmp_path / "bad_expression.py"
    source.write_text(
        f"""
{imports}
from axonscope.membranes.model import Model, currents
from axonscope.membranes.types import CurrentDensity, Voltage
from axonscope.utils.units import mV

class BadExpression(Model):
    model_kind = "bad_expression"

    @currents
    def currents(self, Vm: Voltage, EL: Voltage = -70.0 * mV):
        I_l: CurrentDensity = {expression}
        return I_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(SourceModelCompileError, match=match):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_source_compiler_supports_rates_from_tau_inf_tuple_assignment(tmp_path):
    source = tmp_path / "tau_inf_gate.py"
    source.write_text(
        """
from axonscope.membranes.math import rates_from_tau_inf
from axonscope.membranes.model import Model, currents, rates
from axonscope.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Time,
    Voltage,
)
from axonscope.utils.units import cm2, mS, ms, mV

class TauInfGate(Model):
    model_kind = "tau_inf_gate"

    gbar: ConductanceDensity = 1.0 * mS / cm2
    E: Voltage = -70.0 * mV

    @rates
    def rates(self, Vm: Voltage):
        m_inf: Dimensionless = 0.25
        tau_m: Time = 2.0 * ms
        alpha_m, beta_m = rates_from_tau_inf(m_inf, tau_m)
        self.keep(alpha_m, beta_m)

    @currents
    def currents(self, Vm: Voltage, m: Gate):
        g_l: ConductanceDensity = self.gbar * m
        I_l: CurrentDensity = g_l * (Vm - self.E)
        return I_l, g_l
""".lstrip(),
        encoding="utf-8",
    )

    compiled = compile_model_source_file(source, cache_root=tmp_path / "cache")
    generated = (compiled.cache.directory / "numpy_model.py").read_text(encoding="utf-8")

    assert "def rates_from_tau_inf(x_inf, tau):" in generated

    interpreter = NumpyModelInterpreter(compiled.model, dtype=np.float64)
    alpha, beta = interpreter.rate_constants(np.asarray([-65.0], dtype=np.float64))

    np.testing.assert_allclose(alpha[:, 0], [0.125], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(beta[:, 0], [0.375], rtol=1e-12, atol=1e-12)


def test_source_compiler_rejects_scalar_rates_from_tau_inf_use(tmp_path):
    source = tmp_path / "bad_tau_inf_gate.py"
    source.write_text(
        """
from axonscope.membranes.math import rates_from_tau_inf
from axonscope.membranes.model import Model, currents, rates
from axonscope.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Time,
    Voltage,
)
from axonscope.utils.units import cm2, mS, ms, mV

class BadTauInfGate(Model):
    model_kind = "bad_tau_inf_gate"

    gbar: ConductanceDensity = 1.0 * mS / cm2
    E: Voltage = -70.0 * mV

    @rates
    def rates(self, Vm: Voltage):
        m_inf: Dimensionless = 0.25
        tau_m: Time = 2.0 * ms
        alpha_m = rates_from_tau_inf(m_inf, tau_m)

    @currents
    def currents(self, Vm: Voltage, m: Gate):
        g_l: ConductanceDensity = self.gbar * m
        I_l: CurrentDensity = g_l * (Vm - self.E)
        return I_l, g_l
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceModelCompileError,
        match=r"rates_from_tau_inf\(\.\.\.\) returns alpha and beta",
    ):
        compile_model_source_file(source, cache_root=tmp_path / "cache")


def test_hh_plain_python_source_codegen_keeps_equations_executable(tmp_path):
    compiled = compile_model_source_file(HH_SOURCE, cache_root=tmp_path)

    assert compiled.model.name == "hodgkin_huxley"
    assert [state.name for state in compiled.model.states] == ["m", "h", "n"]
    assert [gate.state for gate in compiled.model.gates] == ["m", "h", "n"]
    assert compiled.cache.cache_hit is False

    numpy_source = (compiled.cache.directory / "numpy_model.py").read_text(encoding="utf-8")
    assert "ARG_NAMES = ('Vm', 'm', 'h', 'n', 'gnabar', 'gkbar', 'gl', 'el', 'ena', 'ek')" in numpy_source
    assert "OUTPUT_NAMES = ('I_na', 'I_k', 'I_l', 'g_na', 'g_k', 'g_l')" in numpy_source
    assert "def model_step(Vm, m, h, n, gnabar, gkbar, gl, el, ena, ek):" in numpy_source
    assert "alpha_m =" not in numpy_source

    spec = importlib.util.spec_from_file_location(
        "axonscope_test_hh_numpy_model",
        compiled.cache.directory / "numpy_model.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    interpreter = NumpyModelInterpreter(compiled.model, dtype=np.float64)
    V = -65.0
    gates = interpreter.init_gates([V])[0]
    params = {parameter.name: parameter.default for parameter in compiled.model.parameters}
    values = module.model_step(
        V,
        float(gates[0]),
        float(gates[1]),
        float(gates[2]),
        params["gnabar"],
        params["gkbar"],
        params["gl"],
        params["el"],
        params["ena"],
        params["ek"],
    )

    expected_currents = interpreter.current_matrix([V], gates.reshape(1, 3))[0]
    expected_conductances = interpreter.conductances(gates.reshape(1, 3))[0]
    np.testing.assert_allclose(values[:3], expected_currents, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(values[3:], expected_conductances, rtol=1e-12, atol=1e-12)


def test_tigerholm_source_exports_stateful_terms_without_return_soup(tmp_path):
    compiled = compile_model_source_file(TIGERHOLM_SOURCE, cache_root=tmp_path)
    model = compiled.model

    assert model.name == "tigerholm"
    assert compiled.function_name == "initials,nav17,nav18,nav19,ks,kf,kdr,hcn,currents,step"
    assert model.metadata["source_function"] == compiled.function_name
    assert model.metadata["display_name"] == "Tigerholm C-fiber"
    assert model.metadata["internal_outputs"] == (
        "i_na_dyn",
        "i_k_dyn",
        "total_outward_current",
        "explicit_outward_current",
        "correction_current",
    )
    assert model.metadata["states"]["nai"]["description"] == (
        "Intracellular sodium concentration."
    )
    spec = importlib.util.spec_from_file_location(
        "axonscope_test_tigerholm_source",
        TIGERHOLM_SOURCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_class = module.Tigerholm
    assert source_class.kind_name() == "tigerholm"
    assert tuple(
        getattr(getattr(source_class, name), "__axonscope_section__")
        for name in (
            "initials",
            "nav17",
            "nav18",
            "nav19",
            "ks",
            "kf",
            "kdr",
            "hcn",
            "currents",
            "step",
        )
    ) == (
        "initials",
        "mechanism:nav17",
        "mechanism:nav18",
        "mechanism:nav19",
        "mechanism:ks",
        "mechanism:kf",
        "mechanism:kdr",
        "mechanism:hcn",
        "currents",
        "step",
    )
    assert [current.name for current in model.currents] == [
        "I_na",
        "I_na",
        "I_na",
        "I_k",
        "I_k",
        "I_k",
        "I_k",
        "I_k",
    ]
    assert [observable.name for observable in model.observables] == ["g_na", "g_k", "w_kna"]
    state_names = [state.name for state in model.states]
    assert all(name in state_names for name in ("nai", "nao", "ki", "ko"))
    assert model.step_program is not None
    assert [update.state for update in model.step_program.prepare_state_updates] == [
        "nai",
        "nao",
        "ki",
        "ko",
    ]

    generated = (compiled.cache.directory / "numpy_model.py").read_text(encoding="utf-8")
    assert "def q10(base, celsius, reference):" in generated
    assert "def alpha_from_inf_tau(x_inf, tau):" in generated
    assert "def rates_from_tau_inf(x_inf, tau):" in generated
    assert "return I_na_nav17, I_na_nav18, I_na_nav19" in generated


def test_schild_sources_export_full_calcium_step_program(tmp_path):
    cases = (
        (SCHILD94_SOURCE, "schild94", 15),
        (SCHILD97_SOURCE, "schild97", 14),
    )
    for source, name, gate_count in cases:
        compiled = compile_model_source_file(source, cache_root=tmp_path)
        model = compiled.model
        assert model.name == name
        assert name + ".py" in model.metadata["source"]
        assert model.metadata["family"] == "schild"
        assert model.metadata["source_contract"] == "plain_python_membrane.v1"
        assert next(
            parameter for parameter in model.parameters if parameter.name == "diameter_um"
        ).quantity.unit == "micrometer"
        module = __import__(
            f"axonscope.membranes.models.{name}",
            fromlist=["Schild94" if name == "schild94" else "Schild97", "derive_parameters"],
        )
        source_class = getattr(module, "Schild94" if name == "schild94" else "Schild97")
        assert callable(module.derive_parameters)
        assert not hasattr(source_class, "parameter_defaults")
        assert [current.name for current in model.currents] == [
            "I_na",
            "I_ca",
            "I_na",
            "I_na",
            "I_k",
            "I_k",
            "I_k",
            "I_ca",
            "I_ca",
        ]
        assert [observable.name for observable in model.observables] == ["g_na", "g_k", "g_ca"]
        assert len(model.gates) == gate_count
        assert model.step_program is not None
        assert model.step_program.prepare_gate_source is LinearizationGateSource.PREVIOUS
        assert model.step_program.linearization_gate_source is LinearizationGateSource.PREVIOUS
        assert [update.state for update in model.step_program.prepare_state_updates] == [
            "cai",
            "Oc",
            "cao",
        ]
        assert [update.state for update in model.step_program.finalize_state_updates] == [
            "c_kca",
            "cai",
            "Oc",
            "cao",
        ]
        assert [diagnostic.name for diagnostic in model.step_program.diagnostics] == [
            "I_na_total_uAcm2",
            "I_k_total_uAcm2",
            "I_ca_total_uAcm2",
            "I_total_rhs_uAcm2",
        ]
        interpreter = NumpyModelInterpreter(model, dtype=np.float64)
        low, high = interpreter.init_membrane_state([-80.0, -40.0])[-1]
        assert low != pytest.approx(high)
        generated = (compiled.cache.directory / "numpy_model.py").read_text(encoding="utf-8")
        assert "def init_state(" in generated
        assert "def prepare_state(" in generated
        assert "def step_current_terms(" in generated
        assert "def finalize_state(" in generated
        assert "def diagnostics(" in generated
        assert "Vm_prev" in generated
        assert "I_ion" in generated
        assert "OUTPUT_NAMES" in generated


def test_schild_public_descriptors_compile_source_and_keep_dynamic_kca_initial():
    for public_model, expected_source in (
        (
            membranes.Schild94(
                diameter=0.8 * axs.um,
                temperature=36.0 * axs.degC,
                v_init=-50.0 * axs.mV,
            ),
            "schild94.py",
        ),
        (
            membranes.Schild97(
                diameter=0.8 * axs.um,
                temperature=36.0 * axs.degC,
                v_init=-50.0 * axs.mV,
            ),
            "schild97.py",
        ),
    ):
        model = lower_membrane_model_to_ir(public_model)
        assert expected_source in model.metadata["source"]
        assert model.metadata["source_contract"] == "plain_python_membrane.v1"
        c_kca = next(state for state in model.states if state.name == "c_kca")
        assert c_kca.initial is not None
        interpreter = NumpyModelInterpreter(model, dtype=np.float64)
        low, high = interpreter.init_membrane_state([-80.0, -40.0])[-1]

        assert low != pytest.approx(high)
        assert model.step_program is not None
        assert model.step_program.prepare_gate_source is LinearizationGateSource.PREVIOUS


def test_generated_schild_stateful_entrypoints_match_model_ir_lowering(tmp_path):
    compiled = compile_model_source_file(
        SCHILD97_SOURCE,
        cache_root=tmp_path,
        generated_targets=("jax",),
        load_generated_modules=("jax",),
    )
    generated = JaxMembraneProgram.from_model_ir(
        compiled.model,
        generated_module=compiled.cache.loaded_modules["jax"],
    )
    interpreted = JaxMembraneProgram.from_model_ir(compiled.model)
    V = jnp.asarray([-70.0, -45.0], dtype=jnp.float32)
    V_new = V + 0.25

    generated_gates = generated.init_gates(V)
    interpreted_gates = interpreted.init_gates(V)
    generated_state = generated.init_membrane_state(2, jnp.float32, V)
    interpreted_state = interpreted.init_membrane_state(2, jnp.float32, V)
    generated_ion = generated.currents(V, generated_gates, generated_state)
    interpreted_ion = interpreted.currents(V, interpreted_gates, interpreted_state)
    background = jnp.asarray([0.1, 0.2], dtype=jnp.float32)
    generated_plan = generated.prepare_membrane_step(
        V, generated_gates, generated_gates, generated_state, 0.01, generated_ion, background
    )
    interpreted_plan = interpreted.prepare_membrane_step(
        V,
        interpreted_gates,
        interpreted_gates,
        interpreted_state,
        0.01,
        interpreted_ion,
        background,
    )
    generated_final = generated.finalize_membrane_step(
        V, V_new, generated_gates, generated_gates, generated_state, generated_plan, 0.01
    )
    interpreted_final = interpreted.finalize_membrane_step(
        V, V_new, interpreted_gates, interpreted_gates, interpreted_state, interpreted_plan, 0.01
    )
    generated_diagnostics = generated.compute_step_diagnostics(
        V, V_new, generated_gates, generated_gates, generated_state,
        generated_final, generated_plan, generated_ion,
    )
    interpreted_diagnostics = interpreted.compute_step_diagnostics(
        V, V_new, interpreted_gates, interpreted_gates, interpreted_state,
        interpreted_final, interpreted_plan, interpreted_ion,
    )

    for actual, expected in zip(generated_state, interpreted_state, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    for actual, expected in zip(generated_plan.state, interpreted_plan.state, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    for actual, expected in (
        (generated_plan.total_outward_current, interpreted_plan.total_outward_current),
        (generated_plan.explicit_outward_current, interpreted_plan.explicit_outward_current),
        (generated_plan.correction_current, interpreted_plan.correction_current),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(generated_final, interpreted_final, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(
        generated_diagnostics, interpreted_diagnostics, strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_hh_model_ir_keeps_gates_currents_and_observables_visible():
    model = _source_model("hodgkin_huxley")

    assert [state.name for state in model.states] == ["m", "h", "n"]
    assert [gate.state for gate in model.gates] == ["m", "h", "n"]
    assert [current.name for current in model.currents] == ["I_na", "I_k", "I_l"]
    assert [observable.name for observable in model.observables] == ["g_na", "g_k", "g_l"]
    assert_valid_model_ir(model)


def test_rattay_model_ir_keeps_specific_rates_visible():
    model = _source_model("rattay_aberham")

    assert model.name == "rattay_aberham"
    assert [state.name for state in model.states] == ["m", "h", "n"]
    assert [current.name for current in model.currents] == ["I_na", "I_k", "I_l"]
    assert model.metadata["final_gate_update"] == "post_solve_voltage"
    assert_valid_model_ir(model)


def test_sundt_component_model_irs_keep_rates_visible():
    na_model = _source_model("na_hh")
    k_model = _source_model("borg_kdr")
    sundt = _source_model("sundt")

    assert [state.name for state in na_model.states] == ["m", "h"]
    assert [current.name for current in na_model.currents] == ["I_na"]
    assert [state.name for state in k_model.states] == ["n", "l"]
    assert [current.name for current in k_model.currents] == ["I_k"]
    assert sundt.name == "sundt"
    assert [state.name for state in sundt.states] == ["m", "h", "n", "l"]
    assert [current.name for current in sundt.currents] == ["I_na", "I_k", "I_l"]
    assert sundt.metadata["final_gate_update"] == "post_solve_voltage"
    assert "sundt.py" in sundt.metadata["source"]
    assert sundt.metadata["display_name"] == "Sundt composite membrane"
    assert_valid_model_ir(na_model)
    assert_valid_model_ir(k_model)
    assert_valid_model_ir(sundt)


def test_axnode_model_ir_keeps_rates_currents_and_observables_visible():
    model = _source_model("axnode")

    assert model.name == "axnode"
    assert [state.name for state in model.states] == ["mp", "m", "h", "s"]
    assert [current.name for current in model.currents] == ["I_nap", "I_na", "I_k", "I_l"]
    assert [observable.name for observable in model.observables] == [
        "g_nap",
        "g_na",
        "g_k",
        "g_l",
    ]
    assert "final_gate_update" not in model.metadata
    assert_valid_model_ir(model)


def test_membrane_descriptions_can_be_adapted_to_model_ir():
    assert lower_membrane_model_to_ir(membranes.Passive()).name == "passive"
    assert lower_membrane_model_to_ir(membranes.HodgkinHuxley()).name == "hodgkin_huxley"
    assert lower_membrane_model_to_ir(membranes.RattayAberham()).name == "rattay_aberham"
    assert lower_membrane_model_to_ir(membranes.Sundt()).name == "sundt"
    assert lower_membrane_model_to_ir(membranes.AxNode()).name == "axnode"
    assert lower_membrane_model_to_ir(membranes.Schild94(diameter=0.8 * axs.um)).name == "schild94"
    assert lower_membrane_model_to_ir(membranes.Schild97(diameter=0.8 * axs.um)).name == "schild97"

    custom = lower_membrane_model_to_ir(
        membranes.HodgkinHuxley(gl=1.0 * axs.mS_per_cm2, ek=-80.0 * axs.mV)
    )
    defaults = {parameter.name: parameter.default for parameter in custom.parameters}

    assert defaults["gl"] == pytest.approx(1.0)
    assert defaults["ek"] == pytest.approx(-80.0)


def test_descriptor_values_affect_parameterized_hash_not_structure():
    base = lower_membrane_model_to_ir(membranes.HodgkinHuxley(gl=0.3 * axs.mS_per_cm2))
    changed = lower_membrane_model_to_ir(membranes.HodgkinHuxley(gl=1.0 * axs.mS_per_cm2))

    assert structural_hash(base) == structural_hash(changed)
    assert parameterized_hash(base) != parameterized_hash(changed)

    passive_base = lower_membrane_model_to_ir(membranes.Passive(Rm=1e4))
    passive_changed = lower_membrane_model_to_ir(membranes.Passive(Rm=2e4))

    assert structural_hash(passive_base) == structural_hash(passive_changed)
    assert parameterized_hash(passive_base) != parameterized_hash(passive_changed)


def test_structural_hash_excludes_dynamic_parameter_values():
    model = _source_model("passive")
    changed_parameter = Parameter(
        model.parameters[0].name,
        model.parameters[0].quantity,
        variability=model.parameters[0].variability,
        default=2e4,
    )
    changed = replace(model, parameters=(changed_parameter, model.parameters[1]))

    assert structural_hash(model) == structural_hash(changed)
    assert parameterized_hash(model) != parameterized_hash(changed)


def test_validation_rejects_dimensional_intrinsic_arguments():
    Vm = symbol("Vm")
    model = ModelIR(
        name="bad_exp",
        inputs=(
            Input(
                "Vm",
                QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
            ),
        ),
        observables=(
            Observable(
                "bad",
                exp(Vm),
                QuantitySpec(unit=DIMENSIONLESS, role=SemanticRole.DIMENSIONLESS),
            ),
        ),
    )

    with pytest.raises(ModelValidationError, match="requires dimensionless"):
        assert_valid_model_ir(model)


def test_validation_rejects_unsupported_intrinsics():
    x = symbol("x")
    model = ModelIR(
        name="bad_intrinsic",
        inputs=(
            Input(
                "x",
                QuantitySpec(unit=DIMENSIONLESS, role=SemanticRole.DIMENSIONLESS),
            ),
        ),
        observables=(
            Observable(
                "bad",
                call("jax_exp", x),
                QuantitySpec(unit=DIMENSIONLESS, role=SemanticRole.DIMENSIONLESS),
            ),
        ),
    )

    with pytest.raises(ModelValidationError, match="unsupported intrinsic"):
        assert_valid_model_ir(model)


def test_validation_rejects_duplicate_recording_output_names():
    x = symbol("x")
    quantity = QuantitySpec(unit=DIMENSIONLESS, role=SemanticRole.DIMENSIONLESS)
    observable_model = ModelIR(
        name="duplicate_observable",
        inputs=(Input("x", quantity),),
        observables=(
            Observable("trace", x, quantity),
            Observable("trace", x, quantity),
        ),
    )

    with pytest.raises(ModelValidationError, match="duplicate observable name 'trace'"):
        assert_valid_model_ir(observable_model)

    diagnostic_model = ModelIR(
        name="duplicate_diagnostic",
        inputs=(Input("x", quantity),),
        step_program=StepProgram(
            diagnostics=(
                Diagnostic("trace", x, quantity),
                Diagnostic("trace", x, quantity),
            ),
        ),
    )

    with pytest.raises(ModelValidationError, match="duplicate diagnostic name 'trace'"):
        assert_valid_model_ir(diagnostic_model)


def test_validation_rejects_wrong_current_linearization_units():
    Vm = symbol("Vm")
    gbar = symbol("gbar")
    E = symbol("E")
    current = gbar * (Vm - E)
    model = ModelIR(
        name="bad_current_terms",
        inputs=(Input("Vm", QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE)),),
        parameters=(
            Parameter(
                "gbar",
                QuantitySpec(
                    unit=CONDUCTANCE_DENSITY_MS_CM2,
                    role=SemanticRole.CONDUCTANCE_DENSITY,
                ),
                default=0.3,
            ),
            Parameter(
                "E",
                QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
                default=-70.0,
            ),
        ),
        currents=(
            Current(
                "I_bad",
                current=current,
                conductance=E,
                reversal=gbar,
                quantity=QuantitySpec(
                    unit=CURRENT_DENSITY_UA_CM2,
                    role=SemanticRole.CURRENT_DENSITY,
                ),
            ),
        ),
    )

    with pytest.raises(
        ModelValidationError,
        match=r"current\.I_bad\.conductance.*expected 'mS/cm2'.*"
        r"current\.I_bad\.reversal.*expected 'mV'",
    ):
        assert_valid_model_ir(model)


def test_validation_rejects_inconsistent_source_outputs_metadata():
    model = _source_model("passive")
    metadata = dict(model.metadata)
    metadata["source_outputs"] = {
        "all": ("I_l", "g_l", "g_l"),
        "currents": ("I_l",),
        "observables": ("g_l", "g_l"),
    }

    with pytest.raises(
        ModelValidationError,
        match=r"metadata\.source_outputs\.observables.*duplicate source output name 'g_l'",
    ):
        assert_valid_model_ir(replace(model, metadata=metadata))


def test_validation_rejects_inconsistent_source_provenance_metadata():
    model = _source_model("passive")
    metadata = dict(model.metadata)
    provenance = dict(metadata["source_provenance"])
    provenance["source_hash"] = "not-the-source-hash"
    metadata["source_provenance"] = provenance

    with pytest.raises(
        ModelValidationError,
        match=r"metadata\.source_provenance\.source_hash.*does not match",
    ):
        assert_valid_model_ir(replace(model, metadata=metadata))


def test_validation_rejects_unknown_source_mechanism_metadata():
    model = _source_model("sundt")
    metadata = dict(model.metadata)
    metadata["source_mechanisms"] = (
        *metadata["source_mechanisms"],
        {
            "name": "missing",
            "function": "missing_rates",
            "assignments": (),
            "depends_on": (),
        },
    )

    with pytest.raises(
        ModelValidationError,
        match=r"metadata\.source_mechanisms\[2\].*unknown source mechanism 'missing'",
    ):
        assert_valid_model_ir(replace(model, metadata=metadata))


def test_validation_rejects_inconsistent_component_public_names_metadata():
    model = _source_model("passive")
    metadata = dict(model.metadata)
    metadata["component_public_names"] = {
        "observables": (("missing_g_l", "passive.g_l"),),
    }
    metadata["gate_trace_observables"] = ("missing_g_l",)

    with pytest.raises(
        ModelValidationError,
        match=r"component_public_names\.observables.*unknown observable 'missing_g_l'",
    ):
        assert_valid_model_ir(replace(model, metadata=metadata))


def test_numpy_interpreter_passive_membrane_primitives_are_analytic():
    model = _source_model("passive", {"Rm": 2e4, "EL": -65.0})
    interpreter = NumpyModelInterpreter(model)
    V = np.asarray([-80.0, -65.0, -40.0], dtype=np.float32)
    gates = interpreter.init_gates(V)
    expected_conductance = np.full((3, 1), 0.05, dtype=np.float32)
    expected_current = 0.05 * (V - (-65.0))

    np.testing.assert_allclose(gates, np.zeros((3, 0), dtype=np.float32))
    np.testing.assert_allclose(interpreter.conductances(gates), expected_conductance)
    np.testing.assert_allclose(interpreter.currents(V, gates), expected_current)


def test_jax_membrane_program_matches_numpy_interpreter_for_hh_and_rattay():
    V = np.linspace(-85.0, 35.0, 9, dtype=np.float32)
    cases = (
        membranes.HodgkinHuxley(),
        membranes.RattayAberham(),
    )

    for public_model in cases:
        model = lower_membrane_model_to_ir(public_model)
        interpreter = NumpyModelInterpreter(model)
        membrane = compile_membrane_model(public_model)
        assert isinstance(membrane, JaxMembraneProgram)
        V_jax = jnp.asarray(V, dtype=jnp.float32)
        gates_np = interpreter.init_gates(V)
        gates_jax = membrane.init_gates(V_jax)
        alpha_np, beta_np = interpreter.rate_constants(V)

        np.testing.assert_allclose(np.asarray(membrane.alpha_funcs(V_jax)), alpha_np, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(membrane.beta_funcs(V_jax)), beta_np, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(gates_jax), gates_np, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(membrane.conductances(gates_jax)),
            interpreter.conductances(gates_np),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(membrane.currents(V_jax, gates_jax)),
            interpreter.currents(V, gates_np),
            rtol=1e-6,
            atol=5e-6,
        )
        np.testing.assert_allclose(
            np.asarray(membrane.cn_gate_update(gates_jax, V_jax, 0.01)),
            interpreter.gate_update(gates_np, V, 0.01),
            rtol=1e-6,
        )


def test_jax_membrane_program_keeps_auxiliary_states_separate_from_gates():
    Vm = symbol("Vm")
    m = symbol("m")
    c = symbol("c")
    gbar = symbol("gbar")
    E = symbol("E")
    dynamic_E = E + literal(5.0, unit=VOLTAGE_MV) * c
    current = gbar * m * (Vm - dynamic_E)
    model = assert_valid_model_ir(
        ModelIR(
            name="aux_state_current",
            inputs=(Input("Vm", QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE)),),
            parameters=(
                Parameter(
                    "gbar",
                    QuantitySpec(
                        unit=CONDUCTANCE_DENSITY_MS_CM2,
                        role=SemanticRole.CONDUCTANCE_DENSITY,
                    ),
                    default=0.3,
                ),
                Parameter(
                    "E",
                    QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
                    default=-70.0,
                ),
            ),
            states=(
                State("m", QuantitySpec(unit=DIMENSIONLESS, role=SemanticRole.GATE)),
                State(
                    "c",
                    QuantitySpec(unit=DIMENSIONLESS, role=SemanticRole.DIMENSIONLESS),
                    initial=literal(2.0),
                ),
            ),
            gates=(
                Gate(
                    "m",
                    state="m",
                    alpha=literal(0.1, unit=RATE_PER_MS),
                    beta=literal(0.2, unit=RATE_PER_MS),
                ),
            ),
            currents=(
                Current(
                    "I_aux",
                    current=current,
                    conductance=gbar * m,
                    reversal=dynamic_E,
                    quantity=QuantitySpec(
                        unit=CURRENT_DENSITY_UA_CM2,
                        role=SemanticRole.CURRENT_DENSITY,
                    ),
                ),
            ),
            observables=(
                Observable(
                    "dynamic_E",
                    dynamic_E,
                    QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
                ),
            ),
        )
    )
    contract = derive_model_step_contract(model)
    interpreter = NumpyModelInterpreter(model)
    membrane = JaxMembraneProgram.from_model_ir(model)
    V = jnp.asarray([-70.0, -60.0], dtype=jnp.float32)
    gates = membrane.init_gates(V)
    state = membrane.init_membrane_state(2, jnp.float32, V)

    assert membrane.gate_names() == ("aux_state_current.m",)
    assert membrane.membrane_state_names() == ("aux_state_current.c",)
    np.testing.assert_allclose(np.asarray(gates), np.full((2, 1), 1.0 / 3.0))
    np.testing.assert_allclose(np.asarray(state[0]), np.full((2,), 2.0))

    expected_current = 0.3 * (1.0 / 3.0) * (np.asarray(V) - (-60.0))
    np.testing.assert_allclose(
        np.asarray(membrane.ionic_current_trace_matrix(V, gates, state)[:, 0]),
        expected_current,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(membrane.lowering.observable_matrix("dynamic_E", gates, state=state)),
        np.full((2,), -60.0, dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        interpreter.currents(np.asarray(V), np.asarray(gates), state=state),
        expected_current,
        rtol=1e-6,
    )


def test_model_ir_step_program_drives_prepare_finalize_and_diagnostics():
    Vm = symbol("Vm")
    m = symbol("m")
    bias = symbol("bias")
    gbar = symbol("gbar")
    E = symbol("E")
    I_aux = symbol("I_aux")
    I_ion = symbol("I_ion")
    I_background = symbol("I_background")
    current = gbar * m * (Vm - E)
    current_quantity = QuantitySpec(
        unit=CURRENT_DENSITY_UA_CM2,
        role=SemanticRole.CURRENT_DENSITY,
    )
    model = assert_valid_model_ir(
        ModelIR(
            name="stateful_step_program",
            inputs=(Input("Vm", QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE)),),
            parameters=(
                Parameter(
                    "gbar",
                    QuantitySpec(
                        unit=CONDUCTANCE_DENSITY_MS_CM2,
                        role=SemanticRole.CONDUCTANCE_DENSITY,
                    ),
                    default=0.3,
                ),
                Parameter(
                    "E",
                    QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
                    default=-70.0,
                ),
            ),
            states=(
                State("m", QuantitySpec(unit=DIMENSIONLESS, role=SemanticRole.GATE)),
                State(
                    "bias",
                    current_quantity,
                    initial=literal(0.0, unit=CURRENT_DENSITY_UA_CM2),
                ),
            ),
            gates=(
                Gate(
                    "m",
                    state="m",
                    alpha=literal(0.1, unit=RATE_PER_MS),
                    beta=literal(0.2, unit=RATE_PER_MS),
                ),
            ),
            currents=(
                Current(
                    "I_aux",
                    current=current,
                    conductance=gbar * m,
                    reversal=E,
                    quantity=current_quantity,
                ),
            ),
            step_program=StepProgram(
                prepare_state_updates=(StateUpdate("bias", I_aux),),
                finalize_state_updates=(
                    StateUpdate(
                        "bias",
                        literal(0.0, unit=CURRENT_DENSITY_UA_CM2),
                    ),
                ),
                total_outward_current=I_background + I_ion + bias,
                explicit_outward_current=I_background + bias,
                correction_current=bias * literal(0.25),
                linearization_gate_source=LinearizationGateSource.PREVIOUS,
                diagnostics=(Diagnostic("bias_current", bias, current_quantity),),
            ),
        )
    )
    contract = derive_model_step_contract(model)
    interpreter = NumpyModelInterpreter(model)
    membrane = JaxMembraneProgram.from_model_ir(model)
    V = jnp.asarray([-60.0, -50.0], dtype=jnp.float32)
    gates_prev = jnp.full((2, 1), 0.25, dtype=jnp.float32)
    gates_new = membrane.init_gates(V)
    state0 = membrane.init_membrane_state(2, jnp.float32, V)
    I_background_values = jnp.full((2,), 0.5, dtype=jnp.float32)
    I_ion_values = membrane.currents(V, gates_new, state0)

    plan = membrane.prepare_membrane_step(
        V,
        gates_prev,
        gates_new,
        state0,
        0.01,
        I_ion_values,
        I_background_values,
    )
    state_final = membrane.finalize_membrane_step(
        V,
        V + 1.0,
        gates_prev,
        gates_new,
        state0,
        plan,
        0.01,
    )
    diagnostics = membrane.compute_step_diagnostics(
        V,
        V + 1.0,
        gates_prev,
        gates_new,
        state0,
        state_final,
        plan,
        I_ion_values,
    )

    expected_current = np.asarray([1.0, 2.0], dtype=np.float32)
    assert contract.total_outward_current == "step.total_outward_current"
    assert contract.explicit_outward_current == "step.explicit_outward_current"
    assert contract.correction_current == "step.correction_current"
    assert contract.linearization_state == ("previous_gates",)
    assert contract.pruning.solver_output_names == ("I_aux",)
    np.testing.assert_allclose(np.asarray(I_ion_values), expected_current, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(plan.state[0]), expected_current, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(plan.linearization_gates), np.asarray(gates_prev))
    np.testing.assert_allclose(
        np.asarray(plan.total_outward_current),
        0.5 + expected_current,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(plan.explicit_outward_current),
        np.full((2,), 0.5, dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(plan.correction_current),
        np.zeros((2,), dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(state_final[0]), np.zeros((2,), dtype=np.float32))
    assert membrane.diagnostic_names() == ("bias_current",)
    np.testing.assert_allclose(np.asarray(diagnostics[0]), expected_current, rtol=1e-6)

    diagnostic_contract = derive_model_step_contract(
        model,
        record_gates=True,
        record_currents=True,
        record_state=True,
        record_diagnostics=True,
    )
    assert diagnostic_contract.pruning.retain_gates == ("m",)
    assert diagnostic_contract.pruning.retain_currents == ("I_aux",)
    assert diagnostic_contract.pruning.retain_state == ("m", "bias")
    assert diagnostic_contract.pruning.retain_recorded_state == ("bias",)
    assert diagnostic_contract.pruning.retain_diagnostics == ("bias_current",)
    assert diagnostic_contract.pruning.recording_output_names == (
        "gates.m",
        "currents.I_aux",
        "states.bias",
        "diagnostics.bias_current",
    )

    np_plan = interpreter.prepare_membrane_step(
        np.asarray(V),
        np.asarray(gates_prev),
        np.asarray(gates_new),
        tuple(np.asarray(value) for value in state0),
        0.01,
        np.asarray(I_ion_values),
        np.asarray(I_background_values),
    )
    np.testing.assert_allclose(np_plan.total_outward_current, plan.total_outward_current)
    np.testing.assert_allclose(np_plan.correction_current, plan.correction_current)


def test_model_ir_step_program_rejects_gate_state_updates():
    Vm = symbol("Vm")
    m = symbol("m")
    gbar = symbol("gbar")
    E = symbol("E")
    current_quantity = QuantitySpec(
        unit=CURRENT_DENSITY_UA_CM2,
        role=SemanticRole.CURRENT_DENSITY,
    )

    with pytest.raises(ModelValidationError, match="cannot update gate state"):
        assert_valid_model_ir(
            ModelIR(
                name="bad_gate_update",
                inputs=(
                    Input(
                        "Vm",
                        QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
                    ),
                ),
                parameters=(
                    Parameter(
                        "gbar",
                        QuantitySpec(
                            unit=CONDUCTANCE_DENSITY_MS_CM2,
                            role=SemanticRole.CONDUCTANCE_DENSITY,
                        ),
                        default=0.3,
                    ),
                    Parameter(
                        "E",
                        QuantitySpec(unit=VOLTAGE_MV, role=SemanticRole.VOLTAGE),
                        default=-70.0,
                    ),
                ),
                states=(State("m", QuantitySpec(unit=DIMENSIONLESS, role=SemanticRole.GATE)),),
                gates=(
                    Gate(
                        "m",
                        state="m",
                        alpha=literal(0.1, unit=RATE_PER_MS),
                        beta=literal(0.2, unit=RATE_PER_MS),
                    ),
                ),
                currents=(
                    Current(
                        "I_aux",
                        current=gbar * m * (Vm - E),
                        conductance=gbar * m,
                        reversal=E,
                        quantity=current_quantity,
                    ),
                ),
                step_program=StepProgram(
                    prepare_state_updates=(StateUpdate("m", literal(0.0)),),
                ),
            )
        )


def test_jax_membrane_program_sundt_primitives_are_well_formed():
    public_model = membranes.Sundt()
    membrane = compile_membrane_model(public_model)
    V = jnp.linspace(-90.0, 30.0, 11, dtype=jnp.float32)

    assert isinstance(membrane, JaxMembraneProgram)
    assert membrane.gate_names() == ("sundt.m", "sundt.h", "sundt.n", "sundt.l")
    assert membrane.current_names() == ("I_na", "I_k", "I_l")
    assert membrane.conductance_names() == ("g_na", "g_k", "g_l")

    gates = membrane.init_gates(V)
    assert gates.shape == (11, 4)
    assert membrane.g_bar.shape == (3,)
    assert membrane.E_rev.shape == (3,)
    np.testing.assert_allclose(np.asarray(membrane.g_bar), [40.0, 40.0, 0.1], rtol=1e-6)
    assert np.isfinite(np.asarray(membrane.alpha_funcs(V))).all()
    assert np.isfinite(np.asarray(membrane.beta_funcs(V))).all()
    assert np.isfinite(np.asarray(membrane.conductances(gates))).all()
    assert np.isfinite(np.asarray(membrane.currents(V, gates))).all()
    assert np.isfinite(np.asarray(membrane.cn_gate_update(gates, V, 0.01))).all()


def test_jax_membrane_program_axnode_primitives_are_well_formed():
    membrane = compile_membrane_model(membranes.AxNode())
    V = jnp.linspace(-100.0, 40.0, 13, dtype=jnp.float32)

    assert isinstance(membrane, JaxMembraneProgram)
    assert membrane.gate_names() == ("axnode.mp", "axnode.m", "axnode.h", "axnode.s")
    assert membrane.current_names() == ("I_nap", "I_na", "I_k", "I_l")
    assert membrane.conductance_names() == ("g_nap", "g_na", "g_k", "g_l")
    assert membrane.supports_stateless_vm_only_fast_path()

    gates = membrane.init_gates(V)
    assert gates.shape == (13, 4)
    assert membrane.g_bar.shape == (4,)
    assert membrane.E_rev.shape == (4,)
    assert np.isfinite(np.asarray(membrane.alpha_funcs(V))).all()
    assert np.isfinite(np.asarray(membrane.beta_funcs(V))).all()
    assert np.isfinite(np.asarray(membrane.conductances(gates))).all()
    assert np.isfinite(np.asarray(membrane.currents(V, gates))).all()
    assert np.isfinite(np.asarray(membrane.cn_gate_update(gates, V, 0.01))).all()


def test_compile_membrane_model_uses_dsl_for_covered_public_builtins():
    for model in (
        membranes.Passive(),
        membranes.HodgkinHuxley(),
        membranes.RattayAberham(),
        membranes.Sundt(),
        membranes.AxNode(),
        membranes.Tigerholm(diameter=1.0 * axs.um),
        membranes.Schild94(diameter=0.8 * axs.um),
        membranes.Schild97(diameter=0.8 * axs.um),
    ):
        compiled = compile_membrane_model(model)
        assert isinstance(compiled, JaxMembraneProgram)
        assert compiled.uses_generated_model_step


def test_compile_membrane_model_reports_source_cache_status_to_benchmark(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AXONSCOPE_MODEL_CODEGEN_CACHE", str(tmp_path / "codegen"))
    public_model = MembraneModel("passive", {})
    axs.enable_benchmark(tmp_path / "benchmark", print_summary=False, save=False)
    try:
        with benchmark_span("compile_membrane"):
            compile_membrane_model(public_model)
        with benchmark_span("compile_membrane"):
            compile_membrane_model(public_model)
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert report is not None
    events = [event for event in report.events if event.name == "compile_membrane"]
    assert [event.metadata["membrane_source_cache"] for event in events] == [
        "miss",
        "hit",
    ]
    assert [event.metadata["membrane_source_cache_reasons"] for event in events] == [
        "manifest_missing",
        "manifest_match",
    ]
    assert all(event.metadata["membrane_source_kind"] == "passive" for event in events)
    assert all(event.metadata["membrane_source_count"] == 1 for event in events)
    assert all(
        event.metadata["membrane_source_generated_module_policy"]
        == "single_source_loaded"
        for event in events
    )
    assert all(event.metadata["membrane_source_loaded_targets"] == ["jax"] for event in events)
    assert events[0].metadata["membrane_source_cache_keys"] == events[1].metadata[
        "membrane_source_cache_keys"
    ]


def test_compile_membrane_model_returns_direct_jax_program_contract():
    model = lower_membrane_model_to_ir(membranes.HodgkinHuxley())
    runtime = JaxMembraneProgram.from_model_ir(model)
    membrane = compile_membrane_model(membranes.HodgkinHuxley())
    V = jnp.linspace(-80.0, 20.0, 5, dtype=jnp.float32)

    assert isinstance(membrane, JaxMembraneProgram)
    assert membrane.static_signature() == runtime.static_signature()
    np.testing.assert_allclose(
        np.asarray(membrane.init_gates(V)),
        np.asarray(runtime.init_gates(V)),
        rtol=1e-6,
    )


def test_jax_membrane_program_caches_derived_static_values():
    membrane = compile_membrane_model(membranes.Passive(Rm=12_000.0, EL=-68.0))

    g_bar = membrane.g_bar
    e_rev = membrane.E_rev
    states = membrane.membrane_state_specs()
    signature = membrane.static_signature()

    assert membrane.g_bar is g_bar
    assert membrane.E_rev is e_rev
    assert membrane.membrane_state_specs() is states
    assert membrane.static_signature() is signature
    assert membrane.g_bar is g_bar
    assert membrane.E_rev is e_rev


def test_membrane_backends_consume_jax_program_directly():
    hh = JaxMembraneProgram.from_model_ir(
        lower_membrane_model_to_ir(membranes.HodgkinHuxley())
    )
    passive = JaxMembraneProgram.from_model_ir(
        lower_membrane_model_to_ir(membranes.Passive())
    )

    uniform = UniformMembraneBackend.from_model(hh, nx=3)
    heterogeneous = HeterogeneousMembraneBackend.from_models([hh, passive, passive])

    assert isinstance(uniform.ion_channel, JaxMembraneProgram)
    assert uniform.ion_channel is hh
    assert all(isinstance(model, JaxMembraneProgram) for model in heterogeneous.membrane_models)
    assert heterogeneous.membrane_models[0] is hh
    assert heterogeneous.membrane_models[1] is passive


def test_composite_of_model_ir_components_compiles_without_legacy_composite():
    public_model = membranes.Composite(
        [
            membranes.RattayAberham(),
            membranes.Passive(Rm=1000.0, EL=-70.0),
        ]
    )
    membrane = compile_membrane_model(public_model)
    V = jnp.linspace(-85.0, 20.0, 7, dtype=jnp.float32)

    assert isinstance(membrane, JaxMembraneProgram)
    assert not membrane.uses_generated_model_step
    assert membrane.gate_names() == (
        "rattay_aberham.m",
        "rattay_aberham.h",
        "rattay_aberham.n",
    )
    assert membrane.current_names() == ("I_na", "I_k", "I_l")
    assert membrane.conductance_names() == ("g_na", "g_k", "g_l")

    gates = membrane.init_gates(V)
    assert gates.shape == (7, 3)
    assert membrane.g_bar.shape == membrane.E_rev.shape
    assert membrane.g_bar.shape[0] >= len(membrane.conductance_names())
    assert np.isfinite(np.asarray(membrane.conductances(gates))).all()
    assert np.isfinite(np.asarray(membrane.currents(V, gates))).all()
    assert np.isfinite(np.asarray(membrane.conductance_trace_matrix(gates))).all()
    assert np.isfinite(np.asarray(membrane.ionic_current_trace_matrix(V, gates))).all()
