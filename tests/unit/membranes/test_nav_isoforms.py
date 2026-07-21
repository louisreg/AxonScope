from __future__ import annotations

import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest

import axonscope as axs
from axonscope.axons import Axon, Layout, LayoutElement, Section
from axonscope.membranes.models.nav_balbi import BalbiNav
from axonscope.runtime.jax.membranes.compile import (
    compile_axon_membrane,
    compile_membrane_model,
)
from axonscope.runtime.jax.membranes.program import is_jax_membrane_program_kind
from axonscope.runtime.solver_axon import build_solver_axon


NAV_MODELS = (
    axs.membranes.Nav11,
    axs.membranes.Nav12,
    axs.membranes.Nav13,
    axs.membranes.Nav14,
    axs.membranes.Nav15,
    axs.membranes.Nav16,
    axs.membranes.Nav17,
    axs.membranes.Nav18,
    axs.membranes.Nav19,
)

# SHA-256 over the 51 normalized ModelDB parameters, sorted by name. These
# references were extracted from ModelDB 230137 Nav11_a.mod through Nav19_a.mod.
MODELDB_PARAMETER_DIGESTS = {
    "nav11": "f3393c97872cee38fed2ced7df751b26c73968ee679a02078af59d63ea90b40d",
    "nav12": "c9879e64888437daa9ea812d1291bfb258d52b1168bcb9591c0af5c46c44e01d",
    "nav13": "8407e29dc657307f3cd8c15fcd27ca79e9b43cf29036ce674622e3d249bfda5e",
    "nav14": "09f0bda60e8855778faeab45d51e4e1f0682ce68e9eb8fb72ad17dff22d25a19",
    "nav15": "bbd60c459e8bd51643dc3d4b40622c93db530098c810fa6a3b2e7204abe6ee86",
    "nav16": "97a085ab12be4363d709dc8f82e84fa99d5f4f3f2a6290ea16f0c64f8ac6b443",
    "nav17": "a0a8f1d06d4b560cc0650f9e32ef9f5dc4ce63fb84b4130da7a7f60dce09328c",
    "nav18": "6f111616be8a7f639dd5a56f851b086bb463c8d443fb55be2ae93537457e23c0",
    "nav19": "22a626f403679251f9e2a75a255ebd5eec57e658da21be4c66a78725eb2d6885",
}

TRANSITIONS = (
    "C1C2",
    "C2C1",
    "C2O1",
    "O1C2",
    "C2O2",
    "O2C2",
    "O1I1",
    "I1O1",
    "I1C1",
    "C1I1",
    "I1I2",
    "I2I1",
)


def _parameter_digest(values: dict[str, object]) -> str:
    payload = json.dumps(
        sorted((name, float(value)) for name, value in values.items()),
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _modeldb_rates(
    values: dict[str, object], voltage_mV: np.ndarray
) -> np.ndarray:
    q10 = 3.0 ** ((float(values["celsius"]) - 20.0) / 10.0)
    rates = []
    for transition in TRANSITIONS:
        rate = np.zeros_like(voltage_mV, dtype=np.float64)
        for suffix in ("1", "2"):
            b_name = f"{transition}b{suffix}"
            if b_name not in values:
                continue
            v = float(values[f"{transition}v{suffix}"])
            k = float(values[f"{transition}k{suffix}"])
            rate += float(values[b_name]) / (1.0 + np.exp((voltage_mV - v) / k))
        rates.append(q10 * rate)
    return np.stack(rates, axis=-1)


def _generated_rates(program, voltage_mV: np.ndarray) -> np.ndarray:
    spec = program.generated_contract.function("kinetic_terms")
    values = program.parameter_values
    args = {
        "Vm": voltage_mV,
        **{name: np.asarray(value) for name, value in values.items()},
    }
    outputs = program._host_module.kinetic_terms(*(args[name] for name in spec.args))
    return np.stack(outputs, axis=-1)


@pytest.mark.parametrize("model_class", NAV_MODELS)
def test_nav_isoform_parameters_match_modeldb_230137(model_class):
    program = compile_membrane_model(model_class())

    assert len(program.parameter_values) == 51
    assert _parameter_digest(program.parameter_values) == MODELDB_PARAMETER_DIGESTS[
        model_class.kind_name()
    ]


def test_nav_isoforms_share_one_generated_source_artifact():
    programs = [compile_membrane_model(model_class()) for model_class in NAV_MODELS]

    assert {model_class.source_path() for model_class in NAV_MODELS} == {
        BalbiNav.source_path()
    }
    assert {model_class.source_class() for model_class in NAV_MODELS} == {"BalbiNav"}
    assert len({program.codegen_cache["key"] for program in programs}) == 1
    assert {program.model_name for program in programs} == {
        model_class.kind_name() for model_class in NAV_MODELS
    }
    for model_class, program in zip(NAV_MODELS, programs, strict=True):
        assert is_jax_membrane_program_kind(program, model_class.kind_name())
        assert program.gate_names() == tuple(
            f"{model_class.kind_name()}.{state}"
            for state in ("C1", "C2", "O1", "O2", "I1", "I2")
        )


def test_nav_explanation_keeps_public_identity_and_shared_source_provenance():
    report = axs.membranes.Nav16().explain()

    assert report.model_kind == "nav16"
    assert report.recording_outputs.gates == tuple(
        f"nav16.{state}" for state in ("C1", "C2", "O1", "O2", "I1", "I2")
    )
    assert report.recording_outputs.observables == (
        "nav16.g_na",
        "nav16.open_probability",
    )
    assert len(report.sources) == 1
    assert report.sources[0].model_name == "balbi_nav"


def test_nav_isoform_uses_generic_axon_composition_and_runtime_path():
    membrane = axs.membranes.Composite(
        {
            "sodium": axs.membranes.Nav16(),
            "leak": axs.membranes.Passive(),
        }
    )
    axon = Axon(
        layout=Layout(
            [
                LayoutElement(
                    Section(
                        "axon",
                        membrane=membrane,
                        diameter=1.0 * axs.um,
                    ),
                    length=100.0 * axs.um,
                    compartments=3,
                )
            ]
        )
    )

    compiled = compile_axon_membrane(axon, solver_axon=build_solver_axon(axon))

    assert compiled.model_name == "composite"
    assert compiled.gate_names() == tuple(
        f"sodium.{state}" for state in ("C1", "C2", "O1", "O2", "I1", "I2")
    )
    assert compiled.current_names() == ("I_na", "I_l")


@pytest.mark.parametrize("model_class", NAV_MODELS)
def test_nav_generated_rates_match_modeldb_equations(model_class):
    program = compile_membrane_model(model_class())
    voltage_mV = np.asarray([-120.0, -80.0, -40.0, 0.0, 40.0])

    np.testing.assert_allclose(
        _generated_rates(program, voltage_mV),
        _modeldb_rates(program.parameter_values, voltage_mV),
        rtol=2e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize("model_class", NAV_MODELS)
def test_nav_stationary_initialization_and_updates_conserve_probability(model_class):
    program = compile_membrane_model(model_class())
    voltage_mV = np.asarray([-100.0, -80.0, -40.0, 20.0], dtype=np.float32)
    initial = program.init_gates_host(voltage_mV, dtype_local=np.dtype(np.float32))
    updated = np.asarray(
        program.cn_gate_update(
            jnp.asarray(initial),
            jnp.asarray(voltage_mV),
            0.005,
        )
    )

    assert np.all(initial >= 0.0)
    assert np.all(updated >= -2e-6)
    np.testing.assert_allclose(initial.sum(axis=1), 1.0, atol=2e-6)
    np.testing.assert_allclose(updated.sum(axis=1), 1.0, atol=2e-6)
