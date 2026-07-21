from __future__ import annotations

import axonscope as axs
from axonscope.axons.unmyelinated import Schild94, Schild97, Tigerholm
from axonscope.runtime.jax.membranes.compile import compile_membrane_model
from axonscope.runtime.jax.membranes.program import is_jax_membrane_program_kind


def test_stateful_public_membranes_compile_to_generated_runtime():
    cases = (
        (
            lambda: Schild94(
                length=300.0 * axs.um,
                diameter=0.8 * axs.um,
                compartments=7,
            ),
            "schild94",
        ),
        (
            lambda: Schild97(
                length=300.0 * axs.um,
                diameter=0.8 * axs.um,
                compartments=7,
            ),
            "schild97",
        ),
        (
            lambda: Tigerholm(length=300.0 * axs.um, diameter=1.0 * axs.um, compartments=9),
            "tigerholm",
        ),
    )
    for factory, kind in cases:
        ax = factory()

        membrane = compile_membrane_model(ax.layout.sections[0].membrane)

        assert is_jax_membrane_program_kind(membrane, kind)
        assert membrane.model_ir is None
        assert membrane.membrane_state_specs()
        assert membrane.generated_contract is not None
        assert membrane.generated_contract.has_step_program
        assert (
            membrane.generated_contract.prepare_state_update_names
            or membrane.generated_contract.finalize_state_update_names
        )
        assert set(membrane.generated_contract.functions) == {
            "init_state",
            "gate_terms",
            "kinetic_terms",
            "kinetic_initials",
            "membrane_terms",
            "reversal_terms",
            "model_step",
            "prepare_state",
            "step_current_terms",
            "finalize_state",
            "diagnostics",
        }


def test_stateless_public_membrane_cache_miss_does_not_retain_model_ir(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AXONSCOPE_MODEL_CODEGEN_CACHE", str(tmp_path / "codegen"))

    membrane = compile_membrane_model(axs.membranes.HodgkinHuxley())

    assert membrane.generated_contract is not None
    assert membrane.model_ir is None
    assert membrane.uses_generated_model_step


def test_stateful_dynamics_stay_out_of_axon_templates():
    ax = Tigerholm(length=300.0 * axs.um, diameter=1.0 * axs.um, compartments=9)
    schild = Schild97(length=300.0 * axs.um, diameter=0.8 * axs.um, compartments=7)

    assert not hasattr(ax, "compute_I_Na_dyn")
    assert not hasattr(ax, "compute_I_K_dyn")
    assert not hasattr(ax, "dynamics_correction")
    assert not hasattr(schild, "init_c_kca")
    assert not hasattr(schild, "compute_I_Ca_budget")
    assert not hasattr(schild, "dynamics_correction_ca")
