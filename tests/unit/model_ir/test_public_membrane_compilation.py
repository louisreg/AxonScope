from __future__ import annotations

import axonscope as axs
from axonscope.axons.unmyelinated import Schild94, Schild97, Tigerholm
from axonscope.backends.jax.membrane_program import is_jax_membrane_program_kind
from axonscope.backends.jax.runtime import compile_membrane_model


def test_stateful_public_membranes_compile_through_model_ir():
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
        assert membrane.membrane_state_specs()


def test_stateful_dynamics_stay_out_of_axon_templates():
    ax = Tigerholm(length=300.0 * axs.um, diameter=1.0 * axs.um, compartments=9)
    schild = Schild97(length=300.0 * axs.um, diameter=0.8 * axs.um, compartments=7)

    assert not hasattr(ax, "compute_I_Na_dyn")
    assert not hasattr(ax, "compute_I_K_dyn")
    assert not hasattr(ax, "dynamics_correction")
    assert not hasattr(schild, "init_c_kca")
    assert not hasattr(schild, "compute_I_Ca_budget")
    assert not hasattr(schild, "dynamics_correction_ca")
