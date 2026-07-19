from __future__ import annotations

import numpy as np
import pytest

import axonscope as axs
from axonscope.preparation.axon_rows import MaterializedAxonRows
from axonscope.runtime.host_preparation import (
    cable_runtime_rows_numpy,
    compartment_area_cm2_numpy,
    diffusion_operator_coeffs_numpy,
    pad_space_array_numpy,
)
from axonscope.runtime.solver_axon import build_solver_axon


def test_materialized_rows_deduplicate_shared_descriptions():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=5,
    )
    solver = build_solver_axon(axon)

    rows = MaterializedAxonRows.from_solver_axons((solver, solver, solver))

    assert rows.size == 3
    assert rows.template_count == 1
    assert rows.translated_row_count == 0
    assert rows.nx == 5
    np.testing.assert_array_equal(rows.row_template_indices, [0, 0, 0])
    np.testing.assert_array_equal(rows.row_x_shifts_um, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rows.x_positions_m, rows.x_um[[0, 0, 0]] * 1e-6)
    assert not rows.x_um.flags.writeable


def test_materialized_rows_pad_heterogeneous_layouts_with_mask():
    short = build_solver_axon(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=3,
        )
    )
    long = build_solver_axon(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.8 * axs.um,
            compartments=5,
        )
    )

    rows = MaterializedAxonRows.from_solver_axons((short, long), target_nx=5)

    assert rows.template_count == 2
    np.testing.assert_array_equal(rows.template_nx, [3, 5])
    np.testing.assert_array_equal(rows.valid_mask.sum(axis=1), [3, 5])
    assert rows.x_um.shape == (2, 5)
    assert rows.h_cm.shape == (2, 4)
    assert rows.x_um[0, -1] == rows.x_um[0, 2]


def test_materialized_rows_accept_shifted_double_cable_and_stateful_models():
    shifted = build_solver_axon(
        axs.axons.MRG(
            diameter=10.0 * axs.um,
            nodes=5,
            x_shift=80.0 * axs.um,
        )
    )
    stateful = build_solver_axon(
        axs.axons.Schild94(
            length=100.0 * axs.um,
            diameter=1.0 * axs.um,
            compartments=5,
        )
    )

    rows = MaterializedAxonRows.from_solver_axons((shifted, stateful))

    assert rows.template_count == 2
    assert rows.formulations == ("double-cable", "single-cable")
    assert rows.membrane_models[0]
    assert rows.membrane_models[1]


def test_materialized_rows_share_explicit_layout_translations():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(),
        diameter=1.0 * axs.um,
    )
    base_layout = axs.axons.Layout.single_uniform(
        section,
        length=100.0 * axs.um,
        compartments=17,
    )
    base = build_solver_axon(axs.axons.Axon(layout=base_layout))
    shifted = build_solver_axon(
        axs.axons.Axon(layout=base_layout.with_x_shift(25.0 * axs.um))
    )

    rows = MaterializedAxonRows.from_solver_axons((base, shifted))

    assert rows.template_count == 1
    assert rows.translated_row_count == 1
    np.testing.assert_array_equal(rows.row_template_indices, [0, 0])
    np.testing.assert_allclose(rows.row_x_shifts_um, [0.0, 25.0])
    np.testing.assert_allclose(rows.x_positions_m[0], base.x_um * 1e-6)
    np.testing.assert_allclose(rows.x_positions_m[1], shifted.x_um * 1e-6)


def test_materialized_rows_share_heterogeneous_custom_layout_translations():
    proximal = axs.axons.Section(
        "proximal",
        membrane=axs.membranes.Passive(Rm=1.0e4 * axs.ohm * axs.cm**2),
        diameter=1.0 * axs.um,
    )
    distal = axs.axons.Section(
        "distal",
        membrane=axs.membranes.Passive(Rm=2.0e4 * axs.ohm * axs.cm**2),
        diameter=0.8 * axs.um,
    )
    base_layout = axs.axons.Layout.sequence(
        [proximal, distal],
        section_lengths=np.asarray([40.0, 60.0]) * axs.um,
        compartments=[2, 3],
        lengths=100.0 * axs.um,
    )
    base = build_solver_axon(axs.axons.Axon(layout=base_layout))
    shifted = build_solver_axon(
        axs.axons.Axon(layout=base_layout.with_x_shift(-17.0 * axs.um))
    )

    rows = MaterializedAxonRows.from_solver_axons((base, shifted))

    assert rows.template_count == 1
    assert rows.translated_row_count == 1
    np.testing.assert_allclose(rows.row_x_shifts_um, [0.0, -17.0])
    np.testing.assert_allclose(rows.x_positions_m[1], shifted.x_um * 1e-6)


def test_materialized_rows_do_not_share_mrg_phase_shifts():
    base = build_solver_axon(axs.axons.MRG(diameter=10.0 * axs.um, nodes=5))
    phase_shifted = build_solver_axon(
        axs.axons.MRG(
            diameter=10.0 * axs.um,
            nodes=5,
            x_shift=80.0 * axs.um,
        )
    )

    rows = MaterializedAxonRows.from_solver_axons((base, phase_shifted))

    assert rows.template_count == 2
    np.testing.assert_array_equal(rows.row_template_indices, [0, 1])
    np.testing.assert_array_equal(rows.row_x_shifts_um, [0.0, 0.0])


def test_materialized_rows_do_not_share_translations_with_cable_overrides():
    base = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5)
    translated = axs.axons.Axon(
        layout=base.layout.with_x_shift(25.0 * axs.um),
        formulation=base.formulation,
        diameter=base.diameter * axs.um,
    )
    base_instance = axs.AxonInstance(base)
    translated_instance = axs.AxonInstance(translated)
    translated_instance.set_extracellular_layer(
        xg_S_per_cm2=np.full((translated.n_compartments,), 1e-3),
    )

    rows = MaterializedAxonRows.from_solver_axons(
        (
            build_solver_axon(base_instance),
            build_solver_axon(translated_instance),
        )
    )

    assert rows.template_count == 2
    assert rows.translated_row_count == 0


def test_materialized_rows_reject_too_small_target_width():
    solver = build_solver_axon(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=5,
        )
    )

    with pytest.raises(ValueError, match="largest row"):
        MaterializedAxonRows.from_solver_axons((solver,), target_nx=4)


def test_vectorized_cable_lowering_matches_individual_solver_rows():
    solvers = (
        build_solver_axon(
            axs.axons.HodgkinHuxley(
                length=100.0 * axs.um,
                diameter=0.5 * axs.um,
                compartments=3,
            )
        ),
        build_solver_axon(
            axs.axons.HodgkinHuxley(
                length=200.0 * axs.um,
                diameter=0.8 * axs.um,
                compartments=5,
            )
        ),
        build_solver_axon(
            axs.axons.MRG(
                diameter=10.0 * axs.um,
                nodes=5,
                x_shift=80.0 * axs.um,
            )
        ),
    )
    rows = MaterializedAxonRows.from_solver_axons(solvers)
    lowered = cable_runtime_rows_numpy(rows, dtype=np.dtype("float64"), include_area=True)

    for index, solver in enumerate(solvers):
        expected_lower, expected_diag, expected_upper = (
            diffusion_operator_coeffs_numpy(solver, dtype=np.dtype("float64"))
        )
        expected_area = compartment_area_cm2_numpy(
            solver,
            dtype=np.dtype("float64"),
        )
        for actual, expected, mode in (
            (lowered.lower[index], expected_lower, "zero"),
            (lowered.diag[index], expected_diag, "zero"),
            (lowered.upper[index], expected_upper, "zero"),
            (lowered.area_cm2[index], expected_area, "edge"),
        ):
            np.testing.assert_allclose(
                actual,
                pad_space_array_numpy(expected, target_nx=rows.nx, mode=mode),
                rtol=1e-12,
                atol=1e-12,
            )
