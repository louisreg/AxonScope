import numpy as np
import pytest

import axonscope as axs


def test_public_unmyelinated_template_and_simulate():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonSimulation(axon)
    sim.add_current_clamp(
        position_um=50.0,
        current=axs.Stimulus.pulse(start=0.02, duration=0.02, amplitude=0.5),
    )

    result = axs.simulate(sim, duration_ms=0.1, dt_ms=0.05)

    assert result.Vm.shape == (2, 11)
    assert np.asarray(result.t).shape == (2,)
    assert isinstance(axon, axs.axons.Unmyelinated)
    assert axon.resolved_formulation == "single-cable"
    assert result.axon is axon
    assert result.simulation is sim


def test_public_axon_is_descriptive_and_simulation_owns_protocol():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    assert not hasattr(axon, "add_current_clamp")
    assert not hasattr(axon, "add_extracellular_context")
    assert not hasattr(axon, "set_position")
    assert not hasattr(axon, "plot_geometry")
    assert callable(axon.layout.plot)

    sim = axs.AxonSimulation(axon, y_um=20.0 * axs.um)
    clamp = axs.IntracellularCurrentClamp(
        position_um=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.02 * axs.ms,
            amplitude=0.5 * axs.nA,
        ),
    )
    sim.add_intracellular_context(context=clamp)

    assert sim.axon is axon
    assert sim.n_compartments == axon.n_compartments
    assert sim.y_um == pytest.approx(20.0)
    assert len(sim.intracellular_clamps) == 1
    assert sim.intracellular_contexts[0] is clamp


def test_public_simulate_rejects_partial_final_time_step():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(ValueError, match="integer multiple"):
        axs.simulate(axon, duration_ms=0.1, dt_ms=0.03)


def test_public_recording_full_requests_observables():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    result = axs.simulate(
        axon,
        duration_ms=0.1,
        dt_ms=0.05,
        recording=axs.Recording.full(),
    )

    assert result.recordings is not None
    assert "Vm" in result.recordings
    assert "gates" in result.recordings
    assert "currents" in result.recordings


def test_public_recording_variables_filter_single_result():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    result = axs.simulate(
        axon,
        duration_ms=0.1,
        dt_ms=0.05,
        recording=axs.Recording(variables=["Vm", "gates"]),
    )

    assert result.recordings is not None
    assert set(result.recordings) == {"Vm", "gates"}


def test_public_single_recording_spatial_filter_is_explicitly_unwired():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(NotImplementedError, match="spatial single-axon"):
        axs.simulate(
            axon,
            duration_ms=0.1,
            dt_ms=0.05,
            recording=axs.Recording.center(["Vm"]),
        )


def test_public_generic_unmyelinated_from_membrane():
    membrane = axs.membranes.HodgkinHuxley(celsius=6.3)
    axon = axs.axons.Unmyelinated(
        membrane=membrane,
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
    )

    assert axon.n_compartments == 11
    assert axon.layout.sections[0].membrane is membrane


def test_public_composite_membrane_can_build_generic_unmyelinated():
    membrane = axs.membranes.Sundt()
    axon = axs.axons.Unmyelinated(
        membrane=membrane,
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
    )

    assert axon.n_compartments == 11
    assert axon.layout.sections[0].membrane is membrane


def test_public_pool_accepts_simulation_protocols():
    axon_a = axs.AxonSimulation(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon_b = axs.AxonSimulation(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon_a.set_position(y_um=20.0, z_um=30.0)
    axon_b.set_position(y_um=-40.0, z_um=10.0)

    result = axs.simulate_pool([axon_a, axon_b], duration_ms=0.1, dt_ms=0.05)

    assert len(result) == 2
    assert result[0].simulation.y_um == 20.0
    assert result[1].simulation.z_um == 10.0


def test_public_pool_recording_center_maps_to_batch_recording():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    result = axs.simulate_pool(
        [axon],
        duration_ms=0.1,
        dt_ms=0.05,
        recording=axs.Recording.center(["Vm"]),
    )

    fiber_result = result[0]
    assert fiber_result.Vm.shape == (2, 1)
    assert fiber_result.record_indices == (5,)


def test_public_pool_recording_indices_maps_to_batch_recording():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    result = axs.simulate_pool(
        [axon],
        duration_ms=0.1,
        dt_ms=0.05,
        recording=axs.Recording.indices([0, 10], "Vm"),
    )

    fiber_result = result[0]
    assert fiber_result.Vm.shape == (2, 2)
    assert fiber_result.record_indices == (0, 10)
    expected_positions_um = axon.layout.position_values(unit=axs.um)[[0, 10]]
    np.testing.assert_allclose(
        fiber_result.position_values(unit=axs.um),
        expected_positions_um,
    )


def test_public_myelinated_mrg_template_and_section_layout():
    layout = axs.membranes.SectionLayout(
        node=axs.membranes.AxNode(),
        stin=axs.membranes.Passive(),
    )

    assert layout.membrane_for("NODE").kind == "axnode"
    assert layout.membrane_for("stin").kind == "passive"

    axon = axs.axons.MRG(diameter=5.7 * axs.um, nodes=3)
    assert axon.nodes == 3
    assert len(axon.node_indices) >= 2


def test_public_myelinated_constructor_accepts_mrg_like_layout():
    section_membranes = axs.membranes.SectionLayout(
        node=axs.membranes.AxNode(),
        mysa=axs.membranes.Passive(Rm=1e6, EL=-80.0),
        flut=axs.membranes.Passive(Rm=1e6, EL=-80.0),
        stin=axs.membranes.Passive(Rm=1e6, EL=-80.0),
    )
    layout = axs.axons.mrg_like_layout(
        diameter=5.7 * axs.um,
        nodes=3,
        membranes=section_membranes,
    )

    axon = axs.axons.Myelinated(
        layout=layout,
    )

    assert axon.nodes == 3
    flat = axs.axons.flatten_layout(axon.layout)
    assert set(kind.lower() for kind in flat.section_names) == {"node", "mysa", "flut", "stin"}
    assert len(flat.membrane_models) == axon.n_compartments
    assert np.asarray(axon.x_nodes_um).shape == np.asarray(axon.node_indices).shape
