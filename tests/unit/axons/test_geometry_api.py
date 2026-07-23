import matplotlib.pyplot as plt
import numpy as np
import pytest

import axonfleet as axs
from axonfleet.axons.flattened import flatten_layout
from axonfleet.axons.templates.mrg_like_double_cable import (
    default_mrg_like_membranes,
    layout_from_mrg_like_geometry,
    mrg_like_length_from_nodes,
    mrg_like_node_spacing,
    mrg_like_nodes_from_length,
)


def test_unmyelinated_template_accepts_public_unit_names():
    axon = axs.axons.HodgkinHuxley(
        length=1.0 * axs.mm,
        diameter=0.5 * axs.um,
        compartments=5,
        Ra=200.0 * axs.ohm_cm,
        Cm=1.0 * axs.uF_per_cm2,
        gnabar=120.0 * axs.mS_per_cm2,
        celsius=6.3 * axs.degC,
    )

    assert axon.length == pytest.approx(1000.0)
    assert axon.diameter == pytest.approx(0.5)
    flat = flatten_layout(axon.layout)
    np.testing.assert_allclose(flat.diam_um, np.full(5, 0.5))
    np.testing.assert_allclose(flat.Ra_ohm_cm, np.full(5, 200.0))
    np.testing.assert_allclose(flat.Cm_uF_cm2, np.full(5, 1.0))
    assert axon.temperature == pytest.approx(6.3)
    np.testing.assert_allclose(
        axon.layout.position_values(unit=axs.mm),
        [0.1, 0.3, 0.5, 0.7, 0.9],
    )
    assert axon.layout.compartment_position(0, unit=axs.mm).magnitude == pytest.approx(0.1)
    assert axon.layout.compartment_position(-1, unit=axs.um).magnitude == pytest.approx(900.0)
    np.testing.assert_allclose(axon.layout.diameter_values(unit=axs.um), np.full(5, 0.5))
    np.testing.assert_allclose(axon.diameter_values(unit=axs.um), np.full(5, 0.5))
    with pytest.raises(IndexError, match="compartment index"):
        axon.layout.compartment_position(5)


def test_layout_reuses_one_read_only_flattened_representation():
    axon = axs.axons.MRG(diameter=10.0 * axs.um, nodes=3)

    first = flatten_layout(axon.layout)
    second = flatten_layout(axon.layout)

    assert second is first
    assert not first.x_um.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first.x_um[0] = -1.0


def test_axon_and_layout_descriptions_are_immutable_templates():
    axon = axs.axons.MRG(diameter=10.0 * axs.um, nodes=3)

    with pytest.raises(AttributeError, match="immutable templates"):
        axon.temperature = 20.0
    with pytest.raises(AttributeError, match="Layout descriptions are immutable"):
        axon.layout.x_shift_um = 20.0


def test_unmyelinated_templates_do_not_expose_legacy_geometry_aliases():
    with pytest.raises(TypeError):
        axs.axons.HodgkinHuxley(L=1000.0, d=0.5, Nx=11)
    assert not hasattr(axs.axons.Unmyelinated, "HH")


def test_public_axon_diameter_is_easy_for_uniform_and_template_models():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(),
        diameter=1.2 * axs.um,
    )
    axon = axs.axons.Axon(
        layout=axs.axons.Layout.single_uniform(
            section,
            length=100.0 * axs.um,
            compartments=3,
        )
    )

    assert axon.diameter == pytest.approx(1.2)
    np.testing.assert_allclose(axon.diameter_values(unit=axs.um), [1.2, 1.2, 1.2])

    mrg = axs.axons.MRG(diameter=10.0 * axs.um, nodes=3)

    assert mrg.diameter == pytest.approx(10.0)


def test_template_axon_diameters_are_quantized_for_cache_reuse():
    small = axs.axons.RattayAberham(
        length=100.0 * axs.um,
        diameter=0.666 * axs.um,
        compartments=5,
    )
    large = axs.axons.RattayAberham(
        length=100.0 * axs.um,
        diameter=1.24 * axs.um,
        compartments=5,
    )
    mrg = axs.axons.MRG(diameter=2.52 * axs.um, nodes=3)

    assert small.diameter == pytest.approx(0.67)
    np.testing.assert_allclose(small.diameter_values(unit=axs.um), np.full(5, 0.67))
    assert large.diameter == pytest.approx(1.2)
    np.testing.assert_allclose(large.diameter_values(unit=axs.um), np.full(5, 1.2))
    assert mrg.diameter == pytest.approx(2.5)
    assert mrg_like_node_spacing(2.52 * axs.um) == pytest.approx(
        mrg_like_node_spacing(2.5 * axs.um)
    )

    equivalent_small = axs.axons.RattayAberham(
        length=100.0 * axs.um,
        diameter=0.674 * axs.um,
        compartments=5,
    )
    assert small.layout is equivalent_small.layout
    assert (
        flatten_layout(small.layout).membrane_models[0]
        is flatten_layout(equivalent_small.layout).membrane_models[0]
    )


def test_default_mrg_layout_reuses_complete_quantized_template_key():
    compartments_a = {"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1}
    compartments_b = {"STIN": 1, "FLUT": 1, "MYSA": 1, "node": 1}

    first = axs.axons.MRG(
        diameter=2.52 * axs.um,
        nodes=4,
        length=1500.0 * axs.um,
        compartments=compartments_a,
    )
    second = axs.axons.MRG(
        diameter=2.5 * axs.um,
        nodes=4,
        length=1500.0 * axs.um,
        compartments=compartments_b,
    )

    assert first.layout is second.layout
    assert flatten_layout(first.layout) is flatten_layout(
        second.layout
    )


def test_default_mrg_layout_key_separates_structural_parameters():
    base = axs.axons.MRG(diameter=10.0 * axs.um, nodes=4)
    shifted = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=4,
        x_shift=25.0 * axs.um,
    )
    warmer = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=4,
        temperature=36.0 * axs.degC,
    )

    assert base.layout is not shifted.layout
    assert base.layout is not warmer.layout
    assert base.node_position_values(unit=axs.um)[0] != pytest.approx(
        shifted.node_position_values(unit=axs.um)[0]
    )
    assert (
        flatten_layout(base.layout).membrane_models[0]
        != flatten_layout(warmer.layout).membrane_models[0]
    )


def test_cached_default_mrg_layout_matches_explicit_uncached_construction():
    template = axs.axons.MRGLikeDoubleCableTemplate(
        diameter=7.3 * axs.um,
        nodes=4,
        length=1500.0 * axs.um,
        compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
    )
    cached = template.layout()
    geometry = template.geometry()
    explicit = layout_from_mrg_like_geometry(
        geometry,
        membranes=default_mrg_like_membranes(geometry),
        compartments=template.compartments,
    )
    cached_flat = flatten_layout(cached)
    explicit_flat = flatten_layout(explicit)

    for field in (
        "x_um",
        "edges_um",
        "lengths_um",
        "diam_um",
        "Ra_ohm_cm",
        "Cm_uF_cm2",
        "section_indices",
    ):
        np.testing.assert_array_equal(
            getattr(cached_flat, field),
            getattr(explicit_flat, field),
        )
    assert cached_flat.membrane_models == explicit_flat.membrane_models
    assert cached_flat.section_names == explicit_flat.section_names
    assert cached_flat.section_tags == explicit_flat.section_tags
    assert cached_flat.periaxonal_layers == explicit_flat.periaxonal_layers


def test_public_axon_diameter_points_to_values_for_non_uniform_layouts():
    narrow = axs.axons.Section(
        "narrow",
        membrane=axs.membranes.Passive(),
        diameter=0.5 * axs.um,
    )
    wide = axs.axons.Section(
        "wide",
        membrane=axs.membranes.Passive(),
        diameter=1.0 * axs.um,
    )
    axon = axs.axons.Axon(
        layout=axs.axons.Layout.sequence(
            [narrow, wide],
            section_lengths=[50.0, 50.0] * axs.um,
            compartments=[1, 1],
            lengths=100.0 * axs.um,
        )
    )

    np.testing.assert_allclose(axon.diameter_values(unit=axs.um), [0.5, 1.0])
    with pytest.raises(ValueError, match="non-uniform diameters"):
        _ = axon.diameter


def test_public_axon_does_not_expose_solver_view_or_solver_vectors():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=5,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonInstance(axon)

    for obj in (axon, sim):
        assert not hasattr(obj, "view")
        assert not hasattr(obj, "diam_vec")
        assert not hasattr(obj, "Ra_vec")
        assert not hasattr(obj, "Cm_vec")
        assert not hasattr(obj, "dx_cm")
        assert not hasattr(obj, "h_cm")
        assert not hasattr(obj, "xraxial_vec")
        assert not hasattr(obj, "xg_vec")
        assert not hasattr(obj, "xc_vec")
        assert not hasattr(obj, "x")
        assert not hasattr(obj, "diameters_um")
        assert not hasattr(obj, "Ra_ohm_cm")
        assert not hasattr(obj, "Cm_uF_cm2")
        assert not hasattr(obj, "length_um")
        assert not hasattr(obj, "membrane_model")
        assert not hasattr(obj, "membrane_models")
        assert not hasattr(obj, "plot")
        assert not hasattr(obj, "L")
        assert not hasattr(obj, "Nx")
    assert not hasattr(axon, "dtype")
    assert hasattr(sim, "dtype")


def test_unmyelinated_template_accepts_unit_aware_custom_mesh():
    x = np.asarray([0.0, 0.25, 1.0]) * axs.mm
    axon = axs.axons.RattayAberham(
        x=x,
        diameter=0.8 * axs.um,
        celsius=37.0 * axs.degC,
    )

    assert axon.n_compartments == 3
    np.testing.assert_allclose(axon.layout.position_values(unit=axs.um), [0.0, 250.0, 1000.0])


def test_layout_x_shift_translates_positions_without_changing_lengths():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(),
        diameter=1.0 * axs.um,
    )
    base = axs.axons.Layout.single_uniform(
        section,
        length=100.0 * axs.um,
        compartments=4,
    )
    shifted = base.with_x_shift(25.0 * axs.um)

    np.testing.assert_allclose(
        shifted.position_values(unit=axs.um),
        base.position_values(unit=axs.um) + 25.0,
    )
    np.testing.assert_allclose(
        shifted.compartment_length_values(unit=axs.um),
        base.compartment_length_values(unit=axs.um),
    )
    assert shifted.length_um == pytest.approx(base.length_um)

    non_uniform = axs.axons.Layout.single_non_uniform(
        section,
        x=np.asarray([0.0, 250.0, 1000.0]) * axs.um,
        x_shift=-10.0 * axs.um,
    )
    np.testing.assert_allclose(non_uniform.position_values(unit=axs.um), [-10.0, 240.0, 990.0])

    with pytest.raises(TypeError, match="x_shift must include units"):
        axs.axons.Layout.single_uniform(
            section,
            length=100.0 * axs.um,
            compartments=4,
            x_shift=25.0,
        )


def test_layout_sequence_phase_shift_rotates_and_crops_repeated_motif():
    node = axs.axons.Section(
        "node",
        membrane=axs.membranes.Passive(),
        diameter=1.0 * axs.um,
    )
    internode = axs.axons.Section(
        "internode",
        membrane=axs.membranes.Passive(),
        diameter=1.0 * axs.um,
    )
    layout = axs.axons.Layout.sequence(
        [node, internode],
        section_lengths=[10.0, 90.0] * axs.um,
        compartments=[1, 3],
        lengths=250.0 * axs.um,
        phase_shift=25.0 * axs.um,
    )
    flat = flatten_layout(layout)

    assert flat.section_names[0] == "internode"
    assert flat.section_names[3] == "node"
    np.testing.assert_allclose(
        layout.compartment_position(3, unit=axs.um).magnitude,
        30.0,
    )
    assert layout.length_um == pytest.approx(250.0)


def test_mrg_x_shift_phases_node_positions_without_world_coordinates():
    base = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5)
    shifted = axs.axons.MRG(diameter=10.0 * axs.um, nodes=5, x_shift=80.0 * axs.um)

    node_spacing_um = mrg_like_node_spacing(10.0 * axs.um)
    np.testing.assert_allclose(
        base.node_position_values(unit=axs.um),
        [0.5 + index * node_spacing_um for index in range(base.nodes)],
        atol=1e-3,
    )
    np.testing.assert_allclose(
        shifted.node_position_values(unit=axs.um),
        [80.5 + index * node_spacing_um for index in range(shifted.nodes)],
        atol=1e-3,
    )
    assert shifted.nodes == base.nodes
    assert shifted.length == pytest.approx(base.length + 80.0)


def test_mrg_non_tabulated_node_count_uses_actual_morphology_spacing():
    expected_length = mrg_like_length_from_nodes(2.52 * axs.um, 47)
    expected_nodes = mrg_like_nodes_from_length(2.52 * axs.um, expected_length * axs.um)

    axon = axs.axons.MRG(diameter=2.52 * axs.um, nodes=47)

    assert axon.length == pytest.approx(expected_length + 1.0)
    assert axon.nodes == expected_nodes


def test_unmyelinated_template_requires_length_units():
    with pytest.raises(TypeError, match="length must include units"):
        axs.axons.HodgkinHuxley(length=1000.0, diameter=0.5 * axs.um, compartments=11)

    with pytest.raises(TypeError, match="diameter must include units"):
        axs.axons.HodgkinHuxley(length=1000.0 * axs.um, diameter=0.5, compartments=11)

    with pytest.raises(TypeError, match="Ra must include units"):
        axs.axons.HodgkinHuxley(
            length=1000.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            Ra=100.0,
        )

    with pytest.raises(TypeError, match="Cm must include units"):
        axs.axons.HodgkinHuxley(
            length=1000.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            Cm=1.0,
        )

    with pytest.raises(TypeError, match="celsius must include units"):
        axs.axons.HodgkinHuxley(
            length=1000.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3,
        )


def test_axon_state_requires_units():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(),
        diameter=1.0 * axs.um,
    )
    layout = axs.axons.Layout.single_uniform(section, length=100.0 * axs.um, compartments=3)

    with pytest.raises(TypeError, match="v_init must include units"):
        axs.axons.Axon(layout=layout, v_init=-70.0)

    with pytest.raises(TypeError, match="temperature must include units"):
        axs.axons.Axon(layout=layout, temperature=37.0)

    with pytest.raises(ValueError, match="Section-building arguments"):
        axs.axons.Unmyelinated(
            layout=layout,
            diameter=1.0 * axs.um,
        )


def test_myelinated_geometry_helpers_and_layout_plotting():
    axon = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=5,
        compartments={"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 3},
        axoplasmic_resistivity=0.7e6 * axs.ohm_um,
        myelin_capacitance=0.1 * axs.uF_per_cm2,
        myelin_conductance=0.001 * axs.S_per_cm2,
    )

    assert axon.node_position_values(unit=axs.um).shape == (5,)
    assert axon.node_position("center", unit=axs.um).magnitude == pytest.approx(
        axon.node_position_values(unit=axs.um)[2]
    )
    flat = flatten_layout(axon.layout)
    assert flat.Nx > len(axon.layout.elements)

    fig, ax_layout = plt.subplots()
    assert axon.layout.plot(ax=ax_layout, position_unit=axs.um, compartment_labels="auto") is ax_layout
    assert ax_layout.patches
    assert ax_layout.texts
    plt.close(fig)

    with pytest.raises(TypeError, match="axoplasmic_resistivity must include units"):
        axs.axons.MRG(
            diameter=10.0 * axs.um,
            nodes=5,
            axoplasmic_resistivity=0.7e6,
        )

    with pytest.raises(ValueError, match="Unknown MRG section"):
        axs.axons.MRG(
            diameter=10.0 * axs.um,
            nodes=5,
            compartments={"bad": 2},
        )


def test_axon_rejects_unknown_formulation():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(),
        diameter=1.0 * axs.um,
    )
    layout = axs.axons.Layout.single_uniform(section, length=100.0 * axs.um, compartments=3)

    with pytest.raises(TypeError, match="CableFormulation"):
        axs.axons.Axon(layout=layout, formulation="triple-cable")
