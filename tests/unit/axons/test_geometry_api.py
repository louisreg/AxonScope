import matplotlib.pyplot as plt
import numpy as np
import pytest

import axonscope as axs


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
    flat = axs.axons.flatten_layout(axon.layout)
    np.testing.assert_allclose(flat.diam_um, np.full(5, 0.5))
    np.testing.assert_allclose(flat.Ra_ohm_cm, np.full(5, 200.0))
    np.testing.assert_allclose(flat.Cm_uF_cm2, np.full(5, 1.0))
    assert axon.temperature == pytest.approx(6.3)
    np.testing.assert_allclose(
        axon.layout.position_values(unit=axs.mm),
        [0.1, 0.3, 0.5, 0.7, 0.9],
    )
    np.testing.assert_allclose(axon.layout.diameter_values(unit=axs.um), np.full(5, 0.5))


def test_unmyelinated_templates_do_not_expose_legacy_geometry_aliases():
    with pytest.raises(TypeError):
        axs.axons.HodgkinHuxley(L=1000.0, d=0.5, Nx=11)
    assert not hasattr(axs.axons.Unmyelinated, "HH")


def test_public_axon_does_not_expose_solver_view_or_solver_vectors():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=5,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonSimulation(axon)

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
    flat = axs.axons.flatten_layout(axon.layout)
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

    with pytest.raises(ValueError, match="Unknown axon formulation"):
        axs.axons.Axon(layout=layout, formulation="triple-cable")
