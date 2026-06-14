import numpy as np
import pytest

import axonscope as axs
from axonscope.results import SimResult


class FakeQuantity:
    def __init__(self, magnitude, unit):
        self.magnitude = magnitude
        self.unit = unit

    def to(self, unit):
        factors = {
            ("millimeter", "micrometer"): 1000.0,
            ("micrometer", "micrometer"): 1.0,
            ("millisecond", "millisecond"): 1.0,
            ("second", "millisecond"): 1000.0,
            ("millivolt", "millivolt"): 1.0,
            ("volt", "millivolt"): 1000.0,
            ("degree_Celsius", "degree_Celsius"): 1.0,
            ("ampere", "microampere"): 1e6,
            ("microampere", "microampere"): 1.0,
            ("ohm * centimeter", "ohm * centimeter"): 1.0,
            ("ohm * centimeter ** 2", "ohm * centimeter ** 2"): 1.0,
            ("microfarad / centimeter ** 2", "microfarad / centimeter ** 2"): 1.0,
            ("siemens / centimeter ** 2", "siemens / centimeter ** 2"): 1.0,
            ("millisiemens / centimeter ** 2", "siemens / centimeter ** 2"): 1e-3,
            ("milliampere / centimeter ** 2", "milliampere / centimeter ** 2"): 1.0,
            ("millimolar", "millimolar"): 1.0,
            ("megaohm / centimeter", "megaohm / centimeter"): 1.0,
        }
        return FakeQuantity(self.magnitude * factors[(self.unit, unit)], unit)


def test_units_accept_plain_canonical_numbers():
    assert axs.units.to_um(12.5) == 12.5
    np.testing.assert_allclose(axs.units.to_ms_array([0.0, 1.0]), [0.0, 1.0])


def test_top_level_unit_aliases_are_available():
    assert (1.0 * axs.ms).to("second").magnitude == pytest.approx(0.001)
    assert (2.0 * axs.uA).to("ampere").magnitude == pytest.approx(2.0e-6)
    assert (3.0 * axs.um).to("meter").magnitude == pytest.approx(3.0e-6)
    assert (4.0 * axs.mS).to("siemens").magnitude == pytest.approx(4.0e-3)
    assert (0.3 * axs.S_per_m).to("siemens / meter").magnitude == pytest.approx(0.3)
    assert (0.3 * axs.S_per_meter).to("siemens / meter").magnitude == pytest.approx(0.3)
    assert (0.3 * axs.siemens_per_m).to("siemens / meter").magnitude == pytest.approx(0.3)
    assert (3.0 * axs.mS_per_cm).to("siemens / meter").magnitude == pytest.approx(0.3)
    assert (100.0 * axs.ohm_cm).to("ohm * centimeter").magnitude == pytest.approx(100.0)
    assert (1.0 * axs.uF_per_cm2).to("microfarad / centimeter ** 2").magnitude == pytest.approx(1.0)
    assert (1.0 * axs.mS_per_cm2).to("siemens / centimeter ** 2").magnitude == pytest.approx(1e-3)
    assert (0.7e6 * axs.ohm_um).to("ohm * micrometer").magnitude == pytest.approx(0.7e6)


def test_units_convert_quantity_like_values():
    assert axs.units.to_um(FakeQuantity(1.5, "millimeter")) == 1500.0
    assert axs.units.to_ms(FakeQuantity(2.0, "second")) == 2000.0
    assert axs.units.to_uA(FakeQuantity(2e-6, "ampere")) == 2.0
    np.testing.assert_allclose(
        axs.units.to_mV_array(
            [
                FakeQuantity(1.0, "volt"),
                FakeQuantity(2.0, "millivolt"),
            ]
        ),
        [1000.0, 2.0],
    )


def test_direct_solver_time_values_accept_quantity_like_values():
    from axonscope.solvers.common import resolve_time_args

    assert resolve_time_args(
        tsim=FakeQuantity(0.002, "second"),
        dt=FakeQuantity(0.001, "second"),
    ) == (2.0, 1.0)


def test_recording_normalizes_quantity_like_filters():
    recording = axs.Recording(
        signals=axs.signals.Vm,
        positions=[FakeQuantity(1.5, "millimeter")],
        sample_dt=FakeQuantity(0.002, "second"),
    )

    assert recording.signals == (axs.signals.Vm,)
    assert recording.positions_um == (1500.0,)
    assert recording.sample_dt_ms == 2.0


def test_recording_sample_dt_requires_units():
    with pytest.raises(TypeError, match="sample_dt must include units compatible with time"):
        axs.Recording(signals=axs.signals.Vm, sample_dt=0.1)


def test_recording_positions_require_units():
    with pytest.raises(TypeError, match="positions must include units compatible with length"):
        axs.Recording(signals=axs.signals.Vm, positions=[100.0])


def test_recording_normalizes_explicit_indices():
    recording = axs.Recording.indices([0, 4], axs.signals.Vm)

    assert recording.spatial is axs.RecordingSpatial.INDICES
    assert recording.record_indices == (0, 4)


def test_simulation_protocol_accepts_unit_quantities():
    axon = axs.axons.HodgkinHuxley(
        length=0.1 * axs.mm,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonInstance(axon, y=0.02 * axs.mm, z=10.0 * axs.um)
    sim.add_current_clamp(
        position=0.05 * axs.mm,
        current=axs.Stimulus.pulse(
            start=20.0 * axs.us,
            duration=20.0 * axs.us,
            amplitude=0.5 * axs.nA,
        ),
    )

    assert sim.y_um == pytest.approx(20.0)
    assert sim.z_um == pytest.approx(10.0)
    assert sim.intracellular_contexts[0].position_um == pytest.approx(50.0)
    np.testing.assert_allclose(sim.intracellular_contexts[0].current.t, [0.0, 0.02, 0.04])


def test_current_clamp_position_requires_units():
    with pytest.raises(TypeError, match="position must include units compatible with length"):
        axs.IntracellularCurrentClamp(position=50.0, current=axs.Stimulus.constant(0.0))


def test_simulation_protocol_position_requires_units():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    with pytest.raises(TypeError, match="y must include units compatible with length"):
        axs.AxonInstance(axon, y=20.0)
    sim = axs.AxonInstance(axon)
    with pytest.raises(TypeError, match="x_offset must include units compatible with length"):
        sim.set_position(x_offset=0.0, y=20.0 * axs.um, z=0.0 * axs.um)


def test_analysis_accepts_quantity_like_thresholds():
    class DummyLayout:
        def position_values(self, *, unit="micrometer"):
            return np.asarray([0.0, 500.0])

    class DummyAxon:
        n_compartments = 2
        layout = DummyLayout()

    t = np.linspace(0.0, 10.0, 1001)
    vm = np.zeros((t.shape[0], 2)) - 70.0
    vm[:, 1] += np.exp(-0.5 * ((t - 5.0) / 0.1) ** 2) * 100.0
    result = SimResult(axon=DummyAxon(), Vm=vm, t=t)

    spike_t_ms, spike_x_um = axs.results.analysis.rasterize(
        result,
        threshold_mV=FakeQuantity(0.0, "millivolt"),
        min_distance_ms=FakeQuantity(0.001, "second"),
    )

    np.testing.assert_allclose(spike_t_ms, [5.0], atol=0.02)
    np.testing.assert_allclose(spike_x_um, [500.0])


def test_section_accepts_quantity_like_geometry():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(Rm=10000.0, EL=-70.0),
        diameter=FakeQuantity(0.5, "micrometer"),
        Ra=FakeQuantity(100.0, "ohm * centimeter"),
        Cm=FakeQuantity(1.0, "microfarad / centimeter ** 2"),
    )
    assert section.diameter_um == 0.5


def test_section_defaults_have_explicit_units_internally():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(Rm=10000.0, EL=-70.0),
        diameter=FakeQuantity(0.5, "micrometer"),
        tags="demo",
    )

    assert section.diameter_um == 0.5
    assert section.Ra_ohm_cm == 100.0
    assert section.Cm_uF_cm2 == 1.0
    assert section.tags == ("demo",)


def test_periaxonal_layer_accepts_unit_aware_aliases():
    layer = axs.axons.PeriaxonalLayer(
        radial_conductance=FakeQuantity(1e-3, "siemens / centimeter ** 2"),
        radial_capacitance=FakeQuantity(0.1, "microfarad / centimeter ** 2"),
        axial_resistance=FakeQuantity(1e8, "megaohm / centimeter"),
    )

    assert layer.radial_conductance_S_cm2 == 1e-3
    assert layer.radial_capacitance_uF_cm2 == 0.1
    assert layer.axial_resistance_MOhm_per_cm == 1e8


def test_section_rejects_plain_user_numbers():
    with pytest.raises(TypeError, match="diameter must include units compatible with length"):
        axs.axons.Section(
            "axon",
            membrane=axs.membranes.Passive(Rm=10000.0, EL=-70.0),
            diameter=0.5,
        )


def test_section_rejects_wrong_unit_dimension():
    with pytest.raises(TypeError, match="diameter must have units compatible with length"):
        axs.axons.Section(
            "axon",
            membrane=axs.membranes.Passive(Rm=10000.0, EL=-70.0),
            diameter=1.0 * axs.ms,
        )


def test_layout_single_non_uniform_requires_unit_aware_coordinates():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(Rm=10000.0, EL=-70.0),
        diameter=0.5 * axs.um,
        Ra=100.0 * axs.ohm_cm,
        Cm=1.0 * axs.uF_per_cm2,
    )
    layout = axs.axons.Layout.single_non_uniform(section, x=np.linspace(0.0, 1.0, 3) * axs.mm)

    flat = axs.axons.flatten_layout(layout)
    np.testing.assert_allclose(flat.x_um, [0.0, 500.0, 1000.0])
    np.testing.assert_allclose(flat.diam_um, [0.5, 0.5, 0.5])


def test_layout_single_non_uniform_rejects_plain_coordinates():
    with pytest.raises(TypeError, match="x must include units compatible with length"):
        axs.axons.Layout.single_non_uniform(
            axs.axons.Section(
                "axon",
                membrane=axs.membranes.Passive(Rm=10000.0, EL=-70.0),
                diameter=0.5 * axs.um,
            ),
            x=np.linspace(0.0, 1000.0, 3),
        )


def test_layout_single_non_uniform_rejects_nonmonotonic_coordinates():
    with pytest.raises(ValueError, match="x must be strictly increasing"):
        axs.axons.Layout.single_non_uniform(
            axs.axons.Section(
                "axon",
                membrane=axs.membranes.Passive(Rm=10000.0, EL=-70.0),
                diameter=0.5 * axs.um,
            ),
            x=np.asarray([0.0, 10.0, 10.0]) * axs.um,
        )


def test_layout_sequence_lengths_requires_units():
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.Passive(Rm=10000.0, EL=-70.0),
        diameter=0.5 * axs.um,
    )

    layout = axs.axons.Layout.sequence(
        [section],
        section_lengths=np.asarray([100.0]) * axs.um,
        compartments=[1],
        lengths=1.0 * axs.mm,
    )
    assert axs.axons.flatten_layout(layout).length_um == pytest.approx(1000.0)

    with pytest.raises(TypeError, match="lengths must include units compatible with length"):
        axs.axons.Layout.sequence(
            [section],
            section_lengths=np.asarray([100.0]) * axs.um,
            compartments=[1],
            lengths=1000.0,
        )


def test_membrane_templates_normalize_quantity_like_parameters():
    hh = axs.membranes.HodgkinHuxley(
        gnabar=FakeQuantity(120.0, "millisiemens / centimeter ** 2"),
        ena=FakeQuantity(0.05, "volt"),
        celsius=FakeQuantity(6.3, "degree_Celsius"),
    )
    assert hh.params["gnabar"] == 0.12
    assert hh.params["ena"] == 50.0
    assert hh.params["celsius"] == 6.3

    passive = axs.membranes.Passive(
        Rm=FakeQuantity(10000.0, "ohm * centimeter ** 2"),
        EL=FakeQuantity(-0.07, "volt"),
    )
    assert passive.params["Rm"] == 10000.0
    assert passive.params["EL"] == -70.0

    tigerholm = axs.membranes.Tigerholm(
        diameter=FakeQuantity(1.2, "millimeter"),
        gbar_nav17=FakeQuantity(106.64, "millisiemens / centimeter ** 2"),
        pump_smalla=FakeQuantity(-0.0047891, "milliampere / centimeter ** 2"),
        pump_ko=FakeQuantity(5.6, "millimolar"),
    )
    assert tigerholm.params["diameter_um"] == 1200.0
    assert tigerholm.params["gbar_nav17"] == pytest.approx(0.10664)
    assert tigerholm.params["pump_ko"] == 5.6


def test_membrane_templates_require_unit_aware_diameter():
    with pytest.raises(TypeError, match="diameter must include units compatible with length"):
        axs.membranes.Tigerholm(diameter=1.2)
    with pytest.raises(TypeError, match="diameter must include units compatible with length"):
        axs.membranes.Schild94(diameter=0.8)
    with pytest.raises(TypeError, match="diameter must include units compatible with length"):
        axs.membranes.Schild97(diameter=0.8)


def test_pint_constructor_reports_missing_dependency_when_absent():
    if axs.units.has_pint():
        pytest.skip("Pint is installed in this environment.")
    with pytest.raises(axs.units.UnitSupportError):
        axs.units.Q_(1.0, "micrometer")
