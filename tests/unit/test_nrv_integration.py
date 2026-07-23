from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import axonfleet as axs
from axonfleet.integrations import nrv as axs_nrv


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        yield from self._rows


class _FakeILoc:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, index):
        return self._values[int(index)]


class _FakeSeries:
    def __init__(self, values):
        self.iloc = _FakeILoc(values)


class _FakeAxons:
    def __init__(self, rows, masks):
        self.axon_pop = _FakeTable(rows)
        self._masks = masks

    def __getitem__(self, key):
        return _FakeSeries(self._masks[key])


class _FakeFascicle:
    def __init__(self, rows, masks):
        self.axons = _FakeAxons(rows, masks)
        self.sim_mask = tuple(masks)
        self.sim_list = [index for index, keep in enumerate(masks["keep"]) if keep]


class _FakeElectrode:
    def __init__(self, footprint, *, ID="electrode"):
        self.ID = ID
        self._footprint = np.asarray(footprint, dtype=float)

    def get_footprint(self):
        return self._footprint


class _FakeExtraStim:
    def __init__(self, footprints):
        array = np.asarray(footprints, dtype=float)
        if array.ndim == 1:
            footprint_rows = [array]
        else:
            footprint_rows = list(array)
        self.electrodes = [
            _FakeElectrode(footprint, ID=f"e{index}")
            for index, footprint in enumerate(footprint_rows)
        ]
        self.model = SimpleNamespace(mesh=SimpleNamespace(n_core=1))
        self.compute_calls = []
        self.clear_count = 0

    def compute_electrodes_footprints(self, positions, y, z, row_id):
        self.compute_calls.append(
            {
                "positions": np.asarray(positions, dtype=float),
                "y": float(y),
                "z": float(z),
                "row_id": int(row_id),
            }
        )

    def clear_electrodes_footprints(self):
        self.clear_count += 1


def test_nrv_integration_exposes_only_bridge_contracts():
    forbidden = {
        "FiberKind",
        "NRVActivationComparison",
        "build_synthetic_4_fascicle_nerve",
        "build_histology_contour_nerve",
        "build_nerve_from_mode",
        "attach_life_fem_electrode",
        "NRVLifeElectrodeSetup",
        "life_pulse_stimulus",
        "replace_life_current",
        "activation_comparisons",
        "nrv_activation_by_row",
        "nrv_fascicle_by_id",
        "nrv_row_id",
        "row_key",
        "stimulation_from_footprint",
    }

    assert forbidden.isdisjoint(set(axs_nrv.__all__))
    for name in forbidden:
        assert not hasattr(axs_nrv, name)


def test_population_from_nrv_honors_nrv_masks_and_node_shift():
    fascicle = _FakeFascicle(
        [
            (0, {"types": 1, "diameters": 8.0, "y": 1.0, "z": 2.0, "node_shift": 0.25}),
            (1, {"types": 0, "diameters": 0.7, "y": 3.0, "z": 4.0}),
            (2, {"types": 1, "diameters": 10.0, "y": 5.0, "z": 6.0, "node_shift": 0.5}),
        ],
        {"keep": [True, False, True]},
    )
    nerve = SimpleNamespace(fascicles={0: fascicle})

    bridge = axs_nrv.population_from_nrv(nerve, nerve_length_um=10_000.0)
    rows = bridge.rows

    assert [row.fiber_index for row in rows] == [0, 2]
    assert [row.kind for row in rows] == ["mrg", "mrg"]
    assert rows[0].fascicle_id == "0"
    assert rows[0].x_shift_um > 0.0


def test_population_from_nrv_builds_axonfleet_population_without_footprints():
    fascicle = _FakeFascicle(
        [
            (0, {"types": 1, "diameters": 8.0, "y": 1.0, "z": 2.0, "node_shift": 0.25}),
            (1, {"types": 0, "diameters": 0.7, "y": 3.0, "z": 4.0}),
        ],
        {"keep": [True, True]},
    )
    nerve = SimpleNamespace(fascicles={0: fascicle})

    bridge = axs_nrv.population_from_nrv(nerve, nerve_length_um=10_000.0)

    assert isinstance(bridge.population, axs.AxonPopulation)
    assert len(bridge) == 2
    assert [row.fiber_index for row in bridge.rows] == [0, 1]
    assert bridge.population.instances[0].extracellular_stimulation is None
    assert bridge.population.axons[0].n_compartments > 0


def test_population_from_nrv_shares_only_exact_axon_templates():
    fascicle = _FakeFascicle(
        [
            (0, {"types": 1, "diameters": 8.0, "y": 1.0, "z": 2.0, "node_shift": 0.25}),
            (1, {"types": 1, "diameters": 8.0, "y": 3.0, "z": 4.0, "node_shift": 0.25}),
            (2, {"types": 1, "diameters": 8.0, "y": 5.0, "z": 6.0, "node_shift": 0.50}),
        ],
        {"keep": [True, True, True]},
    )

    bridge = axs_nrv.population_from_nrv(
        SimpleNamespace(fascicles={0: fascicle}),
        nerve_length_um=10_000.0,
    )

    assert bridge.population.axons[0] is bridge.population.axons[1]
    assert bridge.population.axons[2] is not bridge.population.axons[0]
    assert bridge.population.instances[0] is not bridge.population.instances[1]


def test_footprints_from_nrv_samples_all_electrodes_for_population_bridge():
    rows = (
        axs_nrv.NRVFiberRow("0", 0, "mrg", 8.0, 10.0, 20.0),
        axs_nrv.NRVFiberRow("0", 1, "rattay", 0.8, -5.0, 3.0),
    )
    instances = [
        axs.AxonInstance(
            axs.axons.HodgkinHuxley(
                length=20.0 * axs.um,
                diameter=0.5 * axs.um,
                compartments=3,
            )
        ),
        axs.AxonInstance(
            axs.axons.HodgkinHuxley(
                length=20.0 * axs.um,
                diameter=0.5 * axs.um,
                compartments=3,
            )
        ),
    ]
    bridge = axs_nrv.NRVAxonPopulation(
        population=axs.AxonPopulation(instances),
        rows=rows,
    )
    extra_stim = _FakeExtraStim([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nerve = SimpleNamespace(extra_stim=extra_stim)

    footprints = axs_nrv.footprints_from_nrv(nerve, bridge)

    assert len(footprints.electrode_ids) == 2
    assert footprints.electrode_ids == ("e0", "e1")
    assert len(extra_stim.compute_calls) == 2
    assert extra_stim.clear_count == 2
    np.testing.assert_allclose(footprints.footprints[0][0].values_V_per_A, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(footprints.footprints[0][1].values_V_per_A, [4.0, 5.0, 6.0])

    stimulus = axs.Stimulus.pulse(
        start=0.1 * axs.ms,
        duration=0.1 * axs.ms,
        amplitude=-1.0 * axs.uA,
    )
    stimulated = footprints.stimulated_population(stimulus=stimulus)
    assert isinstance(stimulated, axs.AxonPopulation)
    assert stimulated[0].extracellular_stimulation.drives[0].footprint is footprints.footprints[0][0]


def test_stimulated_population_and_current_replacement_use_one_drive_contract():
    footprint = axs.ExtracellularFootprint.shared(
        values=[1.0, 2.0, 3.0],
        positions=[0.0, 10.0, 20.0] * axs.um,
    )
    axon = axs.axons.HodgkinHuxley(
        length=20.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=3,
    )
    rows = (axs_nrv.NRVFiberRow("0", 0, "rattay", 0.5, 0.0, 0.0),)
    footprints = axs_nrv.NRVFootprints(
        population=axs.AxonPopulation(axon),
        rows=rows,
        footprints=((footprint,),),
        electrode_ids=("life",),
    )
    population = footprints.stimulated_population(
        stimulus=axs.Stimulus.pulse(
            start=0.1 * axs.ms,
            duration=0.2 * axs.ms,
            amplitude=-5.0 * axs.uA,
        ),
        drive_id_prefix="nrv_life",
    )
    simulation = population[0]
    assert simulation.extracellular_stimulation.drives[0].id == axs.DriveId("nrv_life_0")

    updated = simulation.extracellular_stimulation.replace_drive(
        axs.DriveId("nrv_life_0"),
        stimulus=axs.Stimulus.pulse(
            start=0.1 * axs.ms,
            duration=0.2 * axs.ms,
            amplitude=-9.0 * axs.uA,
        ),
    )
    simulation.add_extracellular_stimulation(stimulation=updated, replace=True)

    current_uA = simulation.extracellular_stimulation.drives[0].stimulus.evaluate(
        [0.15] * axs.ms,
        unit=axs.uA,
    )
    np.testing.assert_allclose(current_uA, [-9.0])
