from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import axonscope as axs
from axonscope.integrations import nrv as axs_nrv


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
    def __init__(self, footprint):
        self._footprint = np.asarray(footprint, dtype=float)

    def get_footprint(self):
        return self._footprint


class _FakeExtraStim:
    def __init__(self, footprint):
        self.electrodes = [_FakeElectrode(footprint)]
        self.model = SimpleNamespace(mesh=SimpleNamespace(n_core=1))
        self.compute_calls = []
        self.cleared = False

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
        self.cleared = True


class _FakeAxonResult(dict):
    def __init__(self, recruited):
        super().__init__()
        self._recruited = bool(recruited)

    def is_recruited(self, *, vm_key, t_start):
        assert vm_key == "V_mem"
        assert float(t_start) == 0.1
        return self._recruited


def test_extract_fiber_rows_honors_nrv_masks_and_node_shift():
    fascicle = _FakeFascicle(
        [
            (0, {"types": 1, "diameters": 8.0, "y": 1.0, "z": 2.0, "node_shift": 0.25}),
            (1, {"types": 0, "diameters": 0.7, "y": 3.0, "z": 4.0}),
            (2, {"types": 1, "diameters": 10.0, "y": 5.0, "z": 6.0, "node_shift": 0.5}),
        ],
        {"keep": [True, False, True]},
    )
    nerve = SimpleNamespace(fascicles={0: fascicle})

    rows = axs_nrv.extract_fiber_rows(nerve, include_unmyelinated=True)

    assert [row.fiber_index for row in rows] == [0, 2]
    assert [row.kind for row in rows] == ["mrg", "mrg"]
    assert rows[0].fascicle_id == "0"
    assert rows[0].x_shift_um > 0.0
    assert axs_nrv.row_key(rows[0]) == ("0", 0)
    assert axs_nrv.select_rows(rows, limit=1) == [rows[0]]


def test_sample_life_footprint_uses_intrinsic_positions_and_nrv_metadata():
    row = axs_nrv.NRVFiberRow(
        fascicle_id="2",
        fiber_index=7,
        kind="mrg",
        diameter_um=9.0,
        y_um=12.0,
        z_um=-4.0,
    )
    extra_stim = _FakeExtraStim([1.0, 2.0, 3.0])
    setup = axs_nrv.NRVLifeElectrodeSetup(
        extra_stim=extra_stim,
        diameter_um=25.0,
        length_um=1_000.0,
        x_offset_um=4_500.0,
        y_um=10.0,
        z_um=-5.0,
    )

    footprint = axs_nrv.sample_life_footprint(
        setup,
        positions_um=[0.0, 10.0, 20.0],
        row=row,
    )

    assert extra_stim.cleared
    assert extra_stim.compute_calls[0]["row_id"] == 2_000_007
    np.testing.assert_allclose(extra_stim.compute_calls[0]["positions"], [0.0, 10.0, 20.0])
    np.testing.assert_allclose(footprint.positions_um, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(footprint.values_V_per_A, [1.0, 2.0, 3.0])
    assert footprint.metadata["source"] == "nrv.FEM_stimulation/LIFE_electrode"


def test_life_stimulation_and_current_replacement_use_one_drive_contract():
    footprint = axs.ExtracellularFootprint.shared(
        values=[1.0, 2.0, 3.0],
        positions=[0.0, 10.0, 20.0] * axs.um,
    )
    stimulation = axs_nrv.life_stimulation_from_footprint(
        footprint,
        current=5.0,
        start_ms=0.1,
        pulse_duration_ms=0.2,
    )
    assert stimulation.drives[0].id == axs.DriveId("nrv_life")

    axon = axs.axons.HodgkinHuxley(
        length=20.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=3,
    )
    simulation = axs.AxonInstance(axon)
    simulation.add_extracellular_stimulation(stimulation=stimulation)

    axs_nrv.replace_life_current(
        simulation,
        9.0 * axs.uA,
        start_ms=0.1,
        pulse_duration_ms=0.2,
    )

    current_uA = simulation.extracellular_stimulation.drives[0].stimulus.evaluate(
        [0.15] * axs.ms,
        unit=axs.uA,
    )
    np.testing.assert_allclose(current_uA, [-9.0])


def test_nrv_activation_decoding_and_comparisons():
    rows = [
        axs_nrv.NRVFiberRow("0", 0, "mrg", 8.0, 0.0, 0.0),
        axs_nrv.NRVFiberRow("0", 2, "rattay", 0.8, 0.0, 0.0),
    ]
    fascicle = SimpleNamespace(sim_list=[0, 2])
    nerve = SimpleNamespace(fascicles={0: fascicle})
    nrv_result = {
        "fascicle0": {
            "axon0": {"recruited": True},
            "axon1": _FakeAxonResult(False),
        }
    }

    decoded = axs_nrv.nrv_activation_by_row(
        nrv_result,
        nerve,
        rows,
        t_start_ms=0.1,
    )
    comparisons = axs_nrv.activation_comparisons(
        rows,
        nrv_activated=decoded,
        axonscope_activated=[True, True],
    )

    assert decoded == {("0", 0): True, ("0", 2): False}
    assert [item.matched for item in comparisons] == [True, False]
