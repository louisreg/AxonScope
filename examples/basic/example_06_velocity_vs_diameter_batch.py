"""Example 06: propagation velocity, NRV tutorial style.

Run:
    python examples/basic/example_06_velocity_vs_diameter_batch.py

This example mirrors NRV's propagation-velocity tutorial with AxonScope:

1. stimulate a Hodgkin-Huxley unmyelinated axon and rasterize propagation;
2. stimulate an MRG myelinated axon and rasterize saltatory propagation;
3. sweep diameter for HH and MRG fibers;
4. plot velocity-diameter curves on linear and log-log axes.

The sweeps are executed with ``simulate_pool`` so compatible rows are batched by
the dispatcher when possible. The constants below intentionally follow the NRV
tutorial defaults rather than choosing faster demonstration settings. MRG fibers
are specified by node count directly; AxonScope derives the corresponding
layout length from that topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import axonscope as axs


FiberFamily = Literal["unmyelinated", "myelinated"]

UNMYELINATED_REFERENCE_DIAMETER = 1.0 * axs.um
UNMYELINATED_REFERENCE_LENGTH = 2000.0 * axs.um
UNMYELINATED_REFERENCE_COMPARTMENTS = 201
UNMYELINATED_SWEEP_LENGTH = 5000.0 * axs.um
UNMYELINATED_SWEEP_COMPARTMENTS = 501
UNMYELINATED_DIAMETERS = np.linspace(0.1, 2.0, 10) * axs.um
UNMYELINATED_DURATION = 10.0 * axs.ms

MYELINATED_NODES = 21
MYELINATED_COMPARTMENTS = {"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1}
MYELINATED_DIAMETERS = np.linspace(2.0, 20.0, 10) * axs.um
MYELINATED_REFERENCE_DIAMETER_UM = 10.0
MYELINATED_DURATION = 5.0 * axs.ms

CLAMP_START = 1.0 * axs.ms
UNMYELINATED_CLAMP_DURATION = 0.1 * axs.ms
UNMYELINATED_CLAMP_CURRENT = 5.0 * axs.nA
MYELINATED_CLAMP_DURATION = 0.1 * axs.ms
MYELINATED_CLAMP_CURRENT = 5.0 * axs.nA
UNMYELINATED_DT = 0.001 * axs.ms
MYELINATED_DT = 0.001 * axs.ms
RASTER_THRESHOLD = -40.0 * axs.mV
RASTER_MIN_DISTANCE = 0.5 * axs.ms
NRV_PEAK_HEIGHT_RANGE_MV = (-20.0, 70.0)
NRV_MIN_AP_WIDTH = 0.1 * axs.ms


@dataclass(frozen=True)
class RowInfo:
    """Metadata attached to one simulated fiber."""

    family: FiberFamily
    label: str
    diameter_um: float
    is_reference: bool = False
    include_in_sweep: bool = True


def make_hh_axon(
    *,
    diameter: object,
    length: object,
    compartments: int,
) -> axs.axons.HodgkinHuxley:
    """Return a Hodgkin-Huxley unmyelinated axon."""

    return axs.axons.HodgkinHuxley(
        length=length,
        diameter=diameter,
        compartments=compartments,
        celsius=32.0 * axs.degC,
        v_init=-67.5 * axs.mV,
        include_passive_leak=True,
        g_pas=0.001,
        e_pas=-70.0,
    )


def make_mrg_axon(*, diameter: object) -> axs.axons.MRG:
    """Return an MRG myelinated axon with 21 Ranvier nodes."""

    return axs.axons.MRG(
        diameter=diameter,
        nodes=MYELINATED_NODES,
        compartments=MYELINATED_COMPARTMENTS,
    )


def first_compartment_position(axon: axs.axons.Axon) -> object:
    """Return the first compartment position, like NRV ``insert_I_Clamp(0, ...)``."""

    return float(axon.layout.position_values(unit=axs.um)[0]) * axs.um


def add_current_clamp(
    sim: axs.AxonSimulation,
    *,
    position: object,
    duration: object,
    current: object,
) -> None:
    """Attach the intracellular test pulse used by the NRV tutorial."""

    sim.add_current_clamp(
        position_um=position,
        current=axs.Stimulus.pulse(
            start=CLAMP_START,
            duration=duration,
            amplitude=current,
        ),
    )


def make_unmyelinated_pool() -> tuple[tuple[axs.AxonSimulation, ...], tuple[RowInfo, ...]]:
    """Build the HH reference fiber plus the HH diameter sweep."""

    simulations: list[axs.AxonSimulation] = []
    rows: list[RowInfo] = []

    reference = make_hh_axon(
        diameter=UNMYELINATED_REFERENCE_DIAMETER,
        length=UNMYELINATED_REFERENCE_LENGTH,
        compartments=UNMYELINATED_REFERENCE_COMPARTMENTS,
    )
    reference_sim = axs.AxonSimulation(reference, y_um=0.0 * axs.um, z_um=0.0 * axs.um)
    add_current_clamp(
        reference_sim,
        position=0.0 * axs.um,
        duration=UNMYELINATED_CLAMP_DURATION,
        current=UNMYELINATED_CLAMP_CURRENT,
    )
    simulations.append(reference_sim)
    rows.append(
        RowInfo(
            family="unmyelinated",
            label="HH reference",
            diameter_um=float(UNMYELINATED_REFERENCE_DIAMETER.to(axs.um).magnitude),
            is_reference=True,
            include_in_sweep=False,
        )
    )

    for diameter in UNMYELINATED_DIAMETERS:
        axon = make_hh_axon(
            diameter=diameter,
            length=UNMYELINATED_SWEEP_LENGTH,
            compartments=UNMYELINATED_SWEEP_COMPARTMENTS,
        )
        sim = axs.AxonSimulation(axon, y_um=0.0 * axs.um, z_um=0.0 * axs.um)
        add_current_clamp(
            sim,
            position=0.0 * axs.um,
            duration=UNMYELINATED_CLAMP_DURATION,
            current=UNMYELINATED_CLAMP_CURRENT,
        )
        simulations.append(sim)
        rows.append(
            RowInfo(
                family="unmyelinated",
                label="HH sweep",
                diameter_um=float(diameter.to(axs.um).magnitude),
            )
        )

    return tuple(simulations), tuple(rows)


def make_myelinated_pool() -> tuple[tuple[axs.AxonSimulation, ...], tuple[RowInfo, ...]]:
    """Build the MRG diameter sweep."""

    simulations: list[axs.AxonSimulation] = []
    rows: list[RowInfo] = []
    for diameter in MYELINATED_DIAMETERS:
        axon = make_mrg_axon(diameter=diameter)
        sim = axs.AxonSimulation(axon, y_um=0.0 * axs.um, z_um=0.0 * axs.um)
        add_current_clamp(
            sim,
            position=first_compartment_position(axon),
            duration=MYELINATED_CLAMP_DURATION,
            current=MYELINATED_CLAMP_CURRENT,
        )
        diameter_um = float(diameter.to(axs.um).magnitude)
        simulations.append(sim)
        rows.append(
            RowInfo(
                family="myelinated",
                label="MRG sweep",
                diameter_um=diameter_um,
                is_reference=np.isclose(diameter_um, MYELINATED_REFERENCE_DIAMETER_UM),
            )
        )
    return tuple(simulations), tuple(rows)


def run_pool(
    simulations: tuple[axs.AxonSimulation, ...],
    *,
    duration: object,
    dt: object,
    label: str,
) -> list[axs.SimResult]:
    """Print the dispatch plan and run one pool."""

    print(f"\n=== {label} dispatch ===")
    plan = axs.dispatcher.build_dispatch_plan(simulations)
    axs.dispatcher.print_dispatch_plan(plan)
    return axs.simulate_pool(
        simulations,
        duration_ms=duration,
        dt_ms=dt,
        recording=axs.Recording.voltage(),
        progress=True,
    )


def velocity(result: axs.SimResult) -> float:
    """Return an NRV-style AP speed estimate in m/s."""

    spike_times_ms, spike_positions_um = nrv_like_rasterize(result)
    if spike_times_ms.size < 2:
        return 0.0
    order = np.argsort(spike_times_ms)
    times_ms = np.asarray(spike_times_ms, dtype=float)[order]
    positions_um = np.asarray(spike_positions_um, dtype=float)[order]
    x_start = float(positions_um[0])
    t_start = float(times_ms[0])
    stop_index = int(np.argmax(np.abs(positions_um - x_start)))
    distance_um = abs(float(positions_um[stop_index]) - x_start)
    delay_ms = float(times_ms[stop_index]) - t_start
    if delay_ms <= 0.0:
        return 0.0
    return float(distance_um / delay_ms * 1e-3)


def nrv_like_rasterize(result: axs.SimResult) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize Vm with the same peak-detection settings as NRV's tutorial.

    NRV's current rasterizer detects Vm peaks with a fixed height window
    ``[-20, 70] mV``, a minimum AP width of ``0.1 ms``, and a refractory
    distance of ``0.5 ms``. For myelinated fibers, NRV's default recording is
    node-only, so this helper filters full AxonScope recordings back to Ranvier
    nodes before estimating velocity.
    """

    vm = np.asarray(result.Vm, dtype=float)
    time_ms = np.asarray(result.t, dtype=float)
    positions_um = axs.results.analysis.recorded_positions_um(result)
    if vm.ndim != 2:
        raise ValueError(f"result.Vm must be 2D, got shape {vm.shape}.")
    if time_ms.shape[0] < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if hasattr(result.axon, "node_indices"):
        original_indices = (
            np.arange(result.axon.n_compartments, dtype=int)
            if result.record_indices is None
            else np.asarray(result.record_indices, dtype=int)
        )
        node_indices = set(int(index) for index in result.axon.node_indices)
        keep_columns = [
            column for column, index in enumerate(original_indices) if int(index) in node_indices
        ]
        if keep_columns:
            vm = vm[:, keep_columns]
            positions_um = positions_um[keep_columns]

    dt_ms = float(np.median(np.diff(time_ms)))
    if dt_ms <= 0.0:
        raise ValueError("result.t must be strictly increasing.")
    distance_points = max(1, int(RASTER_MIN_DISTANCE.to(axs.ms).magnitude / dt_ms))
    width_points = max(1, int(NRV_MIN_AP_WIDTH.to(axs.ms).magnitude / dt_ms))

    spike_times: list[float] = []
    spike_positions: list[float] = []
    for column, position_um in enumerate(positions_um):
        peaks, _ = find_peaks(
            vm[:, column],
            height=NRV_PEAK_HEIGHT_RANGE_MV,
            distance=distance_points,
            width=width_points,
        )
        spike_times.extend(time_ms[peaks])
        spike_positions.extend([float(position_um)] * peaks.shape[0])

    return np.asarray(spike_times, dtype=float), np.asarray(spike_positions, dtype=float)


def sweep_curve(
    results: list[axs.SimResult],
    rows: tuple[RowInfo, ...],
    *,
    family: FiberFamily,
) -> tuple[np.ndarray, np.ndarray]:
    """Return diameter and velocity arrays for one family sweep."""

    diameters: list[float] = []
    speeds: list[float] = []
    for row, result in zip(rows, results, strict=True):
        if row.family != family or not row.include_in_sweep:
            continue
        diameters.append(row.diameter_um)
        speeds.append(velocity(result))
    return np.asarray(diameters, dtype=float), np.asarray(speeds, dtype=float)


def reference_result(
    results: list[axs.SimResult],
    rows: tuple[RowInfo, ...],
    *,
    family: FiberFamily,
) -> axs.SimResult:
    """Return the reference result for one family."""

    for row, result in zip(rows, results, strict=True):
        if row.family == family and row.is_reference:
            return result
    raise ValueError(f"No reference result found for {family}.")


def print_summary(
    *,
    unmyelinated_reference: axs.SimResult,
    myelinated_reference: axs.SimResult,
    unmyelinated_curve: tuple[np.ndarray, np.ndarray],
    myelinated_curve: tuple[np.ndarray, np.ndarray],
) -> None:
    """Print the same velocity quantities highlighted by the NRV tutorial."""

    print("\n=== Reference propagation speeds ===")
    print(f"HH d=1 um:      {velocity(unmyelinated_reference):.3f} m/s")
    print(f"MRG d=10 um:    {velocity(myelinated_reference):.3f} m/s")

    print("\n=== Velocity-diameter sweep ===")
    for name, (diameters_um, speeds_m_s) in (
        ("HH unmyelinated", unmyelinated_curve),
        ("MRG myelinated", myelinated_curve),
    ):
        for diameter_um, speed_m_s in zip(diameters_um, speeds_m_s, strict=True):
            print(f"{name:16s} d={diameter_um:>5.2f} um: {speed_m_s:>8.3f} m/s")


def plot_results(
    *,
    unmyelinated_reference: axs.SimResult,
    myelinated_reference: axs.SimResult,
    unmyelinated_curve: tuple[np.ndarray, np.ndarray],
    myelinated_curve: tuple[np.ndarray, np.ndarray],
) -> None:
    """Plot rasterized propagation and velocity-diameter curves."""

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    ax_hh, ax_mrg, ax_linear, ax_log = axes.ravel()

    axs.results.plot_raster(
        unmyelinated_reference,
        ax=ax_hh,
        threshold_mV=RASTER_THRESHOLD,
        min_distance_ms=RASTER_MIN_DISTANCE,
        line_half_height_um=8.0 * axs.um,
    )
    ax_hh.set_title("HH unmyelinated raster")
    ax_hh.set_xlim(0.0, UNMYELINATED_DURATION.to(axs.ms).magnitude)

    axs.results.plot_raster(
        myelinated_reference,
        ax=ax_mrg,
        threshold_mV=RASTER_THRESHOLD,
        min_distance_ms=RASTER_MIN_DISTANCE,
        line_half_height_um=20.0 * axs.um,
    )
    ax_mrg.set_title("MRG myelinated raster")
    ax_mrg.set_xlim(0.0, MYELINATED_DURATION.to(axs.ms).magnitude)

    unmy_d, unmy_v = unmyelinated_curve
    my_d, my_v = myelinated_curve
    ax_linear.plot(unmy_d, unmy_v, "o-", label="unmyelinated HH")
    ax_linear.plot(my_d, my_v, "o-", label="myelinated MRG")
    ax_linear.set_xlabel(r"diameter ($\mu$m)")
    ax_linear.set_ylabel(r"speed (m.s$^{-1}$)")
    ax_linear.set_title("Velocity-diameter relationship")
    ax_linear.grid(True, alpha=0.3)
    ax_linear.legend()

    ax_log.loglog(unmy_d, unmy_v, "o-", label="unmyelinated HH")
    ax_log.loglog(my_d, my_v, "o-", label="myelinated MRG")
    ax_log.set_xlabel(r"diameter ($\mu$m)")
    ax_log.set_ylabel(r"speed (m.s$^{-1}$)")
    ax_log.set_title("Combined log-log view")
    ax_log.grid(True, which="both", alpha=0.3)
    ax_log.legend()

    plt.show()


def main() -> None:
    unmy_sims, unmy_rows = make_unmyelinated_pool()
    my_sims, my_rows = make_myelinated_pool()

    unmy_results = run_pool(
        unmy_sims,
        duration=UNMYELINATED_DURATION,
        dt=UNMYELINATED_DT,
        label="HH unmyelinated",
    )
    my_results = run_pool(
        my_sims,
        duration=MYELINATED_DURATION,
        dt=MYELINATED_DT,
        label="MRG myelinated",
    )

    unmy_ref = reference_result(
        unmy_results,
        unmy_rows,
        family="unmyelinated",
    )
    my_ref = reference_result(
        my_results,
        my_rows,
        family="myelinated",
    )
    unmy_curve = sweep_curve(
        unmy_results,
        unmy_rows,
        family="unmyelinated",
    )
    my_curve = sweep_curve(
        my_results,
        my_rows,
        family="myelinated",
    )

    print_summary(
        unmyelinated_reference=unmy_ref,
        myelinated_reference=my_ref,
        unmyelinated_curve=unmy_curve,
        myelinated_curve=my_curve,
    )
    plot_results(
        unmyelinated_reference=unmy_ref,
        myelinated_reference=my_ref,
        unmyelinated_curve=unmy_curve,
        myelinated_curve=my_curve,
    )


if __name__ == "__main__":
    main()
