"""Validate one excitable Nav composition in complete cable simulations.

This is a runtime validation workload, not a physiologically validated axon
model. It composes existing membrane sources and exercises the canonical public
simulation, analysis, threshold, and recruitment paths for both cable forms.

Run:
    MPLBACKEND=Agg python benchmark/curves/nav_cable_validation.py \
        --output benchmark/results/p18_nav_cable_validation_local
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs
from axonfleet.membranes.models.borg_kdr import BorgKDR
from axonfleet.utils.units import cm2, mS, ohm


PULSE_START = 1.0 * axs.ms
PULSE_WIDTH = 0.1 * axs.ms
SUPRATHRESHOLD_CURRENT = 15.0 * axs.uA
SIGMA = 0.3 * axs.S_per_m
RECRUITMENT_DISTANCES_UM = (15.0, 20.0, 30.0, 40.0, 60.0)
RECRUITMENT_AMPLITUDES_UA = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)


@dataclass(frozen=True)
class CableCase:
    name: str
    duration_ms: float
    dt_ms: float
    compartments: int
    active_compartments: int


CABLE_CASES = {
    "single": CableCase("single", 10.0, 0.005, 201, 201),
    "double": CableCase("double", 5.0, 0.001, 111, 11),
}


def canonical_membrane():
    """Return the benchmark-only Nav1.6 + KDR + leak composition."""

    return axs.membranes.Composite(
        {
            "sodium": axs.membranes.Nav16(
                celsius=22.0 * axs.degC,
                gbar=3000.0 * mS / cm2,
            ),
            "potassium": BorgKDR(
                celsius=22.0 * axs.degC,
                gkdrbar=80.0 * mS / cm2,
                ek=-90.0 * axs.mV,
            ),
            "leak": axs.membranes.Passive(
                Rm=(1000.0 / 7.0) * ohm * cm2,
                EL=-73.20 * axs.mV,
            ),
        }
    )


def build_axon(cable: str):
    """Build one canonical full-cable validation axon."""

    active = canonical_membrane()
    if cable == "single":
        return axs.axons.Unmyelinated(
            membrane=active,
            length=2000.0 * axs.um,
            diameter=1.0 * axs.um,
            compartments=CABLE_CASES[cable].compartments,
            v_init=-70.0 * axs.mV,
            temperature=22.0 * axs.degC,
        )
    if cable == "double":
        passive = axs.membranes.Passive(EL=-70.0 * axs.mV)
        return axs.axons.MRG(
            diameter=10.0 * axs.um,
            nodes=CABLE_CASES[cable].active_compartments,
            compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
            membranes=axs.membranes.SectionLayout(
                node=active,
                mysa=passive,
                flut=passive,
                stin=passive,
            ),
            v_init=-70.0 * axs.mV,
            temperature=22.0 * axs.degC,
        )
    raise ValueError(f"Unknown cable case: {cable!r}.")


def build_simulation(cable: str, *, distance_um: float = 20.0) -> axs.AxonInstance:
    """Attach one zero-current proximal point-source drive."""

    axon = build_axon(cable)
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    electrode_x = (
        positions[0]
        if cable == "single"
        else axon.node_position("proximal", unit=axs.um)
    )
    electrode = axs.analytical.PointSourceElectrode(
        x=electrode_x,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )
    stimulation = axs.analytical.point_source_stimulation(
        electrode,
        positions,
        sigma=SIGMA,
        stimulus=axs.Stimulus.constant(0.0 * axs.uA),
        axon_z=distance_um * axs.um,
    )
    simulation = axs.AxonInstance(axon)
    simulation.add_extracellular_stimulation(stimulation=stimulation)
    return simulation


def waveform_update() -> axs.protocols.ExtracellularWaveformUpdate:
    """Return the typed numeric-axis update shared by every protocol."""

    return axs.protocols.ExtracellularWaveformUpdate(
        lambda current: axs.Stimulus.pulse(
            start=PULSE_START,
            duration=PULSE_WIDTH,
            amplitude=-current,
        )
    )


def activation_criterion() -> axs.analysis.Activation:
    """Count only spikes reaching the distal end after stimulation begins."""

    return axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=PULSE_START,
        target=axs.positions.DISTAL,
    )


def describe_workloads() -> dict[str, Any]:
    """Return stable campaign metadata without executing a solver."""

    return {
        "status": "runtime_validation_only",
        "composition": {
            "components": ["nav16", "borg_kdr", "passive"],
            "temperature_degC": 22.0,
            "v_init_mV": -70.0,
            "nav_gbar_mS_cm2": 3000.0,
            "kdr_gbar_mS_cm2": 80.0,
            "leak_g_mS_cm2": 7.0,
            "leak_reversal_mV": -73.20,
        },
        "cases": {name: asdict(case) for name, case in CABLE_CASES.items()},
        "stimulation": {
            "point_source_distance_um": 20.0,
            "pulse_start_ms": 1.0,
            "pulse_width_ms": 0.1,
            "suprathreshold_current_uA": 15.0,
        },
        "validation": ["waveform", "threshold", "velocity", "recruitment"],
    }


def run_waveform(cable: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case = CABLE_CASES[cable]
    pool = (build_simulation(cable), build_simulation(cable))
    update = waveform_update()
    simulations = tuple(
        update(row, amplitude)
        for row, amplitude in zip(
            pool,
            (0.0 * axs.uA, SUPRATHRESHOLD_CURRENT),
            strict=True,
        )
    )
    started = perf_counter()
    results = axs.AxonSimulation(
        simulations,
        duration=case.duration_ms * axs.ms,
        dt=case.dt_ms * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()
    elapsed_s = perf_counter() - started

    control, stimulated = results
    control_vm = control.voltage_values(unit=axs.mV)
    stimulated_vm = stimulated.voltage_values(unit=axs.mV)
    time_ms = stimulated.time_values(unit=axs.ms)
    positions_um = stimulated.position_values(unit=axs.um)
    criterion = activation_criterion()
    event = criterion.detect(stimulated)
    velocity_definition = axs.analysis.ConductionVelocity(
        threshold=0.0 * axs.mV,
        peak_height=None,
        min_width=None,
        spatial_filter="nodes_if_available",
    )
    velocity = stimulated.analyze(velocity_definition)
    velocity_value = float(np.asarray(velocity.values, dtype=float)[0])

    summary = {
        "elapsed_s": elapsed_s,
        "control_max_abs_drift_mV": float(np.max(np.abs(control_vm + 70.0))),
        "peak_mV": float(np.max(stimulated_vm)),
        "distal_peak_mV": float(np.max(stimulated_vm[:, -1])),
        "distal_activated": bool(event.activated),
        "distal_first_time_ms": event.first_time_ms,
        "velocity_m_s": velocity_value,
        "velocity_status": velocity.statuses[0].value,
    }
    arrays = {
        "time_ms": np.asarray(time_ms, dtype=float),
        "positions_um": np.asarray(positions_um, dtype=float),
        "control_vm_mV": np.asarray(control_vm, dtype=float),
        "stimulated_vm_mV": np.asarray(stimulated_vm, dtype=float),
    }
    return summary, arrays


def run_threshold(cable: str) -> dict[str, Any]:
    case = CABLE_CASES[cable]
    started = perf_counter()
    curve = axs.Runner().run(
        axs.protocols.find_threshold(
            (build_simulation(cable),),
            rows=(cable,),
            update=waveform_update(),
            bounds=(0.0 * axs.uA, 20.0 * axs.uA),
            duration=case.duration_ms * axs.ms,
            dt=case.dt_ms * axs.ms,
            criterion=activation_criterion(),
            tolerance=0.1 * axs.uA,
            relative_tolerance=0.01,
            max_iterations=12,
            recording=axs.Recording.none(),
        )
    )
    return {
        "elapsed_s": perf_counter() - started,
        "threshold_uA": float(curve.threshold_uA[0]),
        "lower_bound_uA": float(curve.lower_bound_uA[0]),
        "upper_bound_uA": float(curve.upper_bound_uA[0]),
        "status": curve.status[0],
        "evaluations": curve.n_iterations,
    }


def run_recruitment(cable: str) -> tuple[dict[str, Any], Any]:
    case = CABLE_CASES[cable]
    pool = tuple(
        build_simulation(cable, distance_um=distance_um)
        for distance_um in RECRUITMENT_DISTANCES_UM
    )
    amplitudes = np.asarray(RECRUITMENT_AMPLITUDES_UA) * axs.uA
    started = perf_counter()
    curve = axs.Runner().run(
        axs.protocols.recruitment_sweep(
            pool,
            update=waveform_update(),
            values=amplitudes,
            duration=case.duration_ms * axs.ms,
            dt=case.dt_ms * axs.ms,
            criterion=activation_criterion(),
            recording=axs.Recording.none(),
            batch_amplitudes=True,
            amplitude_batch_size=None,
        )
    )
    summary = {
        "elapsed_s": perf_counter() - started,
        "distances_um": list(RECRUITMENT_DISTANCES_UM),
        "amplitudes_uA": curve.amplitudes_uA.tolist(),
        "activated": curve.activated.tolist(),
        "fraction": curve.fraction.tolist(),
        "first_activation_uA": curve.first_activation_uA.tolist(),
    }
    return summary, curve


def plot_results(
    waveforms: dict[str, dict[str, np.ndarray]],
    recruitment: dict[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    for column, cable in enumerate(("single", "double")):
        arrays = waveforms[cable]
        time_ms = arrays["time_ms"]
        vm = arrays["stimulated_vm_mV"]
        axes[0, column].plot(time_ms, vm[:, 0], label="proximal")
        axes[0, column].plot(time_ms, vm[:, -1], label="distal")
        axes[0, column].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[0, column].set_title(f"{cable.capitalize()} cable waveform")
        axes[0, column].set_xlabel("Time [ms]")
        axes[0, column].set_ylabel("Vm [mV]")
        axes[0, column].legend()

        curve = recruitment[cable]
        axes[1, column].plot(
            curve.amplitudes_uA,
            curve.fraction,
            marker="o",
        )
        axes[1, column].set_title(f"{cable.capitalize()} cable recruitment")
        axes[1, column].set_xlabel("Cathodic current magnitude [uA]")
        axes[1, column].set_ylabel("Activated fraction")
        axes[1, column].set_ylim(-0.03, 1.03)
        axes[1, column].grid(True, alpha=0.25)
    figure.savefig(output / "nav_cable_validation.png", dpi=160)
    plt.close(figure)


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply numerical-consistency gates without claiming physiological truth."""

    checks: dict[str, bool] = {}
    for cable in ("single", "double"):
        waveform = summary["waveform"][cable]
        threshold = summary["threshold"][cable]
        recruitment = summary["recruitment"][cable]
        fraction = np.asarray(recruitment["fraction"], dtype=float)
        checks[f"{cable}.control_subthreshold"] = (
            waveform["control_max_abs_drift_mV"] < 10.0
        )
        checks[f"{cable}.distal_propagation"] = bool(
            waveform["distal_activated"] and waveform["distal_peak_mV"] > 0.0
        )
        checks[f"{cable}.finite_positive_velocity"] = bool(
            np.isfinite(waveform["velocity_m_s"])
            and waveform["velocity_m_s"] > 0.0
        )
        checks[f"{cable}.bounded_threshold"] = bool(
            threshold["status"] == "threshold"
            and 0.0 < threshold["threshold_uA"] < 20.0
        )
        checks[f"{cable}.monotone_recruitment"] = bool(
            fraction[0] == 0.0
            and fraction[-1] == 1.0
            and np.all(np.diff(fraction) >= 0.0)
        )
    return {"accepted": all(checks.values()), "checks": checks}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    description = describe_workloads()
    if args.dry_run:
        print(json.dumps(description, indent=2))
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used.")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(description)
    summary["waveform"] = {}
    summary["threshold"] = {}
    summary["recruitment"] = {}
    waveform_arrays: dict[str, dict[str, np.ndarray]] = {}
    recruitment_curves: dict[str, Any] = {}
    for cable in ("single", "double"):
        waveform_summary, arrays = run_waveform(cable)
        threshold_summary = run_threshold(cable)
        recruitment_summary, curve = run_recruitment(cable)
        summary["waveform"][cable] = waveform_summary
        summary["threshold"][cable] = threshold_summary
        summary["recruitment"][cable] = recruitment_summary
        waveform_arrays[cable] = arrays
        recruitment_curves[cable] = curve
        np.savez_compressed(args.output / f"{cable}_waveform.npz", **arrays)

    summary["acceptance"] = validate_summary(summary)
    plot_results(waveform_arrays, recruitment_curves, args.output)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Nav cable validation results: {args.output}")
    if not summary["acceptance"]["accepted"]:
        raise RuntimeError("Nav cable validation acceptance gate failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
