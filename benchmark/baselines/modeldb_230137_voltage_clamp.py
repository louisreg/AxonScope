"""Extract voltage-clamp references from a compiled ModelDB 230137 checkout.

The ModelDB MOD files are intentionally not vendored. Compile ``Nav11_a.mod``
through ``Nav19_a.mod`` and ``vclmp_pl.mod`` with ``nrnivmodl``, then run this
script through NEURON's headless Python mode and pass ``--output``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from neuron import h


PROTOCOLS = {
    "nav11": (-120, 15, -80, 60, -10, 100, -140, 0, -10, 100, 1, 10_000),
    "nav12": (-120, 10, -80, 60, -10, 100, -140, -10, -10, 100, 1, 5_000),
    "nav13": (-90, 20, -100, 60, -10, 1_000, -100, 15, -10, 100, 1, 5_000),
    "nav14": (-120, 12, -80, 60, -10, 100, -140, -20, -10, 100, 1, 1_000),
    "nav15": (-120, 20, -90, 60, -10, 500, -120, 0, -20, 1_000, 0.1, 5_000),
    "nav16": (-90, 7.5, -80, 80, 0, 1_000, -120, 0, 0, 100, 0.1, 200),
    "nav17": (-140, 25, -80, 60, -20, 500, -150, -10, -20, 50, 0.1, 2_000),
    "nav18": (-70, 50, -80, 60, 0, 500, -80, 20, 0, 100, 1, 1_000),
    "nav19": (-120, 150, -100, 40, -40, 300, -140, 10, -40, 300, 1, 1_000),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()

    h.load_file("stdrun.hoc")
    h.dt = 0.0125
    h.steps_per_ms = 80
    h.celsius = 22
    result = {
        name: run_isoform(name, protocol)
        for name, protocol in PROTOCOLS.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"ModelDB 230137 voltage-clamp reference: {args.output}")


def run_isoform(name: str, protocol: tuple[float, ...]) -> dict[str, list[float]]:
    (
        holding_mV,
        step_ms,
        start_mV,
        stop_mV,
        availability_test_mV,
        conditioning_ms,
        availability_start_mV,
        availability_stop_mV,
        recovery_condition_mV,
        recovery_condition_ms,
        recovery_min_ms,
        recovery_max_ms,
    ) = protocol
    suffix = f"na{name.removeprefix('nav')}a"
    section = h.Section(name=name)
    section.L = 63.66198
    section.diam = 50
    section.nseg = 1
    section.insert(suffix)
    section.ena = 65
    segment = section(0.5)
    clamp = h.VClamp_plus(segment)

    iv_voltage = np.arange(start_mV, stop_mV + 1, 5, dtype=float)
    iv_current, iv_conductance = run_iv(
        suffix, segment, clamp, holding_mV, step_ms, iv_voltage
    )
    availability_voltage = np.arange(
        availability_start_mV, availability_stop_mV + 1, 5, dtype=float
    )
    availability = run_availability(
        suffix,
        segment,
        clamp,
        holding_mV,
        conditioning_ms,
        availability_voltage,
        availability_test_mV,
        _availability_test_ms(name),
    )
    recovery_ms = modeldb_recovery_intervals(recovery_min_ms, recovery_max_ms)
    recovery = run_recovery(
        suffix,
        segment,
        clamp,
        holding_mV,
        recovery_condition_mV,
        recovery_condition_ms,
        recovery_ms,
        _recovery_test_ms(name),
    )
    return {
        "voltage_mV": iv_voltage.tolist(),
        "peak_current_mA_cm2": iv_current.tolist(),
        "peak_conductance_S_cm2": iv_conductance.tolist(),
        "availability_voltage_mV": availability_voltage.tolist(),
        "availability": availability.tolist(),
        "recovery_ms": recovery_ms.tolist(),
        "recovery": recovery.tolist(),
    }


def run_iv(suffix, segment, clamp, holding_mV, step_ms, voltages):
    clamp.dur[0] = 1
    clamp.amp[0] = holding_mV
    clamp.dur[1] = step_ms
    clamp.dur[2] = 2
    clamp.amp[2] = holding_mV
    h.tstop = step_ms + 3
    currents = []
    conductances = []
    for voltage_mV in voltages:
        clamp.amp[1] = voltage_mV
        time, current, conductance = run_trace(suffix, segment, holding_mV)
        mask = (time > 1) & (time < 1 + step_ms)
        selected = current[mask]
        currents.append(selected[np.argmax(np.abs(selected))])
        conductances.append(np.max(conductance[mask]))
    return np.asarray(currents), np.asarray(conductances)


def run_availability(
    suffix,
    segment,
    clamp,
    holding_mV,
    conditioning_ms,
    conditioning_voltages,
    test_mV,
    test_ms,
):
    clamp.dur[0] = 10
    clamp.amp[0] = holding_mV
    clamp.dur[1] = conditioning_ms
    clamp.dur[2] = test_ms
    clamp.amp[2] = test_mV
    clamp.dur[3] = 10
    clamp.amp[3] = holding_mV
    h.tstop = 20 + conditioning_ms + test_ms
    peaks = []
    test_start = 10 + conditioning_ms
    for conditioning_mV in conditioning_voltages:
        clamp.amp[1] = conditioning_mV
        time, current, _ = run_trace(suffix, segment, holding_mV)
        mask = (time > test_start) & (time < test_start + test_ms)
        peaks.append(np.max(np.abs(current[mask])))
    values = np.asarray(peaks)
    return values / np.max(values)


def run_recovery(
    suffix,
    segment,
    clamp,
    holding_mV,
    conditioning_mV,
    conditioning_ms,
    recovery_ms,
    test_ms,
):
    values = []
    for interval_ms in recovery_ms:
        clamp.dur[0] = 10
        clamp.amp[0] = holding_mV
        clamp.dur[1] = conditioning_ms
        clamp.amp[1] = conditioning_mV
        clamp.dur[2] = interval_ms
        clamp.amp[2] = holding_mV
        clamp.dur[3] = test_ms
        clamp.amp[3] = conditioning_mV
        clamp.dur[4] = 10
        clamp.amp[4] = holding_mV
        h.tstop = 20 + conditioning_ms + interval_ms + test_ms
        first_start = 10
        second_start = 10 + conditioning_ms + interval_ms
        time, current, _ = run_trace(suffix, segment, holding_mV)
        first_mask = (time > first_start) & (time < first_start + 10)
        second_mask = (time > second_start) & (time < second_start + 10)
        first_peak = np.max(np.abs(current[first_mask]))
        second_peak = np.max(np.abs(current[second_mask]))
        values.append(second_peak / first_peak)
    return np.asarray(values)


def run_trace(suffix, segment, holding_mV):
    time = h.Vector().record(h._ref_t)
    current = h.Vector().record(getattr(segment, f"_ref_ina_{suffix}"))
    conductance = h.Vector().record(getattr(segment, f"_ref_g_{suffix}"))
    h.finitialize(holding_mV)
    h.continuerun(h.tstop)
    return np.asarray(time), np.asarray(current), np.asarray(conductance)


def modeldb_recovery_intervals(start_ms: float, stop_ms: float) -> np.ndarray:
    values = []
    value = float(start_ms)
    while value <= stop_ms * (1.0 + 1e-12):
        values.append(value)
        if value < 1.0:
            value += 0.1
        elif value < 10.0:
            value += 1.0
        elif value < 100.0:
            value += 10.0
        elif value < 1_000.0:
            value += 100.0
        elif value < 10_000.0:
            value += 1_000.0
        else:
            value += 10_000.0
    return np.asarray(values, dtype=float)


def _availability_test_ms(name: str) -> float:
    return {"nav14": 50.0, "nav18": 40.0, "nav19": 50.0}.get(name, 20.0)


def _recovery_test_ms(name: str) -> float:
    return {"nav18": 10.0, "nav19": 50.0}.get(name, 20.0)


if __name__ == "__main__":
    main()
