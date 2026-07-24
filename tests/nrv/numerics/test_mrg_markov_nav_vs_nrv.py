from __future__ import annotations

import numpy as np
import nrv

import axonfleet as axs
from tests.nrv._helpers import (
    axonfleet_x_um,
    enable_nrv_recordings,
    normalize_nrv_matrix,
    trace_metrics,
)


DT_MS = 0.005
TSIM_MS = 4.0
OCCUPANCIES = ("C1", "C2", "O1", "O2", "I1", "I2")


def _markov_mrg():
    template = axs.axons.MRGLikeDoubleCableTemplate(
        diameter=10.0 * axs.um,
        nodes=7,
    )
    defaults = template.default_membranes()
    node = axs.membranes.Composite(
        {
            "mrg_k_leak": axs.membranes.AxNode(
                gnapbar=0.0 * axs.mS_per_cm2,
                gnabar=0.0 * axs.mS_per_cm2,
            ),
            "nav11": axs.membranes.Nav11(
                gbar=11_900.0 * axs.mS_per_cm2,
                ena=50.0 * axs.mV,
                celsius=37.0 * axs.degC,
            ),
            "nav16": axs.membranes.Nav16(
                gbar=10.0 * axs.mS_per_cm2,
                ena=50.0 * axs.mV,
                celsius=37.0 * axs.degC,
            ),
        }
    )
    membranes = axs.membranes.SectionLayout(
        node=node,
        mysa=defaults.membrane_for("MYSA"),
        flut=defaults.membrane_for("FLUT"),
        stin=defaults.membrane_for("STIN"),
    )
    axon = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=7,
        membranes=membranes,
    )
    center_node = len(axon.node_indices) // 2
    center_index = int(axon.node_indices[center_node])
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=float(axonfleet_x_um(axon)[center_index]) * axs.um,
        current=axs.Stimulus.pulse(
            start=1.0 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=5.0 * axs.nA,
        ),
    )
    return instance, center_node, center_index


def test_mrg_markov_nav_occupancies_match_nrv() -> None:
    instance, center_node, center_index = _markov_mrg()
    result = axs.AxonSimulation(
        instance,
        duration=TSIM_MS * axs.ms,
        dt=DT_MS * axs.ms,
        recording=axs.Recording.full(),
    ).run().single

    axon_nrv = nrv.myelinated(
        0,
        0,
        10.0,
        float(instance.axon.length),
        model="MRG",
        dt=DT_MS,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )
    axon_nrv.set_Markov_Nav()
    axon_nrv.insert_I_Clamp_node(
        index=center_node,
        t_start=1.0,
        duration=0.1,
        amplitude=5.0,
    )
    enable_nrv_recordings(axon_nrv)
    reference = axon_nrv.simulate(t_sim=TSIM_MS)

    recorded = result.signal(axs.signals.MARKOV_OCCUPANCIES)
    assert recorded
    t = np.asarray(result.t, dtype=float)
    t_nrv = np.asarray(reference["t"], dtype=float)
    metrics: list[str] = []
    failures: list[str] = []
    for channel, nrv_suffix in (("nav11", "nav11"), ("nav16", "nav16")):
        axonfleet_states = []
        nrv_states = []
        for state in OCCUPANCIES:
            axonfleet_trace = np.asarray(
                recorded[f"{channel}.{state}"][:, center_index],
                dtype=float,
            )
            nrv_values = np.asarray(reference[f"{state}_{nrv_suffix}"], dtype=float)
            nrv_trace = np.interp(t, t_nrv, nrv_values[center_node])
            rmse, _, q99_abs = trace_metrics(nrv_trace, axonfleet_trace)
            metrics.append(f"{channel}.{state}: rmse={rmse:.5f}, q99={q99_abs:.5f}")
            if rmse >= 0.05 or q99_abs >= 0.15:
                failures.append(metrics[-1])
            axonfleet_states.append(axonfleet_trace)
            nrv_states.append(nrv_trace)

        np.testing.assert_allclose(
            np.sum(axonfleet_states, axis=0),
            1.0,
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            np.sum(nrv_states, axis=0),
            1.0,
            rtol=2e-5,
            atol=2e-5,
        )

    assert not failures, "\n".join(metrics)

    vm_nrv = normalize_nrv_matrix(
        reference["V_mem"],
        t_nrv,
        np.asarray(reference["x_rec"], dtype=float),
    )
    node_position = float(axonfleet_x_um(instance.axon)[center_index])
    nrv_node_index = int(np.argmin(np.abs(np.asarray(reference["x_rec"]) - node_position)))
    vm_reference = np.interp(t, t_nrv, vm_nrv[nrv_node_index])
    vm_rmse, _, _ = trace_metrics(vm_reference, np.asarray(result.Vm)[:, center_index])
    assert vm_rmse < 2.0
