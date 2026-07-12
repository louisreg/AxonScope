from __future__ import annotations

import numpy as np
import pytest

import nrv

from axonscope import um
from axonscope.axons.myelinated import MRG
from axonscope.runtime.solver_axon import build_solver_axon
from tests.nrv._helpers import (
    axonscope_compartment_lengths_um,
    axonscope_section_names,
    axonscope_x_um,
)


def _ordered_nrv_sections(axon_nrv) -> list[tuple[str, object]]:
    section_lists = {
        "node": axon_nrv.node,
        "MYSA": axon_nrv.MYSA,
        "FLUT": axon_nrv.FLUT,
        "STIN": axon_nrv.STIN,
    }
    return [
        (kind, section_lists[kind][int(idx)])
        for kind, idx in zip(axon_nrv.axon_path_type, axon_nrv.axon_path_index, strict=True)
    ]


def _trim_tiny_terminal_section(sections: list[tuple[str, object]]) -> list[tuple[str, object]]:
    """NRV can emit a terminal section of ~1e-9 µm due to floating-point stop logic.

    This is not a physically meaningful extra compartment, so we trim it before
    strict geometry comparisons.
    """
    if not sections:
        return sections
    _, sec = sections[-1]
    if float(sec.L) <= 1e-6:
        return sections[:-1]
    return sections


def _normalize_terminal_node_convention(
    axon_as: MRG,
    axon_nrv,
    nrv_sections: list[tuple[str, object]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Align AxonScope with NRV when NRV collapses the last node to ~0 length.

    For some exact-length cases, NRV ends the geometry with a numerically tiny
    final node because of floating accumulation in the stop condition. AxonScope
    keeps the intended full last node. For geometry auditing we strip that lone
    endpoint node so we compare the physically meaningful shared prefix.
    """
    x_as = axonscope_x_um(axon_as)
    keep = np.where(x_as <= float(axon_as.length) + 1e-6)[0].astype(int)
    kinds = tuple(axonscope_section_names(axon_as)[keep])
    if (
        len(kinds) == len(nrv_sections) + 1
        and kinds[-1] == "node"
        and getattr(axon_nrv, "last_section_kind", None) == "node"
        and float(getattr(axon_nrv, "last_section_size", 0.0)) <= 1e-6
    ):
        keep = keep[:-1]
        kinds = kinds[:-1]
    return keep, kinds


@pytest.mark.parametrize("diameter_um", [5.7, 8.7, 10.0, 14.0])
@pytest.mark.parametrize("nodes", [3, 5, 11])
def test_mrg_compartment_geometry_matches_nrv(diameter_um: float, nodes: int) -> None:
    axon_as = MRG(diameter=diameter_um * um, nodes=nodes)
    solver_as = build_solver_axon(axon_as)
    axon_nrv = nrv.myelinated(
        0,
        0,
        diameter_um,
        float(axon_as.length),
        model="MRG",
        dt=0.005,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )

    nrv_sections = _trim_tiny_terminal_section(_ordered_nrv_sections(axon_nrv))
    keep_as, as_kinds = _normalize_terminal_node_convention(axon_as, axon_nrv, nrv_sections)
    nrv_kinds = tuple(kind for kind, _ in nrv_sections)

    assert as_kinds == nrv_kinds
    assert keep_as.size == len(nrv_sections)

    lengths_as = axonscope_compartment_lengths_um(axon_as)[keep_as]
    diam_as = np.asarray(solver_as.diam_um, dtype=float)[keep_as]
    ra_as = np.asarray(solver_as.Ra_ohm_cm, dtype=float)[keep_as]
    cm_as = np.asarray(solver_as.Cm_uF_cm2, dtype=float)[keep_as]
    xraxial_as = np.asarray(solver_as.xraxial_MOhm_per_cm, dtype=float)[keep_as]
    xg_as = np.asarray(solver_as.xg_S_cm2, dtype=float)[keep_as]
    xc_as = np.asarray(solver_as.xc_uF_cm2, dtype=float)[keep_as]

    lengths_nrv = np.asarray([float(sec.L) for _, sec in nrv_sections], dtype=float)
    diam_nrv = np.asarray([float(sec.diam) for _, sec in nrv_sections], dtype=float)
    ra_nrv = np.asarray([float(sec.Ra) for _, sec in nrv_sections], dtype=float)
    cm_nrv = np.asarray([float(sec.cm) for _, sec in nrv_sections], dtype=float)
    xraxial_nrv = np.asarray([float(sec.xraxial[0]) for _, sec in nrv_sections], dtype=float)
    xg_nrv = np.asarray([float(sec.xg[0]) for _, sec in nrv_sections], dtype=float)
    xc_nrv = np.asarray([float(sec.xc[0]) for _, sec in nrv_sections], dtype=float)

    np.testing.assert_allclose(lengths_as, lengths_nrv, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(diam_as, diam_nrv, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(ra_as, ra_nrv, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(cm_as, cm_nrv, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(xraxial_as, xraxial_nrv, rtol=0.0, atol=2e-2)
    np.testing.assert_allclose(xg_as, xg_nrv, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(xc_as, xc_nrv, rtol=0.0, atol=1e-8)

    x_as = axonscope_x_um(axon_as)[keep_as]
    x_nrv = np.cumsum(lengths_nrv) - 0.5 * lengths_nrv
    np.testing.assert_allclose(x_as, x_nrv, rtol=0.0, atol=2e-3)

    node_mask_as = np.asarray(axon_as.node_mask, dtype=int)[keep_as]
    assert int(np.sum(node_mask_as)) == sum(kind == "node" for kind in nrv_kinds)
