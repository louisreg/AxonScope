from __future__ import annotations

import math
from typing import Optional, Sequence

import jax.numpy as jnp

from axonscope.axons.multicomp import DoubleCableAxon
from axonscope.channel_models.axnode import AxnodeICM
from axonscope.channel_models.passive import PassiveICM
from axonscope.icm import CompartmentMembraneLayout
from axonscope.morphology import mrg as mrg_morphology
from axonscope.morphology.mrg import get_mrg_morphology


def _mrg_section_sequence() -> list[str]:
    return [
        "node",
        "MYSA",
        "FLUT",
        "STIN",
        "STIN",
        "STIN",
        "STIN",
        "STIN",
        "STIN",
        "FLUT",
        "MYSA",
    ]


def mrg_length_from_nodes(diameter: float, nodes: int) -> float:
    """Return the MRG axon length needed to obtain a requested node count."""
    if nodes < 2:
        raise ValueError(f"nodes must be >= 2, got {nodes}")

    diameter = float(diameter)
    if mrg_morphology._is_exact_tabulated_diameter(diameter):
        deltax = get_mrg_morphology(diameter).deltax
    else:
        deltax = float(mrg_morphology._mrg_polynomials()["deltax_length"](diameter))
    return float(math.ceil(deltax * (nodes - 1)))


def mrg_nodes_from_length(diameter: float, length: float) -> int:
    """Return the approximate MRG node count for a requested axon length."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    morphology = get_mrg_morphology(diameter)
    return int(math.floor(length / morphology.deltax)) + 1


def _passive_from_leak(leak_mS_per_cm2: float, e_rev_mV: float) -> PassiveICM:
    leak = max(float(leak_mS_per_cm2), 1e-12)
    return PassiveICM(Rm=1e3 / leak, EL=e_rev_mV)


def _make_mrg_membrane_layout(
    *,
    is_node: Sequence[bool],
    leak_mS_per_cm2: Sequence[float],
    ena_mV: float = 50.0,
    ek_mV: float = -90.0,
    node_el_mV: float = -90.0,
    internode_el_mV: float = -80.0,
    celsius: float = 37.0,
) -> CompartmentMembraneLayout:
    node_model = AxnodeICM(
        ena_mV=ena_mV,
        ek_mV=ek_mV,
        el_mV=node_el_mV,
        celsius=celsius,
    )
    models = [
        node_model if bool(node) else _passive_from_leak(leak, internode_el_mV)
        for node, leak in zip(is_node, leak_mS_per_cm2, strict=True)
    ]
    return CompartmentMembraneLayout(models)


class Myelinated(DoubleCableAxon):
    """Generic base class for one-dimensional myelinated double-cable axons."""

    def __init__(
        self,
        *,
        ion_channel,
        fiber_d_um: float,
        lengths_um: Sequence[float],
        diam_um: Sequence[float],
        Ra_ohm_cm: Sequence[float],
        Cm_uF_cm2: Sequence[float],
        Vinit: float,
        Temp: float,
        nodes: Optional[int] = None,
        kind_vec: Optional[Sequence[str]] = None,
        is_node: Optional[Sequence[bool]] = None,
        xraxial_MOhm_cm: Optional[Sequence[float]] = None,
        xg_S_cm2: Optional[Sequence[float]] = None,
        xc_uF_cm2: Optional[Sequence[float]] = None,
        Veinit: float = 0.0,
        enable_extracellular: bool = True,
    ) -> None:
        super().__init__(
            ion_channel=ion_channel,
            lengths_um=lengths_um,
            diam_um=diam_um,
            Ra_ohm_cm=Ra_ohm_cm,
            Cm_uF_cm2=Cm_uF_cm2,
            Vinit=Vinit,
            Temp=Temp,
            fiber_d_um=fiber_d_um,
            kind_vec=kind_vec,
            is_node=is_node,
            xraxial_MOhm_cm=xraxial_MOhm_cm,
            xg_S_cm2=xg_S_cm2,
            xc_uF_cm2=xc_uF_cm2,
            Veinit=Veinit,
            enable_extracellular=enable_extracellular,
        )
        self.fiber_d_um = float(fiber_d_um)
        self.nodes = int(nodes) if nodes is not None else int(self.node_indices.shape[0])
        self.prefer_inline_extracellular_solver = True


class MRG(Myelinated):
    """McIntyre-Richardson-Grill myelinated axon model."""

    def __init__(
        self,
        d: float,
        nodes: int,
        L: Optional[float] = None,
        Vinit: float = -80.0,
        Temp: float = 37.0,
        fit_all: bool = False,
        paralength1_um: float = 3.0,
        nodelength_um: float = 1.0,
        rhoa_ohm_um: float = 0.7e6,
        mycm_uF_cm2: float = 0.1,
        mygm_S_cm2: float = 0.001,
        space_p1_um: float = 0.002,
        space_p2_um: float = 0.004,
        space_i_um: float = 0.004,
    ) -> None:
        if L is None:
            L = float(mrg_length_from_nodes(d, nodes))
        morph = get_mrg_morphology(d, fit_all=fit_all)

        paralength1 = float(paralength1_um)
        nodelength = float(nodelength_um)
        paralength2 = float(morph.paralength2)
        interlength = float(
            (morph.deltax - nodelength - 2.0 * paralength1 - 2.0 * paralength2) / 6.0
        )

        rhoa = float(rhoa_ohm_um)
        mycm = float(mycm_uF_cm2)
        mygm = float(mygm_S_cm2)
        space_p1 = float(space_p1_um)
        space_p2 = float(space_p2_um)
        space_i = float(space_i_um)

        Rpn0 = (rhoa * 0.01) / (
            math.pi * ((((morph.nodeD / 2.0) + space_p1) ** 2) - ((morph.nodeD / 2.0) ** 2))
        )
        Rpn1 = (rhoa * 0.01) / (
            math.pi * ((((morph.paraD1 / 2.0) + space_p1) ** 2) - ((morph.paraD1 / 2.0) ** 2))
        )
        Rpn2 = (rhoa * 0.01) / (
            math.pi * ((((morph.paraD2 / 2.0) + space_p2) ** 2) - ((morph.paraD2 / 2.0) ** 2))
        )
        Rpx = (rhoa * 0.01) / (
            math.pi * ((((morph.axonD / 2.0) + space_i) ** 2) - ((morph.axonD / 2.0) ** 2))
        )

        seq = _mrg_section_sequence()
        lengths_um: list[float] = []
        diam_um: list[float] = []
        Ra_ohm_cm: list[float] = []
        Cm_uF_cm2: list[float] = []
        leak_mS_cm2: list[float] = []
        is_node: list[bool] = []
        kind_vec: list[str] = []
        xraxial_MOhm_cm: list[float] = []
        xg_S_cm2: list[float] = []
        xc_uF_cm2: list[float] = []

        x0 = 0.0
        k = 0
        while x0 < L - 1e-12:
            kind = seq[k % len(seq)]
            if kind == "node":
                Lk = nodelength
                dk = float(morph.nodeD)
                ratio = 1.0
                Rak = rhoa / 10000.0
                Cmk = 2.0
                leakk_mS = 0.0
                xr, xg, xc = Rpn0, 1e10, 0.0
            elif kind == "MYSA":
                Lk = paralength1
                dk = float(morph.fiberD)
                ratio = float(morph.paraD1 / morph.fiberD)
                Rak = rhoa * (1.0 / ratio**2) / 10000.0
                Cmk = 2.0 * ratio
                leakk_mS = 0.001 * ratio * 1e3
                xr, xg, xc = Rpn1, mygm / (morph.nl * 2.0), mycm / (morph.nl * 2.0)
            elif kind == "FLUT":
                Lk = paralength2
                dk = float(morph.fiberD)
                ratio = float(morph.paraD2 / morph.fiberD)
                Rak = rhoa * (1.0 / ratio**2) / 10000.0
                Cmk = 2.0 * ratio
                leakk_mS = 0.0001 * ratio * 1e3
                xr, xg, xc = Rpn2, mygm / (morph.nl * 2.0), mycm / (morph.nl * 2.0)
            else:
                Lk = interlength
                dk = float(morph.fiberD)
                ratio = float(morph.axonD / morph.fiberD)
                Rak = rhoa * (1.0 / ratio**2) / 10000.0
                Cmk = 2.0 * ratio
                leakk_mS = 0.0001 * ratio * 1e3
                xr, xg, xc = Rpx, mygm / (morph.nl * 2.0), mycm / (morph.nl * 2.0)

            Lk = min(float(Lk), float(L - x0))
            x1 = x0 + Lk
            lengths_um.append(Lk)
            diam_um.append(dk)
            Ra_ohm_cm.append(Rak)
            Cm_uF_cm2.append(Cmk)
            leak_mS_cm2.append(leakk_mS)
            is_node.append(kind == "node")
            kind_vec.append(kind)
            xraxial_MOhm_cm.append(xr)
            xg_S_cm2.append(float(xg))
            xc_uF_cm2.append(float(xc))
            x0 = x1
            k += 1

        node_mask = jnp.asarray(is_node, dtype=bool)
        if int(jnp.sum(node_mask.astype(jnp.int32))) < 2:
            raise ValueError("Generated MRG mesh has fewer than 2 nodes.")

        membrane_layout = _make_mrg_membrane_layout(
            is_node=is_node,
            leak_mS_per_cm2=leak_mS_cm2,
            node_el_mV=-90.0,
            internode_el_mV=-80.0,
            celsius=Temp,
        )
        ion_channel = membrane_layout.as_membrane_model()

        super().__init__(
            ion_channel=ion_channel,
            fiber_d_um=float(d),
            lengths_um=lengths_um,
            diam_um=diam_um,
            Ra_ohm_cm=Ra_ohm_cm,
            Cm_uF_cm2=Cm_uF_cm2,
            Vinit=Vinit,
            Temp=Temp,
            nodes=nodes,
            kind_vec=kind_vec,
            is_node=is_node,
            xraxial_MOhm_cm=xraxial_MOhm_cm,
            xg_S_cm2=xg_S_cm2,
            xc_uF_cm2=xc_uF_cm2,
            Veinit=0.0,
            enable_extracellular=True,
        )
        self.membrane_layout = membrane_layout


__all__ = ["Myelinated", "MRG", "mrg_length_from_nodes", "mrg_nodes_from_length"]
