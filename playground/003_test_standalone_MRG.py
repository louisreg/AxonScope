# standalone_mrg_extracellular_like.py
# MRG prototype closer to NEURON extracellular mechanism:
#
# Variables:
#   Vi : intracellular / axoplasmic potential
#   Ve : local extracellular / periaxonal potential
#   Vm = Vi - Ve
#
# Equations in absolute units:
#   membrane currents depend on Vm
#   axial intracellular coupling on Vi via Ra
#   extracellular coupling on Ve via xraxial
#   Ve -> ground via xg and xc
#
# This is still a prototype, but the electrical mapping follows NRV/NEURON much more closely.

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


SectionKind = Literal["node", "MYSA", "FLUT", "STIN"]


# =============================================================================
# MRG morphology table
# =============================================================================

MRG_DATA = {
    "fiberD": np.asarray([1, 2, 5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0], dtype=float),
    "g": np.asarray([0.565, 0.585, 0.605, 0.630, 0.661, 0.690, 0.700, 0.719, 0.739, 0.767, 0.791], dtype=float),
    "axonD": np.asarray([0.8, 1.6, 3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7], dtype=float),
    "nodeD": np.asarray([0.7, 1.4, 1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5], dtype=float),
    "paraD1": np.asarray([0.7, 1.4, 1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5], dtype=float),
    "paraD2": np.asarray([0.8, 1.6, 3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7], dtype=float),
    "deltax": np.asarray([100, 200, 500, 750, 1000, 1150, 1250, 1350, 1400, 1450, 1500], dtype=float),
    "paralength2": np.asarray([5, 10, 35, 38, 40, 46, 50, 54, 56, 58, 60], dtype=float),
    "nl": np.asarray([15, 20, 80, 100, 110, 120, 130, 135, 140, 145, 150], dtype=float),
}


@dataclass(frozen=True)
class MRGParams:
    fiberD: float
    g: float
    axonD: float
    nodeD: float
    paraD1: float
    paraD2: float
    deltax: float
    paralength1: float
    paralength2: float
    nodelength: float
    interlength: float
    nl: float
    rhoa: float
    mycm: float
    mygm: float
    space_p1: float
    space_p2: float
    space_i: float
    Rpn0: float
    Rpn1: float
    Rpn2: float
    Rpx: float


@dataclass(frozen=True)
class Section:
    kind: SectionKind
    index: int
    x0_um: float
    x1_um: float
    xc_um: float
    L_um: float
    diam_um: float
    Ra_ohm_cm: float
    cm_uF_cm2: float
    g_pas_S_cm2: float
    e_pas_mV: float
    xraxial_ohm: float
    xg_S_cm2: float
    xc_uF_cm2: float


@dataclass(frozen=True)
class Morphology:
    L_um: float
    fiberD_um: float
    params: MRGParams
    sections: tuple[Section, ...]

    @property
    def x_um(self):
        return np.asarray([s.xc_um for s in self.sections], dtype=float)

    @property
    def node_indices(self):
        return np.asarray([i for i, s in enumerate(self.sections) if s.kind == "node"], dtype=int)

    @property
    def kinds(self):
        return [s.kind for s in self.sections]


@lru_cache(maxsize=1)
def _polys():
    d = MRG_DATA["fiberD"]
    return {
        "g": np.poly1d(np.polyfit(d, MRG_DATA["g"], 3)),
        "axonD": np.poly1d(np.polyfit(d, MRG_DATA["axonD"], 3)),
        "nodeD": np.poly1d(np.polyfit(d, MRG_DATA["nodeD"], 3)),
        "paraD1": np.poly1d(np.polyfit(d, MRG_DATA["paraD1"], 3)),
        "paraD2": np.poly1d(np.polyfit(d, MRG_DATA["paraD2"], 3)),
        "nl": np.poly1d(np.polyfit(d, MRG_DATA["nl"], 3)),
        "deltax": np.poly1d(np.polyfit(d, MRG_DATA["deltax"], 5)),
        "paralength2": np.poly1d(np.polyfit(d, MRG_DATA["paralength2"], 5)),
        "deltax_oob": np.poly1d(np.polyfit(d, MRG_DATA["deltax"], 2)),
        "paralength2_oob": np.poly1d(np.polyfit(d, MRG_DATA["paralength2"], 3)),
    }


def _exact_idx(d):
    hit = np.where(np.isclose(MRG_DATA["fiberD"], d, rtol=0.0, atol=1e-12))[0]
    return int(hit[0]) if len(hit) else None


def get_mrg_params(diameter_um: float) -> MRGParams:
    d = float(diameter_um)
    idx = _exact_idx(d)

    if idx is not None:
        g = float(MRG_DATA["g"][idx])
        axonD = float(MRG_DATA["axonD"][idx])
        nodeD = float(MRG_DATA["nodeD"][idx])
        paraD1 = float(MRG_DATA["paraD1"][idx])
        paraD2 = float(MRG_DATA["paraD2"][idx])
        deltax = float(MRG_DATA["deltax"][idx])
        paralength2 = float(MRG_DATA["paralength2"][idx])
        nl = float(MRG_DATA["nl"][idx])
    else:
        p = _polys()
        oob = d < 1.0 or d > 14.0
        g = float(p["g"](d))
        axonD = float(p["axonD"](d))
        nodeD = float(p["nodeD"](d))
        paraD1 = float(p["paraD1"](d))
        paraD2 = float(p["paraD2"](d))
        deltax = float((p["deltax_oob"] if oob else p["deltax"])(d))
        paralength2 = float((p["paralength2_oob"] if oob else p["paralength2"])(d))
        nl = float(p["nl"](d))

    paralength1 = 3.0
    nodelength = 1.0
    interlength = (deltax - nodelength - 2.0 * paralength1 - 2.0 * paralength2) / 6.0

    rhoa = 0.7e6
    mycm = 0.1
    mygm = 0.001

    space_p1 = 0.002
    space_p2 = 0.004
    space_i = 0.004

    Rpn0 = (rhoa * 0.01) / (math.pi * ((((nodeD / 2) + space_p1) ** 2) - ((nodeD / 2) ** 2)))
    Rpn1 = (rhoa * 0.01) / (math.pi * ((((paraD1 / 2) + space_p1) ** 2) - ((paraD1 / 2) ** 2)))
    Rpn2 = (rhoa * 0.01) / (math.pi * ((((paraD2 / 2) + space_p2) ** 2) - ((paraD2 / 2) ** 2)))
    Rpx = (rhoa * 0.01) / (math.pi * ((((axonD / 2) + space_i) ** 2) - ((axonD / 2) ** 2)))

    return MRGParams(
        fiberD=d,
        g=g,
        axonD=axonD,
        nodeD=nodeD,
        paraD1=paraD1,
        paraD2=paraD2,
        deltax=deltax,
        paralength1=paralength1,
        paralength2=paralength2,
        nodelength=nodelength,
        interlength=interlength,
        nl=nl,
        rhoa=rhoa,
        mycm=mycm,
        mygm=mygm,
        space_p1=space_p1,
        space_p2=space_p2,
        space_i=space_i,
        Rpn0=Rpn0,
        Rpn1=Rpn1,
        Rpn2=Rpn2,
        Rpx=Rpx,
    )


def section_len(kind: SectionKind, p: MRGParams):
    return {
        "node": p.nodelength,
        "MYSA": p.paralength1,
        "FLUT": p.paralength2,
        "STIN": p.interlength,
    }[kind]


def nrv_section_properties(kind: SectionKind, p: MRGParams):
    if kind == "node":
        return dict(
            diam=p.nodeD,
            Ra=p.rhoa / 10000.0,
            cm=2.0,
            g_pas=0.0,
            e_pas=-80.0,
            xraxial=p.Rpn0,
            xg=1e10,
            xc=0.0,
        )

    if kind == "MYSA":
        ratio = p.paraD1 / p.fiberD
        return dict(
            diam=p.fiberD,
            Ra=p.rhoa * (1.0 / ratio**2) / 10000.0,
            cm=2.0 * ratio,
            g_pas=0.001 * ratio,
            e_pas=-80.0,
            xraxial=p.Rpn1,
            xg=p.mygm / (p.nl * 2.0),
            xc=p.mycm / (p.nl * 2.0),
        )

    if kind == "FLUT":
        ratio = p.paraD2 / p.fiberD
        return dict(
            diam=p.fiberD,
            Ra=p.rhoa * (1.0 / ratio**2) / 10000.0,
            cm=2.0 * ratio,
            g_pas=0.0001 * ratio,
            e_pas=-80.0,
            xraxial=p.Rpn2,
            xg=p.mygm / (p.nl * 2.0),
            xc=p.mycm / (p.nl * 2.0),
        )

    if kind == "STIN":
        ratio = p.axonD / p.fiberD
        return dict(
            diam=p.fiberD,
            Ra=p.rhoa * (1.0 / ratio**2) / 10000.0,
            cm=2.0 * ratio,
            g_pas=0.0001 * ratio,
            e_pas=-80.0,
            xraxial=p.Rpx,
            xg=p.mygm / (p.nl * 2.0),
            xc=p.mycm / (p.nl * 2.0),
        )

    raise ValueError(kind)


def build_mrg_morphology(diameter_um=10.0, nodes=11) -> Morphology:
    p = get_mrg_params(diameter_um)
    L_um = float(math.ceil(p.deltax * (nodes - 1)))

    seq: list[SectionKind] = [
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

    sections = []
    x = 0.0
    i = 0

    while x < L_um - 1e-12:
        kind = seq[i % len(seq)]
        Ls = min(section_len(kind, p), L_um - x)
        x1 = x + Ls
        props = nrv_section_properties(kind, p)

        sections.append(
            Section(
                kind=kind,
                index=i,
                x0_um=x,
                x1_um=x1,
                xc_um=0.5 * (x + x1),
                L_um=Ls,
                diam_um=props["diam"],
                Ra_ohm_cm=props["Ra"],
                cm_uF_cm2=props["cm"],
                g_pas_S_cm2=props["g_pas"],
                e_pas_mV=props["e_pas"],
                xraxial_ohm=props["xraxial"],
                xg_S_cm2=props["xg"],
                xc_uF_cm2=props["xc"],
            )
        )

        x = x1
        i += 1

    return Morphology(L_um=L_um, fiberD_um=diameter_um, params=p, sections=tuple(sections))


# =============================================================================
# axnode.mod
# =============================================================================

def Exp(x):
    return jnp.where(x < -100.0, 0.0, jnp.exp(x))


def vtrap1(v):
    x = (v + 27.0) / 10.2
    return jnp.where(
        jnp.abs(x) < 1e-6,
        0.01 * 10.2,
        (0.01 * (v + 27.0)) / (1.0 - Exp(-(v + 27.0) / 10.2)),
    )


def vtrap2(v):
    x = (v + 34.0) / 10.0
    return jnp.where(
        jnp.abs(x) < 1e-6,
        0.00025 * 10.0,
        (0.00025 * (-(v + 34.0))) / (1.0 - Exp((v + 34.0) / 10.0)),
    )


def vtrap6(v):
    x = (v + 21.4) / 10.3
    return jnp.where(
        jnp.abs(x) < 1e-6,
        1.86 * 10.3,
        (1.86 * (v + 21.4)) / (1.0 - Exp(-(v + 21.4) / 10.3)),
    )


def vtrap7(v):
    x = (v + 25.7) / 9.16
    return jnp.where(
        jnp.abs(x) < 1e-6,
        0.086 * 9.16,
        (0.086 * (-(v + 25.7))) / (1.0 - Exp((v + 25.7) / 9.16)),
    )


def vtrap8(v):
    x = (v + 114.0) / 11.0
    return jnp.where(
        jnp.abs(x) < 1e-6,
        0.062 * 11.0,
        (0.062 * (-(v + 114.0))) / (1.0 - Exp((v + 114.0) / 11.0)),
    )


def axnode_rates(Vm, celsius=37.0):
    q10_1 = 2.2 ** ((celsius - 20.0) / 10.0)
    q10_2 = 2.9 ** ((celsius - 20.0) / 10.0)
    q10_3 = 3.0 ** ((celsius - 36.0) / 10.0)

    a_mp = q10_1 * vtrap1(Vm)
    b_mp = q10_1 * vtrap2(Vm)

    a_m = q10_1 * vtrap6(Vm)
    b_m = q10_1 * vtrap7(Vm)

    a_h = q10_2 * vtrap8(Vm)
    b_h = q10_2 * 2.3 / (1.0 + Exp(-(Vm + 31.8) / 13.4))

    v2 = Vm + 80.0
    a_s = q10_3 * 0.3 / (Exp((v2 - 27.0) / -5.0) + 1.0)
    b_s = q10_3 * 0.03 / (Exp((v2 + 10.0) / -1.0) + 1.0)

    alpha = jnp.stack([a_mp, a_m, a_h, a_s], axis=1)
    beta = jnp.stack([b_mp, b_m, b_h, b_s], axis=1)
    return alpha, beta


def init_axnode_gates(Vm, celsius=37.0):
    a, b = axnode_rates(Vm, celsius)
    return a / jnp.maximum(a + b, 1e-12)


def update_axnode_gates(gates, Vm, dt, celsius=37.0):
    a, b = axnode_rates(Vm, celsius)
    ab = jnp.maximum(a + b, 1e-12)
    inf = a / ab
    tau = 1.0 / ab
    return inf - (inf - gates) * jnp.exp(-dt / tau)


def axnode_G_GE_abs(gates, area_cm2):
    mp = gates[:, 0]
    m = gates[:, 1]
    h = gates[:, 2]
    s = gates[:, 3]

    gnapbar = 0.01 * 1e3  # mS/cm2
    gnabar = 3.0 * 1e3
    gkbar = 0.08 * 1e3
    gl = 0.007 * 1e3

    ena = 50.0
    ek = -90.0
    el = -90.0

    gnap = gnapbar * mp**3 * area_cm2
    gna = gnabar * m**3 * h * area_cm2
    gk = gkbar * s * area_cm2
    gleak = gl * area_cm2

    G = gnap + gna + gk + gleak
    GE = (gnap + gna) * ena + gk * ek + gleak * el
    return G, GE


# =============================================================================
# Electrical arrays
# =============================================================================

def membrane_area_cm2(L_um, d_um):
    return math.pi * d_um * L_um * 1e-8


def cross_section_area_cm2(d_um):
    r_cm = 0.5 * d_um * 1e-4
    return math.pi * r_cm * r_cm


def build_arrays(morph: Morphology):
    N = len(morph.sections)

    area = np.zeros(N)
    Cm = np.zeros(N)
    Gpas = np.zeros(N)
    GEpas = np.zeros(N)
    Cx = np.zeros(N)
    Gx = np.zeros(N)

    for i, s in enumerate(morph.sections):
        area[i] = membrane_area_cm2(s.L_um, s.diam_um)

        Cm[i] = s.cm_uF_cm2 * area[i]
        Gpas[i] = s.g_pas_S_cm2 * 1e3 * area[i]
        GEpas[i] = Gpas[i] * s.e_pas_mV

        Cx[i] = s.xc_uF_cm2 * area[i]
        Gx[i] = s.xg_S_cm2 * 1e3 * area[i]

    # intracellular axial conductance Vi_i <-> Vi_{i+1}
    Gax_i = np.zeros(N - 1)
    for i in range(N - 1):
        s0 = morph.sections[i]
        s1 = morph.sections[i + 1]

        L0_cm = 0.5 * s0.L_um * 1e-4
        L1_cm = 0.5 * s1.L_um * 1e-4

        A0 = cross_section_area_cm2(s0.diam_um)
        A1 = cross_section_area_cm2(s1.diam_um)

        R = s0.Ra_ohm_cm * L0_cm / A0 + s1.Ra_ohm_cm * L1_cm / A1
        Gax_i[i] = 1e3 / R  # mS

    # extracellular axial conductance Ve_i <-> Ve_{i+1}
    # NEURON extracellular.xraxial is a longitudinal resistance density [MOhm/cm].
    # Edge resistance is the half-segment path on each side:
    #   R_edge[MOhm] = xraxial_i[MOhm/cm] * (Li_cm/2) + xraxial_j[MOhm/cm] * (Lj_cm/2)
    # Then convert to conductance in mS:
    #   G[mS] = 1e-3 / R[MOhm]
    Gax_e = np.zeros(N - 1)
    for i in range(N - 1):
        s0 = morph.sections[i]
        s1 = morph.sections[i + 1]
        L0_cm = 0.5 * s0.L_um * 1e-4
        L1_cm = 0.5 * s1.L_um * 1e-4
        R_MOhm = s0.xraxial_ohm * L0_cm + s1.xraxial_ohm * L1_cm
        Gax_e[i] = 1e-3 / R_MOhm  # mS

    is_node = np.asarray([s.kind == "node" for s in morph.sections], dtype=bool)

    return {
        "area": jnp.asarray(area),
        "Cm": jnp.asarray(Cm),
        "Gpas": jnp.asarray(Gpas),
        "GEpas": jnp.asarray(GEpas),
        "Cx": jnp.asarray(Cx),
        "Gx": jnp.asarray(Gx),
        "Gax_i": jnp.asarray(Gax_i),
        "Gax_e": jnp.asarray(Gax_e),
        "is_node": jnp.asarray(is_node),
    }


# =============================================================================
# Block tridiagonal solver
# =============================================================================

def inv2(M):
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]
    det = a * d - b * c
    return jnp.array([[d, -b], [-c, a]]) / det


def solve_block_tridiagonal(A_lower, A_diag, A_upper, rhs):
    N = A_diag.shape[0]

    invD0 = inv2(A_diag[0])
    C0 = invD0 @ A_upper[0]
    d0 = invD0 @ rhs[0]

    C = jnp.zeros_like(A_upper)
    d = jnp.zeros_like(rhs)

    C = C.at[0].set(C0)
    d = d.at[0].set(d0)

    def fwd(i, carry):
        C, d = carry
        D = A_diag[i] - A_lower[i] @ C[i - 1]
        invD = inv2(D)
        Ci = jnp.where(i < N - 1, invD @ A_upper[i], jnp.zeros((2, 2)))
        di = invD @ (rhs[i] - A_lower[i] @ d[i - 1])
        C = C.at[i].set(Ci)
        d = d.at[i].set(di)
        return C, d

    C, d = jax.lax.fori_loop(1, N, fwd, (C, d))

    x = jnp.zeros_like(rhs)
    x = x.at[N - 1].set(d[N - 1])

    def bwd(k, x):
        i = N - 2 - k
        xi = d[i] - C[i] @ x[i + 1]
        x = x.at[i].set(xi)
        return x

    x = jax.lax.fori_loop(0, N - 1, bwd, x)
    return x


# =============================================================================
# Extracellular-like assembly
# =============================================================================

def assemble_system(Vi, Ve, gates_node_new, arrays, node_idx, dt):
    N = Vi.shape[0]

    Vm = Vi - Ve

    Cm = arrays["Cm"]
    Gpas = arrays["Gpas"]
    GEpas = arrays["GEpas"]
    Cx = arrays["Cx"]
    Gx = arrays["Gx"]
    Gax_i = arrays["Gax_i"]
    Gax_e = arrays["Gax_e"]
    area = arrays["area"]

    Gnode, GEnode = axnode_G_GE_abs(gates_node_new, area[node_idx])

    Gm = Gpas
    GEm = GEpas

    Gm = Gm.at[node_idx].set(Gnode)
    GEm = GEm.at[node_idx].set(GEnode)

    left_i = jnp.concatenate([jnp.array([0.0]), Gax_i])
    right_i = jnp.concatenate([Gax_i, jnp.array([0.0])])

    left_e = jnp.concatenate([jnp.array([0.0]), Gax_e])
    right_e = jnp.concatenate([Gax_e, jnp.array([0.0])])

    A_diag = jnp.zeros((N, 2, 2))
    A_lower = jnp.zeros((N, 2, 2))
    A_upper = jnp.zeros((N, 2, 2))
    rhs = jnp.zeros((N, 2))

    # Unknown block is [Vi, Ve]
    #
    # Intracellular equation:
    # Cm/dt*(Vi - Ve) + Gm*(Vi - Ve) + axial_i(Vi) = Cm/dt*Vm_old + GEm + Iinj
    A_diag = A_diag.at[:, 0, 0].set(Cm / dt + Gm + left_i + right_i)
    A_diag = A_diag.at[:, 0, 1].set(-(Cm / dt + Gm))
    rhs = rhs.at[:, 0].set((Cm / dt) * Vm + GEm)

    # Extracellular equation:
    # -Cm/dt*(Vi - Ve) - Gm*(Vi - Ve)
    # + Cx/dt*Ve + Gx*Ve + axial_e(Ve)
    # = -Cm/dt*Vm_old - GEm + Cx/dt*Ve_old
    A_diag = A_diag.at[:, 1, 0].set(-(Cm / dt + Gm))
    A_diag = A_diag.at[:, 1, 1].set(
        Cm / dt + Gm + Cx / dt + Gx + left_e + right_e
    )
    rhs = rhs.at[:, 1].set(-(Cm / dt) * Vm - GEm + (Cx / dt) * Ve)

    A_lower = A_lower.at[1:, 0, 0].set(-Gax_i)
    A_upper = A_upper.at[:-1, 0, 0].set(-Gax_i)

    A_lower = A_lower.at[1:, 1, 1].set(-Gax_e)
    A_upper = A_upper.at[:-1, 1, 1].set(-Gax_e)

    return A_lower, A_diag, A_upper, rhs


def step_solver(Vi, Ve, gates_node, arrays, node_idx, dt, Iinj_uA, celsius):
    Vm_guess = Vi[node_idx] - Ve[node_idx]
    gates_node_new = gates_node

    def body(_, state):
        Vm_guess, gates_node_new, Vi_new, Ve_new = state

        gates_node_new = update_axnode_gates(
            gates_node,
            Vm_guess,
            dt,
            celsius,
        )

        A_lower, A_diag, A_upper, rhs = assemble_system(
            Vi=Vi,
            Ve=Ve,
            gates_node_new=gates_node_new,
            arrays=arrays,
            node_idx=node_idx,
            dt=dt,
        )

        rhs = rhs.at[node_idx, 0].add(Iinj_uA[node_idx])

        sol = solve_block_tridiagonal(A_lower, A_diag, A_upper, rhs)

        Vi_new = sol[:, 0]
        Ve_new = sol[:, 1]

        Vm_guess = Vi_new[node_idx] - Ve_new[node_idx]

        return Vm_guess, gates_node_new, Vi_new, Ve_new

    Vi_init = Vi
    Ve_init = Ve

    Vm_guess, gates_node_new, Vi_new, Ve_new = jax.lax.fori_loop(
        0,
        3,
        body,
        (Vm_guess, gates_node_new, Vi_init, Ve_init),
    )

    return Vi_new, Ve_new, gates_node_new

# =============================================================================
# Simulation
# =============================================================================

def simulate(
    diameter_um=10.0,
    nodes=11,
    tsim=5.0,
    dt=0.001,
    stim_node=5,
    stim_amp_nA=5.0,
    stim_start_ms=1.0,
    stim_duration_ms=0.1,
    celsius=37.0,
):
    morph = build_mrg_morphology(diameter_um, nodes)
    arrays = build_arrays(morph)

    N = len(morph.sections)
    Nt = int(math.ceil(tsim / dt))

    node_idx_np = morph.node_indices.astype(np.int32)
    node_idx = jnp.asarray(node_idx_np)

    Vi0 = jnp.full((N,), -80.0)
    Ve0 = jnp.zeros((N,))

    gates0 = init_axnode_gates(Vi0[node_idx] - Ve0[node_idx], celsius)

    stim_comp = int(node_idx_np[stim_node])

    def Iinj(t):
        active = (t >= stim_start_ms) & (t <= stim_start_ms + stim_duration_ms)
        val = jnp.where(active, stim_amp_nA * 1e-3, 0.0)  # nA -> uA
        out = jnp.zeros((N,))
        out = out.at[stim_comp].set(val)
        return out

    def step(carry, n):
        Vi, Ve, gates = carry
        t = n * dt

        Vi_new, Ve_new, gates_new = step_solver(
            Vi=Vi,
            Ve=Ve,
            gates_node=gates,
            arrays=arrays,
            node_idx=node_idx,
            dt=dt,
            Iinj_uA=Iinj(t),
            celsius=celsius,
        )

        Vm_new = Vi_new - Ve_new

        return (Vi_new, Ve_new, gates_new), (Vi_new, Ve_new, Vm_new)

    (_, _, _), (Vi_all, Ve_all, Vm_all) = jax.lax.scan(
        step,
        (Vi0, Ve0, gates0),
        jnp.arange(Nt),
    )

    t = jnp.arange(Nt) * dt

    return morph, np.asarray(t), np.asarray(Vi_all), np.asarray(Ve_all), np.asarray(Vm_all)


# =============================================================================
# Plot
# =============================================================================

def plot_all(morph, t, Vi_all, Ve_all, Vm_all):
    x = morph.x_um
    node_idx = morph.node_indices

    fig, axs = plt.subplots(5, 1, figsize=(12, 12), constrained_layout=True)

    colors = {
        "node": "tab:red",
        "MYSA": "tab:orange",
        "FLUT": "tab:green",
        "STIN": "tab:blue",
    }

    for s in morph.sections:
        axs[0].plot(
            [s.x0_um, s.x1_um],
            [0, 0],
            lw=7,
            solid_capstyle="butt",
            color=colors[s.kind],
        )

    axs[0].set_title("MRG morphology")
    axs[0].set_xlim(0, morph.L_um)
    axs[0].set_yticks([])
    axs[0].grid(axis="x", linestyle=":", alpha=0.4)

    im1 = axs[1].imshow(
        Vi_all.T,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), float(x[0]), float(x[-1])],
        cmap="viridis",
    )
    axs[1].set_title("Vi")
    axs[1].set_ylabel("Position [µm]")
    fig.colorbar(im1, ax=axs[1], label="Vi [mV]")

    im2 = axs[2].imshow(
        Ve_all.T,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), float(x[0]), float(x[-1])],
        cmap="coolwarm",
    )
    axs[2].set_title("Ve / extracellular layer")
    axs[2].set_ylabel("Position [µm]")
    fig.colorbar(im2, ax=axs[2], label="Ve [mV]")

    im3 = axs[3].imshow(
        Vm_all.T,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), float(x[0]), float(x[-1])],
        cmap="viridis",
    )
    axs[3].scatter(
        np.full_like(node_idx, float(t[-1]), dtype=float),
        x[node_idx],
        s=8,
        c="white",
    )
    axs[3].set_title("Vm = Vi - Ve")
    axs[3].set_ylabel("Position [µm]")
    fig.colorbar(im3, ax=axs[3], label="Vm [mV]")

    for i in node_idx:
        k = int(np.where(node_idx == i)[0][0])
        axs[4].plot(t, Vm_all[:, i], label=f"node {k}, x={x[i]:.0f} µm")

    axs[4].set_title("Vm at nodes")
    axs[4].set_xlabel("Time [ms]")
    axs[4].set_ylabel("Vm [mV]")
    axs[4].grid(True)
    axs[4].legend(fontsize=7, ncols=2)

    plt.show()


if __name__ == "__main__":
    
    morph, t, Vi_all, Ve_all, Vm_all = simulate(
        diameter_um=10.0,
        nodes=11,
        tsim=5.0,
        dt=0.001,
        stim_node=5,
        stim_amp_nA=50.0,
        stim_start_ms=1.0,
        stim_duration_ms=0.1,
        celsius=37.0,
    )

    print(f"N sections: {len(morph.sections)}")
    print(f"N nodes: {len(morph.node_indices)}")
    print(f"L: {morph.L_um:.1f} µm")
    print(f"deltax: {morph.params.deltax:.1f} µm")
    print(f"interlength: {morph.params.interlength:.1f} µm")
    print(f"Vi range: {np.nanmin(Vi_all):.3f} to {np.nanmax(Vi_all):.3f} mV")
    print(f"Ve range: {np.nanmin(Ve_all):.3f} to {np.nanmax(Ve_all):.3f} mV")
    print(f"Vm range: {np.nanmin(Vm_all):.3f} to {np.nanmax(Vm_all):.3f} mV")

    plot_all(morph, t, Vi_all, Ve_all, Vm_all)
    