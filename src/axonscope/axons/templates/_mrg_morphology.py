"""Private MRG morphology table used by MRG-like axon layout templates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MRGMorphology:
    """Geometrical parameters of the MRG myelinated axon model.

    Units are micrometers unless otherwise specified.
    """

    fiberD: float
    g: float
    axonD: float
    nodeD: float
    paraD1: float
    paraD2: float
    deltax: float
    paralength2: float
    nl: float


_MRG_DATA: dict[str, NDArray[np.float64]] = {
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


@lru_cache(maxsize=1)
def _mrg_polynomials() -> dict[str, np.poly1d]:
    d = _MRG_DATA["fiberD"]

    return {
        "g": np.poly1d(np.polyfit(d, _MRG_DATA["g"], 3)),
        "axonD": np.poly1d(np.polyfit(d, _MRG_DATA["axonD"], 3)),
        "nodeD": np.poly1d(np.polyfit(d, _MRG_DATA["nodeD"], 3)),
        "paraD1": np.poly1d(np.polyfit(d, _MRG_DATA["paraD1"], 3)),
        "paraD2": np.poly1d(np.polyfit(d, _MRG_DATA["paraD2"], 3)),
        "nl": np.poly1d(np.polyfit(d, _MRG_DATA["nl"], 3)),
        "deltax": np.poly1d(np.polyfit(d, _MRG_DATA["deltax"], 5)),
        "paralength2": np.poly1d(np.polyfit(d, _MRG_DATA["paralength2"], 5)),
        "deltax_oob": np.poly1d(np.polyfit(d, _MRG_DATA["deltax"], 2)),
        "paralength2_oob": np.poly1d(np.polyfit(d, _MRG_DATA["paralength2"], 3)),
        "deltax_length": np.poly1d(np.polyfit(d, _MRG_DATA["deltax"], 4)),
    }


def available_mrg_fiber_diameters() -> NDArray[np.float64]:
    """Return original MRG tabulated fiber diameters in µm."""

    return _MRG_DATA["fiberD"].copy()


def _is_exact_tabulated_diameter(diameter: float) -> bool:
    return bool(np.any(np.isclose(_MRG_DATA["fiberD"], diameter, rtol=0.0, atol=1e-12)))


def _tabulated_morphology(diameter: float) -> MRGMorphology:
    idx = int(np.where(np.isclose(_MRG_DATA["fiberD"], diameter, rtol=0.0, atol=1e-12))[0][0])

    return MRGMorphology(
        fiberD=float(_MRG_DATA["fiberD"][idx]),
        g=float(_MRG_DATA["g"][idx]),
        axonD=float(_MRG_DATA["axonD"][idx]),
        nodeD=float(_MRG_DATA["nodeD"][idx]),
        paraD1=float(_MRG_DATA["paraD1"][idx]),
        paraD2=float(_MRG_DATA["paraD2"][idx]),
        deltax=float(_MRG_DATA["deltax"][idx]),
        paralength2=float(_MRG_DATA["paralength2"][idx]),
        nl=float(_MRG_DATA["nl"][idx]),
    )


def get_mrg_morphology(diameter: float, *, fit_all: bool = False) -> MRGMorphology:
    """Return MRG morphology parameters for a fiber diameter.

    Parameters
    ----------
    diameter:
        Fiber diameter in µm.
    fit_all:
        If False, exact tabulated diameters return original table values.
        If True, all values are obtained from polynomial fits.

    Returns
    -------
    MRGMorphology
        Typed morphology object containing MRG geometrical parameters.

    Notes
    -----
    Based on the MRG morphology table from:

    McIntyre CC, Richardson AG, Grill WM.
    Modeling the excitability of mammalian nerve fibers: influence of
    afterpotentials on the recovery cycle.
    Journal of Neurophysiology, 87:995-1006, 2002.
    """
    diameter = float(diameter)

    if diameter <= 0:
        raise ValueError(f"fiber diameter must be positive, got {diameter}")

    if not fit_all and _is_exact_tabulated_diameter(diameter):
        return _tabulated_morphology(diameter)

    p = _mrg_polynomials()

    out_of_original_range = diameter < 1.0 or diameter > 14.0

    deltax_poly = p["deltax_oob"] if out_of_original_range else p["deltax"]
    paralength2_poly = p["paralength2_oob"] if out_of_original_range else p["paralength2"]

    return MRGMorphology(
        fiberD=diameter,
        g=float(p["g"](diameter)),
        axonD=float(p["axonD"](diameter)),
        nodeD=float(p["nodeD"](diameter)),
        paraD1=float(p["paraD1"](diameter)),
        paraD2=float(p["paraD2"](diameter)),
        deltax=float(deltax_poly(diameter)),
        paralength2=float(paralength2_poly(diameter)),
        nl=float(p["nl"](diameter)),
    )


def get_mrg_length_node_spacing(diameter: float, *, fit_all: bool = False) -> float:
    """Return the NRV `get_length_from_nodes` spacing in micrometers.

    NRV uses a separate degree-4 polynomial for `get_length_from_nodes`, while
    `get_MRG_parameters` uses the morphology `deltax` interpolation.
    """

    diameter = float(diameter)

    if diameter <= 0:
        raise ValueError(f"fiber diameter must be positive, got {diameter}")

    if not fit_all and _is_exact_tabulated_diameter(diameter):
        return float(_tabulated_morphology(diameter).deltax)

    return float(_mrg_polynomials()["deltax_length"](diameter))


def get_mrg_morphologies(
    diameters: ArrayLike,
    *,
    fit_all: bool = True,
) -> list[MRGMorphology]:
    """Vector-friendly helper returning one morphology object per diameter."""

    d_arr = np.asarray(diameters, dtype=float)
    return [get_mrg_morphology(float(d), fit_all=fit_all) for d in d_arr]


__all__ = [
    "MRGMorphology",
    "available_mrg_fiber_diameters",
    "get_mrg_length_node_spacing",
    "get_mrg_morphologies",
    "get_mrg_morphology",
]
