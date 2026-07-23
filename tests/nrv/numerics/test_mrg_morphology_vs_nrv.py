from __future__ import annotations

import numpy as np
import pytest

import nrv

from axonfleet import um
from axonfleet.axons.templates._mrg_morphology import (
    available_mrg_fiber_diameters,
    get_mrg_morphology,
)
from axonfleet.axons.templates.mrg_like_double_cable import mrg_like_length_from_nodes


FIELDS = ("g", "axonD", "nodeD", "paraD1", "paraD2", "deltax", "paralength2", "nl")


def _nrv_mrg_parameters(diameter: float, *, fit_all: bool) -> dict[str, float]:
    values = nrv.get_MRG_parameters(diameter, fit_all=fit_all)
    return {name: float(value) for name, value in zip(FIELDS, values, strict=True)}


@pytest.mark.parametrize("fit_all", [False, True], ids=["table_or_fit", "fit_all"])
@pytest.mark.parametrize("diameter_um", [1.0, 2.0, 3.0, 5.7, 6.0, 8.7, 10.0, 12.8, 14.0, 15.0, 16.0])
def test_mrg_morphology_matches_nrv(diameter_um: float, fit_all: bool) -> None:
    morphology = get_mrg_morphology(diameter_um, fit_all=fit_all)
    reference = _nrv_mrg_parameters(diameter_um, fit_all=fit_all)

    for field in FIELDS:
        value = getattr(morphology, field)
        assert value == pytest.approx(reference[field], abs=1e-12), (
            f"Mismatch on {field} for diameter={diameter_um} fit_all={fit_all}: "
            f"{value} vs {reference[field]}"
        )


@pytest.mark.parametrize("diameter_um", [1.0, 2.0, 3.0, 5.7, 6.0, 8.7, 10.0, 12.8, 14.0, 15.0, 16.0])
@pytest.mark.parametrize("nodes", [2, 3, 7, 11])
def test_mrg_like_length_from_nodes_matches_nrv(diameter_um: float, nodes: int) -> None:
    assert mrg_like_length_from_nodes(diameter_um * um, nodes) == nrv.get_length_from_nodes(diameter_um, nodes)


def test_available_mrg_fiber_diameters_matches_nrv_table() -> None:
    assert np.array_equal(
        available_mrg_fiber_diameters(),
        np.asarray([1.0, 2.0, 5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]),
    )
