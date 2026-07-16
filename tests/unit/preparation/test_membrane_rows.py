from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from axonscope.preparation.membrane_rows import MembraneRowPlan


def _item(signature, *, nx=5, v_init=-70.0):
    return SimpleNamespace(
        membrane_signature=signature,
        solver_axon=SimpleNamespace(
            n_compartments=nx,
            membrane_models=signature,
        ),
        simulation=SimpleNamespace(v_init=v_init),
    )


def test_membrane_row_plan_deduplicates_equivalent_parameter_rows():
    items = (
        _item(("active", "leak-a")),
        _item(("active", "leak-b")),
        _item(("active", "leak-a")),
    )

    plan = MembraneRowPlan.from_dispatch_items(items)

    assert plan.size == 3
    assert plan.unique_count == 2
    assert plan.cache_hits == 1
    assert plan.unique_model_count == 3
    np.testing.assert_array_equal(plan.row_parameter_indices, [0, 1, 0])
    np.testing.assert_array_equal(plan.representative_item_indices, [0, 1])
    assert not plan.row_parameter_indices.flags.writeable


def test_membrane_row_plan_keeps_shape_and_initial_voltage_distinct():
    signature = ("active", "leak")
    plan = MembraneRowPlan.from_dispatch_items(
        (
            _item(signature, nx=5, v_init=-70.0),
            _item(signature, nx=7, v_init=-70.0),
            _item(signature, nx=5, v_init=-65.0),
        )
    )

    assert plan.unique_count == 3
    assert plan.unique_model_count == 2
    np.testing.assert_array_equal(plan.row_parameter_indices, [0, 1, 2])
