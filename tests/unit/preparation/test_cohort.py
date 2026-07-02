import numpy as np

import axonscope as axs
from axonscope.dispatcher import build_dispatch_plan
from axonscope.preparation.cohort import PreparedCohort


def test_prepared_cohort_collects_group_rows():
    model = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_a = axs.AxonInstance(model)
    axon_b = axs.AxonInstance(model)

    plan = build_dispatch_plan([axon_a, axon_b])
    cohort = PreparedCohort.from_dispatch_group(plan.groups[0])

    assert cohort.size == 2
    assert cohort.nx == 11
    assert cohort.axons == (axon_a, axon_b)
    assert cohort.solver_axons[0] is cohort.solver_axons[1]
    assert cohort.stimulations == ((), ())
    assert np.asarray(cohort.x_positions_m).shape == (2, 11)
    np.testing.assert_allclose(cohort.axon_y_um, [0.0, 0.0])
    np.testing.assert_allclose(cohort.axon_z_um, [0.0, 0.0])
