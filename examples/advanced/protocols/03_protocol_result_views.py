"""Inspect protocol result views without running a solver.

Run:
    python examples/advanced/protocols/03_protocol_result_views.py

Protocol helpers return small summary objects after the simulation work is
done. This example builds synthetic results so the display contract is visible
without paying for a batch run: rows, pandas tables, compact text, and plots
all go through the same public result objects.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    recruitment = axs.protocols.RecruitmentCurve(
        amplitudes_uA=np.asarray([0.0, 5.0, 10.0, 15.0, 20.0]),
        activated=np.asarray(
            [
                [False, False, False],
                [False, False, False],
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            dtype=bool,
        ),
    )

    sweep = axs.protocols.PoolSweepResult(
        values=tuple(np.asarray([0.0, 10.0, 20.0]) * axs.uA),
        observations=np.asarray(
            [
                [-70.0, -70.0, -70.0],
                [-20.0, -48.0, -66.0],
                [25.0, -5.0, -35.0],
            ],
            dtype=float,
        ),
    )

    threshold_curve = axs.protocols.ThresholdCurve(
        row_labels=tuple(np.asarray([0.4, 0.8, 1.2]) * axs.um),
        threshold_uA=np.asarray([18.0, 12.0, 8.0]),
        lower_bound_uA=np.asarray([17.5, 11.5, 7.5]),
        upper_bound_uA=np.asarray([18.5, 12.5, 8.5]),
        status=("threshold", "threshold", "threshold"),
        tested_uA=(np.asarray([10.0, 15.0, 20.0]), np.asarray([8.0, 12.0, 16.0])),
        satisfied=(np.asarray([False, True, True]), np.asarray([False, True, True])),
    )

    print(recruitment.to_dataframe(unit=axs.uA).to_string(index=False))
    print()
    print(
        sweep.to_dataframe(value_name="current_uA", value_unit=axs.uA).to_string(
            index=False
        )
    )
    print()
    print(
        threshold_curve.format(
            row_name="diameter_um",
            row_unit=axs.um,
            threshold_unit=axs.uA,
        )
    )
    print()
    print(threshold_curve.to_analysis_result().to_dataframe().to_string(index=False))

    rows = recruitment.rows(unit=axs.uA)
    print(f"\nFirst recruitment view row: {rows[0]}")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    recruitment.plot(ax=axes[0], unit=axs.uA)
    axes[0].set_title("Recruitment")
    sweep.plot(ax=axes[1], value_unit=axs.uA)
    axes[1].set_title("Pool sweep observation")
    threshold_curve.plot(ax=axes[2], row_unit=axs.um, threshold_unit=axs.uA)
    axes[2].set_title("Threshold curve")
    plt.show()


if __name__ == "__main__":
    main()
