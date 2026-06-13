from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.nrv_performance.plot_results import write_performance_plot


def test_write_nrv_performance_plot_from_grid_csv(tmp_path: Path):
    csv_path = tmp_path / "grid.csv"
    pd.DataFrame(
        [
            {
                "model": "hh_intracellular",
                "dt_ms": 0.01,
                "tsim_ms": 20.0,
                "diameter_um": 0.5,
                "input_nx": 21,
                "nodes": None,
                "axon_nx": 21,
                "as_warm_total_median_s": 0.03,
                "nrv_repeat_total_median_s": 0.05,
                "speedup_nrv_over_as_total_warm": 1.67,
            },
            {
                "model": "hh_intracellular",
                "dt_ms": 0.01,
                "tsim_ms": 20.0,
                "diameter_um": 0.5,
                "input_nx": 51,
                "nodes": None,
                "axon_nx": 51,
                "as_warm_total_median_s": 0.07,
                "nrv_repeat_total_median_s": 0.13,
                "speedup_nrv_over_as_total_warm": 1.86,
            },
        ]
    ).to_csv(csv_path, index=False)

    report = write_performance_plot([csv_path], tmp_path / "report", prefix="hh_sweep")

    assert report["plot"].exists()
    assert report["summary_csv"].exists()
    summary = pd.read_csv(report["summary_csv"])
    assert list(summary["axon_nx"]) == [21, 51]
