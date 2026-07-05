from pathlib import Path

from benchmark.solvers.summarize_double_cable_linear_solvers import (
    load_rows,
    summarize_rows,
)


def test_summarize_linear_solver_csv_selects_fastest_and_ratios(tmp_path):
    path = tmp_path / "gpu" / "summary.csv"
    path.parent.mkdir()
    path.write_text(
        "\n".join(
            [
                "requested_solver,resolved_solver,kernel_solver,batch_size,nx,dtype,steady_median_ms,node_solves_per_s",
                "thomas,thomas,thomas,128,51,float32,10.0,652800.0",
                "pcr,pcr,pcr,128,51,float32,4.0,1632000.0",
                "pcr_soa,pcr_soa,pcr_soa,128,51,float32,3.0,2176000.0",
                "pcr_adaptive,pcr_adaptive,pcr_soa,128,51,float32,3.2,2040000.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_rows(path)
    summary = summarize_rows(rows)

    assert len(summary) == 1
    row = summary[0]
    assert row["source"] == "gpu"
    assert row["fastest_requested_solver"] == "pcr_soa"
    assert row["fastest_kernel_solver"] == "pcr_soa"
    assert row["thomas_over_fastest_x"] == 10.0 / 3.0
    assert row["adaptive_over_fastest_x"] == 3.2 / 3.0

