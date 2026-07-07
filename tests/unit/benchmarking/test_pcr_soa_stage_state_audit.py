import csv

from benchmark.analysis.pcr_soa_stage_state_audit import (
    pcr_soa_stage_state_rows,
    write_outputs,
)


def test_pcr_soa_stage_state_rows_report_powers_of_two_strides():
    rows = pcr_soa_stage_state_rows(batch_size=512, nx=89, dtype="float32")

    assert [row["stride"] for row in rows] == [1, 2, 4, 8, 16, 32, 64]
    assert all(row["variant"] == "pcr_soa_batched" for row in rows)
    assert all(row["state_arrays"] == 14 for row in rows)
    assert rows[0]["state_mib"] == 14 * 512 * 89 * 4 / (1024 * 1024)
    assert rows[0]["explicit_neighbor_reads"] == 28


def test_pcr_soa_stage_state_rows_support_symmetric_candidate():
    rows = pcr_soa_stage_state_rows(
        batch_size=512,
        nx=89,
        dtype="float32",
        variant="pcr_soa_symmetric_batched",
    )

    assert [row["stride"] for row in rows] == [1, 2, 4, 8, 16, 32, 64]
    assert all(row["variant"] == "pcr_soa_symmetric_batched" for row in rows)
    assert all(row["state_arrays"] == 10 for row in rows)
    assert rows[0]["state_mib"] == 10 * 512 * 89 * 4 / (1024 * 1024)
    assert rows[0]["explicit_neighbor_reads"] == 24
    assert "upper00" not in rows[0]["state_fields"]


def test_pcr_soa_stage_state_rows_attach_hlo_fusion_metrics():
    rows = pcr_soa_stage_state_rows(
        batch_size=512,
        nx=89,
        dtype="float32",
        hlo_rows=(
            {
                "stage": "block_solve",
                "variant": "pcr_soa_batched",
                "fusion_name": "%loop_select_subtract_fusion.1",
                "line": "10",
                "output_arrays": "22",
                "output_bytes_estimate": str(22 * 512 * 89 * 4),
                "computation_input_bytes_estimate": str(22 * 512 * 89 * 4),
                "computation_io_bytes_estimate": str(44 * 512 * 89 * 4),
                "count_gather": "28",
                "count_select": "18",
            },
        ),
    )

    assert rows[0]["hlo_fusion_name"] == "%loop_select_subtract_fusion.1"
    assert rows[0]["hlo_output_arrays"] == 22
    assert rows[0]["hlo_gather"] == 28
    assert rows[0]["hlo_select"] == 18


def test_pcr_soa_stage_state_rows_attach_variant_hlo_fusion_metrics():
    rows = pcr_soa_stage_state_rows(
        batch_size=512,
        nx=89,
        dtype="float32",
        variant="pcr_soa_symmetric_batched",
        hlo_rows=(
            {
                "stage": "block_solve",
                "variant": "pcr_soa_batched",
                "fusion_name": "%loop_select_subtract_fusion.1",
                "line": "10",
                "output_arrays": "22",
            },
            {
                "stage": "block_solve",
                "variant": "pcr_soa_symmetric_batched",
                "fusion_name": "%loop_select_subtract_fusion.1",
                "line": "10",
                "output_arrays": "14",
                "output_bytes_estimate": str(14 * 512 * 89 * 4),
                "computation_input_bytes_estimate": str(14 * 512 * 89 * 4),
                "computation_io_bytes_estimate": str(28 * 512 * 89 * 4),
                "count_gather": "20",
                "count_select": "13",
            },
        ),
    )

    assert rows[0]["hlo_output_arrays"] == 14
    assert rows[0]["hlo_gather"] == 20
    assert rows[0]["hlo_select"] == 13


def test_pcr_soa_stage_state_write_outputs(tmp_path):
    rows = pcr_soa_stage_state_rows(batch_size=8, nx=9, dtype="float32")

    write_outputs(
        tmp_path,
        rows=rows,
        metadata={"batch_size": 8, "nx": 9, "dtype": "float32", "variant": "pcr_soa_batched"},
    )

    summary = tmp_path / "pcr_soa_stage_state_summary.csv"
    report = tmp_path / "pcr_soa_stage_state_report.md"
    metrics = tmp_path / "pcr_soa_stage_state_metrics.json"
    assert summary.is_file()
    assert report.is_file()
    assert metrics.is_file()
    with summary.open("r", encoding="utf-8", newline="") as fh:
        written = list(csv.DictReader(fh))
    assert [int(row["stride"]) for row in written] == [1, 2, 4, 8]
    assert all(row["variant"] == "pcr_soa_batched" for row in written)
