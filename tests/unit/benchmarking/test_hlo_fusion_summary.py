from benchmark.analysis.hlo_fusion_summary import analyze_hlo_files


def test_hlo_fusion_summary_estimates_fusion_io_bytes(tmp_path):
    hlo = """HloModule test, entry_computation_layout={(f32[2,3]{1,0}, f32[2,3]{1,0})->f32[2,3]{1,0}}, allow_spmd_sharding_propagation_to_output={true}

%fused_add (p0: f32[2,3]{1,0}, p1: f32[2,3]{1,0}) -> f32[2,3]{1,0} {
  %p0 = f32[2,3]{1,0} parameter(0)
  %p1 = f32[2,3]{1,0} parameter(1)
  %exp = f32[2,3]{1,0} exponential(%p0)
  ROOT %add = f32[2,3]{1,0} add(%exp, %p1)
}

ENTRY %main (Arg_0.1: f32[2,3]{1,0}, Arg_1.2: f32[2,3]{1,0}) -> f32[2,3]{1,0} {
  %Arg_0.1 = f32[2,3]{1,0} parameter(0)
  %Arg_1.2 = f32[2,3]{1,0} parameter(1)
  ROOT %fusion = f32[2,3]{1,0} fusion(%Arg_0.1, %Arg_1.2), kind=kLoop, calls=%fused_add
}
"""
    path = tmp_path / "block_solve_pcr_soa_batched.compiled.optimized_hlo.txt"
    path.write_text(hlo, encoding="utf-8")

    analysis = analyze_hlo_files((path,))

    assert len(analysis.fusion_rows) == 1
    row = analysis.fusion_rows[0]
    assert row["stage"] == "block_solve"
    assert row["variant"] == "pcr_soa_batched"
    assert row["output_bytes_estimate"] == 24
    assert row["computation_input_bytes_estimate"] == 48
    assert row["computation_output_bytes_estimate"] == 24
    assert row["computation_io_bytes_estimate"] == 72
    assert row["count_exponential"] == 1
    assert analysis.module_rows[0]["count_exponential"] == 1
