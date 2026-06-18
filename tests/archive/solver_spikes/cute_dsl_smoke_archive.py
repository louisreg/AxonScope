from benchmark.cute_dsl.run_cute_dsl_smoke import (
    parse_compute_capability,
    parse_nvidia_smi_gpu_line,
)


def test_parse_compute_capability():
    assert parse_compute_capability("8.9") == (8, 9)
    assert parse_compute_capability("9.0") == (9, 0)


def test_parse_nvidia_smi_gpu_line():
    assert parse_nvidia_smi_gpu_line("NVIDIA L4, 8.9") == ("NVIDIA L4", "8.9")
    assert parse_nvidia_smi_gpu_line("bad") == (None, None)
