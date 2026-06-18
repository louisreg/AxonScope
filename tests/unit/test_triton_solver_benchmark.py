import pytest

from benchmark.triton_solver.bench_double_cable_triton import (
    TritonCase,
    percentile,
)


def test_triton_case_label_fields():
    case = TritonCase(batch_size=1024, nx=51, dtype="float32")

    assert case.batch_size == 1024
    assert case.nx == 51
    assert case.dtype == "float32"


def test_percentile_interpolates():
    assert percentile([1.0, 3.0, 5.0], 50.0) == 3.0
    assert percentile([1.0, 3.0], 95.0) == pytest.approx(2.9)
