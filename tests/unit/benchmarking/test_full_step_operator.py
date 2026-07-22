from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.analysis import full_step_operator


@pytest.mark.parametrize("cable", ("single", "double"))
def test_staged_production_solves_match_assembled_operator(cable: str) -> None:
    proof = full_step_operator.prove_block_elimination(cable=cable)

    assert proof.accepted
    assert proof.cross_axon_nonzeros == 0
    assert proof.staged_vs_dense_max_abs < 2e-11
    assert proof.direct_sum_vs_independent_max_abs < 2e-11


def test_explicit_operator_storage_scales_with_independent_axons() -> None:
    rows = full_step_operator.operator_storage_estimates(populations=(1, 1024))

    for cable in ("single", "double"):
        selected = [row for row in rows if row["cable"] == cable]
        assert selected[1]["explicit_csr_bytes"] > selected[0]["explicit_csr_bytes"]
        assert selected[1]["matrix_free_core_bytes"] == (
            1024 * selected[0]["matrix_free_core_bytes"]
        )
        assert selected[1]["storage_ratio"] > 1.0


def test_cli_writes_accepted_evidence(tmp_path: Path) -> None:
    assert full_step_operator.main(["--output", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "full_step_operator.json").read_text())
    assert payload["accepted"] is True
    assert {proof["cable"] for proof in payload["proofs"]} == {"single", "double"}
    assert (tmp_path / "full_step_operator.md").is_file()
