from benchmark.realistic_examples.bench_basic_examples import (
    child_command,
    parse_args,
    planned_cases,
)


def test_realistic_examples_smoke_plan_is_small():
    args = parse_args(["--preset", "smoke"])

    cases = planned_cases(args)

    assert len(cases) == 5
    assert {(case.workflow, case.fiber_type, case.run_count) for case in cases} == {
        ("example06_velocity", "hh", 2),
        ("example06_velocity", "mrg", 2),
        ("example07_threshold", "rattay", 2),
        ("example07_threshold", "mrg", 2),
        ("example08_recruitment", "mixed", 4),
    }


def test_realistic_examples_child_command_for_platform_spawn(tmp_path):
    args = parse_args(
        [
            "--preset",
            "standard",
            "--platforms",
            "cpu",
            "gpu",
            "--workflows",
            "example08_recruitment",
            "--family-counts",
            "5",
            "--repeats",
            "2",
            "--out-dir",
            str(tmp_path),
            "--prefix",
            "workflow",
        ]
    )

    command = child_command(args, platform_label="gpu")

    assert "--platforms" in command
    assert command[command.index("--platforms") + 1] == "current"
    assert command[command.index("--platform-label") + 1] == "gpu"
    assert command[command.index("--family-counts") + 1] == "5"
    assert command[command.index("--prefix") + 1] == "workflow"
