from benchmark.realistic_examples.bench_basic_examples import (
    child_command,
    parse_args,
    planned_cases,
    write_platform_comparison,
    write_profile_comparison,
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


def test_realistic_examples_stress_plan_uses_larger_diameter_sets():
    args = parse_args(["--preset", "stress"])

    cases = planned_cases(args)

    assert len(cases) == 14
    assert 20 in {case.run_count for case in cases}
    assert {
        case.run_count
        for case in cases
        if case.workflow == "example08_recruitment"
    } == {50, 100}


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
            "--no-plots",
            "--profile",
        ]
    )

    command = child_command(args, platform_label="gpu")

    assert "--platforms" in command
    assert command[command.index("--platforms") + 1] == "current"
    assert command[command.index("--platform-label") + 1] == "gpu"
    assert command[command.index("--family-counts") + 1] == "5"
    assert command[command.index("--prefix") + 1] == "workflow"
    assert "--no-plots" in command
    assert "--profile" in command


def test_realistic_examples_accepts_center_vm_recruitment_recording():
    args = parse_args(
        [
            "--preset",
            "stress",
            "--workflows",
            "example08_recruitment",
            "--example08-recording",
            "center",
        ]
    )

    cases = planned_cases(args)

    assert {case.recording for case in cases} == {"center"}


def test_realistic_examples_writes_cpu_gpu_comparison(tmp_path):
    fieldnames = [
        "workflow",
        "fiber_type",
        "run_count",
        "duration_ms",
        "dt_ms",
        "recording",
        "protocol_steps",
        "first_run_s",
        "total_first_s",
        "warm.mean_s",
        "jax_backend",
    ]
    cpu = tmp_path / "workflow_cpu.csv"
    gpu = tmp_path / "workflow_gpu.csv"
    cpu.write_text(
        ",".join(fieldnames)
        + "\nexample06_velocity,hh,2,10.0,0.001,full,1,4.0,5.0,2.0,cpu\n",
        encoding="utf-8",
    )
    gpu.write_text(
        ",".join(fieldnames)
        + "\nexample06_velocity,hh,2,10.0,0.001,full,1,1.0,2.0,0.5,gpu\n",
        encoding="utf-8",
    )

    comparison = write_platform_comparison(
        out_dir=tmp_path,
        prefix="workflow",
        platforms=["cpu", "gpu"],
    )

    assert comparison == tmp_path / "workflow_cpu_vs_gpu.csv"
    text = comparison.read_text(encoding="utf-8")
    assert "first_run_speedup_cpu_over_gpu" in text
    assert "4" in text


def test_realistic_examples_writes_profile_cpu_gpu_comparison(tmp_path):
    fieldnames = [
        "workflow",
        "fiber_type",
        "run_count",
        "duration_ms",
        "dt_ms",
        "recording",
        "protocol_steps",
        "phase",
        "repeat_index",
        "event_name",
        "event_count",
        "total_ms",
        "self_ms",
        "profile_dir",
    ]
    cpu = tmp_path / "workflow_cpu_profile.csv"
    gpu = tmp_path / "workflow_gpu_profile.csv"
    cpu.write_text(
        ",".join(fieldnames)
        + "\nexample06_velocity,hh,2,10.0,0.001,full,1,warm_repeat,1,kernel.wait,1,8.0,8.0,/tmp/cpu\n",
        encoding="utf-8",
    )
    gpu.write_text(
        ",".join(fieldnames)
        + "\nexample06_velocity,hh,2,10.0,0.001,full,1,warm_repeat,1,kernel.wait,1,2.0,2.0,/tmp/gpu\n",
        encoding="utf-8",
    )

    comparison = write_profile_comparison(out_dir=tmp_path, prefix="workflow")

    assert comparison == tmp_path / "workflow_profile_cpu_vs_gpu.csv"
    text = comparison.read_text(encoding="utf-8")
    assert "total_speedup_cpu_over_gpu" in text
    assert "kernel.wait" in text
    assert "4" in text
