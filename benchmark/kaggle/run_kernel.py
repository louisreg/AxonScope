"""Submit AxonScope benchmark scripts and campaigns to Kaggle."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.run import SCRIPTS
from benchmark.workloads.curve_options import PRESETS


KAGGLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = KAGGLE_DIR.parents[1]
KERNEL_ENTRY = KAGGLE_DIR / "kernel_entry.py"
DEFAULT_REPO_URL = "https://github.com/louisreg/AxonScope.git"
DEFAULT_SLUG = "axonscope-p11a-benchmarks"
DEFAULT_TITLE = None
DEFAULT_BRANCH_PREFIX = "kaggle-bench"
TIME_CHUNK_SWEEP_CAMPAIGN = "time_chunk_sweep"
DOUBLE_CABLE_SOLVER_POLICY_CAMPAIGN = "double_cable_solver_policy"
SOLVER_STAGE_PROFILE_CAMPAIGN = "double_cable_solver_stage_profile"
REAL_STAGE_PROFILE_CAMPAIGN = "double_cable_real_stage_profile"
SOLVER_LOWERING_AUDIT_CAMPAIGN = "double_cable_solver_lowering_audit"
LARGE_POPULATION_SOLVER_PROFILE_CAMPAIGN = "large_population_double_cable_solver_profile"
JAX_TRITON_COLD_START_AUDIT_CAMPAIGN = "jax_triton_cold_start_audit"
STANDALONE_CAMPAIGNS = (
    DOUBLE_CABLE_SOLVER_POLICY_CAMPAIGN,
    SOLVER_STAGE_PROFILE_CAMPAIGN,
    REAL_STAGE_PROFILE_CAMPAIGN,
    SOLVER_LOWERING_AUDIT_CAMPAIGN,
    LARGE_POPULATION_SOLVER_PROFILE_CAMPAIGN,
    JAX_TRITON_COLD_START_AUDIT_CAMPAIGN,
)
CAMPAIGNS = (TIME_CHUNK_SWEEP_CAMPAIGN, *STANDALONE_CAMPAIGNS)


def main(argv: list[str] | None = None) -> int:
    args, benchmark_args = parse_args(argv)
    run_dir = args.run_dir or make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    kernel_ref = f"{args.username}/{args.slug}"

    if args.attach:
        kaggle_bin = resolve_kaggle_bin(args.kaggle_bin)
        return monitor_kernel(args, kaggle_bin, kernel_ref, run_dir)

    if not args.no_publish_branch:
        args.branch = publish_git_branch(args, run_dir)

    package_dir = prepare_kernel_package(args, benchmark_args, run_dir)
    if args.dry_run:
        print(f"dry-run: wrote Kaggle package to {package_dir}")
        print(f"dry-run: kernel ref would be {kernel_ref}")
        return 0

    kaggle_bin = resolve_kaggle_bin(args.kaggle_bin)
    push_command = [
        kaggle_bin,
        "kernels",
        "push",
        "-p",
        str(package_dir),
        "--timeout",
        str(args.timeout),
    ]
    if not is_cpu_machine_shape(args.machine_shape):
        push_command.extend(["--accelerator", args.machine_shape])
    push = run(push_command, capture=True, check=False)
    write_text(run_dir / "push.log", command_output(push))
    if push.returncode != 0:
        print(f"Kaggle push failed. See {run_dir / 'push.log'}")
        return push.returncode
    if args.no_wait:
        print(f"Kaggle push submitted: https://www.kaggle.com/code/{kernel_ref}")
        print(f"Run artifacts: {run_dir}")
        return 0
    return monitor_kernel(args, kaggle_bin, kernel_ref, run_dir)


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Kaggle username slug.")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle kernel slug.")
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Kaggle kernel title. Defaults to the slug to avoid Kaggle rewriting the id.",
    )
    parser.add_argument("--campaign", choices=CAMPAIGNS)
    parser.add_argument("--script", choices=tuple(SCRIPTS))
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--platform", choices=("cpu", "gpu", "nrv"))
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Shortcut for a Kaggle CPU run without an accelerator.",
    )
    parser.add_argument(
        "--machine-shape",
        help=(
            "Kaggle accelerator shape. Omit it for the default P100 GPU run, "
            "use cpu/no-accelerator for a CPU-only Kaggle run, or combine "
            "--platform cpu with a GPU shape to run AxonScope's CPU path on a "
            "Kaggle GPU machine."
        ),
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--publish-branch")
    parser.add_argument("--no-publish-branch", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--timeout", type=int, default=43_200)
    parser.add_argument("--wait-timeout", type=int, default=43_200)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--max-status-fetch-failures", type=int, default=5)
    parser.add_argument("--kaggle-bin")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "benchmark" / "results" / "kaggle",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-require-gpu", action="store_true")
    parser.add_argument(
        "--jax-cuda-extra",
        default="cuda12",
        help="Optional JAX CUDA extra installed in Kaggle; use '' to skip.",
    )
    parser.add_argument(
        "--pip-package",
        action="append",
        default=[],
        help=(
            "Extra pip package installed inside the Kaggle kernel after the "
            "repo and optional JAX CUDA wheel. Repeat for several packages."
        ),
    )
    parser.add_argument(
        "--output-file-pattern",
        default=".*axonscope_benchmark_results.*",
        help="Regex passed to `kaggle kernels output --file-pattern`.",
    )
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    preset_explicit = _flag_present(raw_args, "--preset")
    machine_shape_explicit = _flag_present(raw_args, "--machine-shape")
    args, benchmark_args = parser.parse_known_args(argv)
    if benchmark_args[:1] == ["--"]:
        benchmark_args = benchmark_args[1:]
    validate_benchmark_target(parser, args)
    normalize_compute_args(
        args,
        preset_explicit=preset_explicit,
        machine_shape_explicit=machine_shape_explicit,
    )
    return args, benchmark_args


def validate_benchmark_target(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.campaign in STANDALONE_CAMPAIGNS:
        if args.script is not None:
            parser.error(
                f"--script is not used with --campaign {args.campaign}."
            )
        return
    if args.script is None:
        parser.error(
            "--script is required unless a standalone --campaign is used."
        )


def normalize_compute_args(
    args: argparse.Namespace,
    *,
    preset_explicit: bool,
    machine_shape_explicit: bool,
) -> None:
    if args.cpu:
        args.platform = "cpu"
        args.machine_shape = "cpu"
        args.no_require_gpu = True
        if not preset_explicit:
            args.preset = "quick"
        return

    if args.platform is None:
        args.platform = "gpu"
    if args.machine_shape is None:
        args.machine_shape = "cpu" if args.platform == "cpu" else "NvidiaTeslaP100"
    if args.preset is None:
        if args.campaign in STANDALONE_CAMPAIGNS:
            args.preset = "quick"
        else:
            args.preset = "quick" if args.platform == "cpu" else "gpu_smoke"

    if is_cpu_machine_shape(args.machine_shape):
        args.machine_shape = "cpu"
        args.platform = "cpu"
        args.no_require_gpu = True
        if not preset_explicit:
            args.preset = "quick"
    elif args.platform == "cpu":
        args.no_require_gpu = True
        if not machine_shape_explicit:
            args.machine_shape = "cpu"


def _flag_present(argv: list[str], flag: str) -> bool:
    return flag in argv or any(value.startswith(flag + "=") for value in argv)


def prepare_kernel_package(
    args: argparse.Namespace,
    benchmark_args: list[str],
    run_dir: Path,
) -> Path:
    package_dir = run_dir / "kernel"
    package_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(args)
    require_gpu = bool(not args.no_require_gpu and args.platform == "gpu")
    config = {
        "campaign": args.campaign,
        "repo_url": args.repo_url,
        "branch": args.branch,
        "script": args.script,
        "preset": args.preset,
        "platform": args.platform,
        "benchmark_args": benchmark_args,
        "run_id": run_id,
        "require_gpu": require_gpu,
        "jax_cuda_extra": args.jax_cuda_extra,
        "pip_packages": list(args.pip_package or ()),
    }
    metadata: dict[str, Any] = {
        "id": f"{args.username}/{args.slug}",
        "title": args.title or args.slug,
        "code_file": "axonscope_benchmark_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false" if is_cpu_machine_shape(args.machine_shape) else "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if not is_cpu_machine_shape(args.machine_shape):
        metadata["machine_shape"] = args.machine_shape

    write_json(package_dir / "kernel-metadata.json", metadata)
    write_json(package_dir / "kaggle_config.json", config)
    write_text(
        package_dir / "axonscope_benchmark_kernel.py",
        render_kernel_entry(config),
    )
    write_json(
        run_dir / "submission.json",
        {
            "kernel_ref": metadata["id"],
            "package_dir": str(package_dir),
            "metadata": metadata,
            "config": config,
            "benchmark_args": benchmark_args,
        },
    )
    return package_dir


def render_kernel_entry(config: dict[str, Any]) -> str:
    source = KERNEL_ENTRY.read_text(encoding="utf-8")
    marker = "EMBEDDED_CONFIG: dict[str, Any] = {}"
    payload = json.dumps(config, sort_keys=True)
    replacement = f"EMBEDDED_CONFIG: dict[str, Any] = json.loads({payload!r})"
    if marker not in source:
        raise RuntimeError(f"Could not find embedded config marker in {KERNEL_ENTRY}")
    return source.replace(marker, replacement, 1)


def make_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    machine = safe_token(args.machine_shape)
    target = args.campaign or args.script
    slug = safe_token(args.slug)
    return args.output_root / (
        f"{timestamp}_{target}_{args.preset}_{args.platform}_{machine}_{slug}"
    )


def make_run_id(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = args.campaign or args.script
    return (
        f"{safe_token(target)}_{safe_token(args.preset)}_"
        f"{safe_token(args.platform)}_{timestamp}"
    )


def resolve_kaggle_bin(kaggle_bin: str | None) -> str:
    if kaggle_bin:
        return kaggle_bin
    env_kaggle = Path(sys.executable).with_name("kaggle")
    if env_kaggle.exists():
        return str(env_kaggle)
    resolved = shutil.which("kaggle")
    if resolved:
        return resolved
    raise SystemExit("Could not find `kaggle` on PATH. Activate the project env first.")


def publish_git_branch(args: argparse.Namespace, run_dir: Path) -> str:
    short_sha = git_output(["git", "rev-parse", "--short", "HEAD"]).strip()
    branch = args.publish_branch or f"{DEFAULT_BRANCH_PREFIX}/{short_sha}"
    validate_git_branch_name(branch)
    status = git_output(["git", "status", "--short"])
    write_text(run_dir / "git_status.log", status)
    if status.strip():
        message = (
            "Local worktree has uncommitted changes. Kaggle clones committed files "
            f"from HEAD {short_sha}; commit first if this run must include them."
        )
        if args.require_clean_git:
            raise SystemExit(message)
        print(f"Warning: {message}")
    result = run(
        ["git", "push", args.remote, f"HEAD:{branch}"],
        capture=True,
        check=False,
        cwd=REPO_ROOT,
    )
    write_text(run_dir / "git_publish.log", command_output(result))
    if result.returncode != 0:
        raise SystemExit(f"Could not publish Kaggle branch. See {run_dir / 'git_publish.log'}")
    print(f"Published HEAD {short_sha} to {args.remote}/{branch}")
    return branch


def monitor_kernel(
    args: argparse.Namespace,
    kaggle_bin: str,
    kernel_ref: str,
    run_dir: Path,
) -> int:
    status, kind = poll_status(
        kaggle_bin=kaggle_bin,
        kernel_ref=kernel_ref,
        poll_interval=args.poll_interval,
        wait_timeout=args.wait_timeout,
        max_status_fetch_failures=args.max_status_fetch_failures,
        run_dir=run_dir,
    )
    fetch_logs(kaggle_bin, kernel_ref, run_dir)
    fetch_files(kaggle_bin, kernel_ref, run_dir)
    if kind == "success":
        if not args.no_download:
            code = download_outputs(
                kaggle_bin,
                kernel_ref,
                run_dir,
                file_pattern=args.output_file_pattern,
            )
            if code != 0:
                return code
        print(f"Kaggle run completed: {status}")
        print(f"Run artifacts: {run_dir}")
        return 0
    print(f"Kaggle run ended with {status}. See {run_dir / 'kernel.log'}")
    return 1


def poll_status(
    *,
    kaggle_bin: str,
    kernel_ref: str,
    poll_interval: int,
    wait_timeout: int,
    max_status_fetch_failures: int,
    run_dir: Path,
) -> tuple[str, str]:
    deadline = time.monotonic() + wait_timeout
    status_log = run_dir / "status.log"
    failures = 0
    saw_status = False
    printed_log_bytes = 0
    while True:
        result = run([kaggle_bin, "kernels", "status", kernel_ref], capture=True, check=False)
        output = command_output(result)
        append_text(status_log, f"\n[{datetime.now().isoformat(timespec='seconds')}]\n{output}\n")
        if result.returncode != 0:
            failures += 1
            if saw_status and failures <= max_status_fetch_failures:
                time.sleep(poll_interval)
                continue
            raise SystemExit(f"Could not fetch Kaggle status. See {status_log}")
        failures = 0
        saw_status = True
        status = parse_kernel_status(output)
        print(output.strip())
        printed_log_bytes = stream_log_snapshot(
            kaggle_bin,
            kernel_ref,
            run_dir,
            printed_log_bytes,
        )
        kind = status_kind(status)
        if kind != "running":
            return status, kind
        if time.monotonic() >= deadline:
            raise SystemExit(f"Timed out waiting for Kaggle kernel. See {status_log}")
        time.sleep(poll_interval)


def stream_log_snapshot(
    kaggle_bin: str,
    kernel_ref: str,
    run_dir: Path,
    printed_bytes: int,
) -> int:
    result = run([kaggle_bin, "kernels", "logs", kernel_ref], capture=True, check=False)
    raw = command_output(result)
    if result.returncode != 0:
        return printed_bytes
    formatted = format_kaggle_logs(raw)
    write_text(run_dir / "kernel.log", raw)
    write_text(run_dir / "kernel.txt", formatted)
    encoded = formatted.encode("utf-8")
    if len(encoded) > printed_bytes:
        delta = encoded[printed_bytes:].decode("utf-8", errors="replace")
        if delta.strip():
            print(delta, end="" if delta.endswith("\n") else "\n")
    return len(encoded)


def fetch_logs(kaggle_bin: str, kernel_ref: str, run_dir: Path) -> None:
    result = run([kaggle_bin, "kernels", "logs", kernel_ref], capture=True, check=False)
    raw = command_output(result)
    write_text(run_dir / "kernel.log", raw)
    write_text(run_dir / "kernel.txt", format_kaggle_logs(raw))


def fetch_files(kaggle_bin: str, kernel_ref: str, run_dir: Path) -> None:
    result = run([kaggle_bin, "kernels", "files", kernel_ref], capture=True, check=False)
    write_text(run_dir / "files.log", command_output(result))


def download_outputs(
    kaggle_bin: str,
    kernel_ref: str,
    run_dir: Path,
    *,
    file_pattern: str | None,
) -> int:
    output_dir = run_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [kaggle_bin, "kernels", "output", kernel_ref, "-p", str(output_dir), "-o"]
    if file_pattern:
        command.extend(["--file-pattern", file_pattern])
    result = run(command, capture=True, check=False)
    write_text(run_dir / "download.log", command_output(result))
    if result.returncode != 0:
        print(f"Could not download Kaggle outputs. See {run_dir / 'download.log'}")
    return result.returncode


def parse_kernel_status(output: str) -> str:
    match = re.search(r'status\s+"([^"]+)"', output)
    if match:
        return match.group(1)
    return output.strip() or "UNKNOWN"


def status_kind(status: str) -> str:
    value = status.upper()
    if "COMPLETE" in value or "SUCCESS" in value:
        return "success"
    if "ERROR" in value or "FAIL" in value or "CANCEL" in value:
        return "failure"
    return "running"


def format_kaggle_logs(raw_output: str) -> str:
    try:
        entries = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output
    if not isinstance(entries, list):
        return raw_output
    lines: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            return raw_output
        stream = entry.get("stream_name", "stream")
        timestamp = entry.get("time", "")
        data = str(entry.get("data", ""))
        prefix = f"[{timestamp} {stream}] " if timestamp else f"[{stream}] "
        for line in data.splitlines() or [""]:
            lines.append(prefix + line)
    return "\n".join(lines) + "\n"


def validate_git_branch_name(branch: str) -> None:
    result = run(
        ["git", "check-ref-format", "--branch", branch],
        capture=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(f"Invalid git branch name: {branch!r}")


def git_output(command: list[str]) -> str:
    result = run(command, capture=True, check=True, cwd=REPO_ROOT)
    return command_output(result)


def run(
    command: list[str],
    *,
    capture: bool,
    check: bool,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(shlex.quote(str(part)) for part in command), flush=True)
    return subprocess.run(command, cwd=cwd, text=True, capture_output=capture, check=check)


def command_output(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout or "") + (result.stderr or "")


def write_json(path: Path, data: dict[str, Any]) -> None:
    write_text(path, json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def is_cpu_machine_shape(value: str) -> bool:
    return str(value).strip().lower() in {"cpu", "none", "no_accelerator", "no-accelerator"}


def safe_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
