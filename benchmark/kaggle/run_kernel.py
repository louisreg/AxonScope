"""Run the AxonScope Kaggle solver benchmark kernel end to end."""

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
from typing import Sequence


KERNEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = KERNEL_DIR.parents[1]
DEFAULT_SLUG = "axonscope-double-cable-solver-benchmarks"
DEFAULT_TITLE = "AxonScope Double-Cable Solver Benchmarks"
DEFAULT_REPO_URL = "https://github.com/louisreg/AxonScope.git"
DEFAULT_PUBLISH_BRANCH_PREFIX = "kaggle-bench"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    kaggle_bin = resolve_kaggle_bin(args.kaggle_bin)
    kernel_ref = f"{args.username}/{args.slug}"
    run_dir = args.run_dir or make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        return run_kernel(args, kaggle_bin, kernel_ref, run_dir)
    except KeyboardInterrupt:
        print("\nInterrupted locally with Ctrl+C.")
        print("The remote Kaggle kernel may still be running.")
        fetch_logs(kaggle_bin, kernel_ref, run_dir)
        fetch_files(kaggle_bin, kernel_ref, run_dir)
        if args.delete_kernel_on_interrupt:
            return delete_kernel(kaggle_bin, kernel_ref, run_dir)
        print(f"Check/stop it in Kaggle: https://www.kaggle.com/code/{kernel_ref}")
        print(f"Run artifacts: {run_dir}")
        print(
            "CLI fallback, destructive: "
            f"{kaggle_bin} kernels delete {kernel_ref} --yes"
        )
        return 130


def run_kernel(
    args: argparse.Namespace,
    kaggle_bin: str,
    kernel_ref: str,
    run_dir: Path,
) -> int:
    if args.attach:
        return monitor_kernel(args, kaggle_bin, kernel_ref, run_dir)

    published_branch = publish_git_branch(args, run_dir)
    if published_branch:
        args.branch = published_branch

    prepare_metadata(args)

    push = run(
        [
            kaggle_bin,
            "kernels",
            "push",
            "-p",
            str(args.kernel_dir),
            "--accelerator",
            args.machine_shape,
            "--timeout",
            str(args.timeout),
        ],
        capture=True,
        check=False,
    )
    write_text(run_dir / "push.log", command_output(push))
    if push.returncode != 0:
        print(f"Kaggle push failed. See {run_dir / 'push.log'}")
        return push.returncode

    if args.no_wait:
        print(f"Kaggle push submitted: {kernel_ref}")
        print(f"Run artifacts: {run_dir}")
        return 0

    return monitor_kernel(args, kaggle_bin, kernel_ref, run_dir)


def monitor_kernel(
    args: argparse.Namespace,
    kaggle_bin: str,
    kernel_ref: str,
    run_dir: Path,
) -> int:
    if args.no_wait:
        fetch_logs(kaggle_bin, kernel_ref, run_dir)
        fetch_files(kaggle_bin, kernel_ref, run_dir)
        print(f"Fetched current Kaggle artifacts for {kernel_ref}")
        print(f"Run artifacts: {run_dir}")
        return 0

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
            output_code = download_outputs(
                kaggle_bin,
                kernel_ref,
                run_dir,
                file_pattern=args.output_file_pattern,
            )
            if output_code != 0:
                return output_code
        print(f"Kaggle run completed: {status}")
        print(f"Run artifacts: {run_dir}")
        return 0

    print(f"Kaggle run ended with {status}. See {run_dir / 'kernel.log'}")
    return 1


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Kaggle username slug.")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle kernel slug.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Kaggle kernel title.")
    parser.add_argument(
        "--benchmark",
        default="smoke",
        choices=(
            "smoke",
            "linear",
            "linear_pcr_soa_trace",
            "e2e",
            "e2e_full",
            "realistic_smoke",
            "realistic",
            "realistic_stress",
            "both",
        ),
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--machine-shape",
        default="NvidiaTeslaP100",
        help="Kaggle accelerator, e.g. NvidiaTeslaP100 or NvidiaTeslaT4.",
    )
    parser.add_argument(
        "--branch",
        default="bench-colab",
        help="Existing repo branch cloned by Kaggle when branch publishing is disabled.",
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Repo URL cloned by Kaggle.")
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote used to publish the Kaggle clone branch.",
    )
    parser.add_argument(
        "--publish-branch",
        default=None,
        help=(
            "Remote branch to push HEAD to before submitting Kaggle. "
            "Defaults to kaggle-bench/<short-sha>."
        ),
    )
    parser.add_argument(
        "--no-publish-branch",
        action="store_true",
        help="Do not push a dedicated Kaggle branch; use --branch as-is.",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Fail if the local worktree has uncommitted changes before publishing HEAD.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=43_200,
        help="Maximum Kaggle kernel runtime in seconds.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=43_200,
        help="Maximum local polling time in seconds.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks.",
    )
    parser.add_argument(
        "--max-status-fetch-failures",
        type=int,
        default=5,
        help=(
            "Consecutive Kaggle status fetch failures tolerated after at least one "
            "successful status response. This handles transient Kaggle API errors "
            "while still failing immediately for wrong slugs/auth before any status "
            "was fetched."
        ),
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        default=KERNEL_DIR,
        help="Folder uploaded by `kaggle kernels push`.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "benchmark" / "results" / "kaggle",
        help="Local folder where runner logs and downloaded outputs are saved.",
    )
    parser.add_argument(
        "--kaggle-bin",
        default=None,
        help="Path to the Kaggle executable. Defaults to the active PATH.",
    )
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail inside Kaggle if JAX does not select a GPU backend.",
    )
    parser.add_argument("--no-wait", action="store_true", help="Return after submitting the run.")
    parser.add_argument(
        "--attach",
        action="store_true",
        help="Attach to the latest existing Kaggle run; do not publish or push a new kernel.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Local artifact directory to use, useful with --attach after Ctrl+C.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download outputs after a successful run.",
    )
    parser.add_argument(
        "--output-file-pattern",
        default=".*axonscope_solver_results.*",
        help="Regex for Kaggle output files to download. Defaults to benchmark result files only.",
    )
    parser.add_argument(
        "--delete-kernel-on-interrupt",
        action="store_true",
        help="On Ctrl+C, delete the remote Kaggle kernel. Destructive.",
    )
    return parser.parse_args(argv)


def resolve_kaggle_bin(kaggle_bin: str | None) -> str:
    if kaggle_bin:
        return kaggle_bin
    env_kaggle = Path(sys.executable).with_name("kaggle")
    if env_kaggle.exists():
        return str(env_kaggle)
    resolved = shutil.which("kaggle")
    if not resolved:
        raise SystemExit("Could not find `kaggle` on PATH. Activate Axonscope-env first.")
    return resolved


def make_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    machine = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.machine_shape)
    return args.output_root / f"{timestamp}_{args.benchmark}_{machine}"


def publish_git_branch(args: argparse.Namespace, run_dir: Path) -> str | None:
    if args.no_publish_branch:
        return None

    head = git_output(["git", "rev-parse", "--short", "HEAD"]).strip()
    branch = args.publish_branch or default_publish_branch(head)
    validate_git_branch_name(branch)

    status = git_output(["git", "status", "--short"])
    write_text(run_dir / "git_status.log", status)
    if status.strip():
        message = (
            "Local worktree has uncommitted changes. Kaggle will clone only committed files "
            f"from HEAD {head}; uploaded Kaggle wrapper files are still sent by `kaggle kernels push`."
        )
        if args.require_clean_git:
            print(message)
            print(f"See {run_dir / 'git_status.log'}")
            raise SystemExit(1)
        print(f"Warning: {message}")

    push = run(
        ["git", "push", args.remote, f"HEAD:{branch}"],
        capture=True,
        check=False,
        cwd=REPO_ROOT,
    )
    write_text(run_dir / "git_publish.log", command_output(push))
    if push.returncode != 0:
        print(f"Could not publish Kaggle branch. See {run_dir / 'git_publish.log'}")
        raise SystemExit(push.returncode)

    print(f"Published HEAD {head} to {args.remote}/{branch}")
    return branch


def default_publish_branch(short_sha: str) -> str:
    return f"{DEFAULT_PUBLISH_BRANCH_PREFIX}/{short_sha.strip()}"


def validate_git_branch_name(branch: str) -> None:
    result = run(
        ["git", "check-ref-format", "--branch", branch],
        capture=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(f"Invalid git branch name: {branch!r}")


def git_output(command: Sequence[str]) -> str:
    result = run(command, capture=True, check=True, cwd=REPO_ROOT)
    return command_output(result)


def prepare_metadata(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(KERNEL_DIR / "prepare_kernel_metadata.py"),
        "--username",
        args.username,
        "--slug",
        args.slug,
        "--title",
        args.title,
        "--machine-shape",
        args.machine_shape,
        "--benchmark",
        args.benchmark,
        "--branch",
        args.branch,
        "--repo-url",
        args.repo_url,
        "--path",
        str(args.kernel_dir),
    ]
    if not args.require_gpu:
        command.append("--no-require-gpu")
    run(command, capture=False, check=True)


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
    consecutive_fetch_failures = 0
    saw_successful_status = status_log.exists() and bool(
        re.search(r'status\s+"[^"]+"', status_log.read_text(encoding="utf-8"))
    )
    while True:
        result = run(
            [kaggle_bin, "kernels", "status", kernel_ref],
            capture=True,
            check=False,
        )
        output = command_output(result)
        append_text(status_log, f"\n[{datetime.now().isoformat(timespec='seconds')}]\n{output}\n")
        if result.returncode != 0:
            if saw_successful_status:
                consecutive_fetch_failures += 1
                if consecutive_fetch_failures <= max_status_fetch_failures:
                    print(
                        "Could not fetch Kaggle status "
                        f"({consecutive_fetch_failures}/{max_status_fetch_failures}); "
                        f"retrying. See {status_log}"
                    )
                    if time.monotonic() >= deadline:
                        raise SystemExit(f"Timed out waiting for Kaggle kernel. See {status_log}")
                    time.sleep(poll_interval)
                    continue
            raise SystemExit(f"Could not fetch Kaggle status. See {status_log}")

        consecutive_fetch_failures = 0
        saw_successful_status = True
        status = parse_kernel_status(output)
        kind = status_kind(status)
        print(output.strip())
        if kind != "running":
            return status, kind

        if time.monotonic() >= deadline:
            raise SystemExit(f"Timed out waiting for Kaggle kernel. See {status_log}")
        time.sleep(poll_interval)


def parse_kernel_status(output: str) -> str:
    match = re.search(r'status\s+"([^"]+)"', output)
    if match:
        return match.group(1)
    stripped = output.strip()
    return stripped or "UNKNOWN"


def status_kind(status: str) -> str:
    status_upper = status.upper()
    if "COMPLETE" in status_upper or "SUCCESS" in status_upper:
        return "success"
    if "ERROR" in status_upper or "FAIL" in status_upper or "CANCEL" in status_upper:
        return "failure"
    return "running"


def fetch_logs(kaggle_bin: str, kernel_ref: str, run_dir: Path) -> None:
    result = run([kaggle_bin, "kernels", "logs", kernel_ref], capture=True, check=False)
    raw_output = command_output(result)
    write_text(run_dir / "kernel.log", raw_output)
    formatted = format_kaggle_logs(raw_output)
    if formatted != raw_output:
        write_text(run_dir / "kernel.txt", formatted)
    if result.returncode != 0:
        print(f"Could not fetch Kaggle logs. See {run_dir / 'kernel.log'}")


def format_kaggle_logs(raw_output: str) -> str:
    try:
        entries = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output
    if not isinstance(entries, list):
        return raw_output

    lines = []
    for entry in entries:
        if not isinstance(entry, dict):
            return raw_output
        stream = entry.get("stream_name", "stream")
        timestamp = entry.get("time", "")
        data = str(entry.get("data", ""))
        prefix = f"[{timestamp} {stream}] " if timestamp != "" else f"[{stream}] "
        for line in data.splitlines() or [""]:
            lines.append(prefix + line)
    return "\n".join(lines) + "\n"


def fetch_files(kaggle_bin: str, kernel_ref: str, run_dir: Path) -> None:
    result = run([kaggle_bin, "kernels", "files", kernel_ref], capture=True, check=False)
    write_text(run_dir / "files.log", command_output(result))
    if result.returncode != 0:
        print(f"Could not list Kaggle output files. See {run_dir / 'files.log'}")


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
    result = run(
        command,
        capture=True,
        check=False,
    )
    write_text(run_dir / "download.log", command_output(result))
    if result.returncode != 0:
        print(f"Could not download Kaggle outputs. See {run_dir / 'download.log'}")
    return result.returncode


def delete_kernel(kaggle_bin: str, kernel_ref: str, run_dir: Path) -> int:
    result = run(
        [kaggle_bin, "kernels", "delete", kernel_ref, "--yes"],
        capture=True,
        check=False,
    )
    write_text(run_dir / "delete.log", command_output(result))
    if result.returncode == 0:
        print(f"Deleted remote Kaggle kernel: {kernel_ref}")
    else:
        print(f"Could not delete Kaggle kernel. See {run_dir / 'delete.log'}")
    return result.returncode


def run(
    command: Sequence[str],
    *,
    capture: bool,
    check: bool,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(shlex.quote(str(part)) for part in command), flush=True)
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=check,
    )


def command_output(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout or "") + (result.stderr or "")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
