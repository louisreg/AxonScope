"""Kaggle entry point for AxonScope double-cable solver benchmarks.

This script is meant to run as a Kaggle script kernel. It clones the configured
AxonScope branch, installs the benchmark extra, runs the selected solver
benchmark(s), and writes zipped outputs under /kaggle/working.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime


CONFIG_PATH = pathlib.Path(__file__).with_name("kaggle_config.json")
EMBEDDED_CONFIG: dict[str, object] = {}


def load_config() -> dict[str, object]:
    config = dict(EMBEDDED_CONFIG)
    if not CONFIG_PATH.exists():
        return config
    config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return config


CONFIG = load_config()


def string_setting(env_name: str, config_key: str, default: str) -> str:
    if env_name in os.environ:
        return os.environ[env_name]
    value = CONFIG.get(config_key, default)
    return str(value)


def bool_setting(env_name: str, config_key: str, default: bool) -> bool:
    if env_name in os.environ:
        value: object = os.environ[env_name]
    else:
        value = CONFIG.get(config_key, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


REPO_URL = string_setting(
    "AXONSCOPE_REPO_URL", "repo_url", "https://github.com/louisreg/AxonScope.git"
)
BRANCH = string_setting("AXONSCOPE_BRANCH", "branch", "bench-colab")
BENCHMARK = string_setting("AXONSCOPE_KAGGLE_BENCHMARK", "benchmark", "smoke")
REQUIRE_GPU = bool_setting("AXONSCOPE_REQUIRE_GPU", "require_gpu", True)
JAX_CUDA_EXTRA = string_setting("AXONSCOPE_JAX_CUDA_EXTRA", "jax_cuda_extra", "cuda12")

WORK_DIR = pathlib.Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
CHECKOUT_DIR = pathlib.Path(os.environ.get("AXONSCOPE_CHECKOUT_DIR", "/tmp/AxonScope"))
RESULTS_ROOT = WORK_DIR / "axonscope_solver_results"


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    setup_repo()
    verify_backend()

    run_id = f"kaggle_{BENCHMARK}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if BENCHMARK == "smoke":
        run_linear(out_dir, smoke=True)
        run_e2e(out_dir, smoke=True)
    elif BENCHMARK == "linear":
        run_linear(out_dir, smoke=False)
    elif BENCHMARK == "linear_assoc_focus":
        run_linear_assoc_focus(out_dir)
    elif BENCHMARK == "linear_pcr_soa_trace":
        run_linear_pcr_soa_trace(out_dir)
    elif BENCHMARK == "linear_pcr_soa_nomask_focus":
        run_linear_pcr_soa_nomask_focus(out_dir)
    elif BENCHMARK == "linear_pallas_focus":
        run_linear_pallas_focus(out_dir)
    elif BENCHMARK == "e2e":
        run_e2e(out_dir, mode="standard")
    elif BENCHMARK == "e2e_full":
        run_e2e(out_dir, mode="full")
    elif BENCHMARK == "both":
        run_linear(out_dir, smoke=False)
        run_e2e(out_dir, smoke=False)
    else:
        raise ValueError(
            "AXONSCOPE_KAGGLE_BENCHMARK must be smoke, linear, "
            "linear_assoc_focus, linear_pcr_soa_trace, "
            "linear_pcr_soa_nomask_focus, linear_pallas_focus, e2e, "
            "e2e_full, or both."
        )

    archive = shutil.make_archive(str(out_dir), "zip", out_dir)
    print(f"\nResults folder: {out_dir}")
    print(f"Archive: {archive}")


def setup_repo() -> None:
    if CHECKOUT_DIR.exists():
        shutil.rmtree(CHECKOUT_DIR)
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(CHECKOUT_DIR)])
    run(["git", "rev-parse", "--short", "HEAD"], cwd=CHECKOUT_DIR)
    run([sys.executable, "-m", "pip", "install", "-U", "pip"])
    run([sys.executable, "-m", "pip", "install", "-e", ".[benchmark]"], cwd=CHECKOUT_DIR)
    if REQUIRE_GPU and JAX_CUDA_EXTRA:
        install_jax_gpu_extra()


def install_jax_gpu_extra() -> None:
    jax_version = installed_package_version("jax")
    requirement = f"jax[{JAX_CUDA_EXTRA}]=={jax_version}"
    run([sys.executable, "-m", "pip", "install", "--upgrade", requirement], cwd=CHECKOUT_DIR)


def installed_package_version(package_name: str) -> str:
    command = [
        sys.executable,
        "-c",
        (
            "import importlib.metadata as metadata; "
            f"print(metadata.version({package_name!r}))"
        ),
    ]
    print("\n$", " ".join(str(part) for part in command), flush=True)
    return subprocess.check_output(command, cwd=CHECKOUT_DIR, text=True).strip()


def verify_backend() -> None:
    run(["bash", "-lc", "nvidia-smi || true"], cwd=CHECKOUT_DIR)
    code = """
import jax
backend = jax.default_backend()
print("jax backend:", backend)
print("jax devices:", jax.devices())
if {require_gpu!r} and backend != "gpu":
    raise SystemExit(f"Expected a Kaggle GPU backend, got {{backend!r}}.")
""".format(require_gpu=REQUIRE_GPU)
    run([sys.executable, "-c", code], cwd=CHECKOUT_DIR)


def run_linear(out_dir: pathlib.Path, *, smoke: bool) -> None:
    command = [
        sys.executable,
        "benchmark/solvers/bench_double_cable_linear_solvers.py",
        "--out-dir",
        str(out_dir),
        "--prefix",
        "linear",
    ]
    if smoke:
        command.extend(
            [
                "--batch-sizes",
                "8",
                "--nx",
                "16",
                "--dtypes",
                "float32",
                "--solvers",
                "thomas",
                "pcr_soa",
                "--warmups",
                "0",
                "--repeats",
                "1",
            ]
        )
    else:
        command.extend(
            [
                "--batch-sizes",
                "128",
                "512",
                "1024",
                "2048",
                "4096",
                "--nx",
                "32",
                "51",
                "64",
                "96",
                "--dtypes",
                "float32",
                "--solvers",
                "thomas",
                "pcr",
                "pcr_soa",
                "pcr_adaptive",
                "--warmups",
                "1",
                "--repeats",
                "5",
            ]
        )
    run(command, cwd=CHECKOUT_DIR)
    print_summary(out_dir / "linear" / "summary.csv", mode="linear")


def run_linear_assoc_focus(out_dir: pathlib.Path) -> None:
    command = [
        sys.executable,
        "benchmark/solvers/bench_double_cable_linear_solvers.py",
        "--out-dir",
        str(out_dir),
        "--prefix",
        "linear_assoc_focus",
        "--batch-sizes",
        "1024",
        "2048",
        "4096",
        "--nx",
        "51",
        "64",
        "96",
        "--dtypes",
        "float32",
        "--solvers",
        "thomas",
        "thomas_batched",
        "assoc_backward",
        "pcr_soa",
        "pcr_adaptive",
        "--warmups",
        "1",
        "--repeats",
        "5",
    ]
    run(command, cwd=CHECKOUT_DIR)
    print_summary(out_dir / "linear_assoc_focus" / "summary.csv", mode="linear")


def run_linear_pcr_soa_trace(out_dir: pathlib.Path) -> None:
    trace_dir = out_dir / "linear_pcr_soa_trace" / "jax_traces"
    command = [
        sys.executable,
        "benchmark/solvers/bench_double_cable_linear_solvers.py",
        "--out-dir",
        str(out_dir),
        "--prefix",
        "linear_pcr_soa_trace",
        "--batch-sizes",
        "2048",
        "4096",
        "--nx",
        "51",
        "96",
        "--dtypes",
        "float32",
        "--solvers",
        "pcr",
        "pcr_soa",
        "pcr_adaptive",
        "--warmups",
        "1",
        "--repeats",
        "2",
        "--skip-reference",
        "--jax-trace",
        "--jax-trace-dir",
        str(trace_dir),
    ]
    run(command, cwd=CHECKOUT_DIR)
    print_summary(out_dir / "linear_pcr_soa_trace" / "summary.csv", mode="linear")


def run_linear_pcr_soa_nomask_focus(out_dir: pathlib.Path) -> None:
    command = [
        sys.executable,
        "benchmark/solvers/bench_double_cable_linear_solvers.py",
        "--out-dir",
        str(out_dir),
        "--prefix",
        "linear_pcr_soa_nomask_focus",
        "--batch-sizes",
        "2048",
        "4096",
        "--nx",
        "51",
        "96",
        "--dtypes",
        "float32",
        "--solvers",
        "pcr_soa",
        "pcr_soa_nomask",
        "pcr_soa_shift",
        "--warmups",
        "1",
        "--repeats",
        "5",
    ]
    run(command, cwd=CHECKOUT_DIR)
    print_summary(out_dir / "linear_pcr_soa_nomask_focus" / "summary.csv", mode="linear")


def run_linear_pallas_focus(out_dir: pathlib.Path) -> None:
    command = [
        sys.executable,
        "benchmark/solvers/bench_double_cable_linear_solvers.py",
        "--out-dir",
        str(out_dir),
        "--prefix",
        "linear_pallas_focus",
        "--batch-sizes",
        "1024",
        "2048",
        "4096",
        "--nx",
        "51",
        "64",
        "96",
        "--dtypes",
        "float32",
        "--solvers",
        "thomas",
        "thomas_batched",
        "assoc_backward",
        "pcr",
        "pcr_soa",
        "pallas_pcr_128",
        "pcr_adaptive",
        "--warmups",
        "1",
        "--repeats",
        "5",
    ]
    run(command, cwd=CHECKOUT_DIR)
    print_summary(out_dir / "linear_pallas_focus" / "summary.csv", mode="linear")


def run_e2e(out_dir: pathlib.Path, *, smoke: bool = False, mode: str = "standard") -> None:
    command = [
        sys.executable,
        "benchmark/solvers/bench_double_cable_end_to_end.py",
        "--out-dir",
        str(out_dir),
        "--prefix",
        "e2e",
    ]
    if smoke:
        command.extend(
            [
                "--batch-sizes",
                "2",
                "--nx",
                "51",
                "--nt",
                "3",
                "--dt",
                "0.05",
                "--recordings",
                "center",
                "--iinj-modes",
                "none",
                "--solvers",
                "thomas",
                "--warmups",
                "0",
                "--repeats",
                "1",
            ]
        )
    elif mode == "standard":
        command.extend(
            [
                "--batch-sizes",
                "512",
                "2048",
                "--nx",
                "51",
                "96",
                "--nt",
                "500",
                "--dt",
                "0.01",
                "--recordings",
                "none",
                "center",
                "--iinj-modes",
                "none",
                "dense_zero",
                "--solvers",
                "auto",
                "thomas",
                "pcr_adaptive",
                "--warmups",
                "1",
                "--repeats",
                "2",
            ]
        )
    elif mode == "full":
        command.extend(
            [
                "--batch-sizes",
                "512",
                "1024",
                "2048",
                "--nx",
                "51",
                "64",
                "96",
                "--nt",
                "500",
                "1000",
                "--dt",
                "0.01",
                "--recordings",
                "none",
                "center",
                "full",
                "--iinj-modes",
                "none",
                "dense_zero",
                "--solvers",
                "auto",
                "thomas",
                "pcr_adaptive",
                "--warmups",
                "1",
                "--repeats",
                "3",
            ]
        )
    else:
        raise ValueError(f"unknown e2e mode: {mode!r}")
    run(command, cwd=CHECKOUT_DIR)
    print_summary(out_dir / "e2e" / "summary.csv", mode="e2e")


def print_summary(path: pathlib.Path, *, mode: str, limit: int = 12) -> None:
    if not path.exists():
        print(f"summary missing: {path}")
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"\n{mode.upper()} summary: {path}")
    for row in rows[:limit]:
        if mode == "linear":
            print(
                f"{row['requested_solver']}({row['kernel_solver']}) "
                f"B={row['batch_size']} Nx={row['nx']} {row['dtype']}: "
                f"median={float(row['steady_median_ms']):.3f} ms"
            )
        elif mode == "validation":
            status = "PASS" if row["passed_thresholds"] == "True" else "CHECK"
            print(
                f"{status} {row['candidate_solver']} vs {row['reference_solver']} "
                f"B={row['batch_size']} Nx={row['actual_nx']} Nt={row['nt']} "
                f"rec={row['recording']} iinj={row['iinj_mode']}: "
                f"max_abs={float(row['max_abs_mV']):.4g} mV "
                f"rms={float(row['rms_mV']):.4g} mV "
                f"activation={float(row['activation_agreement']):.3f}"
            )
        else:
            kernel_ms = float(row["kernel_enqueue_median_ms"]) + float(
                row["kernel_wait_median_ms"]
            )
            print(
                f"{row['requested_solver']}->{row['resolved_solver']} "
                f"B={row['batch_size']} Nx={row['actual_nx']} Nt={row['nt']} "
                f"rec={row['recording']} iinj={row['iinj_mode']}: "
                f"kernel={kernel_ms:.3f} ms total+inputs={float(row['total_with_inputs_ms']):.3f} ms"
            )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows")


def run(command: list[str], *, cwd: pathlib.Path | None = None) -> None:
    print("\n$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
