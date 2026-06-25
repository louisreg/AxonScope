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
import shlex
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
PYTHON_EXECUTABLE = pathlib.Path(sys.executable)
os.environ.setdefault("MAMBA_ROOT_PREFIX", str(WORK_DIR / "micromamba_root"))


def python_executable() -> str:
    return str(PYTHON_EXECUTABLE)


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
    elif BENCHMARK == "linear_pcr_soa_trace":
        run_linear_pcr_soa_trace(out_dir)
    elif BENCHMARK == "e2e":
        run_e2e(out_dir, mode="standard")
    elif BENCHMARK == "e2e_full":
        run_e2e(out_dir, mode="full")
    elif BENCHMARK == "realistic_smoke":
        run_realistic_examples(out_dir, smoke=True)
    elif BENCHMARK == "realistic":
        run_realistic_examples(out_dir, smoke=False)
    elif BENCHMARK == "realistic_stress":
        run_realistic_examples(out_dir, smoke=False, stress=True)
    elif BENCHMARK == "realistic_stress_cpu":
        run_realistic_examples(
            out_dir,
            smoke=False,
            stress=True,
            platforms=("cpu",),
            progress=True,
        )
    elif BENCHMARK == "realistic_stress_gpu":
        run_realistic_examples(
            out_dir,
            smoke=False,
            stress=True,
            platforms=("gpu",),
            progress=True,
        )
    elif BENCHMARK == "realistic_stress_single_vm":
        run_realistic_examples(
            out_dir,
            smoke=False,
            stress=True,
            example08_recording="center",
        )
    elif BENCHMARK == "realistic_stress_single_vm_cpu":
        run_realistic_examples(
            out_dir,
            smoke=False,
            stress=True,
            platforms=("cpu",),
            example08_recording="center",
            progress=True,
        )
    elif BENCHMARK == "realistic_stress_single_vm_gpu":
        run_realistic_examples(
            out_dir,
            smoke=False,
            stress=True,
            platforms=("gpu",),
            example08_recording="center",
            progress=True,
        )
    elif BENCHMARK == "realistic_stress_observer":
        run_realistic_examples(
            out_dir,
            smoke=False,
            stress=True,
            example08_recording="observer_only",
            example08_observer_cpu_chunk_size=1,
        )
    elif BENCHMARK == "realistic_stress_observer_cpu":
        run_realistic_examples(
            out_dir,
            smoke=False,
            stress=True,
            platforms=("cpu",),
            example08_recording="observer_only",
        )
    elif BENCHMARK == "realistic_stress_observer_gpu":
        run_realistic_examples(
            out_dir,
            smoke=False,
            stress=True,
            platforms=("gpu",),
            example08_recording="observer_only",
            progress=True,
        )
    elif BENCHMARK == "population_tsim_gpu":
        run_population_tsim_gpu(out_dir)
    elif BENCHMARK == "population_tsim_gpu_1000":
        run_population_tsim_gpu(
            out_dir,
            suite="population_tsim_gpu_1000",
            prefix="population_tsim_gpu_1000",
        )
    elif BENCHMARK == "realistic_fascicle_nrv_gpu":
        run_realistic_fascicle_nrv_gpu(out_dir)
    elif BENCHMARK == "both":
        run_linear(out_dir, smoke=False)
        run_e2e(out_dir, smoke=False)
    else:
        raise ValueError(
            "AXONSCOPE_KAGGLE_BENCHMARK must be smoke, linear, "
            "linear_pcr_soa_trace, e2e, e2e_full, realistic_smoke, "
            "realistic, realistic_stress, realistic_stress_cpu, "
            "realistic_stress_gpu, realistic_stress_single_vm, "
            "realistic_stress_single_vm_cpu, realistic_stress_single_vm_gpu, "
            "realistic_stress_observer, "
            "realistic_stress_observer_cpu, realistic_stress_observer_gpu, "
            "population_tsim_gpu, population_tsim_gpu_1000, "
            "realistic_fascicle_nrv_gpu, or both."
        )

    archive = shutil.make_archive(str(out_dir), "zip", out_dir)
    print(f"\nResults folder: {out_dir}")
    print(f"Archive: {archive}")


def setup_repo() -> None:
    global PYTHON_EXECUTABLE
    if CHECKOUT_DIR.exists():
        shutil.rmtree(CHECKOUT_DIR)
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(CHECKOUT_DIR)])
    run(["git", "rev-parse", "--short", "HEAD"], cwd=CHECKOUT_DIR)
    if benchmark_needs_nrv_stack():
        PYTHON_EXECUTABLE = install_nrv_stack()
    run([python_executable(), "-m", "pip", "install", "-U", "pip"])
    run([python_executable(), "-m", "pip", "install", "-e", ".[benchmark]"], cwd=CHECKOUT_DIR)
    if REQUIRE_GPU and JAX_CUDA_EXTRA:
        install_jax_gpu_extra()
    if benchmark_needs_nrv_stack():
        verify_nrv_stack()


def benchmark_needs_nrv_stack() -> bool:
    return BENCHMARK in {"realistic_fascicle_nrv_gpu"}


def install_nrv_stack() -> pathlib.Path:
    micromamba = install_micromamba()
    env_dir = WORK_DIR / "axonscope_nrv_env"
    env_yaml = WORK_DIR / "nrv_linux.yaml"
    run(
        [
            "curl",
            "-L",
            "-o",
            str(env_yaml),
            "https://raw.githubusercontent.com/nrv-framework/NRV/refs/heads/master/conda/nrv_linux.yaml",
        ],
        cwd=CHECKOUT_DIR,
    )
    run(
        [
            str(micromamba),
            "create",
            "-y",
            "-p",
            str(env_dir),
            "-f",
            str(env_yaml),
        ],
        cwd=CHECKOUT_DIR,
    )
    env_yaml.unlink(missing_ok=True)
    env_python = env_dir / "bin" / "python"
    run([str(env_python), "-m", "pip", "install", "-U", "pip"])
    run([str(env_python), "-m", "pip", "install", "nrv-py"], cwd=CHECKOUT_DIR)
    return env_python


def install_micromamba() -> pathlib.Path:
    micromamba = WORK_DIR / "micromamba" / "bin" / "micromamba"
    if micromamba.exists():
        return micromamba
    micromamba.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        f"| tar -xj -C {shlex.quote(str(WORK_DIR / 'micromamba'))} bin/micromamba"
    )
    run(["bash", "-lc", command], cwd=CHECKOUT_DIR)
    return micromamba


def verify_nrv_stack() -> None:
    code = """
import dolfinx
import mpi4py
import nrv
print("dolfinx:", getattr(dolfinx, "__version__", "unknown"))
print("mpi4py import ok")
print("nrv import ok")
"""
    run([python_executable(), "-c", code], cwd=CHECKOUT_DIR)


def install_jax_gpu_extra() -> None:
    jax_version = installed_package_version("jax")
    requirements = [
        f"jax=={jax_version}",
        f"jaxlib=={jax_version}",
        f"jax[{JAX_CUDA_EXTRA}]=={jax_version}",
    ]
    if JAX_CUDA_EXTRA == "cuda12":
        requirements.append(f"jax-cuda12-plugin[with_cuda]=={jax_version}")
        requirements.extend(
            [
                "nvidia-cublas-cu12",
                "nvidia-cuda-cupti-cu12",
                "nvidia-cuda-runtime-cu12",
                "nvidia-cudnn-cu12",
                "nvidia-cufft-cu12",
                "nvidia-cusolver-cu12",
                "nvidia-cusparse-cu12",
                "nvidia-nccl-cu12",
                "nvidia-nvjitlink-cu12",
            ]
        )
    run(
        [
            python_executable(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            *requirements,
        ],
        cwd=CHECKOUT_DIR,
    )
    configure_cuda_library_path()
    print_cuda_package_diagnostics()


def configure_cuda_library_path() -> None:
    code = """
import pathlib
import site

roots = []
for site_dir in site.getsitepackages():
    nvidia_dir = pathlib.Path(site_dir) / "nvidia"
    if not nvidia_dir.exists():
        continue
    roots.extend(str(path) for path in sorted(nvidia_dir.glob("*/lib")) if path.exists())
print("\\n".join(roots))
"""
    output = subprocess.check_output(
        [python_executable(), "-c", code],
        cwd=CHECKOUT_DIR,
        text=True,
    )
    paths = [line.strip() for line in output.splitlines() if line.strip()]
    if not paths:
        print("No pip NVIDIA CUDA library directories found.")
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    combined = os.pathsep.join(paths + ([existing] if existing else []))
    os.environ["LD_LIBRARY_PATH"] = combined
    print("LD_LIBRARY_PATH CUDA entries:")
    for path in paths:
        print(f"  {path}")


def print_cuda_package_diagnostics() -> None:
    code = """
import importlib.metadata as metadata
for name in (
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "nvidia-cusparse-cu12",
    "nvidia-cublas-cu12",
):
    try:
        print(f"{name}: {metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}: not installed")
"""
    run([python_executable(), "-c", code], cwd=CHECKOUT_DIR)


def installed_package_version(package_name: str) -> str:
    command = [
        python_executable(),
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
    run([python_executable(), "-c", code], cwd=CHECKOUT_DIR)


def run_linear(out_dir: pathlib.Path, *, smoke: bool) -> None:
    command = [
        python_executable(),
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


def run_linear_pcr_soa_trace(out_dir: pathlib.Path) -> None:
    trace_dir = out_dir / "linear_pcr_soa_trace" / "jax_traces"
    command = [
        python_executable(),
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


def run_e2e(out_dir: pathlib.Path, *, smoke: bool = False, mode: str = "standard") -> None:
    command = [
        python_executable(),
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


def run_realistic_examples(
    out_dir: pathlib.Path,
    *,
    smoke: bool,
    stress: bool = False,
    platforms: tuple[str, ...] = ("cpu", "gpu"),
    example08_recording: str = "full",
    example08_observer_cpu_chunk_size: int = 0,
    progress: bool = False,
) -> None:
    command = [
        python_executable(),
        "benchmark/realistic_examples/bench_basic_examples.py",
        "--out-dir",
        str(out_dir / "realistic_examples"),
        "--prefix",
        "realistic_examples",
        "--platforms",
        *platforms,
        "--example08-recording",
        example08_recording,
    ]
    if example08_observer_cpu_chunk_size:
        command.extend(
            [
                "--example08-observer-cpu-chunk-size",
                str(example08_observer_cpu_chunk_size),
            ]
        )
    if progress:
        command.append("--progress")
    if smoke:
        command.extend(
            [
                "--preset",
                "smoke",
                "--repeats",
                "1",
                "--warmups",
                "0",
            ]
        )
    elif stress:
        command.extend(
            [
                "--preset",
                "stress",
                "--run-counts",
                "5",
                "10",
                "20",
                "--family-counts",
                "25",
                "50",
                "--example07-max-iterations",
                "20",
                "--example08-amplitude-count",
                "8",
                "--repeats",
                "3",
                "--warmups",
                "1",
                "--profile",
            ]
        )
    else:
        command.extend(
            [
                "--preset",
                "standard",
                "--run-counts",
                "2",
                "5",
                "--family-counts",
                "5",
                "25",
                "--example07-max-iterations",
                "8",
                "--example08-amplitude-count",
                "4",
                "--repeats",
                "2",
                "--warmups",
                "1",
            ]
        )
    run(command, cwd=CHECKOUT_DIR)


def run_population_tsim_gpu(
    out_dir: pathlib.Path,
    *,
    suite: str = "population_tsim_gpu",
    prefix: str = "population_tsim_gpu",
) -> None:
    result_dir = out_dir / prefix
    profile_dir = out_dir / f"{prefix}_profiles"
    command = [
        python_executable(),
        "benchmark/nrv_performance/run.py",
        "--suite",
        suite,
        "--out-dir",
        str(result_dir),
        "--prefix",
        prefix,
        "--",
        "--report-dir",
        str(profile_dir),
    ]
    run(command, cwd=CHECKOUT_DIR)
    print_population_tsim_summary(result_dir / f"{prefix}.csv")


def run_realistic_fascicle_nrv_gpu(out_dir: pathlib.Path) -> None:
    prefix = "realistic_fascicle_nrv_gpu"
    result_dir = out_dir / prefix
    profile_dir = out_dir / f"{prefix}_profiles"
    command = [
        python_executable(),
        "benchmark/nrv_performance/run.py",
        "--suite",
        "realistic_fascicle_smoke",
        "--out-dir",
        str(result_dir),
        "--prefix",
        prefix,
        "--",
        "--report-dir",
        str(profile_dir),
        "--validate-nrv",
        "--solver-progress",
        "plain",
        "--time-chunk-steps",
        "1000",
    ]
    run(command, cwd=CHECKOUT_DIR)
    print_realistic_fascicle_summary(result_dir / f"{prefix}_summary.csv")


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


def print_population_tsim_summary(path: pathlib.Path, *, limit: int = 12) -> None:
    if not path.exists():
        print(f"population summary missing: {path}")
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"\nPOPULATION TSIM GPU summary: {path}")
    for row in rows[:limit]:
        print(
            f"n={int(row['fiber_count_simulated']):3d} "
            f"mrg={int(row['mrg_count']):3d} rattay={int(row['rattay_count']):3d} "
            f"tsim={float(row['tsim_ms']):g} ms backend={row.get('jax_backend', '')} "
            f"AS_first={float(row['as_total_first_s']):.3f}s "
            f"AS_warm={float(row['as_total_warm_median_s']):.3f}s "
            f"groups={row.get('as_group_count')} padded={row.get('as_padded_groups')}"
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows")


def print_realistic_fascicle_summary(path: pathlib.Path) -> None:
    if not path.exists():
        print(f"realistic fascicle summary missing: {path}")
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"realistic fascicle summary empty: {path}")
        return
    row = rows[0]
    print(f"\nREALISTIC FASCICLE NRV GPU summary: {path}")
    for key in (
        "simulated_fibers",
        "amplitudes_count",
        "jax_backend",
        "nrv_fem_first_footprint_s",
        "nrv_cached_footprints_s",
        "axonscope_cold_step_s",
        "axonscope_warm_step_s",
        "axonscope_steps_total_s",
        "nrv_validation_single_amplitude_s",
        "nrv_full_sweep_estimated_s",
        "nrv_estimate_over_axonscope_steps",
        "nrv_estimate_over_axonscope_with_footprints",
        "maxrss_mib",
    ):
        if key in row and row[key] != "":
            print(f"{key}: {row[key]}")


def run(command: list[str], *, cwd: pathlib.Path | None = None) -> None:
    print("\n$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
