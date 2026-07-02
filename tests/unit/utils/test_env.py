# tests/test_env.py

import json

from axonscope.utils.env import (
    collect_environment_info,
    save_environment_info,
)


def test_collect_environment_info_has_expected_sections():
    info = collect_environment_info()

    expected_sections = [
        "timestamp_unix",
        "timestamp_iso",
        "os",
        "python",
        "cpu",
        "memory",
        "disk",
        "packages",
        "jax",
        "gpu",
        "mlx",
        "environment_variables",
        "git",
    ]

    for section in expected_sections:
        assert section in info


def test_collect_environment_info_does_not_include_torch():
    info = collect_environment_info()

    assert "torch" not in info
    assert "torch" not in info["packages"]
    assert "pytorch" not in info["packages"]


def test_collect_environment_info_python_section():
    info = collect_environment_info()

    assert "version" in info["python"]
    assert "implementation" in info["python"]
    assert "executable" in info["python"]
    assert isinstance(info["python"]["is_venv"], bool)


def test_collect_environment_info_os_section():
    info = collect_environment_info()

    assert "system" in info["os"]
    assert "platform" in info["os"]
    assert "hostname" in info["os"]


def test_collect_environment_info_cpu_section():
    info = collect_environment_info()

    assert "logical_cores" in info["cpu"]
    assert info["cpu"]["logical_cores"] is None or info["cpu"]["logical_cores"] > 0
    assert "physical_cores" in info["cpu"]


def test_collect_environment_info_memory_section():
    info = collect_environment_info()

    assert "total_gb" in info["memory"]
    assert info["memory"]["total_gb"] > 0
    assert "available_gb" in info["memory"]


def test_package_versions_contains_expected_packages():
    info = collect_environment_info()

    assert "numpy" in info["packages"]
    assert "scipy" in info["packages"]
    assert "jax" in info["packages"]
    assert "jaxlib" in info["packages"]
    assert "mlx" in info["packages"]
    assert "psutil" in info["packages"]


def test_jax_section_has_stable_shape():
    info = collect_environment_info()

    assert "available" in info["jax"]

    if info["jax"]["available"]:
        assert "default_backend" in info["jax"]
        assert "devices" in info["jax"]
        assert "device_details" in info["jax"]
        assert isinstance(info["jax"]["devices"], list)
        assert isinstance(info["jax"]["device_details"], list)
    else:
        assert "error" in info["jax"]


def test_gpu_section_has_stable_shape():
    info = collect_environment_info()

    assert "available" in info["gpu"]
    assert info["gpu"].get("source") == "nvidia-smi"
    if info["gpu"]["available"]:
        assert "devices" in info["gpu"]
        assert isinstance(info["gpu"]["devices"], list)


def test_mlx_section_has_stable_shape():
    info = collect_environment_info()

    assert "available" in info["mlx"]

    if info["mlx"]["available"]:
        assert "default_device" in info["mlx"]
    else:
        assert "error" in info["mlx"]


def test_environment_variables_section():
    info = collect_environment_info()

    expected_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "XLA_FLAGS",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "CUDA_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
    ]

    for var in expected_vars:
        assert var in info["environment_variables"]


def test_save_environment_info(tmp_path):
    path = tmp_path / "env.json"

    info = save_environment_info(path)

    assert path.exists()

    loaded = json.loads(path.read_text())

    assert loaded["python"]["version"] == info["python"]["version"]
    assert "cpu" in loaded
    assert "memory" in loaded
    assert "jax" in loaded
    assert "mlx" in loaded
    assert "torch" not in loaded
