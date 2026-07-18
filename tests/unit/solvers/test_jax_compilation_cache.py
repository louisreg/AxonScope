from __future__ import annotations

from axonscope.runtime.jax import compilation_cache


class _ConfigRecorder:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, name: str, value: object) -> None:
        self.values[name] = value


def test_jax_compilation_cache_defaults_under_axonscope_cache(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AXONSCOPE_JAX_COMPILATION_CACHE", raising=False)

    policy = compilation_cache.jax_compilation_cache_policy()

    assert policy.enabled is True
    assert policy.directory == tmp_path / ".axonscope_cache" / "runtime" / "jax" / "xla"
    assert policy.min_compile_time_s == 0.5
    assert policy.min_entry_size_bytes == 0
    assert policy.max_size_bytes == 2 * 1024**3
    assert policy.xla_caches == "xla_gpu_per_fusion_autotune_cache_dir"


def test_jax_compilation_cache_can_be_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AXONSCOPE_JAX_COMPILATION_CACHE", "off")
    config = _ConfigRecorder()

    policy = compilation_cache.configure_jax_compilation_cache(jax_config=config)

    assert policy.enabled is False
    assert policy.directory is None
    assert config.values == {"jax_enable_compilation_cache": False}


def test_jax_compilation_cache_honors_policy_overrides(monkeypatch, tmp_path) -> None:
    directory = tmp_path / "shared-xla"
    monkeypatch.setenv("AXONSCOPE_JAX_COMPILATION_CACHE", str(directory))
    monkeypatch.setenv("AXONSCOPE_JAX_CACHE_MIN_COMPILE_TIME_S", "0")
    monkeypatch.setenv("AXONSCOPE_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    monkeypatch.setenv("AXONSCOPE_JAX_CACHE_MAX_SIZE_BYTES", "4096")
    monkeypatch.setenv("AXONSCOPE_JAX_PERSISTENT_XLA_CACHES", "all")
    config = _ConfigRecorder()

    policy = compilation_cache.configure_jax_compilation_cache(jax_config=config)

    assert policy.directory == directory
    assert directory.is_dir()
    assert config.values == {
        "jax_enable_compilation_cache": True,
        "jax_compilation_cache_dir": str(directory),
        "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": -1,
        "jax_compilation_cache_max_size": 4096,
        "jax_persistent_cache_enable_xla_caches": "all",
    }


def test_jax_compilation_cache_rejects_unknown_xla_cache_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AXONSCOPE_JAX_PERSISTENT_XLA_CACHES", "mystery")

    try:
        compilation_cache.jax_compilation_cache_policy()
    except ValueError as exc:
        assert "AXONSCOPE_JAX_PERSISTENT_XLA_CACHES" in str(exc)
    else:
        raise AssertionError("unknown XLA cache mode should be rejected")
