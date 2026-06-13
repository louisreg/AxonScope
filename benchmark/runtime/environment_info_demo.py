# benchmark/runtime/environment_info_demo.py

"""
Example showing how to collect and save environment metadata.

Run:
    python benchmark/runtime/environment_info_demo.py
"""

from pathlib import Path
import json

from axonscope.utils.env import (
    collect_environment_info,
    save_environment_info,
)


def main():
    info = collect_environment_info()

    print("AxonScope environment summary")
    print("-----------------------------")

    # --------------------------------------------------
    # OS / Python
    # --------------------------------------------------
    print(f"OS:        {info['os']['platform']}")
    print(f"Machine:   {info['os']['machine']}")
    print(f"Python:    {info['python']['version']}")
    print(f"Venv:      {info['python']['is_venv']}")

    # --------------------------------------------------
    # CPU / Memory
    # --------------------------------------------------
    print(
        f"CPU:       {info['cpu']['physical_cores']} physical cores / "
        f"{info['cpu']['logical_cores']} logical cores"
    )

    if info["cpu"]["frequency_mhz"] is not None:
        print(
            f"CPU freq:  {info['cpu']['frequency_mhz']['current']:.0f} MHz"
        )

    print(f"Memory:    {info['memory']['total_gb']} GB")

    # --------------------------------------------------
    # JAX
    # --------------------------------------------------
    print(
        f"JAX:       available={info['jax']['available']}"
    )

    if info["jax"]["available"]:
        print(
            f"           backend={info['jax']['default_backend']}"
        )
        print(
            f"           devices={len(info['jax']['devices'])}"
        )

    # --------------------------------------------------
    # MLX
    # --------------------------------------------------
    print(
        f"MLX:       available={info['mlx']['available']}"
    )

    if info["mlx"]["available"]:
        print(
            f"           device={info['mlx']['default_device']}"
        )

    # --------------------------------------------------
    # Package versions
    # --------------------------------------------------
    print()
    print("Key package versions:")
    for pkg in ["numpy", "scipy", "jax", "jaxlib", "mlx", "psutil"]:
        print(f"  {pkg:<8} {info['packages'].get(pkg)}")

    # --------------------------------------------------
    # Git info
    # --------------------------------------------------
    print()
    print("Git:")
    print(f"  branch:  {info['git']['branch']}")
    print(f"  commit:  {info['git']['commit']}")
    print(f"  dirty:   {info['git']['is_dirty']}")

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------
    out = Path("benchmark/results/environment_info.json")
    save_environment_info(out)

    print()
    print(f"Full environment information saved to: {out}")

    # --------------------------------------------------
    # Small preview
    # --------------------------------------------------
    preview = {
        "os": info["os"],
        "python": info["python"],
        "cpu": info["cpu"],
        "memory": info["memory"],
        "jax": info["jax"],
        "mlx": info["mlx"],
    }

    print()
    print("JSON preview:")
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
