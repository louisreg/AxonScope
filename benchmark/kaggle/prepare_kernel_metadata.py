"""Write Kaggle kernel metadata for AxonScope solver benchmark scripts."""

from __future__ import annotations

import argparse
import json
import pprint
from pathlib import Path
from typing import Sequence


SCRIPT_TEMPLATE = Path(__file__).resolve().parent / "axonscope_solver_benchmarks.py"
GENERATED_CODE_FILE = "axonscope_solver_benchmarks_generated.py"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Kaggle username slug.")
    parser.add_argument(
        "--slug",
        default="axonscope-double-cable-solver-benchmarks",
        help="Kernel slug under the Kaggle username.",
    )
    parser.add_argument(
        "--title",
        default="AxonScope Double-Cable Solver Benchmarks",
    )
    parser.add_argument(
        "--machine-shape",
        default="NvidiaTeslaP100",
        help="Kaggle accelerator/machine shape, e.g. NvidiaTeslaT4 or NvidiaTeslaP100.",
    )
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
            "both",
        ),
        help="Benchmark suite to run inside the Kaggle kernel.",
    )
    parser.add_argument(
        "--branch",
        default="bench-colab",
        help="Git branch cloned by the Kaggle kernel.",
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/louisreg/AxonScope.git",
        help="Repository cloned by the Kaggle kernel.",
    )
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail the Kaggle run if JAX does not select a GPU backend.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder where kernel-metadata.json will be written.",
    )
    parser.add_argument(
        "--code-file",
        default=GENERATED_CODE_FILE,
        help="Generated self-contained script referenced by kernel-metadata.json.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=SCRIPT_TEMPLATE,
        help="Source script template used to generate the uploaded Kaggle script.",
    )
    args = parser.parse_args(argv)
    args.path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": f"{args.username}/{args.slug}",
        "title": args.title,
        "code_file": args.code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": args.machine_shape,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    config = {
        "repo_url": args.repo_url,
        "branch": args.branch,
        "benchmark": args.benchmark,
        "require_gpu": args.require_gpu,
    }

    metadata_output = args.path / "kernel-metadata.json"
    config_output = args.path / "kaggle_config.json"
    code_output = args.path / args.code_file
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config_output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_embedded_script(template=args.template, output=code_output, config=config)
    print(metadata_output)
    print(config_output)
    print(code_output)


def write_embedded_script(*, template: Path, output: Path, config: dict[str, object]) -> None:
    source = template.read_text(encoding="utf-8")
    needle = "EMBEDDED_CONFIG: dict[str, object] = {}\n"
    embedded = pprint.pformat(config, sort_dicts=True, width=100)
    replacement = f"EMBEDDED_CONFIG: dict[str, object] = {embedded}\n"
    if needle not in source:
        raise ValueError(f"Could not find embedded config marker in {template}")
    output.write_text(source.replace(needle, replacement, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
