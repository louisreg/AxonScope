"""Inspection helpers for generated membrane model code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from axonscope.membranes.compiler import lower_membrane_model_with_sources
from axonscope.membranes.model import MembraneModel, ensure_membrane_model


@dataclass(frozen=True, slots=True)
class GeneratedCodeFileInspection:
    """One file produced by the membrane source compiler."""

    name: str
    path: str
    kind: str
    size_bytes: int
    sha1: str
    text: str | None = None

    def read_text(self) -> str:
        """Read the generated file from disk."""

        return Path(self.path).read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class GeneratedMembraneCodeInspection:
    """Generated-code report for one standalone membrane source file."""

    model_name: str
    source_path: str
    source_hash: str
    function_name: str
    cache_status: str
    cache_reason: str
    cache_key: str
    cache_directory: str
    manifest_path: str
    manifest: Mapping[str, Any]
    files: tuple[GeneratedCodeFileInspection, ...]

    def file(self, name: str) -> GeneratedCodeFileInspection:
        """Return one generated file by name."""

        for generated in self.files:
            if generated.name == name:
                return generated
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class GeneratedMembraneCodeReport:
    """Generated-code report for a public membrane model."""

    model_kind: str
    sources: tuple[GeneratedMembraneCodeInspection, ...]

    def format(self, *, include_text: bool | None = None) -> str:
        """Return a compact text report."""

        return format_generated_membrane_code_report(self, include_text=include_text)

    def print(
        self,
        file: TextIO | None = None,
        *,
        include_text: bool | None = None,
    ) -> None:
        """Print the generated-code report."""

        print(self.format(include_text=include_text), file=file)


def inspect_generated_code(
    model: MembraneModel,
    *,
    include_text: bool = False,
    files: Sequence[str] | None = None,
    max_text_chars: int | None = 6000,
) -> GeneratedMembraneCodeReport:
    """Inspect the generated code artifacts for a membrane model.

    `model` is a public membrane description such as
    `axs.membranes.HodgkinHuxley()`. The compiler may generate missing cache
    artifacts as part of this inspection.
    """

    membrane = ensure_membrane_model(model)
    lowered = lower_membrane_model_with_sources(membrane)
    requested_files = None if files is None else tuple(str(name) for name in files)
    sources = tuple(
        _inspect_source_result(
            result,
            include_text=include_text,
            requested_files=requested_files,
            max_text_chars=max_text_chars,
        )
        for result in lowered.source_results
    )
    return GeneratedMembraneCodeReport(model_kind=membrane.kind, sources=sources)


def format_generated_membrane_code_report(
    report: GeneratedMembraneCodeReport,
    *,
    include_text: bool | None = None,
) -> str:
    """Format a generated-code report as plain text."""

    lines = [
        "AxonScope generated membrane code",
        f"model={report.model_kind}",
        "sources:",
    ]
    for source in report.sources:
        lines.extend(
            [
                f"  {source.model_name}:",
                f"    source={source.source_path}",
                f"    source_hash={source.source_hash}",
                (
                    f"    cache={source.cache_status}, reason={source.cache_reason}, "
                    f"key={source.cache_key}"
                ),
                f"    directory={source.cache_directory}",
                f"    manifest={source.manifest_path}",
                f"    targets={tuple(source.manifest.get('targets', ()))!r}",
                "    files:",
            ]
        )
        for generated in source.files:
            lines.append(
                f"      {generated.name}: kind={generated.kind}, "
                f"size={generated.size_bytes} B, sha1={generated.sha1}"
            )
            if include_text is True or (include_text is None and generated.text is not None):
                text = generated.text if generated.text is not None else generated.read_text()
                lines.append("        text:")
                lines.extend(f"          {line}" for line in text.splitlines())
    return "\n".join(lines)


def _inspect_source_result(
    result: Any,
    *,
    include_text: bool,
    requested_files: tuple[str, ...] | None,
    max_text_chars: int | None,
) -> GeneratedMembraneCodeInspection:
    manifest = _read_manifest(result.cache.manifest_path)
    files = tuple(
        _inspect_generated_file(
            path,
            include_text=include_text and _include_file(path.name, requested_files),
            max_text_chars=max_text_chars,
        )
        for path in result.cache.generated_files
        if requested_files is None or path.name in requested_files
    )
    return GeneratedMembraneCodeInspection(
        model_name=str(result.model.name),
        source_path=str(result.source_path),
        source_hash=str(result.source_hash),
        function_name=str(result.function_name),
        cache_status="hit" if result.cache.cache_hit else "miss",
        cache_reason=str(result.cache.cache_reason),
        cache_key=str(result.cache.key),
        cache_directory=str(result.cache.directory),
        manifest_path=str(result.cache.manifest_path),
        manifest=manifest,
        files=files,
    )


def _inspect_generated_file(
    path: Path,
    *,
    include_text: bool,
    max_text_chars: int | None,
) -> GeneratedCodeFileInspection:
    data = path.read_bytes()
    text = None
    if include_text:
        decoded = data.decode("utf-8")
        text = decoded if max_text_chars is None else decoded[:max_text_chars]
    return GeneratedCodeFileInspection(
        name=path.name,
        path=str(path),
        kind=_generated_file_kind(path.name),
        size_bytes=len(data),
        sha1=hashlib.sha1(data).hexdigest(),
        text=text,
    )


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _include_file(name: str, requested_files: tuple[str, ...] | None) -> bool:
    return requested_files is None or name in requested_files


def _generated_file_kind(name: str) -> str:
    if name == "source_snapshot.py":
        return "source"
    if name in {"graph.json", "optimized_graph.json"}:
        return "graph"
    if name == "jax_model.py":
        return "jax"
    if name == "numpy_model.py":
        return "numpy"
    return "artifact"


__all__ = [
    "GeneratedCodeFileInspection",
    "GeneratedMembraneCodeInspection",
    "GeneratedMembraneCodeReport",
    "format_generated_membrane_code_report",
    "inspect_generated_code",
]
