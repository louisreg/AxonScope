"""Stream live logs from the latest Kaggle kernel run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="Kernel ref in owner/slug format.")
    parser.add_argument(
        "--wait-for-stream",
        type=int,
        default=300,
        help="Seconds Kaggle should wait for the live stream URL while queued.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw stream payloads instead of formatted log data.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional file where formatted output is appended.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Stop streaming after this many seconds. Defaults to streaming until Kaggle ends.",
    )
    args = parser.parse_args(argv)

    owner, slug = parse_kernel_ref(args.kernel)
    api = KaggleApi()
    api.authenticate()

    request = ApiGetKernelSessionLogsStreamRequest()
    request.user_name = owner
    request.kernel_slug = slug
    request.wait_for_logs_url_seconds = int(args.wait_for_stream)

    with api.build_kaggle_client() as kaggle:
        response = kaggle.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
        response.raise_for_status()
        stream_response(
            response,
            raw=bool(args.raw),
            save_path=args.save,
            max_seconds=args.max_seconds,
        )
    return 0


def parse_kernel_ref(kernel: str) -> tuple[str, str]:
    parts = [part for part in kernel.split("/") if part]
    if len(parts) < 2:
        raise ValueError("kernel must be in owner/slug format.")
    return parts[0], parts[1]


def stream_response(
    response,
    *,
    raw: bool,
    save_path: Path | None,
    max_seconds: float | None,
) -> None:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        text = response.text
        output = text if raw else format_persisted_log(text)
        emit(output, save_path=save_path)
        return

    deadline = None if max_seconds is None else time.monotonic() + float(max_seconds)
    for raw_line in response.iter_lines(decode_unicode=True):
        if deadline is not None and time.monotonic() >= deadline:
            break
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            continue
        payload = line[5:].strip() if line.startswith("data:") else line
        if payload == "END_OF_LOG":
            break
        output = payload + "\n" if raw else format_stream_payload(payload)
        emit(output, save_path=save_path)


def format_stream_payload(payload: str) -> str:
    try:
        entry = json.loads(payload)
    except json.JSONDecodeError:
        return payload + "\n"
    if not isinstance(entry, dict):
        return payload + "\n"
    data = str(entry.get("data", ""))
    if data:
        return data if data.endswith("\n") else data + "\n"
    message = entry.get("message")
    if message is not None:
        return str(message) + "\n"
    return payload + "\n"


def format_persisted_log(raw_output: str) -> str:
    try:
        entries = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output
    if not isinstance(entries, list):
        return raw_output
    chunks = []
    for entry in entries:
        if not isinstance(entry, dict):
            return raw_output
        chunks.append(str(entry.get("data", "")))
    return "".join(chunks)


def emit(text: str, *, save_path: Path | None) -> None:
    print(text, end="", flush=True)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("a", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
