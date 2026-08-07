from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def require_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"Required tool not found on PATH: {name}")
    return resolved


def require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"Input file not found: {resolved}")
    return resolved


def prepare_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def probe_dimensions(path: Path) -> tuple[int, int]:
    ffmpeg = require_tool("ffmpeg")
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(path),
            "-vf",
            "showinfo",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise SystemExit(process.stderr.strip() or f"Could not decode image: {path}")
    match = re.search(r"\bs:(\d+)x(\d+)\b", process.stderr)
    if match is None:
        raise SystemExit(f"Could not determine image dimensions: {path}")
    return int(match.group(1)), int(match.group(2))


def require_equirectangular(path: Path) -> tuple[int, int]:
    width, height = probe_dimensions(path)
    if width != height * 2:
        raise SystemExit(
            f"Expected strict 2:1 equirectangular dimensions, got {width}x{height}: {path}"
        )
    if width % 2 != 0:
        raise SystemExit(f"Panorama width must be even, got {width}: {path}")
    return width, height


def run(command: list[str], *, input_bytes: bytes | bytearray | None = None) -> bytes:
    process = subprocess.run(
        command,
        check=True,
        input=input_bytes,
        capture_output=True,
    )
    return process.stdout
