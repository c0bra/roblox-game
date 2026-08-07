#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _panorama import (
    prepare_output,
    probe_dimensions,
    require_equirectangular,
    require_file,
    require_tool,
    run,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_webp(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    quality: int,
) -> None:
    ffmpeg = require_tool("ffmpeg")
    run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int",
            "-frames:v",
            "1",
            "-c:v",
            "libwebp",
            "-preset",
            "picture",
            "-quality",
            str(quality),
            "-compression_level",
            "6",
            str(output),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export exact 4K and 8K WebP panorama variants."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "output_prefix",
        type=Path,
        help="Path prefix; -4k.webp and -8k.webp are appended",
    )
    parser.add_argument("--quality-4k", type=int, default=92)
    parser.add_argument("--quality-8k", type=int, default=90)
    parser.add_argument("--allow-small-source", action="store_true")
    args = parser.parse_args()

    source = require_file(args.input)
    source_width, _ = require_equirectangular(source)
    if source_width < 4096 and not args.allow_small_source:
        raise SystemExit(
            "Source is below 4096 pixels wide. Reconstruct/upscale it first, or pass "
            "--allow-small-source only when intentional."
        )
    for value in (args.quality_4k, args.quality_8k):
        if value < 1 or value > 100:
            raise SystemExit("WebP quality values must be between 1 and 100")

    prefix = args.output_prefix.expanduser().resolve()
    output_4k = prepare_output(Path(f"{prefix}-4k.webp"))
    output_8k = prepare_output(Path(f"{prefix}-8k.webp"))
    export_webp(source, output_4k, width=4096, height=2048, quality=args.quality_4k)
    export_webp(source, output_8k, width=8192, height=4096, quality=args.quality_8k)

    results = []
    for path, expected in (
        (output_4k, (4096, 2048)),
        (output_8k, (8192, 4096)),
    ):
        dimensions = probe_dimensions(path)
        if dimensions != expected:
            raise SystemExit(
                f"Unexpected export dimensions for {path}: {dimensions[0]}x{dimensions[1]}"
            )
        results.append(
            {
                "path": str(path),
                "width": dimensions[0],
                "height": dimensions[1],
                "sha256": sha256(path),
            }
        )

    print(json.dumps({"source": str(source), "outputs": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
