#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _panorama import (
    prepare_output,
    require_equirectangular,
    require_file,
    require_tool,
    run,
)


def smoothstep(value: float) -> float:
    return value * value * (3 - 2 * value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply a narrow symmetric feather to panorama longitude edges."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pixels", type=int, default=64)
    args = parser.parse_args()

    source = require_file(args.input)
    output = prepare_output(args.output)
    if output.suffix.lower() != ".png":
        raise SystemExit("Use a .png output to keep the feathered master lossless")

    width, height = require_equirectangular(source)
    if args.pixels < 2 or args.pixels > width // 8:
        raise SystemExit(f"--pixels must be between 2 and {width // 8}")

    ffmpeg = require_tool("ffmpeg")
    decoded = run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            "format=rgb24",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    expected = width * height * 3
    if len(decoded) != expected:
        raise SystemExit(
            f"Unexpected decoded image size: expected {expected}, got {len(decoded)}"
        )

    pixels = bytearray(decoded)
    row_bytes = width * 3
    for row in range(height):
        row_start = row * row_bytes
        for distance in range(args.pixels):
            progress = distance / (args.pixels - 1)
            weight = 0.5 * (1 - smoothstep(progress))
            left_index = row_start + distance * 3
            right_index = row_start + (width - 1 - distance) * 3
            for channel in range(3):
                left = pixels[left_index + channel]
                right = pixels[right_index + channel]
                pixels[left_index + channel] = round(
                    left * (1 - weight) + right * weight
                )
                pixels[right_index + channel] = round(
                    right * (1 - weight) + left * weight
                )

    run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            str(output),
        ],
        input_bytes=pixels,
    )
    require_equirectangular(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
