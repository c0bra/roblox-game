#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from _panorama import prepare_output, require_equirectangular, require_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upscale one lossless equirectangular image with Real-ESRGAN."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", default="realesrgan-x4plus")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()

    source = require_file(args.input)
    output = prepare_output(args.output)
    binary = require_file(args.binary)
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"Real-ESRGAN binary is not executable: {binary}")
    if source.suffix.lower() != ".png" or output.suffix.lower() != ".png":
        raise SystemExit("Use PNG input and output for the reconstruction stage")
    if args.scale < 1 or args.tile < 0:
        raise SystemExit("--scale must be positive and --tile cannot be negative")

    input_width, input_height = require_equirectangular(source)
    subprocess.run(
        [
            str(binary),
            "-i",
            str(source),
            "-o",
            str(output),
            "-n",
            args.model,
            "-s",
            str(args.scale),
            "-t",
            str(args.tile),
            "-f",
            "png",
        ],
        check=True,
    )
    output_width, output_height = require_equirectangular(output)
    expected = (input_width * args.scale, input_height * args.scale)
    if (output_width, output_height) != expected:
        raise SystemExit(
            f"Unexpected upscale dimensions: expected {expected[0]}x{expected[1]}, "
            f"got {output_width}x{output_height}"
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
