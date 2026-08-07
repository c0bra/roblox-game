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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rotate an equirectangular panorama by half its width."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = require_file(args.input)
    output = prepare_output(args.output)
    if output.suffix.lower() != ".png":
        raise SystemExit("Use a .png output to keep the seam-edit intermediate lossless")

    width, height = require_equirectangular(source)
    half = width // 2
    ffmpeg = require_tool("ffmpeg")
    run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            (
                f"[0:v]crop={half}:{height}:{half}:0[right];"
                f"[0:v]crop={half}:{height}:0:0[left];"
                "[right][left]hstack=inputs=2[out]"
            ),
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(output),
        ]
    )
    require_equirectangular(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
