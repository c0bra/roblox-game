#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _panorama import require_equirectangular, require_file, require_tool, run


def decode_edge_strip(
    path: Path,
    *,
    x: int,
    strip: int,
    height: int,
) -> bytes:
    ffmpeg = require_tool("ffmpeg")
    payload = run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"crop={strip}:{height}:{x}:0,format=gray",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    expected = strip * height
    if len(payload) != expected:
        raise SystemExit(
            f"Unexpected decoded strip size: expected {expected}, got {len(payload)}"
        )
    return payload


def mean_absolute_error(pairs: list[tuple[int, int]]) -> float:
    return sum(abs(left - right) for left, right in pairs) / len(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate 2:1 dimensions and quantify panorama longitude continuity."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--strip", type=int, default=32)
    parser.add_argument("--max-boundary-mae", type=float)
    parser.add_argument("--max-strip-mae", type=float)
    args = parser.parse_args()

    image = require_file(args.image)
    width, height = require_equirectangular(image)
    if args.strip < 1 or args.strip > width // 4:
        raise SystemExit(f"--strip must be between 1 and {width // 4}")

    left = decode_edge_strip(image, x=0, strip=args.strip, height=height)
    right = decode_edge_strip(
        image,
        x=width - args.strip,
        strip=args.strip,
        height=height,
    )

    boundary_pairs = [
        (left[row * args.strip], right[row * args.strip + args.strip - 1])
        for row in range(height)
    ]
    strip_pairs = [
        (
            left[row * args.strip + distance],
            right[row * args.strip + args.strip - 1 - distance],
        )
        for row in range(height)
        for distance in range(args.strip)
    ]
    boundary_mae = mean_absolute_error(boundary_pairs)
    strip_mae = mean_absolute_error(strip_pairs)
    failures: list[str] = []

    if (
        args.max_boundary_mae is not None
        and boundary_mae > args.max_boundary_mae
    ):
        failures.append(
            f"boundary_mae {boundary_mae:.4f} exceeds {args.max_boundary_mae:.4f}"
        )
    if args.max_strip_mae is not None and strip_mae > args.max_strip_mae:
        failures.append(
            f"strip_mae {strip_mae:.4f} exceeds {args.max_strip_mae:.4f}"
        )

    print(
        json.dumps(
            {
                "image": str(image),
                "width": width,
                "height": height,
                "strict_2_to_1": True,
                "strip_pixels": args.strip,
                "boundary_mae": round(boundary_mae, 4),
                "strip_mae": round(strip_mae, 4),
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
