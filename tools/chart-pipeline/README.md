# Chart pipeline

This root-owned TypeScript tool turns either one mixed song or a directory of stems
into a platform-neutral rhythm-game bundle. It is independent of the web game, so the
same output can be adapted for Roblox, native mobile, or another renderer.

The [`SYSTEMS_MAP.md`](../../SYSTEMS_MAP.md#81-song-chart--encounter-authoring)
Content Authoring entry defines this toolchain's design responsibility and the
required follow-on specification. This package is the existing foundation for that
offline production system, not a Roblox runtime subsystem.

## Commands

Run these from the repository root:

```bash
./chart build --song /path/to/song.mp3
./chart build --stems /path/to/stems
./chart validate /path/to/build/song
```

Without `--output`, the bundle is written to `build/<input-name>`. Pass
`--output /path/to/build/song` to override that location.

Optional build flags:

- `--start 60` starts the playable clip at 60 seconds in the source.
- `--duration 90` limits the playable clip to 90 seconds.
- `--snap-ms 80` sets the maximum melodic correction to the local beat grid.
- `--model htdemucs.yaml` selects the audio-separator model.

Stem directories need files whose names identify drums, vocals, guitar, and bass.
A four-stem Demucs export is supported: `other` is selected as guitar when no
guitar-specific stem exists.

## Output contract

```text
song-build/
├── manifest.json
├── audio/stems/
│   ├── drums.wav
│   ├── vocals.wav
│   ├── guitar.wav
│   └── bass.wav
└── charts/
    ├── drums-{easy,medium,hard}.json
    ├── vocals-{easy,medium,hard}.json
    ├── guitar-{easy,medium,hard}.json
    ├── bass-{easy,medium,hard}.json
    └── validation.json
```

`manifest.json` is the entry point. Its `schemaVersion` identifies the contract,
all paths are relative to the bundle, and all timing values use seconds. The copied
stems retain source timing; a consumer starts playback at `timing.sourceOffset` while
chart note times start at zero.

Drums retain detected transient timestamps. Beat tracking controls density but does
not move a kick or snare to a synthetic beat. Melodic events snap to the local
piecewise beat grid and preserve detected durations as sustain notes when possible.

The validation report includes note counts, rejected off-grid events, duplicate grid
events, and grid-error statistics. Generation is intentionally deterministic, but a
new song still needs a listening pass before release.

## Requirements

- Bun
- `ffmpeg`
- `sonic-annotator` with BeatRoot, Aubio onset, and pYIN Vamp plugins
- Docker when using `--song`; the command runs `beveradb/audio-separator`

Install and verify the package itself with:

```bash
cd tools/chart-pipeline
bun install
bun test
bun run check
```

The web project links this package as `@bands-battle/chart-pipeline`. Platform
exporters should consume the bundle contract, not import web-game source files.
