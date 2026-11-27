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
- Without `--duration`, the clip runs from `--start` to the decoded end of the
  master mix, or the longest selected layer for aligned stem-only input.
  Each role shares that timeline, including its rests and the song's decay.
- `--snap-ms 80` sets the maximum melodic correction to the local beat grid.
- `--outro-role guitar` continues the final measured tempo after the last
  detected beat for that role's draft. Use it for an audible part that continues
  through a drum dropout. It snaps detected candidates within the usual tolerance.
  Repeat the flag if several parts continue. Review timing by ear if the outro slows.
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

## Recovering missing notes

Every build and editor chart run checks the four instrument stems independently. Pitch recovery uses beats with no accepted primary detection. Repeated bass attacks may also use available subdivisions. Both passes keep each difficulty's density cap and preserve established notes and holds. The editor also reserves authored transcription windows and honors marked rests.

| Instrument | Evidence for a recovered note |
| --- | --- |
| Bass | pYIN detects a low note on a filtered, octave-raised copy. At least 100 ms of bass energy must support the pitch detection on the original clock. A separate onset pass detects repeated plucks, checking for periodic low-frequency energy just after each attack. |
| Guitar | The QM onset detector checks the stem and a temporary copy with locally balanced volume for quiet attacks. The sound after it must have a repeating waveform or a strong frequency shared across two short windows. A repeated chord can produce several strums. |
| Vocals | Voiced pYIN pitch frames form a phrase. A break in voicing or a sustained dip and return in level can begin a new syllable. Pitch wobble alone keeps the hold intact. Hard also listens for distinct attacks followed by a settled sung pitch and either a fresh rise in volume or a pitch step of at least two semitones. These extra taps fit around established holds. |
| Drums | A rapid rise above the recent amplitude and noise floor identifies a candidate hit. The drum classifier assigns its lane, and the chart uses its detected attack time. |

In `charts/validation.json`, each instrument's `recovery.added` gives additions by difficulty. `recovery.activeGaps` gives source-time ranges where measured stem activity spans at least two seconds without a Normal note or hold. The check groups activity separated by up to 350 ms so repeated short hits can be reviewed. The bass report also retains its previous `bassRecovery` key. Audition flagged passages against the stem and master. Bleed, decay, or an indistinct chord can trigger a review.

## Requirements

- Bun 1.3.12 for the verified local build
- `ffmpeg`
- `sonic-annotator` with BeatRoot, Aubio onset, pYIN, and QM Onset Detector Vamp plugins
- Docker when using `--song`; the command runs `beveradb/audio-separator`

Install and verify the package itself with:

```bash
cd tools/chart-pipeline
bun install --frozen-lockfile
bun test
bun run check
```

The web project links this package as `@bands-battle/chart-pipeline`. Platform
exporters should consume the bundle contract, not import web-game source files.
