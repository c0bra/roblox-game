# Heaven's Edge — HTML5 Battle

A portrait-first rhythm boss battle built with strict TypeScript, Vite, Canvas 2D, Web Audio, and Babylon.js.

## Run

```bash
bun install
bun run dev
```

Open the printed local URL, choose an instrument and difficulty, and press **Enter the breach**. Easy is selected by default. On desktop, lanes also respond to `D/F/K` or `1/2/3`.

## Commands

```bash
bun test       # note judgment and attack-window rules
bun run check  # Biome and strict TypeScript
bun run build  # production bundle in dist/
bun run assets # rebuild charts and 90-second stem pairs from ../../audio/Heavens_Edge
bun run pipeline --stems /path/to/stems --output /path/to/level
```

Append `?qa=1` for the 12-second browser-QA version of the encounter. The normal route plays the full 90-second level.

## Level data

- Drums: 201 easy / 380 medium / 411 hard notes
- Vocals: 135 notes
- Guitar: 178 notes
- Bass: 94 notes
- Perfect / Great / Good windows: ±60 ms / ±110 ms / ±170 ms
- Four boss attack phrases; a failed phrase costs 28 ward health

The selected instrument is a separate Web Audio channel. Misses duck that channel for 350 ms and add a short dissonant cue while the other stems continue.

## Reusable song-to-chart pipeline

The chart compiler uses one timing representation end to end: absolute seconds from the
source audio. Melodic notes snap to a piecewise 16th-note grid between detected beats.
Drums use Aubio onsets from the exact playable stem and preserve those audible transient
timestamps; the beat grid only groups them for difficulty density. The pipeline does not
convert through a constant-BPM MIDI timeline or substitute legacy clustered drum events.

If your music service already supplies stems, put audio files containing `drum`,
`vocal`, `bass`, and `guitar` in their names into one directory. A four-stem Demucs
export is also accepted; `other` is used for guitar when no guitar-specific stem exists.

```bash
bun run pipeline \
  --stems "/absolute/path/to/stems" \
  --output "/absolute/path/to/generated-level" \
  --start 0 \
  --duration 90
```

To start from one mixed song, omit `--stems` and pass `--song`. This runs the same
`beveradb/audio-separator` Docker image and `htdemucs.yaml` model already used by the
repository, then analyzes the resulting stems.

```bash
bun run pipeline \
  --song "/absolute/path/to/song.wav" \
  --output "/absolute/path/to/generated-level" \
  --start 0 \
  --duration 90
```

Requirements for automatic analysis:

- Docker for optional stem separation.
- `sonic-annotator` with BeatRoot, Aubio onset, and pYIN plugins.
- Bun and this workspace's installed dependencies.

The command writes `charts/{instrument}-{easy|medium|hard}.json` plus
`charts/validation.json`. Easy, medium, and hard cap density at one, two, and four
notes per beat respectively; sparse parts are not padded with invented notes, so a
sparse bass line can legitimately have the same count at multiple difficulties.

The validation report records rejected off-grid events, duplicate grid events,
distance to the local grid, and note counts. For melodic instruments that distance is
the timestamp correction; for drums it is only the grouping distance because the onset
timestamp is preserved. Treat a large rejection count or distance near the `--snap-ms`
limit as a request for human listening, a better beat grid, or a better transcription
source. For polyphonic guitar, a Basic Pitch MIDI transcription usually gives better
pitch segmentation than pYIN; the Heaven's Edge-specific `bun run assets` adapter
already uses its checked-in Basic Pitch MIDI.

## Timing guarantees

- Beat tracking, onset detection, transcription, chart output, audio cropping, and
  browser playback share the same zero point and seconds unit.
- Drum notes retain their detected transient timestamps instead of being moved onto a
  theoretical subdivision.
- Melodic quantization follows the detected beat timestamps locally, so live tempo
  drift is preserved instead of flattened to one BPM.
- Events farther than 80 ms from the nearest grid point are rejected by default. Change
  this with `--snap-ms`; they are never silently emitted at their raw timestamps.
- The Heaven's Edge audio builder uses `atrim` plus `asetpts=PTS-STARTPTS` for exact,
  matching stem/backing crop origins.
- The browser judges notes against `AudioContext.currentTime`, not animation-frame or
  wall-clock time.
