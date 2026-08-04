# Bands Battle — HTML5 Battle

A portrait-first rhythm boss battle built with strict TypeScript, Vite, Canvas 2D, Web Audio, and Babylon.js.

## Run

```bash
bun install
bun run dev
```

Open the printed local URL, choose a song, instrument, and difficulty, then press **Enter the breach**. Heaven's Edge and Easy are selected by default. Tap short notes and keep the lane pressed for the full length of melodic ribbon notes. On desktop, lanes also respond to `D/F/K` or `1/2/3`.

## Commands

```bash
bun test       # note judgment and attack-window rules
bun run check  # Biome and strict TypeScript
bun run build  # production bundle in dist/
bun run assets # rebuild charts and 90-second stem pairs from ../../audio/Heavens_Edge
bun run level:import -- /path/to/bundle level-id "Display Title"
../../chart build --stems /path/to/stems --output /path/to/level
```

Append `?qa=1` for a deterministic 12-second browser-QA version of the selected encounter. The normal route plays the selected level's full authored duration.

## Add a song to the web game

Build a platform-neutral song bundle from an MP3 or WAV at the project root, then import that bundle into the web app:

```bash
./chart build --song "/absolute/path/to/song.mp3" --output "/absolute/path/to/song-bundle"
cd roblox/web
bun run level:import -- "/absolute/path/to/song-bundle" song-id "Song Title"
```

The importer validates the bundle, copies its 12 instrument/difficulty charts, creates a stem and backing M4A for each instrument, and adds the song to `src/data/levels.json`. It publishes transactionally, so a failed audio encode does not leave a partial level behind. Level IDs use lowercase kebab case, such as `blackened-crown`, and an existing ID is never overwritten.

## Heaven's Edge level data

- Drums: 201 easy / 380 medium / 411 hard notes
- Vocals: 113 easy / 135 medium / 135 hard notes; 44 / 59 / 59 sustains
- Guitar: 119 easy / 178 medium / 207 hard notes; 27 / 16 / 15 sustains
- Bass: 94 notes on each difficulty; 71 sustains
- Perfect / Great / Good windows: ±60 ms / ±110 ms / ±170 ms
- Four boss attack phrases; a failed phrase costs 28 ward health

The selected instrument is a separate Web Audio channel. Misses duck that channel for 350 ms and add a short dissonant cue while the other stems continue.

## Reusable song-to-chart pipeline

The chart compiler uses one timing representation end to end: absolute seconds from the
source audio. Melodic notes snap to a piecewise 16th-note grid between detected beats.
Drums use Aubio onsets from the exact playable stem and preserve those audible transient
timestamps; the beat grid only groups them for difficulty density. The pipeline does not
convert through a constant-BPM MIDI timeline or substitute legacy clustered drum events.
Vocals, guitar, and bass preserve pYIN or MIDI note duration. Notes at least 350 ms long
become holdable sustain ribbons and are clipped before the next playable note; drum
transients always remain taps.

If your music service already supplies stems, put audio files containing `drum`,
`vocal`, `bass`, and `guitar` in their names into one directory. A four-stem Demucs
export is also accepted; `other` is used for guitar when no guitar-specific stem exists.

```bash
../../chart build \
  --stems "/absolute/path/to/stems" \
  --output "/absolute/path/to/generated-level" \
  --start 0 \
  --duration 90
```

To start from one mixed song, omit `--stems` and pass `--song`. This runs the same
`beveradb/audio-separator` Docker image and `htdemucs.yaml` model already used by the
repository, then analyzes the resulting stems.

```bash
../../chart build \
  --song "/absolute/path/to/song.wav" \
  --output "/absolute/path/to/generated-level" \
  --start 0 \
  --duration 90
```

Requirements for automatic analysis:

- Docker for optional stem separation.
- `sonic-annotator` with BeatRoot, Aubio onset, and pYIN plugins.
- Bun and this workspace's installed dependencies.

The command writes a versioned `manifest.json`, portable `audio/stems`,
`charts/{instrument}-{easy|medium|hard}.json`, and `charts/validation.json`.
Easy, medium, and hard cap density at one, two, and four
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
- Melodic note durations come from the transcription source, remain in seconds, and
  never overlap the next playable note after difficulty filtering.
- Events farther than 80 ms from the nearest grid point are rejected by default. Change
  this with `--snap-ms`; they are never silently emitted at their raw timestamps.
- The Heaven's Edge audio builder uses `atrim` plus `asetpts=PTS-STARTPTS` for exact,
  matching stem/backing crop origins.
- The browser judges notes against `AudioContext.currentTime`, not animation-frame or
  wall-clock time.
