# Roblox Bands Battle

Roblox Bands Battle is an early-stage rhythm boss-battle game concept plus a
working offline audio-to-chart toolchain.

The game vision is a combat-oriented Guitar Hero for Roblox: players perform as a
band, hit rhythm notes for individual instruments, fight bosses, earn loot drops,
and upgrade gear between battles. The style target is dark high-energy K-pop with
EDM, trap or 808 bass, soaring vocals, distorted guitar, cinematic supernatural
bosses, and mobile-friendly controls.

Current reality: this repo contains Python audio-processing tools and project
notes. It does not yet contain Roblox/Luau game code, Rojo config, or `.rbxl`
place files.

## New developer quick start

Read these first:

1. `README.md`, this file, for repo structure and the current workflow.
2. `GAME_VISION.md`, the product north star, audience, tone, and scope.
3. `GAME_DESIGN.md`, the canonical player-facing mechanics and behavior.
4. `SYSTEMS_MAP.md`, canonical system ownership, dependencies, boundaries, and
   detailed-spec sequence.
5. `SYSTEM_SPECIFICATIONS_TRACKER.md`, links and completion status for all 13
   approved detailed design specifications.
6. `ART_DIRECTION.md`, canonical look and feel for characters, environments, UI,
   materials, lighting, VFX, and visual production.
7. `audio/Heavens_Edge/stems/spec.md`, lane-chart pipeline spec.
8. `audio/Heavens_Edge/stems/sonic-annotator.md`, command cookbook for BeatRoot,
   pYIN, Aubio, and CREPE experiments.
9. `audio/Heavens_Edge/stems/AGENTS.md`, local notes for the main audio workflow
   subtree.

If you are here to build the Roblox game layer, start with `GAME_DESIGN.md` and
`SYSTEMS_MAP.md`, then use `SYSTEM_SPECIFICATIONS_TRACKER.md` to enter the
completed detailed-spec set before defining technical architecture.

If you are here to generate charts from songs, start with `Current audio pipeline`.

## Repo map

```text
roblox-bands-battle/
|-- README.md
|-- chart                         # repository-root chart pipeline command
|-- tools/chart-pipeline/         # platform-neutral compiler and bundle format
|-- GAME_VISION.md               # product north star and scope authority
|-- GAME_DESIGN.md
|-- SYSTEMS_MAP.md                # system ownership and spec sequence
|-- SYSTEM_SPECIFICATIONS_TRACKER.md # links to all 13 approved system specs
|-- CONTENT_AUTHORING.md          # reconciled offline content contract
|-- PLAYER_DATA.md                # durable profile and transaction contract
|-- ART_DIRECTION.md              # canonical visual style and asset guidance
|-- trello_notes.md                 # raw brainstorming notes
|-- chatgpt_chat.md                 # raw core-loop discussion
|-- cool-dangerous-electric-guitar-roblox-game-asset.png
|-- audio/
|   |-- Heavens_Edge/
|   |   |-- stem2midi.py             # standalone Basic Pitch + madmom converter
|   |   |-- beats_dump.py            # madmom beat/downbeat inspector
|   |   |-- vocals.py                # old hardcoded MIDI simplifier
|   |   `-- stems/
|   |       |-- kick_snare_hat_separator.py
|   |       |-- lane_assign_melody.py
|   |       |-- pyin_csv_to_midi_quant.py
|   |       |-- pyin_bass_shift_to_midi.py
|   |       |-- pyin_to_lanes.py
|   |       |-- spec.md
|   |       `-- sonic-annotator.md
|   `-- Blackened Crown/
|       |-- README.md                # alternate song-specific recipe
|       |-- stems_to_midi.py
|       `-- *.sh                     # older shell conversion helpers
```

This repo is asset-heavy. Many `.mp3`, `.wav`, `.mid`, and `.csv` files are
generated examples or intermediate outputs, not source of truth.

## What exists today

### 1. Audio-to-chart pipeline

The working part of this repo takes a song and turns it into timing data:

1. Split a full song into stems with Demucs / `audio-separator`.
2. Use the drum stem to detect beats and build a timing grid.
3. Convert vocals, bass, guitar, synth, or drums into MIDI or note/event CSVs.
4. Quantize notes to the beat grid.
5. Map melodic notes into 2 or 3 game lanes.

The intended chart output is CSV data shaped like:

```csv
time_s,lane,pitch,dur_s
7.836735,1,62,0.348299
```

### 2. Game design notes

The design notes define a Roblox rhythm/boss-battle game:

- 75 to 90 second battles.
- Each player plays an instrument part.
- Notes or prompts are simple enough for mobile.
- Accurate play fills a hype meter and drives damage or score.
- Misses can duck the player audio, trigger flub animations, reduce crowd hype,
  or lower health.
- Bosses attack during the song and can knock players out of bonus positions.
- Players earn item drops after boss battles and upgrade instruments or gear.
- Timed bosses and dangling story clues can support retention.

See `GAME_DESIGN.md` for the approved player-facing rules and `SYSTEMS_MAP.md`
for the systems that own and implement those rules.

## Implementation status

The repository now includes two separate gameplay tracks:

- `roblox/web/` contains the playable browser demo, including the preserved
  Classic rhythm-highway mode and the opt-in Arena V2 boss-battle vertical slice.
- The native Roblox implementation still has not been started.

The browser demo has its own Bun/Vite package, TypeScript runtime, automated
tests, responsive UI, encounter data, and browser-ready Arena assets. See
`roblox/web/README.md` and run it from that directory.

The following native Roblox pieces do not exist yet:

- no `.lua` or `.luau` scripts
- no Rojo project file
- no Roblox place file
- no native Roblox UI implementation
- no server/client networking layer
- no DataStore persistence
- no native Roblox boss battle runtime
- no chart loader in Roblox

The audio pipeline is wired into a playable web prototype, but not into a Roblox
place. The next native-platform milestone is a minimal Roblox prototype that can
load one exported chart and play one 60 to 90 second battle.

## Recommended first Roblox milestone

Build the smallest playable vertical slice:

1. Add a Roblox project structure, likely Rojo-based.
2. Import one song and one generated lane chart from `audio/Heavens_Edge/stems/`.
3. Render a client-side 3 or 4 lane rhythm UI with `ScreenGui`.
4. Bind inputs across keyboard, gamepad, and touch.
5. Play the song locally and judge notes against chart timestamps.
6. Show hit, miss, combo, score, and hype meter feedback.
7. Add one boss with scripted chart events.
8. End the round with a simple loot/drop screen.

Roblox implementation guidance from official docs and common rhythm-game patterns:

- Use a client UI lane first, not world physics. `ScreenGui` is the practical
  starting point for mobile and desktop.
- Use Roblox's Input Action System or `ContextActionService` style abstractions so
  keyboard, touch, and gamepad share one action model.
- Keep note movement and immediate hit feedback responsive on the client.
- Keep authoritative rewards, inventory, unlocks, and boss outcomes on the server.
- Use RemoteEvents for discrete gameplay messages. Avoid flooding remotes every
  frame.
- Use DataStoreService server-side for persistent loot, instruments, unlocks, and
  progression.
- Test on low-end mobile early. This game is only viable if the note UI remains
  readable and responsive on a phone.

## Current audio pipeline

### Reusable pipeline

The maintained entry point now lives at the repository root and produces a neutral,
versioned bundle rather than writing directly into one game client:

```bash
./chart build --song "/absolute/path/to/song.mp3"
./chart build --stems "/absolute/path/to/stems"
./chart validate "/absolute/path/to/build"
```

The default output is `build/<input-name>`; use `--output` to override it.

Use `--song` to run Docker stem separation automatically, or `--stems` when a music
service already supplied drums, vocals, guitar, and bass. Both routes create four
portable audio stems, easy/medium/hard charts for every instrument, validation data,
and `manifest.json`. See `tools/chart-pipeline/README.md` for the output contract,
timing rules, and requirements.

The older song-specific Python recipes below remain useful for analysis experiments
and manual recovery, but new game clients should consume the root bundle format.

### Environment setup

The older audio libraries are picky. Start with Python 3.10 or 3.9 and install the
legacy-compatible pins before `madmom`.

```bash
# install a compatible Python
mise install python@3.10.14
# or: mise install python@3.9.19

# make and activate a fresh venv with that Python
~/.local/share/mise/installs/python/3.10.14/bin/python3 -m venv .venv
source .venv/bin/activate

# pin older deps that play nicely with madmom
pip install "numpy<2" "cython<3" "scipy<1.12"

# now install madmom
pip install madmom==0.16.1
```

Other scripts may also need:

```bash
pip install librosa soundfile pretty_midi scikit-learn basic-pitch crepe aubio
```

External CLI tools used by the workflow:

| Tool | Role |
| --- | --- |
| `audio-separator` / Demucs | Split a full song into vocals, drums, bass, other, and sometimes more stems. |
| `sonic-annotator` + BeatRoot Vamp plugin | Create beat-time grids, usually from the drum stem. |
| `sonic-annotator` + pYIN Vamp plugin | Extract monophonic notes from vocals or bass as `time,duration,frequency` rows. |
| `ffmpeg` | Convert formats and do quick EQ-based drum splitting experiments. |
| `midicsv` / `csvmidi` | Convert MIDI to editable CSV and back for cleanup workflows. |

### Practical recovery path

If you are starting from a full song and want the current intended chart workflow,
run this:

```bash
# 1. Split full song to stems.
docker run -it -v "$(pwd)":/workdir -w /workdir beveradb/audio-separator \
  --output_format wav --model_filename htdemucs.yaml \
  "audio/Heavens_Edge/heavens_edge.wav"

# 2. Move into the main working stem folder.
cd "audio/Heavens_Edge/stems"

# 3. Create the drum beat grid.
sonic-annotator -d vamp:beatroot-vamp:beatroot:beats \
  -w csv --csv-omit-filename --csv-one-file drum_beats.csv \
  --csv-force --force "input_(Drums)_htdemucs.wav"

# 4. Split and classify drums.
python kick_snare_hat_separator.py \
  --audio "input_(Drums)_htdemucs.wav" \
  --beats "drum_beats.csv" \
  --outdir ./drums --subdiv 4

# 5. Extract vocal notes with pYIN.
sonic-annotator -d vamp:pyin:pyin:notes \
  --force -w csv "input_(Vocals)_htdemucs.wav"

# 6. Convert vocal notes to quantized MIDI.
python pyin_csv_to_midi_quant.py \
  "input_(Vocals)_htdemucs_vamp_pyin_pyin_notes.csv" \
  drum_beats.csv pyin_notes.mid

# 7. Convert vocal notes to game lanes.
python pyin_to_lanes.py

# 8. Convert bass with the bass-specific helper.
python pyin_bass_shift_to_midi.py \
  "input_(Bass)_htdemucs.wav" bass_notes.mid --semitones 24
```

Then inspect:

```text
drums/events.csv
drums/events.mid
pyin_notes.mid
vocal_lanes.csv
bass_notes.mid
```

### Beat grid format

Most scripts expect beat times as plain seconds, one float per line:

```text
0.512345
1.021678
1.530912
```

If `sonic-annotator` writes extra columns on your machine, normalize the file to
the first timestamp column before passing it to Python scripts.

`beats_dump.py` is the exception. It writes:

```csv
time_s,is_downbeat
0.512345,1
1.021678,0
```

### Stem splitting

Use the Docker image for `python-audio-separator`. It wraps Ultimate Vocal Remover
models, including Demucs models such as `htdemucs`.

```bash
docker run -it -v "$(pwd)":/workdir -w /workdir beveradb/audio-separator \
  --output_format wav \
  --model_filename htdemucs.yaml \
  "audio/Heavens_Edge/heavens_edge.wav"
```

Expected outputs are stem files like:

```text
input_(Vocals)_htdemucs.wav
input_(Drums)_htdemucs.wav
input_(Bass)_htdemucs.wav
input_(Other)_htdemucs.wav
```

### Beat inspection

Use BeatRoot through `sonic-annotator` for the main drum beat grid:

```bash
cd "audio/Heavens_Edge/stems"

sonic-annotator -d vamp:beatroot-vamp:beatroot:beats \
  -w csv --csv-omit-filename --csv-one-file drum_beats.csv \
  --csv-force --force "input_(Drums)_htdemucs.wav"
```

Use `beats_dump.py` when you want a madmom beat/downbeat CSV or click track:

```bash
python "audio/Heavens_Edge/beats_dump.py" \
  "audio/Heavens_Edge/Heaven's Edge (Vocals).mp3" \
  --meter 4 \
  --click-wav "audio/Heavens_Edge/vocal_click.wav"
```

### Drums

Preferred script:

```bash
cd "audio/Heavens_Edge/stems"

python kick_snare_hat_separator.py \
  --audio "input_(Drums)_htdemucs.wav" \
  --beats "drum_beats.csv" \
  --outdir ./drums \
  --subdiv 4
```

Outputs:

```text
drums/kick.wav
drums/snare.wav
drums/hats.wav
drums/events.csv       # time_s,cluster,label,conf
drums/events.mid       # kick/snare/hats as short drum notes
```

The lane convention from `spec.md` is:

```text
kick  -> lane 0
snare -> lane 1
hats  -> lane 2
```

For 2-lane accessibility, hats collapse onto one of the two available lanes.

### Vocals and melody

The pYIN path creates note rows from the vocal stem:

```bash
cd "audio/Heavens_Edge/stems"

sonic-annotator -d vamp:pyin:pyin:notes \
  --force -w csv "input_(Vocals)_htdemucs.wav"
```

Expected pYIN CSV shape, no header:

```csv
time_seconds,duration_seconds,frequency_hz
7.836734694,0.348299319,292.796
```

Quantize those notes to the drum beat grid and write MIDI:

```bash
python pyin_csv_to_midi_quant.py \
  "input_(Vocals)_htdemucs_vamp_pyin_pyin_notes.csv" \
  "drum_beats.csv" \
  pyin_notes.mid
```

Convert pYIN notes into game lanes:

```bash
python pyin_to_lanes.py
```

Current caveat: `pyin_to_lanes.py` has hardcoded filenames and should become a
real CLI before production use.

### CREPE alternative

Use CREPE when pYIN is not giving usable vocal notes:

```bash
cd "audio/Heavens_Edge/stems"

python crepe_to_notes_quant.py \
  "input_(Vocals)_htdemucs.wav" \
  --grid-csv drum_beats.csv \
  --snap-ms 120 \
  --out vocal_notes.mid
```

If you already have a CREPE f0 CSV with header `time,frequency,confidence`, use:

```bash
python crepe_f0_to_midi_quant.py \
  "input_(Vocals)_htdemucs.f0.csv" \
  "drum_beats.csv" \
  vocal_notes.mid
```

### Bass

Bass is hard for pYIN because low fundamentals can be missed. The bass helper
pitch-shifts the bass up, runs pYIN, then transposes MIDI notes back down.

```bash
cd "audio/Heavens_Edge/stems"

python pyin_bass_shift_to_midi.py \
  "input_(Bass)_htdemucs.wav" \
  bass_notes.mid \
  --semitones 24
```

### Any single stem with Basic Pitch

`audio/Heavens_Edge/stem2midi.py` is a standalone path that does not require a
precomputed `drum_beats.csv`. It tracks beats/downbeats with `madmom`, transcribes
notes with Spotify Basic Pitch, cleans them, snaps them to a grid, and writes MIDI.

```bash
python "audio/Heavens_Edge/stem2midi.py" \
  --input "audio/Heavens_Edge/Heaven's Edge (Vocals).mp3" \
  --meter 4 \
  --subdiv 4 \
  --save-debug
```

Default output:

```text
<input>.quant.mid
```

With `--save-debug`, it also writes:

```text
<input>.quant.raw_notes.csv
<input>.quant.clean_notes.csv
<input>.quant.quantized_notes.csv
<input>.quant.beat_grid.csv
```

## Python script inventory

### Preferred scripts

| Script | Purpose | Inputs | Outputs |
| --- | --- | --- | --- |
| `audio/Heavens_Edge/beats_dump.py` | Inspect beats/downbeats with madmom. | Any audio file. | `<input>.beats.csv`, optional click WAV. |
| `audio/Heavens_Edge/stem2midi.py` | One-stem Basic Pitch plus madmom quantized MIDI. | Any audio stem. | `<input>.quant.mid`, optional debug CSVs. |
| `audio/Heavens_Edge/stems/kick_snare_hat_separator.py` | Adaptive drum K/S/H separator. | Drum stem plus beat times. | `kick.wav`, `snare.wav`, `hats.wav`, `events.csv`, `events.mid`. |
| `audio/Heavens_Edge/stems/pyin_csv_to_midi_quant.py` | pYIN note CSV to beat-quantized MIDI. | pYIN CSV plus beat times. | `.mid`. |
| `audio/Heavens_Edge/stems/pyin_bass_shift_to_midi.py` | Bass transcription helper. | Bass stem. | Bass MIDI transposed back to original octave. |
| `audio/Heavens_Edge/stems/crepe_to_notes_quant.py` | CREPE audio-to-MIDI path. | Audio plus grid CSV or grid MIDI. | Quantized MIDI. |
| `audio/Heavens_Edge/stems/lane_assign_melody.py` | Reusable melody lane mapper. | Notes plus beat times. | Python list of `(time_s,lane,pitch,dur_s)`. |
| `audio/Heavens_Edge/stems/pyin_to_lanes.py` | Hardcoded pYIN-to-vocal-lanes driver. | Hardcoded pYIN CSV plus `drum_beats.csv`. | `vocal_lanes.csv`. |

### Supporting or experimental scripts

| Script | Status |
| --- | --- |
| `audio/Heavens_Edge/stems/pyin_csv_to_midi.py` | Older non-quantized pYIN to MIDI converter. |
| `audio/Heavens_Edge/stems/crepe_f0_to_midi_quant.py` | Converts an already-created CREPE f0 CSV to quantized MIDI. |
| `audio/Heavens_Edge/stems/onsets_to_midi.py` | Onset-only MIDI helper, useful for syllables or percussive timing. |
| `audio/Heavens_Edge/stems/split_drum_wav.py` | Quick fixed-EQ drum splitter, less good than adaptive clustering. |
| `audio/Heavens_Edge/stems/brian_script.py` | Hardcoded vocal onset experiment. |
| `audio/Heavens_Edge/vocals.py` | Hardcoded old vocal MIDI simplifier. |
| `audio/Blackened Crown/stems_to_midi.py` | Song-specific drum stem to General MIDI utility. |

### Legacy hotspot

`audio/Heavens_Edge/stems/lane_assign.py` is a mixed-responsibility legacy file. It
contains a copy of the adaptive drum separator, then embedded drum and melody lane
assignment helpers. It also has visible indentation drift around the CLI section.
Use these cleaner files first:

```text
kick_snare_hat_separator.py
lane_assign_melody.py
pyin_to_lanes.py
```

## Source files vs generated artifacts

Treat these as generated outputs unless you are explicitly inspecting examples:

```text
*.wav
*.mp3
*.mid
*.csv
audio/Heavens_Edge/stems/drums/
audio/Heavens_Edge/stems/index.html
audio/Heavens_Edge/.venv/
```

Useful generated examples already checked in:

```text
audio/Heavens_Edge/drum_beats.csv
audio/Heavens_Edge/Heaven's Edge (Vocals).quant.mid
audio/Heavens_Edge/Heaven's Edge (Vocals).quant.beat_grid.csv
audio/Heavens_Edge/stems/drum_beats.csv
audio/Heavens_Edge/stems/input_(Vocals)_htdemucs_vamp_pyin_pyin_notes.csv
audio/Heavens_Edge/stems/pyin_notes.mid
audio/Heavens_Edge/stems/vocal_lanes.csv
audio/Heavens_Edge/stems/drums/events.csv
audio/Heavens_Edge/stems/drums/events.mid
```

## Song-specific notes

### Heavens Edge

Primary working song folder. Most active tooling and generated examples are here.
Use `audio/Heavens_Edge/stems/AGENTS.md` before changing scripts in the `stems/`
subtree.

### Blackened Crown

Older alternate workflow. It uses `ffmpeg` filters to split drums, then
`stems_to_midi.py` to create a drum MIDI.

```bash
cd "audio/Blackened Crown"

ffmpeg -y -i "1_Blackened Crown(1)_(Drums).mp3" \
  -af "highpass=f=150,lowpass=f=3000,volume=2" snare.wav
ffmpeg -y -i "1_Blackened Crown(1)_(Drums).mp3" \
  -af "lowpass=f=150,volume=2" kick.wav
ffmpeg -y -i "1_Blackened Crown(1)_(Drums).mp3" \
  -af "highpass=f=5000,volume=3" hats.wav

./stems_to_midi.py --kick kick.wav --snare snare.wav --hats hats.wav \
  -o drums.mid --bpm 128 --ppq 480 --hold-ms 60 --vel 105 \
  --min-sep-ms 45 --quant-div 4 --delta 0.18
```

## Known drift and TODOs

- `audio/Heavens_Edge/stems/spec.md` mentions future-looking names such as
  `pipeline.py`, `lane_assign_drums.py`, and `quantize.py`. Those are not current
  on-disk scripts.
- `pyin_to_lanes.py` should become a real CLI instead of using hardcoded filenames.
- BeatRoot output should be normalized to one float per line if Sonic Annotator
  writes a richer CSV on a given machine.
- There is no repo-wide Python package or CI suite; `roblox/web/` has its own
  package scripts and automated tests.
- There is no native Roblox game code yet; the implemented gameplay demo is web-only.

## Asset and model notes

Raw notes in `trello_notes.md` mention these 3D asset experiments:

- Meshy worked better than the other tried tools.
- hitem3d.ai generated usable-looking output but with too many faces/vertices and
  imported worse into Roblox than Meshy.
- Figuro, nlevel.ai, and Rodin were considered but not selected in the notes.

Potential item drops from the notes:

- guitar pedals
- kick pedals
- drum sticks
- guitar picks
- microphone
- mic stand
- instrument cable
- guitar, bass, drums, keyboard

Sound effect sources:

- https://pixabay.com/sound-effects/search
- https://elevenlabs.io/sound-effects
- https://directory.audio/

# Resources

## Assets

- https://www.cgtrader.com/free-3d-models/textures/miscellaneous/the-stylized-vault-375-stylized-pbr-mega-pack-texture-library

## Textures

- https://www.textures.com/free

## Animation

- https://mesh2motion.org/
