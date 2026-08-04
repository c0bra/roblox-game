# STEMS WORKFLOW KNOWLEDGE BASE

## OVERVIEW
This subtree is the main workflow hotspot: most of the drum separation, pYIN/CREPE conversion, lane assignment, quantization, and CSV/MIDI export logic lives here, while a few song-level helpers remain one directory up.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Pipeline contract | `spec.md` | Intended architecture, presets, validation rules, known code/spec drift |
| Command cookbook | `sonic-annotator.md` | Real command lines used for BeatRoot, pYIN, CREPE, drum separation |
| Drum separation | `kick_snare_hat_separator.py` | Adaptive K/S/H separator, WAV + CSV + MIDI outputs |
| Large hotspot | `lane_assign.py` | Monolithic mixed logic; highest-risk file to edit |
| Melody lane mapping | `lane_assign_melody.py` | Cleaner melody-specific assignment helpers |
| pYIN to MIDI | `pyin_csv_to_midi_quant.py`, `pyin_csv_to_midi.py` | Quantized vs simpler legacy-style converter |
| CREPE to MIDI | `crepe_to_notes_quant.py`, `crepe_f0_to_midi_quant.py` | Vocal/f0 workflows |
| Bass workflow | `pyin_bass_shift_to_midi.py` | Shift-up/analyze/shift-down path |
| Onset export | `onsets_to_midi.py` | madmom/aubio-style onset-to-MIDI utility |
| Quick splitter | `split_drum_wav.py` | Simpler ffmpeg/EQ-based helper |

## CONVENTIONS
- Treat `spec.md` and `sonic-annotator.md` as the local docs of record before changing script behavior.
- Keep compatibility shims for older audio libraries unless you verify the entire toolchain still works.
- Beat-grid files are plain text float-per-line inputs; pYIN and CREPE CSV formats are script-sensitive.
- Lane outputs and note exports follow the local rhythm-game workflow, not generic MIDI-tool assumptions.
- Prefer editing the cleaner specialized file when possible (`lane_assign_melody.py`) before touching the monolithic hotspot.

## ANTI-PATTERNS
- Do not create nested docs in `drums/`, `__pycache__/`, or other generated-output folders.
- Do not assume `pyin_to_lanes.py` is a clean reusable CLI; it is example-style and uses hardcoded workflow assumptions.
- Do not treat every converter as canonical; some files are overlapping utilities or older simpler paths.
- Do not trust spec names blindly: `spec.md` mentions files like `pipeline.py` and `lane_assign_drums.py` that are not the current on-disk filenames.

## CANONICAL VS SUPPORTING
- Canonical workflow references: `spec.md`, `sonic-annotator.md`, `kick_snare_hat_separator.py`, `lane_assign_melody.py`
- High-risk hotspot: `lane_assign.py` is a mixed-responsibility legacy file whose name understates its scope; it still carries drum-separator-style structure plus embedded drum and melody lane-assignment logic.
- Supporting / overlapping utilities: `pyin_csv_to_midi.py`, `split_drum_wav.py`, `brian_script.py`

## OUTPUT BOUNDARY
- `drums/` is output-heavy and inherits this file; do not add another AGENTS.md below it.
- `.mid`, `.wav`, `.csv`, and the checked-in `index.html` here are mostly artifacts or inspection outputs unless a task explicitly targets them.

## COMMANDS
```bash
sonic-annotator -d vamp:pyin:pyin:notes --force -w csv "input_(Vocals)_htdemucs.wav"
python pyin_csv_to_midi_quant.py "input_(Vocals)_htdemucs_vamp_pyin_pyin_notes.csv" drum_beats.csv pyin_notes.mid
python crepe_to_notes_quant.py "input_(Vocals)_htdemucs.wav" --grid-csv drum_beats.csv --out vocal_notes.mid
python pyin_bass_shift_to_midi.py "input_(Bass)_htdemucs.wav" bass_notes.mid --semitones 24
python kick_snare_hat_separator.py --audio "blackend_crown_drums.wav" --beats "blackened_crown_drum_beats.csv" --outdir ./drums --subdiv 4
```

## NOTES
- This folder mixes maintained scripts with generated artifacts; identify targets by file role, not extension alone.
- If you touch `lane_assign.py`, read neighboring specialized files first so you do not duplicate or regress logic already split elsewhere.
