# PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-08
**Commit:** a999b70
**Branch:** main

## OVERVIEW
Python audio-processing workspace for turning song stems into beat grids, MIDI, and lane-chart data for a rhythm-game workflow.

This is not a packaged app. The repo is mostly song-specific assets plus a small number of standalone Python and shell tools.

## STRUCTURE
```text
roblox-bands-battle/
├── README.md                  # Environment setup + top-level workflow notes
├── models/                    # Blender source files and rendered previews for Roblox assets
├── audio/
│   ├── Heavens_Edge/          # Main working song folder; mostly assets plus a few scripts
│   │   └── stems/             # Main code-heavy processing subtree; see child AGENTS.md
│   └── Blackened Crown/       # Smaller song-specific workflow area; asset-heavy
└── .gitignore                 # Local artifact rules
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Environment setup | `README.md` | Python version, venv, pinned deps, madmom install |
| Repo-wide workflow | `README.md` | Demucs/audio-separator, BeatRoot, pYIN steps |
| Beat grid inspection | `audio/Heavens_Edge/beats_dump.py` | madmom beat/downbeat dump + optional click track |
| Single-stem transcription | `audio/Heavens_Edge/stem2midi.py` | Basic Pitch + madmom + quantization |
| Stems pipeline details | `audio/Heavens_Edge/stems/` | Main code-heavy subtree; check child AGENTS.md |
| Blackened Crown workflow | `audio/Blackened Crown/README.md` | ffmpeg + shell-script commands |
| Blackened Crown drum MIDI | `audio/Blackened Crown/stems_to_midi.py` | Smaller sibling script, root-covered |
| Roblox Blender assets | `models/` | Source `.blend` files plus PNG/MP4 previews |

## CONVENTIONS
- Treat this repo as a script workspace, not a Python package.
- Prefer direct CLI execution of standalone scripts over introducing package structure.
- Root setup expects a fresh `.venv` and pinned legacy-compatible deps before installing `madmom`.
- External tools are part of the normal workflow: Docker audio-separator/Demucs, `sonic-annotator`, `ffmpeg`, `midicsv`/`csvmidi`.
- Python compatibility shims are intentional in audio scripts; do not remove them casually.

## ANTI-PATTERNS (THIS PROJECT)
- Do not treat generated `.csv`, `.mid`, `.wav`, or `.mp3` artifacts as source of truth when code or docs disagree.
- Do not add child documentation under output-only folders such as generated drum/output directories.
- Do not assume CI, package scripts, or automated tests exist; they do not.
- Do not rely on checked-in local environments; `audio/Heavens_Edge/.venv/` is noise, not the intended contributor setup.

## UNIQUE STYLES
- Song folders mix source scripts with generated artifacts; use filenames and nearby docs to distinguish maintained tools from outputs.
- `audio/Heavens_Edge/` is asset-heavy but contains a few maintained scripts at the song root.
- `audio/Blackened Crown/` is mostly a recipe/workflow folder with shell helpers and one main Python script.

## BLENDER 5.2 AUTOMATION
- For Blender 5.2 automation, use `BLENDER_EEVEE` as the Eevee render-engine enum; `BLENDER_EEVEE_NEXT` is rejected.
- This Blender 5.2 build does not expose `FFMPEG` as an image output format. Render a PNG sequence, then encode it with the `ffmpeg` CLI.
- Blender 5.2 actions do not expose the legacy `action.fcurves` attribute used by older scripts; avoid scripts that depend on that API.

## COMMANDS
```bash
# environment
mise install python@3.10.14
~/.local/share/mise/installs/python/3.10.14/bin/python3 -m venv .venv
source .venv/bin/activate
pip install "numpy<2" "cython<3" "scipy<1.12"
pip install madmom==0.16.1

# stem separation
docker run -it -v `pwd`:/workdir beveradb/audio-separator --output_format wav --model_filename htdemucs.yaml input.wav

# beat extraction
sonic-annotator -d vamp:beatroot-vamp:beatroot:beats -w csv --csv-omit-filename --csv-one-file drum_beats.csv --force "input_(Drums)_htdemucs.wav"

# vocal quantization / transcription
python audio/Heavens_Edge/stem2midi.py --input "audio/Heavens_Edge/heavens_edge.wav"
python audio/Heavens_Edge/beats_dump.py "audio/Heavens_Edge/Heaven's Edge (Vocals).mp3" --meter 4
```

## NOTES
- No `AGENTS.md` existed before this pass.
- No repo CI, no `pyproject.toml`, no `package.json`, no `Makefile`, and no formal test suite were found.
- For most logic changes, the main working area is `audio/Heavens_Edge/stems/`; use its child AGENTS.md for local rules.
