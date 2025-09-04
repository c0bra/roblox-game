# Python crap

```
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

# Music Processing

Split the mp3 from Suno.ai into stems using python-audio-separator (Docker image), which is based on Ultimate Voice Remover,
which in turns uses demucs (and other models) internally (it takes about as long as the song is, on my Mac).

```
docker run -it -v `pwd`:/workdir beveradb/audio-separator --output_format wav --model_filename htdemucs.yaml input.wav
```

## Drums

Get the beats first (TODO: try using librosa first for tempo):
```
sonic-annotator -d vamp:beatroot-vamp:beatroot:beats \
  -w csv --csv-omit-filename --csv-one-file drum_beats.csv
  --force \
  1_heavens_edge_\(Drums\).mp3
```

OK wait. So instead of the below, I've got a script called `kick_snare_hat_separator.py` that uses librosa (or aubio if
available) for onsets, then clusters around the beats found with the BeatRoot sonic-annotator plugin. It separates into
3 wav files (kick, snare, hat) and does the midi conversion and events.csv so it's pretty good!


## 

## Vocals

Once you have the beats from the drums, you can extract the vocal notes with sonic-annotator and the pyin plugin:

```
sonic-annotator -d vamp:pyin:pyin:notes \
  --force -w csv "input_(Vocals)_htdemucs.wav"
```

And then quantize the vocal notes to our beat with this script:

```
python pyin_csv_to_midi_quant.py "input_(Vocals)_htdemucs_vamp_pyin_pyin_notes.csv" drum_beats.csv pyin_notes.mid
```

# Sound Effects

* Find some here: https://pixabay.com/sound-effects/search
  * Bad ukelele strumming: https://pixabay.com/sound-effects/bad-ukulele-strumming-2-40667/
* Generate them here: https://elevenlabs.io/sound-effects and use my extension to download them freely.
* https://directory.audio/