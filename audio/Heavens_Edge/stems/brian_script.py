import numpy as np

# Shim for madmom / older libs expecting np.float etc.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object
    
from madmom.features.onsets import RNNOnsetProcessor, CNNOnsetProcessor, OnsetPeakPickingProcessor

audio_path = "input_(Vocals)_htdemucs.wav"

# 1) Pick a model. RNN is a good default; try CNN if RNN misses too much.
fps = 200  # finer time resolution; 100–200 is typical
rnn = RNNOnsetProcessor()        # or: cnn = CNNOnsetProcessor()

act = rnn(audio_path)            # activation curve at `fps` Hz

# 2) Peak picking tuned for vocals
picker = OnsetPeakPickingProcessor(
    fps=fps,
    threshold=0.15,   # vocals: 0.10–0.30; lower = more sensitive
    smooth=0.05,      # seconds; smooth activation to reduce jitter
    combine=0.03,     # min gap (s) to merge double-triggers from the same syllable
    pre_max=0.02,     # local max window before (s)
    post_max=0.02,    # local max window after (s)
    pre_avg=0.10,     # adaptive threshold window before (s)
    post_avg=0.10,    # adaptive threshold window after (s)
)

onset_times = picker(act)  # numpy array of seconds
print(len(onset_times), "onsets")
for t in onset_times[:20]:
    print(f"{t:.3f}")
