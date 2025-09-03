import subprocess
import sys
import os

def split_drum_stem(input_file):
  # Output filenames
  snare_file = "snare.wav"
  kick_file = "kick.wav"
  hats_file = "hats.wav"

  # FFmpeg filter commands
  snare_filter = "highpass=f=150,lowpass=f=3000,volume=2"
  kick_filter = "lowpass=f=150,volume=2"
  hats_filter = "highpass=f=5000,volume=3"

  commands = [
    ["ffmpeg", "-y", "-i", input_file, "-af", snare_filter, snare_file],
    ["ffmpeg", "-y", "-i", input_file, "-af", kick_filter, kick_file],
    ["ffmpeg", "-y", "-i", input_file, "-af", hats_filter, hats_file],
  ]

  for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print(f"Usage: python {os.path.basename(__file__)} <drum_stem_file>")
    sys.exit(1)
  input_file = sys.argv[1]
  split_drum_stem(input_file)