#!/usr/bin/env bash
# Convert aubionotes 3-col output (PITCH START_SEC END_SEC) -> cleaned MIDI
# Requires: csvmidi (from midicsv)
# Usage:
#   ./aubio_clean2midi.sh notes_3col.txt out.mid [BPM] [PPQ] [QUANT_DIV] [MIN_DUR_MS] [P_MIN] [P_MAX] [PROGRAM]
# Defaults:
#   BPM=120, PPQ=480, QUANT_DIV=4 (1/16), MIN_DUR_MS=80, P_MIN=0, P_MAX=127, PROGRAM=81 (Lead 2 Saw)

IN="$1"; OUT="$2"
BPM="${3:-120}"
PPQ="${4:-480}"
QDIV="${5:-4}"        # 1/QDIV note grid (4=16ths, 8=32nds, 2=8ths)
MINMS="${6:-80}"      # drop notes shorter than this
PMIN="${7:-0}"
PMAX="${8:-127}"
PROGRAM="${9:-81}"

if [ -z "$IN" ] || [ -z "$OUT" ]; then
  echo "Usage: $0 notes_3col.txt out.mid [BPM] [PPQ] [QUANT_DIV] [MIN_DUR_MS] [P_MIN] [P_MAX] [PROGRAM]" >&2
  exit 1
fi
if ! command -v csvmidi >/dev/null 2>&1; then
  echo "csvmidi not found. Install midicsv (macOS: brew install midicsv; Debian/Ubuntu: sudo apt install midicsv)" >&2
  exit 1
fi

CSV="$(mktemp)"

awk -v BPM="$BPM" -v PPQ="$PPQ" -v QDIV="$QDIV" -v MINMS="$MINMS" \
    -v PMIN="$PMIN" -v PMAX="$PMAX" -v PROGRAM="$PROGRAM" -v VEL=96 '
function sec_to_ticks(sec){ return int(sec * PPQ * BPM / 60.0 + 0.5) }
function grid(){ return int(PPQ / QDIV + 0.5) }
function qround(t){ g=grid(); return int((t + g/2)/g)*g }
function flush_note(p){
  if (!(p in on)) return
  s = on[p]; e = off[p]
  if (e <= s) e = s + 1
  dur = e - s
  if (dur >= min_ticks) {
    printf "1, %d, Note_on_c, 0, %d, %d\n", s, p, VEL
    printf "1, %d, Note_off_c, 0, %d, 0\n", e, p
    if (e > last) last = e
  }
  delete on[p]; delete off[p]
}
BEGIN{
  min_ticks = int((MINMS/1000.0) * PPQ * BPM / 60.0 + 0.5)
  print  "0, 0, Header, 1, 2, " PPQ
  print  "1, 0, Start_track"
  printf "1, 0, Tempo, %d\n", int(60000000 / BPM)
  printf "1, 0, Program_c, 0, %d\n", PROGRAM+0
}
# Expect exactly 3 columns: PITCH START_SEC END_SEC
NF>=3 {
  pitch = int($1 + 0)
  if (pitch < PMIN || pitch > PMAX) next

  t_on  = qround(sec_to_ticks($2))
  t_off = qround(sec_to_ticks($3))
  if (t_off <= t_on) t_off = t_on + 1

  # Merge adjacent same-pitch segments if separated by <= one grid
  if ((pitch in on) && (t_on - off[pitch] <= grid())) {
    if (t_off > off[pitch]) off[pitch] = t_off
  } else {
    # flush previous note of same pitch, then start new
    flush_note(pitch)
    on[pitch]  = t_on
    off[pitch] = t_off
  }
}
END{
  # flush any remaining notes
  for (p in on) flush_note(p)
  printf "1, %d, End_track\n", (last ? last : 0) + 1
  print  "0, 0, End_of_file"
}
' "$IN" > "$CSV"

csvmidi "$CSV" "$OUT" >/dev/null 2>&1 && echo "Wrote $OUT (BPM=$BPM, PPQ=$PPQ, grid=1/$QDIV, min=${MINMS}ms, pitch[$PMIN..$PMAX], program=$PROGRAM)"
rm -f "$CSV"
