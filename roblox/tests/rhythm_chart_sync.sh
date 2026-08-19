#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
module="$repo_root/roblox/src/shared/BandsBattle/RhythmChart.luau"
source_path="$(sed -n 's/^RhythmChart.sourcePath = "\(.*\)"$/\1/p' "$module")"
audio_path="$(sed -n 's/^RhythmChart.audioSourcePath = "\(.*\)"$/\1/p' "$module")"
clip_start="$(sed -n 's/^RhythmChart.audioClipStart = \([0-9.]*\)$/\1/p' "$module")"
audio_id="$(sed -n 's/^RhythmChart.audioId = "rbxassetid:\/\/\([0-9]*\)"$/\1/p' "$module")"

if [[ -z "$source_path" ]]; then
	printf 'FAIL: RhythmChart.luau does not identify its source JSON\n' >&2
	exit 1
fi

if [[ "$audio_path" != "audio/Heavens_Edge/heavens_edge.mp3" || "$clip_start" != "60" ]]; then
	printf 'FAIL: Roblox must use Heaven\x27s Edge MP3 at the chart builder\x27s 60-second clip origin\n' >&2
	exit 1
fi

if [[ ! -f "$repo_root/$audio_path" ]]; then
	printf 'FAIL: configured chart MP3 does not exist: %s\n' "$audio_path" >&2
	exit 1
fi

if [[ -z "$audio_id" ]]; then
	printf 'FAIL: RhythmChart.luau does not have an uploaded Roblox audio asset ID\n' >&2
	exit 1
fi

source_file="$repo_root/$source_path"
canonical_json="$(mktemp)"
embedded_json="$(mktemp)"
trap 'rm -f "$canonical_json" "$embedded_json"' EXIT
jq -c . "$source_file" > "$canonical_json"
awk '
	/^local SOURCE_JSON = \[==\[$/ { capture = 1; next }
	/^\]==\]$/ { capture = 0 }
	capture { print }
' "$module" > "$embedded_json"

source_hash="$(shasum -a 256 "$canonical_json" | awk '{print $1}')"
embedded_hash="$(shasum -a 256 "$embedded_json" | awk '{print $1}')"

if [[ "$source_hash" != "$embedded_hash" ]]; then
	printf 'FAIL: embedded Roblox chart differs from %s\n' "$source_path" >&2
	printf 'source=%s embedded=%s\n' "$source_hash" "$embedded_hash" >&2
	exit 1
fi

printf 'PASS: Roblox chart exactly matches %s (%s)\n' "$source_path" "$source_hash"
