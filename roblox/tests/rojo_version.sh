#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
manifest="$repo_root/roblox/rokit.toml"
supported_version="7.7.0"
manifest_version="$(sed -n 's/^rojo = "rojo-rbx\/rojo@\([^"]*\)"$/\1/p' "$manifest")"

if [[ "$manifest_version" != "$supported_version" ]]; then
	printf 'FAIL: Roblox must pin Rojo %s to match the supported Studio plugin; found %s\n' \
		"$supported_version" "${manifest_version:-no valid Rojo pin}" >&2
	exit 1
fi

printf 'PASS: Roblox pins the supported Rojo version %s\n' "$supported_version"
