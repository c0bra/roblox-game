# Roblox Studio development

This directory is the Roblox Studio game project. Gameplay code lives in `src/` and syncs into Studio with [Rojo](https://rojo.space/).

Before implementing gameplay, use [`GAME_DESIGN.md`](../GAME_DESIGN.md) for
player-facing behavior and [`SYSTEMS_MAP.md`](../SYSTEMS_MAP.md) for system
ownership, dependencies, and the required detailed-spec sequence. The systems map
defines design boundaries, not a one-to-one Roblox service or module layout.

## One-time setup

1. Install [Rokit](https://github.com/rojo-rbx/rokit), then run `rokit install` in this directory. The checked-in `rokit.toml` pins the supported Rojo version.
2. Install the matching Studio plugin with `rojo plugin install`. Restart Studio if it was open during installation.
3. In Roblox Studio, open `Place1.rbxl`.
4. For AI-assisted Studio work, open **Assistant > ... > Manage MCP Servers** and enable **Studio as MCP server**. The project MCP command is already configured in `.codex/config.yml`.

## Daily workflow

From this directory, start Rojo:

```bash
rojo serve
```

In Studio, open the Rojo plugin and connect to `localhost:34872`. Rojo owns only these source-backed paths:

- `ReplicatedStorage/BandsBattle`
- `ServerScriptService/BandsBattle`
- `StarterPlayer/StarterPlayerScripts/BandsBattle`

The project uses `$ignoreUnknownInstances` so existing Studio-authored terrain, models, UI, and scripts are preserved when Rojo connects. Keep new gameplay code under `src/`; keep the map and other binary-only content in `Place1.rbxl` until it is intentionally migrated.

Use Studio MCP for inspecting the DataModel, manipulating scene objects, and running playtests. Edit Rojo-owned script source in `src/`, not through MCP or Studio's script editor, because the next Rojo sync will restore the filesystem version.

Build a clean place containing the source-backed tree with:

```bash
rojo build --output build/BandsBattle.rbxl
```

Generate a sourcemap for editor tooling with:

```bash
rojo sourcemap --output sourcemap.json
```

`build/` and `sourcemap.json` are generated and not committed.
