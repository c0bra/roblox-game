# Roblox Studio development

This directory is the Roblox Studio game project. Gameplay code lives in `src/` and syncs into Studio with [Rojo](https://rojo.space/).

Before implementing gameplay, use [`GAME_DESIGN.md`](../GAME_DESIGN.md) for
player-facing behavior and [`SYSTEMS_MAP.md`](../SYSTEMS_MAP.md) for system
ownership and dependencies, then follow the approved documents linked from
[`SYSTEM_SPECIFICATIONS_TRACKER.md`](../SYSTEM_SPECIFICATIONS_TRACKER.md) and the
shared numeric contract in
[`BALANCE_FRAMEWORK.md`](../BALANCE_FRAMEWORK.md) and the lightweight evidence
contract in
[`PLAYTEST_AND_ANALYTICS.md`](../PLAYTEST_AND_ANALYTICS.md). Player-facing strings
follow [`NAMING_AND_TONE.md`](../NAMING_AND_TONE.md); `Bands Battle` remains a
working project title. The systems map defines design boundaries, not a one-to-
one Roblox service or module layout.

Use [`TECHNICAL_ARCHITECTURE.md`](../TECHNICAL_ARCHITECTURE.md) for runtime
authority, source ownership, content/export contracts, deployment, persistence,
testing, and release boundaries. Its
[`question plan`](../TECHNICAL_ARCHITECTURE_QUESTIONS.md) and
[`working record`](../TECHNICAL_ARCHITECTURE_WORKING.md) preserve interview
history but do not override the canonical architecture.

Use the [implementation plan](../IMPLEMENTATION_PLAN.md) for the Blackened Crown
staging tasks and their acceptance checks. Record results in [Implementation Progress](../IMPLEMENTATION_PROGRESS.md).

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

For the foundation sync check, open a disposable copy of `Place1.rbxl` and
create a Folder named `T02StudioMarker` in ReplicatedStorage. Connect the
Rojo plugin to the server above. Verify that the synced RhythmChart module
appears under ReplicatedStorage/BandsBattle and that T02StudioMarker survives.
Disconnect the plugin and record the observed result in [Implementation Progress](../IMPLEMENTATION_PROGRESS.md).

For the shared timeline contract check, keep Rojo connected to the disposable
place. From the repository root on macOS, copy the following Studio Command Bar
check to the clipboard, then paste and run it in edit mode:

```bash
{ echo 'local run = (function()'; cat roblox/tests/timeline_contract.luau; echo 'end)()'; echo 'print(run([==['; cat roblox/tests/fixtures/instrumental-timeline.json; echo ']==]))'; } | pbcopy
```

The check loads the synced reader and reports whether the fixture and six
rejection cases passed. Run `bun test tests/timeline-contract.test.ts` from
`tools/chart-pipeline` to check the same fixture with the Bun reader.
