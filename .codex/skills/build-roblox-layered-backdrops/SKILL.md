---
name: build-roblox-layered-backdrops
description: Build and repair convincing walkable Roblox environments that combine a cubemap Sky with real 3D foreground, midground, and rear-transition scenery. Use when Codex is creating or debugging Roblox Studio skybox arenas, panoramic backdrops, fake or overly distant scenery, hard floor-to-horizon seams, missing movement parallax, repeated generated props, or MCP-driven environment tests.
---

# Build Roblox Layered Backdrops

Treat the skybox as the infinity layer and create perceived proximity with real geometry. Build a layered environment that remains convincing while the player translates through the level, not only while the camera rotates at spawn.

## Non-negotiable rules

- Reserve the cubemap for sky and terrain that should remain infinitely distant. A skybox has rotational response but no translational parallax.
- Put any object intended to feel nearby into 3D. Enlarging painted mountains or rocks can change apparent angular size, but cannot make them behave correctly while walking.
- Keep the playable floor as real geometry and extend its visual surface beyond the intended movement area.
- Use at least three distinct silhouette families for repeated scenery. Scaling and yaw alone do not hide a recognizable cloned mesh.
- Anchor decorative scenery and disable collisions and touch interactions unless gameplay explicitly requires them.
- Keep main and fallback traversal routes clear even when scenery overlaps them visually.
- Validate the final composition from spawn and displaced player positions. Hierarchy inspection alone cannot approve the visual transition.

## Depth model

Compose four bands:

1. **Foreground accents:** small rocks, ice shards, vegetation, or debris that provide immediate parallax without crowding the player.
2. **Midground masses:** cliffs, boulders, ruins, or trees that establish scale and break the empty-floor silhouette.
3. **Rear transition:** broad, low, overlapping ridges that obscure the ruler-straight floor-to-sky boundary without forming a continuous wall.
4. **Infinite background:** a restrained cubemap with distant landforms, atmosphere, and sky.

Derive distances from the actual gameplay radius and camera instead of copying fixed stud values. As a starting point, distribute foreground forms near the outer edge of ordinary movement, place midground forms beyond them, and place rear ridges just before the visual horizon. Test every allowed camera height and movement extreme.

## Workflow

### 1. Audit the live place

List connected Roblox Studio instances and explicitly select the intended place before editing. Inspect:

- the playable center, maximum travel distance, and floor bounds;
- camera height, field of view, and expected viewpoints;
- the current `Sky`, `Atmosphere`, and `Lighting` configuration;
- existing scenery positions, dimensions, materials, meshes, and collision settings;
- a spawn screenshot and at least one displaced screenshot when capture is available.

Quantify the depth gap. Count existing scenery in foreground, midground, and rear-transition bands before deciding what to add.

### 2. Prepare the far background

Use a Roblox `Sky` cubemap for the infinite layer. When starting from an equirectangular panorama, convert it into six consistently oriented square faces and verify every cubemap edge.

Keep the lower background visually quiet and color-compatible with the real floor. Include distant mountains or skyline forms, but move close rocks, trees, buildings, and other parallax-sensitive objects into 3D.

Do not try to solve a walking-level transition by adding an enormous empty foreground band to the panorama. That makes the world feel farther away while preserving the underlying lack of parallax.

### 3. Build real depth layers

Create foreground, midground, and rear-transition folders beneath one environment Model or Folder. Preserve a clear route through the scene.

Mix at least three archetypes, such as:

- tall irregular cliffs;
- broad low ridges;
- narrow spires or crystals.

Vary proportions, yaw, depth, spacing, material treatment, and snow or vegetation coverage. Arrange pieces as asymmetrical overlapping clusters. Avoid evenly spaced rings and repeated left/right pairs.

Use low ridges to interrupt exposed horizon gaps. Overlap their screen-space silhouettes slightly from important viewpoints, while leaving enough variation that they do not read as a fence.

When using Studio AI mesh generation, retain each generated source Model as a template and clone the Model. Directly assigning `MeshId` from Luau can fail with a `NotAccessible` capability error. Store accepted templates in `ServerStorage`; archive rejected or replaced prototypes there instead of deleting them.

Read [references/studio-mcp.md](references/studio-mcp.md) before modifying a place through Roblox Studio MCP.

### 4. Blend floor, geometry, and sky

- Match the floor's dominant hue and value to the quiet lower portion of the cubemap.
- Reduce floor texture contrast with distance through lighting, atmosphere, or deliberate material treatment.
- Let low 3D ridges and midground masses cross the floor-to-sky boundary in screen space.
- Avoid a glowing rim, hard platform edge, exposed straight horizon, or fog thick enough to erase every depth cue.
- Keep the skybox distant. Use geometry and atmosphere, not skybox scaling, to make the environment feel closer.

### 5. Verify in Play mode

Run structural assertions before visual review:

- expected layer and object counts;
- at least three unique mesh or silhouette families;
- every decorative part anchored;
- every non-gameplay scenery part non-collidable and non-touching;
- intended floor and spawn still present.

Then perform the real interaction:

1. start Play mode and wait for the character;
2. walk from spawn toward multiple movement extremes and back;
3. confirm the foreground and midground move with distinct parallax;
4. inspect the horizon at spawn and at displaced positions;
5. check that no prop blocks navigation and no floor edge becomes visible;
6. inspect the Studio console for asset, script, and rendering errors;
7. capture fresh frames after the last scene change;
8. stop Play mode and leave Studio in Edit mode.

If the screenshot bridge times out, retry only after returning to Edit mode and starting a fresh Play session. Continue structural and navigation checks, but report the missing pixel-level evidence explicitly and do not claim a visual-fidelity pass.

## Completion contract

Finish only when:

- the cubemap serves only as the infinite layer;
- foreground, midground, rear transition, and sky read as separate depth bands;
- nearby scenery exhibits real translational parallax;
- no obvious repeated-mesh cadence or ruler-straight horizon remains in fresh captures;
- all intended movement routes pass with no decorative collisions;
- the Studio console is clean or unrelated pre-existing errors are named;
- the exact final scene is visually reviewed from spawn and a displaced position;
- Studio is returned to Edit mode and the user is reminded to save the place when MCP cannot save it directly.

