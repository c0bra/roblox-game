---
name: build-babylon-photodomes
description: Create, upscale, repair, implement, and visually validate seamless equirectangular 360-degree Babylon.js PhotoDome or inside-sphere environment backdrops with adaptive 4K/8K delivery and a blendable 3D floor. Use for panoramic game backgrounds, sky domes, circular arenas, non-interactive scenery, blurry or visibly seamed PhotoDomes, curved or color-mismatched floor junctions, walkable panorama limitations, and generated-image-to-Babylon workflows.
---

# Build Babylon PhotoDomes

Build a production panorama pipeline, not merely a large image. Preserve spherical geometry, repair the longitude seam, create real detail before resizing, and verify the result while rotating inside the rendered dome.

## Non-negotiable rules

- Use a strict `2:1` equirectangular master: width must equal twice the height.
- Do not apply a second spherical or dome warp in post-processing. Babylon performs the equirectangular-to-sphere mapping at runtime.
- Map a full equirectangular texture onto a full sphere (`slice: 1`). A partial sphere compresses latitude and lifts the texture equator above the geometric horizon, producing a curved bowl at the floor junction.
- Keep the walkable floor as Babylon geometry when the player stands or moves on it. Do not bake a second platform into the panorama.
- Create semantic detail before or during AI reconstruction. Lanczos resizing alone does not create detail, and Real-ESRGAN cannot repair incorrect 360-degree geometry.
- Treat the left and right edges as adjacent pixels. Repair the longitude seam before final export and inspect it again after upscaling.
- Preserve an opaque image with smooth poles. Do not leave alpha holes, black caps, text, UI, people, or a visible vertical seam.
- Use 8K selectively. An `8192 × 4096` texture decodes to roughly 128 MiB as RGBA before mipmaps.

## Workflow

### 1. Audit the target

Inspect the existing renderer, asset dimensions, camera FOV, WebGL texture limit, device-pixel-ratio handling, floor geometry, CSS color tokens, and current screenshots. Record the accepted composition before changing it.

Choose the surface:

- Use `PhotoDome` when the project already uses it and only needs a panoramic background.
- Use an inward-facing sphere when texture lifecycle, material flags, fog exclusion, or adaptive asset selection need explicit control.
- Keep the camera near the sphere center. A panorama supplies rotational parallax only; moving far from center reveals that it is not real geometry.

### 2. Generate or reconstruct a panorama master

Use image generation for semantic detail and composition. State the intended use as a production 360-degree game environment and require:

- strict `2:1` equirectangular projection;
- seamless left/right boundary and smooth poles;
- open central ground matching the future floor color;
- detailed nearby rocks, boulders, ice, or vegetation;
- progressively softer distant layers;
- no baked platform, camera, text, characters, or UI;
- composition and palette preservation when editing an accepted source.

For a walkable scene, interpret “nearby” conservatively: keep the panorama's immediate near-ground quiet and place truly close rocks or other parallax-sensitive objects as 3D geometry. Do not generate a large arbitrary empty band to compensate for projection errors.

When a generated source is too soft, run a reconstruction pass that adds believable material detail while preserving all large silhouettes and horizon placement. Do this before mechanical upscaling.

Read [references/image-pipeline.md](references/image-pipeline.md) for prompt templates, seam repair, Real-ESRGAN, edge feathering, export commands, and failure modes.

### 3. Repair the longitude seam

Move the original left/right boundary to the center:

```bash
python scripts/shift_longitude.py input.png seam-centered.png
```

Use image editing to repair only a narrow center band. Preserve the scene outside that band. Shift the repaired result back with the same command; a half-width shift is self-inverse.

After AI repair, apply only a narrow symmetric edge feather if the boundary still differs slightly:

```bash
python scripts/feather_longitude.py repaired.png seam-safe.png --pixels 64
```

Do not use a broad feather to conceal mismatched composition. Return to the centered seam edit when rocks, mountains, or the horizon do not connect.

### 4. Upscale without inventing false geometry

If the detailed master is below 4K, upscale a lossless PNG with the official Real-ESRGAN portable build and the general-purpose `realesrgan-x4plus` model:

```bash
python scripts/upscale_realesrgan.py detailed.png reconstructed.png \
  --binary /path/to/realesrgan-ncnn-vulkan \
  --model realesrgan-x4plus --scale 4 --tile 256
```

Inspect for halos, waxy snow, repeated crack patterns, altered color, and a reopened seam. Upscaling is a detail-reconstruction step, not a substitute for equirectangular generation or seam editing.

Apply the final narrow edge feather after upscaling, then export exact runtime variants:

```bash
python scripts/export_variants.py seam-safe.png public/assets/backdrops/ice-mountains
```

This creates `ice-mountains-4k.webp` and `ice-mountains-8k.webp`. Keep the lossless seam-safe master and record SHA-256 hashes and the complete prompt/edit/upscale history beside the assets.

### 5. Validate the image before implementation

Run:

```bash
python scripts/validate_panorama.py seam-safe.png --strip 64
python scripts/validate_panorama.py ice-mountains-4k.webp --strip 32
python scripts/validate_panorama.py ice-mountains-8k.webp --strip 64
```

Require exact `2:1` dimensions. Use the reported boundary and strip MAE as comparison signals, not universal aesthetic thresholds. Lower is better; a sudden regression between master and export requires inspection.

### 6. Implement one projection in Babylon

Read [references/babylon-implementation.md](references/babylon-implementation.md) and follow its version-compatible patterns.

At minimum:

- use the equirectangular texture directly on `PhotoDome` or one inward-facing sphere;
- use the complete sphere latitude range; set `slice: 1` explicitly when using `CreateSphere`;
- wrap longitude (`U`) and clamp latitude (`V`);
- keep dome material unaffected by scene fog;
- load 4K by default;
- load 8K only for high-density desktop viewports whose `maxTextureSize >= 8192`;
- cap Babylon render density at `2×` using the inverse `hardwareScalingLevel`;
- show a poster or loading state until a fully rendered frame is ready;
- use a moderately wide camera, commonly `1.25–1.35` radians vertically, then inspect edges for fisheye-like stretching.

Treat texture row `v = 0.5` as the geometric horizon. For a camera `h` units above a flat floor and a floor blend at radius `r`, calculate the visible floor boundary instead of eyeballing it:

```text
downward angle = atan(h / r)
panorama v     = 0.5 + atan(h / r) / pi
pixel row      = panorama v * image height
```

Calculate both fade radii. The resulting narrow row interval is the physical handoff band that should contain low-contrast, color-matched ground. Add only a modest art-direction safety margin. Read [references/babylon-implementation.md](references/babylon-implementation.md) for the worked example, partial-sphere failure formula, and movement rules.

### 7. Blend the floor into the panorama

Sample the dominant near-ground panorama color into explicit floor and haze tokens. Build the player floor as real textured geometry, but extend its visual surface well beyond the playable radius.

Use all three layers:

1. low-contrast tileable albedo/normal/roughness textures;
2. distance fog that removes texture contrast before the dome wall;
3. a radial opacity feather that reveals the panorama ground before the mesh edge.

The panorama should contain open matching ground, not another visible disc. Remove glowing rims and hard platform edges unless the art direction explicitly requires them.

For walking levels, keep close scenery as 3D props and the panorama as middle/far scenery. A panorama has no translational parallax, so increasing the authored “empty” foreground merely makes the world feel distant; it does not make nearby backdrop rocks behave correctly while the player moves.

### 8. Verify through the real browser

Do not approve from the flat source image alone.

- Capture desktop DPR1, desktop DPR2, tablet, mobile DPR1, and mobile DPR2.
- Confirm desktop DPR2 requests 8K and mobile requests 4K.
- Confirm the drawing buffer reaches the capped DPR size.
- Rotate repeatedly through the longitude seam and capture it centered.
- Check the horizon, zenith, nadir/floor junction, and near objects for pinching or blur.
- At an ultrawide viewport, verify that the floor junction remains level across the frame. A pronounced U-shaped transition usually means latitude compression from `slice < 1`, not insufficient empty ground in the image.
- Drag away, reset, and compare the reset frame with the initial frame.
- Inspect console and asset requests for texture or WebGL failures.
- Verify no black frame, alpha hole, duplicate platform, hard floor arc, clipped controls, or text overlap.
- Re-capture every viewport after the last rendered-source edit.

## Completion contract

Finish only when:

- every panorama asset is exact `2:1` and documented;
- every full equirectangular asset uses a full sphere (`slice: 1`), with a regression check when the sphere configuration is code-driven;
- the seam is clean in the rotated browser view, not only in an edge metric;
- the high-density desktop path demonstrably loads 8K and renders at capped DPR;
- mobile demonstrably uses the 4K fallback;
- the floor transition is visually continuous;
- tests, type checks, lint, and production build pass or unrelated pre-existing warnings are named;
- fresh screenshots and an independent visual review approve the final build.

## Bundled resources

- `scripts/shift_longitude.py`: rotate a panorama by half its width for centered seam editing.
- `scripts/feather_longitude.py`: apply a narrow symmetric longitude-edge feather without image-library dependencies.
- `scripts/upscale_realesrgan.py`: invoke and verify a Real-ESRGAN file upscale.
- `scripts/export_variants.py`: export exact 4K and 8K WebP variants with ffmpeg.
- `scripts/validate_panorama.py`: verify projection dimensions and quantify boundary continuity.
- [references/image-pipeline.md](references/image-pipeline.md): generation, repair, upscaling, export, and troubleshooting.
- [references/babylon-implementation.md](references/babylon-implementation.md): Babylon implementation and floor-blending patterns.
