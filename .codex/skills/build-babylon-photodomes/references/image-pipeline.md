# Equirectangular image pipeline

## Contents

1. [Choose the source path](#choose-the-source-path)
2. [Generate a master](#generate-a-master)
3. [Repair the seam](#repair-the-seam)
4. [Reconstruct and upscale detail](#reconstruct-and-upscale-detail)
5. [Feather and export](#feather-and-export)
6. [Interpret quality-control metrics](#interpret-quality-control-metrics)
7. [Troubleshoot common failures](#troubleshoot-common-failures)

## Choose the source path

Use one of these paths:

- **New stylized environment:** generate a strict 2:1 equirectangular concept with enough open ground for the separate 3D floor.
- **Accepted but soft panorama:** use image editing to reconstruct material detail while preserving composition, then upscale.
- **Real 360 photograph:** preserve the original stitch geometry; denoise or upscale without generative changes unless the user explicitly wants them.
- **Ordinary perspective image:** do not stretch it to 2:1 and call it equirectangular. Generate or stitch missing longitude and poles with a geometry-aware workflow.

Keep lossless PNG intermediates until the final runtime WebP export.

## Generate a master

Use this prompt skeleton for a new environment:

```text
Use case: production 360-degree game environment panorama
Asset type: strict 2:1 equirectangular master for projection on the inside of a Babylon.js sphere

Create [ENVIRONMENT]. Keep the player viewpoint at the center of an open [GROUND MATERIAL] basin. Keep the immediate near-ground quiet and color-matched for a separate 3D floor. Place detailed [ROCKS / ICE / VEGETATION / ARCHITECTURE] from the middle distance outward, with progressively softer and lower-contrast distant layers. If the player will walk, reserve truly close objects for 3D geometry.

Projection requirements: exact 2:1 equirectangular composition, full 360-degree longitude, seamless left/right boundary, smooth zenith and nadir, continuous horizon, no perspective frame, no visible vertical seam.

Do not include a circular platform, baked gameplay disc, people, characters, camera, text, logos, UI, borders, black caps, or transparency.
```

Use this prompt skeleton to reconstruct an accepted source:

```text
Use case: stylized environment reconstruction
Asset type: high-detail 360-degree game panorama edit
Input image: composition, palette, horizon, silhouette, and object-placement reference

Reconstruct the input as a substantially sharper panoramic master. Preserve every large silhouette, the open center, horizon height, palette, lighting direction, and near/middle/far depth organization. Add believable fine detail to rock, ice, snow, ground, mountains, and clouds without adding or moving major objects.

Maintain a strict 2:1 equirectangular layout, seamless left/right boundary, smooth poles, and continuous lower ground. Do not add a platform, text, UI, characters, or a second horizon.
```

Image generators may produce a 2:1 file whose content is still perspective-like. Reject these symptoms before upscaling:

- duplicated landmarks near both edges;
- objects cut by one edge without continuing at the other;
- converging perspective that implies a single forward-facing camera;
- a pinched or detailed zenith that collapses when mapped to a sphere;
- a baked oval or circular floor.

Do not guess a large empty foreground percentage. First preserve the image equator as the true horizon, then calculate the narrow floor handoff band from camera height and visual-floor fade radii using [babylon-implementation.md](babylon-implementation.md). Extra blank ground beyond the calculated band is an art-direction choice and can make the scenery feel unnecessarily distant.

## Repair the seam

Run `shift_longitude.py` to exchange the left and right halves. This moves the wrap boundary to the image center without changing equirectangular geometry.

Edit the shifted image with a narrow-scope prompt:

```text
Use case: precise panorama seam repair
Asset type: equirectangular longitude seam-repair pass
Input image: the original wrap boundary has been shifted to the vertical center

Repair only the narrow vertical center seam so terrain, horizon, cloud bands, lighting, color, and texture continue naturally across it. Preserve the exact scene, composition, silhouettes, and palette everywhere outside the center repair band. Do not add, remove, or move objects. Do not change the horizon or ground height. Keep the image strict 2:1 equirectangular with smooth poles.
```

Shift the edited image again to restore the original longitude. Inspect both the flat edges and a rotated in-dome view.

Use edge feathering only for small pixel/color discrepancies. A 32–64 pixel feather is usually enough at 4K–8K. If a mountain or boulder changes shape across the seam, repeat the centered semantic repair instead.

## Reconstruct and upscale detail

Follow this order:

1. Fix projection, composition, and semantic seam continuity.
2. Reconstruct meaningful surface detail with image editing when the source is visibly soft.
3. Upscale the lossless result with Real-ESRGAN.
4. Inspect and, if necessary, apply a final narrow edge feather.
5. Downsample or slightly upsample to exact runtime dimensions with high-quality Lanczos filtering.

Do not reverse steps 1 and 2. Upscaling a broken seam makes the defect sharper and more expensive to repair.

The tested portable Real-ESRGAN command is:

```bash
./realesrgan-ncnn-vulkan \
  -i detailed-master.png \
  -o reconstructed.png \
  -n realesrgan-x4plus \
  -s 4 \
  -t 256 \
  -f png
```

Use the official Real-ESRGAN repository and releases: <https://github.com/xinntao/Real-ESRGAN>. Treat `v0.2.5.0` as a tested example, not a promise that it is the newest release. Ask before downloading a binary or model when network access or installation needs approval.

Model selection guidance:

- Prefer `realesrgan-x4plus` for realistic or semi-stylized environments.
- Test an anime model only for strongly cel-shaded line art; it can flatten snow and rock texture.
- Reduce tile size when GPU memory is limited.
- Keep the scale at the model's supported value and resize to exact deliverable dimensions afterward.

Real-ESRGAN can sharpen plausible edges but cannot know where the 360 seam, poles, or horizon should be. Always repeat seam and spherical checks after upscaling.

## Feather and export

Apply the feather to the high-detail lossless intermediate:

```bash
python scripts/feather_longitude.py reconstructed.png seam-safe.png --pixels 64
```

The script blends corresponding pixels inward from the two longitude edges. At the boundary the pixels become identical; the blend smoothly reaches zero at the inner edge of the feather.

Normalize the master to sRGB before this step. The feather script preserves RGB pixels but intentionally does not preserve arbitrary ICC profiles or ancillary metadata.

Export runtime variants:

```bash
python scripts/export_variants.py seam-safe.png output/environment
```

Default outputs:

- `environment-4k.webp`: `4096 × 2048`, WebP quality 92.
- `environment-8k.webp`: `8192 × 4096`, WebP quality 90.

Use 4K as the safe default. Make 8K conditional because an 8K panorama has four times the pixels of 4K and can exceed mobile memory budgets despite a small compressed file size.

Record:

- generation and edit prompts;
- source and export dimensions;
- seam repair method and feather width;
- Real-ESRGAN version, model, scale, and tile size;
- ffmpeg export settings;
- SHA-256 for source, 4K, and 8K files;
- license or provenance for non-generated sources.

## Interpret quality-control metrics

Run `validate_panorama.py` before and after each lossy or reconstructive operation.

It reports:

- `strict_2_to_1`: the hard projection gate;
- `boundary_mae`: grayscale mean absolute error between the outermost left and right columns;
- `strip_mae`: grayscale mean absolute error between mirrored edge strips, measuring how similarly content develops away from the seam.

Use metrics comparatively:

- A low boundary MAE with a visible duplicated rock is still a failure.
- A higher strip MAE can be valid when the immediate boundary connects but nearby composition differs.
- A large jump after WebP export indicates compression or scaling damage near the seam.
- A clean flat-image metric never replaces rotating through the seam inside Babylon.

## Troubleshoot common failures

| Symptom | Likely cause | Correction |
| --- | --- | --- |
| Dome looks blurry | Low semantic detail, low runtime texture, DPR1 drawing buffer, or browser downscale | Reconstruct detail, upscale the lossless master, add adaptive 8K, and verify drawing-buffer dimensions |
| Dome looks zoomed in | Camera FOV too narrow | Test roughly `1.25–1.35` radians vertically and inspect edge distortion |
| Scene looks fisheye | FOV too wide or source is not truly equirectangular | Reduce FOV and fix source projection; do not pre-warp again |
| Vertical line appears when rotating | Longitude edges differ | Shift seam to center, repair semantically, shift back, then feather narrowly |
| Top or bottom pinches | Detailed/incorrect pole content | Regenerate or edit smooth poles; avoid high-frequency objects at zenith/nadir |
| Floor looks like a floating disc | Hard mesh edge, color mismatch, or baked second platform | Remove rim, match floor/haze colors, extend surface, add fog and radial opacity fade |
| Floor becomes transparent too close | Fade starts inside playable radius | Start opacity fade well beyond playable area and finish before the dome wall |
| Ground transition forms a U-shaped bowl | Full equirectangular texture mapped to a partial sphere | Set Babylon sphere `slice: 1`; do not regenerate or add empty ground until latitude mapping is correct |
| Scenery feels too far away after fixing the floor | Arbitrary oversized empty foreground band | Restore the composition and reserve only the calculated quiet handoff band; use 3D geometry for truly close objects |
| 8K works on desktop but crashes mobile | Decoded texture and drawing-buffer memory | Force 4K below desktop width, cap DPR, and check `maxTextureSize` |
| Upscale looks waxy or haloed | Wrong model or excessive reconstruction | Compare models, preserve lossless source, or reduce generative strength |
