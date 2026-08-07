# Backdrop textures

`ice-mountains-equirectangular-v6-source.png` is the current project-local
image-generation source for the centered Babylon arena preview. It keeps the
strict 2:1 projection and mountain palette from v5, but moves the first rock ring
into the middle distance and reserves the lower 46% of the image for low-contrast
ice. That ground reserve gives the separate Babylon floor room to fade before a
dark silhouette can reach the junction.

The floor-to-panorama band is derived from the scene geometry rather than an
eyeballed crop. With a 1.6-unit camera height, a 70-unit visual floor radius, and
opacity-fade radii of 56–68.6 units, a downward ray maps through
`v = 0.5 + atan2(height, radius) / pi`. The exact fade occupies normalized
latitude `v = 0.50742–0.50909`. The v6 art reserve starts at `v = 0.54`, leaving
a normalized 0.03091 safety band: about 63 runtime rows at 4K or 127 at 8K before
the first rock base.

The runtime uses `ice-mountains-equirectangular-v6-4k.webp` by default and
selects `ice-mountains-equirectangular-v6-8k.webp` only on high-density desktop
viewports whose WebGL texture limit supports it. The source was reconstructed 4x
with the official Real-ESRGAN `realesrgan-x4plus` model, given a 64px symmetric
longitude-edge feather, then encoded at exact 4K and 8K dimensions. The real 3D
floor is not baked into this image; Babylon renders it separately with the shared
`--ice-floor` design token. The v5 files remain as the accepted pre-clearance
baseline.

`ice-floor-albedo-v1.png`, `ice-floor-normal-v1.png`, and
`ice-floor-roughness-v1.png` are the project-local generated sources for that 3D
floor. The albedo is a flat top-down frost-and-crack pattern; the tangent-space
normal and grayscale roughness maps preserve the same feature layout. Babylon
keeps the pattern at an 8-unit world scale with wrap addressing, so the 24-unit
playable footprint contains the original 3 × 3 repeat. The matching visual
ground continues beyond that footprint, then loses texture contrast through
`--ice-haze` distance fog and a radial opacity feather before it reaches the
panorama wall. The albedo remains multiplied by `--ice-floor` so the near field
stays within the environment palette without exposing a platform edge.

Generation prompt set:

- Panorama v6 transition edit: preserve sky, clouds, horizon, mountains, palette,
  and lighting; remove foreground objects; keep every rock/crystal base at or
  above 54% image height; reserve the bottom 46% as uninterrupted low-contrast
  glacier ice; maintain strict 2:1 equirectangular geometry and wrap continuity.
- Panorama v6 ring pass: strengthen only spaced middle-distance dark rock and
  translucent ice clusters along the horizon while keeping the complete lower
  46% object-free; prohibit a continuous barricade, platform, rim, or cliff.
- Panorama seam repair: repair only the narrow vertical discontinuity introduced
  by the generator while preserving the open-ground reserve and all established
  silhouettes outside the repair band.

- Albedo: seamless top-down square frosted ice, cloudy frozen layers,
  wind-brushed frost, fine branching hairline cracks, sparse snow dusting,
  neutral grayscale, flat shadowless albedo, no focal point or baked lighting.
- Normal: convert the exact albedo layout to a tangent-space normal map with
  shallow recessed cracks, subtle granular frost, and nearly flat broad ice.
- Roughness: convert the exact layout to grayscale roughness, with powder frost
  light/high-roughness and clear ice mid-gray, using no crushed blacks.

```bash
# Real-ESRGAN v0.2.5.0 portable build, run from its extracted directory.
./realesrgan-ncnn-vulkan \
  -i ice-mountains-equirectangular-v6-source.png \
  -o ice-mountains-equirectangular-v6-reconstructed.png \
  -n realesrgan-x4plus -s 4 -t 256 -f png

# Apply the documented 64px symmetric longitude-edge feather to the
# reconstructed intermediate before these exports.
ffmpeg -i ice-mountains-equirectangular-v6-seam-safe.png \
  -vf scale=4096:2048:flags=lanczos+accurate_rnd+full_chroma_int \
  -c:v libwebp -preset picture -quality 92 -compression_level 6 \
  ice-mountains-equirectangular-v6-4k.webp

ffmpeg -i ice-mountains-equirectangular-v6-seam-safe.png \
  -vf scale=8192:4096:flags=lanczos+accurate_rnd+full_chroma_int \
  -c:v libwebp -preset picture -quality 90 -compression_level 6 \
  ice-mountains-equirectangular-v6-8k.webp

ffmpeg -i ice-floor-albedo-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 88 -compression_level 6 ice-floor-albedo-v1.webp
ffmpeg -i ice-floor-normal-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 92 -compression_level 6 ice-floor-normal-v1.webp
ffmpeg -i ice-floor-roughness-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 90 -compression_level 6 ice-floor-roughness-v1.webp
```

SHA-256:

- Panorama v6 source PNG: `8d30af3e4debda3b07b2c8a393636705b4d39129aac620d7feff91da1068cb48`
- Panorama v6 4K WebP: `95aa98c9ebc7fb71f9072ac34b184ca3c0e62f39ffdadf56c89102c720eaf6a7`
- Panorama v6 8K WebP: `20fda61844c3ecb0d873706aa85a4625e1b4253361cfbe0d1c2f4c669d4dee34`
- Floor albedo PNG: `206655d122075d482fe44110f895cf368f971d6048eb11d8ae24eb4c3819e2c9`
- Floor albedo WebP: `cd5e709ce7b711846228f6fdc0d73e3d6065ab90841a7281b2fbc9b8de1a7d05`
- Floor normal PNG: `eee8522bb254a2b6b5a77f730b2fcfb05dea99e4efa06512320b9e76275febd5`
- Floor normal WebP: `2be14b5a5d84a7418a7a7c8f263ce16bde9842a4a999bbbdff5756ce0c2b8aba`
- Floor roughness PNG: `cabd7b0b52f95cc71545012fa0aa25929b4d9058cf9383a361976661a733b202`
- Floor roughness WebP: `9255feccb863fea4a74bd2ef08b836935f0db808111c870d82c5e1ab2ed1f9c0`
