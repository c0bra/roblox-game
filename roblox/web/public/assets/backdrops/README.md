# Backdrop textures

`ice-mountains-equirectangular-v5-source.png` is the project-local
image-generation source for the centered Babylon arena preview. It preserves the
open basin from v4, reconstructs sharper rock, ice, snow, mountain, and cloud
detail, and uses a strict 2:1 equirectangular layout. The longitude seam was
rotated into the center for an image-generation repair pass, rotated back, then
given a narrow symmetric edge feather before export.

The runtime uses `ice-mountains-equirectangular-v5-4k.webp` by default and
selects `ice-mountains-equirectangular-v5-8k.webp` only on high-density desktop
viewports whose WebGL texture limit supports it. The source was reconstructed 4x
with the official Real-ESRGAN `realesrgan-x4plus` model before the exact 4K and
8K exports were encoded. The real 3D floor is not baked into this image; Babylon
renders it separately with the shared `--ice-floor` design token.

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
  -i ice-mountains-equirectangular-v5-source.png \
  -o ice-mountains-equirectangular-v5-reconstructed.png \
  -n realesrgan-x4plus -s 4 -t 256 -f png

# Apply the documented 64px symmetric longitude-edge feather to the
# reconstructed intermediate before these exports.
ffmpeg -i ice-mountains-equirectangular-v5-seam-safe.png \
  -vf scale=4096:2048:flags=lanczos+accurate_rnd+full_chroma_int \
  -c:v libwebp -preset picture -quality 92 -compression_level 6 \
  ice-mountains-equirectangular-v5-4k.webp

ffmpeg -i ice-mountains-equirectangular-v5-seam-safe.png \
  -vf scale=8192:4096:flags=lanczos+accurate_rnd+full_chroma_int \
  -c:v libwebp -preset picture -quality 90 -compression_level 6 \
  ice-mountains-equirectangular-v5-8k.webp

ffmpeg -i ice-floor-albedo-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 88 -compression_level 6 ice-floor-albedo-v1.webp
ffmpeg -i ice-floor-normal-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 92 -compression_level 6 ice-floor-normal-v1.webp
ffmpeg -i ice-floor-roughness-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 90 -compression_level 6 ice-floor-roughness-v1.webp
```

SHA-256:

- Panorama v5 source PNG: `dac8b356a8f788bca7811da3bfb46f7bdffafa3b7627a08669ba48f96f223daa`
- Panorama v5 4K WebP: `dca5d537394c7f989dc64e5582512fdbf7a612e858615639d2533ca3b0515ee4`
- Panorama v5 8K WebP: `c71cc8fe3ac26bcf2207ded8019b7300ccd63eb485549985a1f5ff46bb51273c`
- Floor albedo PNG: `206655d122075d482fe44110f895cf368f971d6048eb11d8ae24eb4c3819e2c9`
- Floor albedo WebP: `cd5e709ce7b711846228f6fdc0d73e3d6065ab90841a7281b2fbc9b8de1a7d05`
- Floor normal PNG: `eee8522bb254a2b6b5a77f730b2fcfb05dea99e4efa06512320b9e76275febd5`
- Floor normal WebP: `2be14b5a5d84a7418a7a7c8f263ce16bde9842a4a999bbbdff5756ce0c2b8aba`
- Floor roughness PNG: `cabd7b0b52f95cc71545012fa0aa25929b4d9058cf9383a361976661a733b202`
- Floor roughness WebP: `9255feccb863fea4a74bd2ef08b836935f0db808111c870d82c5e1ab2ed1f9c0`
