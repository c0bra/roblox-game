# Backdrop textures

`ice-mountains-equirectangular-v4.png` is the project-local image-generation
source for the centered Babylon arena preview. It was generated as a strict 2:1
icy panorama with nearby slate boulders and fractured ice around an open center,
then progressively softer glacier valleys and distant mountain layers.

The runtime uses `ice-mountains-equirectangular-v4.webp`. It is resized and
encoded with the command below. The real 3D floor is not baked into this image;
Babylon renders it separately with the shared `--ice-floor` design token.

`ice-floor-albedo-v1.png`, `ice-floor-normal-v1.png`, and
`ice-floor-roughness-v1.png` are the project-local generated sources for that 3D
floor. The albedo is a flat top-down frost-and-crack pattern; the tangent-space
normal and grayscale roughness maps preserve the same feature layout. Babylon
tiles each runtime texture 3 × 3 with wrap addressing, then multiplies the
albedo by `--ice-floor` so the material stays within the environment palette.

Generation prompt set:

- Albedo: seamless top-down square frosted ice, cloudy frozen layers,
  wind-brushed frost, fine branching hairline cracks, sparse snow dusting,
  neutral grayscale, flat shadowless albedo, no focal point or baked lighting.
- Normal: convert the exact albedo layout to a tangent-space normal map with
  shallow recessed cracks, subtle granular frost, and nearly flat broad ice.
- Roughness: convert the exact layout to grayscale roughness, with powder frost
  light/high-roughness and clear ice mid-gray, using no crushed blacks.

```bash
ffmpeg -i ice-mountains-equirectangular-v4.png \
  -vf scale=2048:1024 \
  -c:v libwebp -quality 86 ice-mountains-equirectangular-v4.webp

ffmpeg -i ice-floor-albedo-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 88 -compression_level 6 ice-floor-albedo-v1.webp
ffmpeg -i ice-floor-normal-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 92 -compression_level 6 ice-floor-normal-v1.webp
ffmpeg -i ice-floor-roughness-v1.png -vf scale=1024:1024 \
  -c:v libwebp -quality 90 -compression_level 6 ice-floor-roughness-v1.webp
```

SHA-256:

- Source PNG: `dddeb81b7088c621ae4bb3e973d4c7f8e2c2f79155614e46094f88f6c886eb7e`
- Runtime WebP: `41e7e2da012eed642157d0ffb87a784b2f5204bde09e11dae74aac5bc52f87e6`
- Floor albedo PNG: `206655d122075d482fe44110f895cf368f971d6048eb11d8ae24eb4c3819e2c9`
- Floor albedo WebP: `cd5e709ce7b711846228f6fdc0d73e3d6065ab90841a7281b2fbc9b8de1a7d05`
- Floor normal PNG: `eee8522bb254a2b6b5a77f730b2fcfb05dea99e4efa06512320b9e76275febd5`
- Floor normal WebP: `2be14b5a5d84a7418a7a7c8f263ce16bde9842a4a999bbbdff5756ce0c2b8aba`
- Floor roughness PNG: `cabd7b0b52f95cc71545012fa0aa25929b4d9058cf9383a361976661a733b202`
- Floor roughness WebP: `9255feccb863fea4a74bd2ef08b836935f0db808111c870d82c5e1ab2ed1f9c0`
