# Babylon implementation patterns

## Contents

1. [Select PhotoDome or an inward sphere](#select-photodome-or-an-inward-sphere)
2. [Preserve full-sphere latitude](#preserve-full-sphere-latitude)
3. [Build an explicit inward sphere](#build-an-explicit-inward-sphere)
4. [Select 4K or 8K](#select-4k-or-8k)
5. [Render at capped device density](#render-at-capped-device-density)
6. [Blend a real floor](#blend-a-real-floor)
7. [Design for player movement](#design-for-player-movement)
8. [Prevent black loading frames](#prevent-black-loading-frames)
9. [Verify runtime behavior](#verify-runtime-behavior)

## Select PhotoDome or an inward sphere

Use the API already established by the project when it exposes the controls needed by the task.

Use `PhotoDome` for a simple panorama with default lifecycle behavior. Pass the original equirectangular image directly. Confirm constructor options against the installed `@babylonjs/core` version rather than copying options from a different release.

Use an inward-facing sphere when the implementation needs:

- adaptive asset selection before texture construction;
- explicit wrap and sampling modes;
- a material excluded from scene fog;
- custom load/error callbacks;
- predictable mesh placement and disposal;
- other geometry under direct control.

Do not nest PhotoDome and another textured sphere for the same panorama. Do not pre-warp the image for either path.

## Preserve full-sphere latitude

A strict 2:1 equirectangular image encodes the complete `180°` latitude range. Map it to a complete sphere. With Babylon `CreateSphere`, set `slice: 1` explicitly and lock that value with a regression test when it comes from application configuration.

A partial sphere is not a harmless crop. Babylon redistributes the full texture height over the remaining spherical arc. For sphere slice fraction `s`, texture equator `v = 0.5` is lifted above the geometric equator by:

```text
equator error = (0.5 - 0.5 * s) * 180 degrees
```

For `slice: 0.82`, the painted horizon moves `16.2°` upward. At wide aspect ratios this appears as a strong U-shaped ground wall. Fix the sphere geometry; do not add more empty foreground, move the camera, or pre-warp the image to counteract it.

If the application intentionally needs a cropped dome, crop or remap the source latitude deliberately and document the projection. Do not feed a full equirectangular master to a partial sphere.

## Build an explicit inward sphere

Use granular Babylon imports and adjust only version-specific signatures:

```ts
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import { Color3 } from "@babylonjs/core/Maths/math.color";
import { CreateSphere } from "@babylonjs/core/Meshes/Builders/sphereBuilder";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import type { Scene } from "@babylonjs/core/scene";

export function buildPanoramaSphere(scene: Scene, url: string): void {
  const material = new StandardMaterial("panorama-material", scene);
  const texture = new Texture(url, scene, {
    invertY: false,
    noMipmap: false,
    samplingMode: Texture.TRILINEAR_SAMPLINGMODE,
  });

  texture.wrapU = Texture.WRAP_ADDRESSMODE;
  texture.wrapV = Texture.CLAMP_ADDRESSMODE;
  material.backFaceCulling = false;
  material.diffuseTexture = texture;
  material.disableLighting = true;
  material.emissiveColor = Color3.White();
  material.emissiveTexture = texture;
  material.fogEnabled = false;

  const sphere = CreateSphere(
    "panorama-sphere",
    {
      diameter: 120,
      segments: 64,
      slice: 1,
      sideOrientation: Mesh.BACKSIDE,
    },
    scene,
  );
  sphere.material = material;
}
```

If the image appears vertically flipped, verify the asset orientation and installed Babylon texture constructor before changing `invertY`. Avoid compensating with arbitrary rotations until the seam and horizon have been checked.

Place the sphere at the intended stationary viewpoint. Keep interactive camera translation limited; a panorama has no true positional parallax.

## Select 4K or 8K

Use a pure selector so automated tests can cover memory policy without constructing WebGL:

```ts
export const panoramaUrls = {
  standard: "/assets/environment-4k.webp",
  highDetail: "/assets/environment-8k.webp",
} as const;

type RenderProfile = {
  readonly devicePixelRatio: number;
  readonly maxTextureSize: number;
  readonly viewportWidth: number;
};

export function panoramaUrlForProfile(profile: RenderProfile): string {
  const supportsHighDetail =
    profile.devicePixelRatio > 1 &&
    profile.viewportWidth >= 1_024 &&
    profile.maxTextureSize >= 8_192;

  return supportsHighDetail
    ? panoramaUrls.highDetail
    : panoramaUrls.standard;
}
```

Resolve the URL when the canvas and engine capabilities are available:

```ts
const url = panoramaUrlForProfile({
  devicePixelRatio: window.devicePixelRatio,
  maxTextureSize: engine.getCaps().maxTextureSize,
  viewportWidth: canvas.clientWidth,
});
```

Do not select 8K solely because a device reports a large texture limit. Viewport size and DPR determine whether the extra memory produces visible benefit.

## Render at capped device density

Babylon's `hardwareScalingLevel` is inverse: `1` renders at CSS-pixel density and `0.5` renders at 2× density.

```ts
function cappedRenderScale(devicePixelRatio: number): number {
  return Math.min(2, Math.max(1, devicePixelRatio));
}

const engine = new Engine(
  canvas,
  true,
  {
    powerPreference: "high-performance",
    preserveDrawingBuffer: false,
    stencil: false,
  },
  true,
);

function applyRenderScale(): void {
  engine.setHardwareScalingLevel(
    1 / cappedRenderScale(window.devicePixelRatio),
  );
}

applyRenderScale();
const resizeObserver = new ResizeObserver(() => {
  applyRenderScale();
  engine.resize();
});
resizeObserver.observe(canvas);
```

On disposal, disconnect the observer, detach camera controls, stop listeners, and dispose the engine.

## Blend a real floor

Separate gameplay dimensions from visual blending dimensions:

```ts
const arenaLayout = {
  playableDiameter: 24,
  visualSurfaceDiameter: 116,
  floorHeight: 0.36,
  textureWorldSize: 8,
  fogStart: 18,
  fogEnd: 54,
  opacityFadeStart: 0.68,
  opacityFadeEnd: 0.98,
} as const;
```

Use `playableDiameter` for movement/collision logic. Use the much larger `visualSurfaceDiameter` for the rendered ground skirt. Maintain the original tile scale:

```ts
const repeat =
  arenaLayout.visualSurfaceDiameter / arenaLayout.textureWorldSize;
albedoTexture.uScale = repeat;
albedoTexture.vScale = repeat;
```

Configure linear scene fog with a haze color sampled from the panorama's lower near-field. Keep the panorama material's `fogEnabled` false so only real geometry loses contrast.

Create a radial opacity texture using `DynamicTexture`:

```ts
const size = 256;
const opacity = new DynamicTexture(
  "floor-opacity",
  { width: size, height: size },
  scene,
  false,
  Texture.BILINEAR_SAMPLINGMODE,
);
const context = opacity.getContext();
const radius = size / 2;
const gradient = context.createRadialGradient(
  radius,
  radius,
  radius * arenaLayout.opacityFadeStart,
  radius,
  radius,
  radius * arenaLayout.opacityFadeEnd,
);
gradient.addColorStop(0, "white");
gradient.addColorStop(1, "black");
context.fillStyle = gradient;
context.fillRect(0, 0, size, size);
opacity.gammaSpace = false;
opacity.getAlphaFromRGB = true;
opacity.update(false);
```

Attach it to a PBR material in alpha-blend mode. Disable specular-over-alpha when it creates a bright edge. Ensure the opacity fade starts beyond the playable radius and finishes before the visual mesh reaches the dome wall.

Calculate which panorama rows meet the floor. With image-space `v = 0` at the zenith, `v = 0.5` at the geometric horizon, camera height `h` above the floor, and radial distance `r` from the camera:

```text
downward angle = atan(h / r)
v(r)           = 0.5 + atan(h / r) / pi
row(r)         = v(r) * panoramaHeight
```

For example, with `h = 1.6`, fade radii `56–68.6`, and a 4K panorama whose height is `2048`, the floor handoff spans approximately `v = 0.50909` to `v = 0.50742`, or rows `1042.6` to `1039.2`. At 8K height `4096`, it spans approximately rows `2085.2` to `2078.4`. Author low-contrast matching ground through this interval plus a modest safety margin. This is usually tens of rows, not a large fraction of the image.

The formula defines the physical handoff, not the first allowable rock row. Keep silhouettes and high-contrast bases outside the handoff plus the chosen safety margin, then confirm in the real camera because fog, alpha, camera pitch, and uneven geometry affect the perceived blend.

The floor's albedo should remain low contrast. Let the panorama's near rocks and distant mountains carry most environmental detail.

## Design for player movement

An equirectangular panorama is captured from one point. Rotation is correct; translation cannot produce true parallax.

- Keep the panorama centered on the intended viewing region and limit player displacement relative to the apparent distance of backdrop features.
- Put close rocks, crystals, trees, and other objects that should shift during walking into the 3D scene.
- Use the panorama for featureless near-ground continuation, middle-distance terrain, mountains, and sky.
- Size the real floor and its fade for the movement envelope so the player cannot reach or see its hard edge.
- Recalculate the handoff for the highest and lowest permitted camera heights. Use the worst-case row interval when authoring the quiet ground band.
- Test from the center and movement extremes. If nearby panorama objects appear glued to the camera, move them into geometry instead of increasing the empty band.

## Prevent black loading frames

Render an inline or small raster poster behind the canvas until the high-resolution texture has loaded and the scene has produced one complete frame.

Use texture callbacks to distinguish load and error states. Reveal the canvas after `scene.executeWhenReady` and an `onAfterRenderObservable.addOnce` callback rather than immediately after the network request completes.

Keep accessible status text such as:

- `Loading panorama…`
- `Drag across the scene to look around.`
- `The panorama could not be loaded.`

## Verify runtime behavior

Collect these values in a real browser:

```ts
const canvas = document.querySelector("canvas");
const gl = canvas?.getContext("webgl2") ?? canvas?.getContext("webgl");

const evidence = {
  dpr: window.devicePixelRatio,
  client: [canvas?.clientWidth, canvas?.clientHeight],
  drawingBuffer: [canvas?.width, canvas?.height],
  maxTextureSize: gl?.getParameter(gl.MAX_TEXTURE_SIZE),
  resources: performance
    .getEntriesByType("resource")
    .map((entry) => entry.name)
    .filter((name) => name.includes("/assets/")),
};
```

Require the following evidence:

- desktop DPR2: drawing buffer approximately 2× CSS dimensions and 8K requested when supported;
- mobile DPR2: drawing buffer capped at 2× but 4K requested;
- zero texture, WebGL, and JavaScript console errors;
- floor albedo, normal, and roughness assets all requested;
- ready state reached after a rendered frame;
- repeated horizontal drags reveal no longitude seam;
- the texture equator remains on the geometric horizon and the sphere uses `slice: 1`;
- the floor junction stays approximately level across an ultrawide frame rather than bowing upward at the sides;
- movement extremes do not reveal the floor mesh edge or implausible near-object parallax;
- reset restores the accepted framing;
- screenshots remain fully composited at every viewport.
