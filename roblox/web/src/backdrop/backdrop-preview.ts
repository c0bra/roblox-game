export type BackdropId = "ice";
export type BackdropViewState = "loading" | "ready" | "error";

export const iceArenaLayout = {
  cameraPosition: { x: 0, y: 0, z: 0 },
  cameraFov: 1.3,
  panoramaSphereSlice: 1,
  cameraSpeed: 0.16,
  cameraTravelRadius: 10,
  floorCenter: { x: 0, y: -1.78, z: 0 },
  floorColorToken: "--ice-floor",
  floorDiameter: 24,
  floorHeight: 0.36,
  surfaceDiameter: 140,
  textureWorldSize: 8,
  hazeColorToken: "--ice-haze",
  fogStart: 30,
  fogEnd: 58,
  albedoLift: 1.12,
  emissiveStrength: 0.2,
  opacityFadeStart: 0.8,
  opacityFadeEnd: 0.98,
} as const;

const floorTop = iceArenaLayout.floorCenter.y + iceArenaLayout.floorHeight / 2;
const cameraHeight = iceArenaLayout.cameraPosition.y - floorTop;
const surfaceRadius = iceArenaLayout.surfaceDiameter / 2;
const fadeStartRadius = surfaceRadius * iceArenaLayout.opacityFadeStart;
const fadeEndRadius = surfaceRadius * iceArenaLayout.opacityFadeEnd;

export const icePanoramaFloorTransition = {
  cameraHeight,
  fadeStartRadius,
  fadeEndRadius,
  fadeStartV: 0.5 + Math.atan2(cameraHeight, fadeStartRadius) / Math.PI,
  fadeEndV: 0.5 + Math.atan2(cameraHeight, fadeEndRadius) / Math.PI,
} as const;

const groundReserveFraction = 0.46;
const groundReserveStartV = 1 - groundReserveFraction;

export const icePanoramaComposition = {
  groundReserveFraction,
  groundReserveStartV,
  transitionClearanceV:
    groundReserveStartV - icePanoramaFloorTransition.fadeStartV,
} as const;

export const icePanoramaUrls = {
  standard: "/assets/backdrops/ice-mountains-equirectangular-v6-4k.webp",
  highDetail: "/assets/backdrops/ice-mountains-equirectangular-v6-8k.webp",
} as const;

type BackdropRenderProfile = {
  readonly devicePixelRatio: number;
  readonly maxTextureSize: number;
  readonly viewportWidth: number;
};

export function backdropRenderScaleForDevicePixelRatio(
  devicePixelRatio: number,
): number {
  return Math.min(2, Math.max(1, devicePixelRatio));
}

export function icePanoramaUrlForProfile(
  profile: BackdropRenderProfile,
): (typeof icePanoramaUrls)[keyof typeof icePanoramaUrls] {
  const supportsHighDetail =
    profile.devicePixelRatio > 1 &&
    profile.viewportWidth >= 1_024 &&
    profile.maxTextureSize >= 8_192;
  return supportsHighDetail
    ? icePanoramaUrls.highDetail
    : icePanoramaUrls.standard;
}

export const iceFloorTextureUrls = {
  albedo: "/assets/backdrops/ice-floor-albedo-v1.webp",
  normal: "/assets/backdrops/ice-floor-normal-v1.webp",
  roughness: "/assets/backdrops/ice-floor-roughness-v1.webp",
} as const;

const icePanoramaPosterUrl =
  "data:image/jpeg;base64,/9j//gAQTGF2YzYyLjExLjEwMAD/2wBDAAgKCgsKCw0NDQ0NDRAPEBAQEBAQEBAQEBASEhIVFRUSEhIQEBISFBQVFRcXFxUVFRUXFxkZGR4eHBwjIyQrKzP/xAB5AAEBAQEBAAAAAAAAAAAAAAAFBAMGAAEBAQEBAQEAAAAAAAAAAAAAAQIAAwUEEAACAQIDBgUDBQEBAAAAAAABAhEDACEEMRJhE0EigXFRBTIUkaHB8OFigrFC8REBAQEAAwEAAwEAAAAAAAAAAAERAiFhMXFRkRL/wAARCAAgAEADARIAAhIAAxIA/9oADAMBAAIRAxEAPwDoOKBvtJaOXZOIGAF/biu3i6rIL42676FNK3EiF2TzGo899ziqnTIjSoGNqHIwRiouMLaf8sOHIkRdgy2z/wBL4/r83JJwfE2k1FQPeO37WEKsEthaJyyxJZY33lIbBe2LqX4pDEPpyiNo/wAZswg5EcnzAvwzNCUUp7iwbAnZjTxndZisCsc96f6s2aJpuVWfbsKEII5xvG+bHy6vTpT8biVg2DOhBK/1Gu+Rel0R05cc9Xc/bt3T5ChS9VFwnpAnxbDDwwsrJ16604q0Xpx7IBaAeXTGE+YN0J+HGRd8rpKaAADiu+ESXk/W4gMy4DpVp051DIZ+kjAjkRO+8yc7VPS9OjRTQHUk9RxJ88ce9ycYUhLnaPPZUn6QCbCMUpq5WhVMsskGR1NgfMY4dryFYMJE/wCfYxYU4Wb0EIbqeCMes/8At5tVk6Hvp+bWRikiU6NFWC1oGOpVis89JHhZuaz2wWVMvmC2jVUpgEY6oSpndPa1P9T9dM9iXPeomjVCCGVFOLpO23I4YjZ+5sr1GlWzR4lMVjAAJZYJ8CYM9ovXqi7U8ZsXx6+3X//Z" as const;

export type BackdropPresentation = {
  readonly rootState: BackdropViewState;
  readonly status: string;
};

export function backdropIdFromSearchParams(
  params: URLSearchParams,
): BackdropId | null {
  return params.get("backdrop") === "ice" ? "ice" : null;
}

export function backdropPresentationForState(
  state: BackdropViewState,
): BackdropPresentation {
  switch (state) {
    case "loading":
      return { rootState: state, status: "Loading panorama…" };
    case "ready":
      return {
        rootState: state,
        status: "WASD to walk. Drag to look around.",
      };
    case "error":
      return {
        rootState: state,
        status: "The panorama could not be loaded.",
      };
    default:
      return state satisfies never;
  }
}

export function backdropShell(): string {
  return `
    <main class="backdrop-preview" data-state="loading" aria-labelledby="backdrop-title">
      <img
        class="backdrop-poster"
        src="${icePanoramaPosterUrl}"
        alt=""
        aria-hidden="true"
      />
      <canvas
        id="backdrop-canvas"
        tabindex="0"
        aria-label="Walkable 360-degree ice arena. Use WASD to move and drag to look around."
      ></canvas>
      <div class="backdrop-vignette" aria-hidden="true"></div>
      <header class="backdrop-copy">
        <p class="backdrop-kicker"><span></span>Environment test</p>
        <h1 id="backdrop-title">Ice Mountain <em>Backdrop</em></h1>
        <p>Walk the ice. Nearby rocks and crystals shift with your movement while the mountains stay distant.</p>
      </header>
      <section class="backdrop-controls" aria-label="Backdrop controls">
        <div class="backdrop-spec">
          <span>Movement</span>
          <strong>WASD · drag to look</strong>
        </div>
        <button id="backdrop-reset" type="button">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 8V4m0 0h4M4 4l3.4 3.4A7 7 0 1 1 5 15" />
          </svg>
          Reset position
        </button>
      </section>
      <div class="backdrop-reticle" aria-hidden="true"></div>
      <p id="backdrop-status" class="backdrop-status" role="status">Loading panorama…</p>
    </main>
  `;
}
