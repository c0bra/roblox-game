export type BackdropId = "ice";
export type BackdropViewState = "loading" | "ready" | "error";

export const iceArenaLayout = {
  cameraPosition: { x: 0, y: 0, z: 0 },
  cameraFov: 1.3,
  floorCenter: { x: 0, y: -1.78, z: 0 },
  floorColorToken: "--ice-floor",
  floorDiameter: 24,
  floorHeight: 0.36,
  surfaceDiameter: 116,
  textureWorldSize: 8,
  hazeColorToken: "--ice-haze",
  fogStart: 18,
  fogEnd: 54,
  albedoLift: 1.12,
  emissiveStrength: 0.2,
  opacityFadeStart: 0.68,
  opacityFadeEnd: 0.98,
} as const;

export const icePanoramaUrls = {
  standard: "/assets/backdrops/ice-mountains-equirectangular-v5-4k.webp",
  highDetail: "/assets/backdrops/ice-mountains-equirectangular-v5-8k.webp",
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
        status: "Drag across the scene to look around.",
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
        aria-label="Interactive 360-degree ice mountain backdrop"
      ></canvas>
      <div class="backdrop-vignette" aria-hidden="true"></div>
      <header class="backdrop-copy">
        <p class="backdrop-kicker"><span></span>Environment test</p>
        <h1 id="backdrop-title">Ice Mountain <em>Backdrop</em></h1>
        <p>Stand at the center of the ice floor. Drag to inspect the near rock ring and distant mountain layers.</p>
      </header>
      <section class="backdrop-controls" aria-label="Backdrop controls">
        <div class="backdrop-spec">
          <span>Projection</span>
          <strong>Centered 360° arena</strong>
        </div>
        <button id="backdrop-reset" type="button">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 8V4m0 0h4M4 4l3.4 3.4A7 7 0 1 1 5 15" />
          </svg>
          Reset view
        </button>
      </section>
      <div class="backdrop-reticle" aria-hidden="true"></div>
      <p id="backdrop-status" class="backdrop-status" role="status">Loading panorama…</p>
    </main>
  `;
}
