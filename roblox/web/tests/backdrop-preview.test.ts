import { describe, expect, test } from "bun:test";
import {
  backdropIdFromSearchParams,
  backdropPresentationForState,
  backdropRenderScaleForDevicePixelRatio,
  backdropShell,
  iceArenaLayout,
  iceFloorTextureUrls,
  icePanoramaComposition,
  icePanoramaFloorTransition,
  icePanoramaUrlForProfile,
  icePanoramaUrls,
} from "../src/backdrop/backdrop-preview";

describe("backdrop preview", () => {
  test("Given the ice backdrop query, when parsed, then the ice preview is selected", () => {
    const params = new URLSearchParams("backdrop=ice");

    const result = backdropIdFromSearchParams(params);

    expect(result).toBe("ice");
  });

  test("Given an unknown backdrop query, when parsed, then no preview is selected", () => {
    const params = new URLSearchParams("backdrop=unknown");

    const result = backdropIdFromSearchParams(params);

    expect(result).toBeNull();
  });

  test("Given the ice arena, when its layout is configured, then the viewer stands at the center of the floor", () => {
    const floorTop =
      iceArenaLayout.floorCenter.y + iceArenaLayout.floorHeight / 2;

    expect(iceArenaLayout.cameraPosition.x).toBe(iceArenaLayout.floorCenter.x);
    expect(iceArenaLayout.cameraPosition.z).toBe(iceArenaLayout.floorCenter.z);
    expect(iceArenaLayout.cameraPosition.y).toBeGreaterThan(floorTop);
    expect(iceArenaLayout.floorDiameter).toBeGreaterThan(20);
  });

  test("Given the panoramic arena, when its camera is configured, then the lens presents a wide view without fisheye distortion", () => {
    expect(iceArenaLayout.cameraFov).toBeGreaterThan(1.2);
    expect(iceArenaLayout.cameraFov).toBeLessThan(1.4);
  });

  test("Given a full equirectangular panorama, when mapped to the backdrop sphere, then its equator remains on the geometric horizon", () => {
    expect(iceArenaLayout.panoramaSphereSlice).toBe(1);
  });

  test("Given the playable ice disc, when it meets the panorama, then its visual surface fades beyond the arena instead of ending at a hard edge", () => {
    const playableRadius = iceArenaLayout.floorDiameter / 2;
    const surfaceRadius = iceArenaLayout.surfaceDiameter / 2;

    expect(iceArenaLayout.surfaceDiameter).toBeGreaterThan(
      iceArenaLayout.floorDiameter * 4,
    );
    expect(iceArenaLayout.fogStart).toBeGreaterThan(playableRadius);
    expect(iceArenaLayout.fogEnd).toBeLessThan(surfaceRadius);
    expect(iceArenaLayout.cameraTravelRadius).toBe(10);
    expect(iceArenaLayout.fogEnd - iceArenaLayout.fogStart).toBeGreaterThan(
      playableRadius,
    );
    expect(iceArenaLayout.textureWorldSize).toBe(8);
    expect(iceArenaLayout.albedoLift).toBeGreaterThan(1);
    expect(iceArenaLayout.emissiveStrength).toBeGreaterThan(0.1);
    expect(iceArenaLayout.hazeColorToken).toBe("--ice-haze");
    expect(iceArenaLayout.opacityFadeStart).toBeGreaterThan(0.5);
    expect(iceArenaLayout.opacityFadeEnd).toBeLessThan(1);
    expect(iceArenaLayout.opacityFadeEnd).toBeGreaterThan(
      iceArenaLayout.opacityFadeStart,
    );
  });

  test("Given the floor fade geometry, when projected into equirectangular latitude, then its exact transition band remains below the horizon", () => {
    expect(icePanoramaFloorTransition.cameraHeight).toBeCloseTo(1.6);
    expect(icePanoramaFloorTransition.fadeStartRadius).toBeCloseTo(56);
    expect(icePanoramaFloorTransition.fadeEndRadius).toBeCloseTo(68.6);
    expect(icePanoramaFloorTransition.fadeEndV).toBeGreaterThan(0.5);
    expect(icePanoramaFloorTransition.fadeStartV).toBeGreaterThan(
      icePanoramaFloorTransition.fadeEndV,
    );
    expect(icePanoramaFloorTransition.fadeStartV).toBeLessThan(0.52);
  });

  test("Given the panorama composition, when the floor transition is projected, then the empty ground reserve leaves a visible safety band before the rocks", () => {
    expect(icePanoramaComposition.groundReserveFraction).toBe(0.46);
    expect(icePanoramaComposition.groundReserveStartV).toBeCloseTo(0.54);
    expect(icePanoramaComposition.transitionClearanceV).toBeGreaterThan(0.02);
  });

  test("Given the ice arena floor, when its material loads, then it uses the complete tileable texture set", () => {
    expect(iceFloorTextureUrls).toEqual({
      albedo: "/assets/backdrops/ice-floor-albedo-v1.webp",
      normal: "/assets/backdrops/ice-floor-normal-v1.webp",
      roughness: "/assets/backdrops/ice-floor-roughness-v1.webp",
    });
  });

  test("Given a high-density desktop with 8K texture support, when the panorama is selected, then the high-detail asset is used", () => {
    const result = icePanoramaUrlForProfile({
      devicePixelRatio: 2,
      maxTextureSize: 16_384,
      viewportWidth: 1_200,
    });

    expect(result).toBe(icePanoramaUrls.highDetail);
  });

  test("Given the ice panorama assets, when selected for runtime, then the projection-aware v6 variants are used", () => {
    expect(icePanoramaUrls).toEqual({
      standard: "/assets/backdrops/ice-mountains-equirectangular-v6-4k.webp",
      highDetail: "/assets/backdrops/ice-mountains-equirectangular-v6-8k.webp",
    });
  });

  test("Given a mobile viewport, when the panorama is selected, then the memory-safe 4K asset is used", () => {
    const result = icePanoramaUrlForProfile({
      devicePixelRatio: 3,
      maxTextureSize: 16_384,
      viewportWidth: 768,
    });

    expect(result).toBe(icePanoramaUrls.standard);
  });

  test("Given a dense display, when its render scale is resolved, then Retina detail is enabled without exceeding two times", () => {
    expect(backdropRenderScaleForDevicePixelRatio(2)).toBe(2);
    expect(backdropRenderScaleForDevicePixelRatio(3)).toBe(2);
    expect(backdropRenderScaleForDevicePixelRatio(0.75)).toBe(1);
  });

  test("Given the ice backdrop, when its shell renders, then it exposes labelled interactive controls", () => {
    const html = backdropShell();

    expect(html).toContain('id="backdrop-canvas"');
    expect(html).toContain('tabindex="0"');
    expect(html).toContain(
      'aria-label="Walkable 360-degree ice arena. Use WASD to move and drag to look around."',
    );
    expect(html).toContain('id="backdrop-reset"');
    expect(html).toContain('id="backdrop-status"');
  });

  test("Given Babylon is still loading, when the shell renders, then an ice panorama poster prevents a black frame", () => {
    const html = backdropShell();

    expect(html).toContain('class="backdrop-poster"');
    expect(html).toContain('src="data:image/jpeg;base64,');
  });

  test("Given the texture finishes loading, when view state changes, then drag guidance is announced", () => {
    const result = backdropPresentationForState("ready");

    expect(result).toEqual({
      rootState: "ready",
      status: "WASD to walk. Drag to look around.",
    });
  });

  test("Given the texture fails to load, when view state changes, then recovery status is announced", () => {
    const result = backdropPresentationForState("error");

    expect(result).toEqual({
      rootState: "error",
      status: "The panorama could not be loaded.",
    });
  });
});
