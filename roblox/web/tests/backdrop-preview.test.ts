import { describe, expect, test } from "bun:test";
import {
  backdropIdFromSearchParams,
  backdropPresentationForState,
  backdropShell,
  iceArenaLayout,
  iceFloorTextureUrls,
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

  test("Given the ice arena floor, when its material loads, then it uses the complete tileable texture set", () => {
    expect(iceFloorTextureUrls).toEqual({
      albedo: "/assets/backdrops/ice-floor-albedo-v1.webp",
      normal: "/assets/backdrops/ice-floor-normal-v1.webp",
      roughness: "/assets/backdrops/ice-floor-roughness-v1.webp",
    });
  });

  test("Given the ice backdrop, when its shell renders, then it exposes labelled interactive controls", () => {
    const html = backdropShell();

    expect(html).toContain('id="backdrop-canvas"');
    expect(html).toContain(
      'aria-label="Interactive 360-degree ice mountain backdrop"',
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
      status: "Drag across the scene to look around.",
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
