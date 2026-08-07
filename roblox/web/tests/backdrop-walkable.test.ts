import { describe, expect, test } from "bun:test";
import { icePerimeterClusters } from "../src/backdrop/backdrop-environment-layout";
import { clampBackdropPosition } from "../src/backdrop/backdrop-movement";
import { iceArenaLayout } from "../src/backdrop/backdrop-preview";

describe("walkable ice backdrop", () => {
  test("Given a camera inside the playable travel bound, when clamped, then its position is preserved", () => {
    const result = clampBackdropPosition({ x: 3, z: -4 }, 10);

    expect(result).toEqual({ x: 3, z: -4 });
  });

  test("Given a camera outside the playable travel bound, when clamped, then it remains on the circular boundary", () => {
    const result = clampBackdropPosition({ x: 12, z: 16 }, 10);

    expect(result.x).toBeCloseTo(6);
    expect(result.z).toBeCloseTo(8);
    expect(Math.hypot(result.x, result.z)).toBeCloseTo(10);
  });

  test("Given the walking radius and fog distance, when sizing the real floor, then no legal camera can reach an unfogged mesh edge", () => {
    const surfaceRadius = iceArenaLayout.surfaceDiameter / 2;

    expect(surfaceRadius).toBeGreaterThanOrEqual(
      iceArenaLayout.cameraTravelRadius + iceArenaLayout.fogEnd,
    );
  });

  test("Given the hybrid environment, when its perimeter is authored, then every cluster is real middle-distance scenery outside the playable disc", () => {
    const playableRadius = iceArenaLayout.floorDiameter / 2;

    expect(icePerimeterClusters.length).toBeGreaterThanOrEqual(12);
    for (const cluster of icePerimeterClusters) {
      expect(cluster.radius).toBeGreaterThan(playableRadius + 6);
      expect(cluster.radius).toBeLessThan(iceArenaLayout.fogEnd);
    }
    expect(
      icePerimeterClusters.some((cluster) => cluster.crystalHeight > 2),
    ).toBe(true);
  });
});
