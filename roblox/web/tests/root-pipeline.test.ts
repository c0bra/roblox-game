import { describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeChartBundle } from "@bands-battle/chart-pipeline";
import { createBundleManifest } from "@bands-battle/chart-pipeline/bundle";
import { bundleManifestSchema } from "@bands-battle/chart-pipeline/format";
import { parsePipelineArgs } from "@bands-battle/chart-pipeline/options";

describe("root chart pipeline", () => {
  test("Given a song build request, when arguments are parsed, then the platform-neutral build is selected", () => {
    const args = parsePipelineArgs([
      "build",
      "--song",
      "/songs/example.mp3",
      "--output",
      "/builds/example",
      "--duration",
      "90",
    ]);

    expect(args).toEqual({
      command: "build",
      song: "/songs/example.mp3",
      output: "/builds/example",
      start: 0,
      duration: 90,
      model: "htdemucs.yaml",
      snapMs: 80,
    });
  });

  test("Given compiled song metadata, when a bundle manifest is created, then every consumer receives versioned chart paths", () => {
    const manifest = createBundleManifest({
      source: "example.mp3",
      start: 12,
      duration: 90,
      model: "htdemucs.yaml",
    });

    expect(bundleManifestSchema.parse(manifest)).toEqual(manifest);
    expect(manifest.schemaVersion).toBe(1);
    expect(manifest.timing).toEqual({ unit: "seconds", sourceOffset: 12 });
    expect(manifest.charts.drums.easy).toBe("charts/drums-easy.json");
    expect(manifest.charts.vocals.hard).toBe("charts/vocals-hard.json");
  });

  test("Given analyzed stems, when a bundle is written, then it is complete and validates independently of the web app", async () => {
    const directory = await mkdtemp(join(tmpdir(), "chart-bundle-test-"));
    try {
      const sourceStems = {
        drums: join(directory, "source-drums.wav"),
        vocals: join(directory, "source-vocals.wav"),
        guitar: join(directory, "source-guitar.wav"),
        bass: join(directory, "source-bass.wav"),
      } as const;
      await Promise.all(
        Object.values(sourceStems).map((path) => Bun.write(path, "audio")),
      );
      const output = join(directory, "bundle");
      const manifest = await writeChartBundle({
        output,
        source: "example.mp3",
        model: "htdemucs.yaml",
        clip: { start: 0, duration: 2 },
        maxSnapSeconds: 0.08,
        sourceStems,
        beatTimes: [0, 0.5, 1, 1.5, 2],
        events: {
          drums: [
            { time: 0.5, pitch: 36, strength: 1, duration: 0, label: "kick" },
          ],
          vocals: [{ time: 0.5, pitch: 60, strength: 1, duration: 0.75 }],
          guitar: [{ time: 1, pitch: 64, strength: 1, duration: 0 }],
          bass: [{ time: 1.5, pitch: 40, strength: 1, duration: 0 }],
        },
      });

      expect(manifest.schemaVersion).toBe(1);
      expect(
        JSON.parse(
          await readFile(join(output, manifest.charts.vocals.easy), "utf8"),
        ),
      ).toMatchObject({ instrument: "vocals", difficulty: "easy" });
      expect(await Bun.file(join(output, manifest.stems.drums)).text()).toBe(
        "audio",
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
