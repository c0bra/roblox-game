import { describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createBundleManifest } from "../src/bundle";
import { bundleManifestSchema } from "../src/chart-format";
import { parsePipelineArgs } from "../src/options";
import { buildChartBundle, writeChartBundle } from "../src/song-pipeline";
import { separateSong } from "../src/stem-separator";

describe("root chart pipeline", () => {
  test("retains the full audio tail after the final drum beat", async () => {
    const directory = await mkdtemp(join(tmpdir(), "chart-tail-test-"));
    try {
      for (const name of ["bass", "drums", "guitar", "vocals"])
        await Bun.write(join(directory, `${name}.wav`), "fixture");
      const options = {
        command: "build",
        stems: directory,
        output: join(directory, "bundle"),
        start: 1,
        model: "fixture",
        snapMs: 80,
      } as const;
      const runner = {
        text: async (command: readonly string[]) => {
          if (command.includes("-progress"))
            return command.includes(join(directory, "guitar.wav"))
              ? "out_time_us=8000000\nprogress=end"
              : "out_time_us=6000000\nprogress=end";
          return command.includes("vamp:beatroot-vamp:beatroot:beats")
            ? "1\n2\n3"
            : "";
        },
        bytes: async () => new ArrayBuffer(0),
      };
      const full = await buildChartBundle(options, runner);
      expect(full.duration).toBe(7);
      expect(full.timing.sourceOffset).toBe(1);
      const clip = await buildChartBundle({ ...options, duration: 2 }, runner);
      expect(clip.duration).toBe(2);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
  test("parses a platform-neutral song build", () => {
    expect(
      parsePipelineArgs([
        "build",
        "--song",
        "/songs/example.mp3",
        "--output",
        "/builds/example",
        "--duration",
        "90",
      ]),
    ).toEqual({
      command: "build",
      song: "/songs/example.mp3",
      output: "/builds/example",
      start: 0,
      duration: 90,
      model: "htdemucs.yaml",
      snapMs: 80,
    });
  });

  test("defaults song output to a slugged build directory", () => {
    expect(
      parsePipelineArgs(["build", "--song", "Blackened Crown - New.mp3"]),
    ).toEqual({
      command: "build",
      song: "Blackened Crown - New.mp3",
      output: "build/blackened-crown-new",
      start: 0,
      model: "htdemucs.yaml",
      snapMs: 80,
    });
  });

  test("shows help from build and validate subcommands", () => {
    expect(parsePipelineArgs(["build", "--help"])).toEqual({ command: "help" });
    expect(parsePipelineArgs(["validate", "--help"])).toEqual({
      command: "help",
    });
  });

  test("creates a versioned manifest for every chart consumer", () => {
    const manifest = createBundleManifest({
      source: "example.mp3",
      start: 12,
      duration: 90,
      model: "htdemucs.yaml",
    });
    expect(bundleManifestSchema.parse(manifest)).toEqual(manifest);
    expect(manifest.timing).toEqual({ unit: "seconds", sourceOffset: 12 });
    expect(manifest.charts.drums.easy).toBe("charts/drums-easy.json");
    expect(manifest.charts.vocals.hard).toBe("charts/vocals-hard.json");
  });

  test("writes a self-contained bundle from analyzed stems", async () => {
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

  test("stages a song in an isolated directory before mounting it into the separator", async () => {
    const directory = await mkdtemp(join(tmpdir(), "chart-separator-test-"));
    try {
      const song = join(directory, "example.mp3");
      const output = join(directory, "bundle");
      await Bun.write(song, "audio");
      let command: readonly string[] = [];
      await separateSong({
        song,
        output,
        model: "htdemucs.yaml",
        runner: {
          text: async (nextCommand) => {
            command = nextCommand;
            return "";
          },
          bytes: async () => new ArrayBuffer(0),
        },
      });
      const isolatedInput = join(output, "analysis", "input");
      expect(command).toContain(`${isolatedInput}:/input:ro`);
      expect(await Bun.file(join(isolatedInput, "example.mp3")).text()).toBe(
        "audio",
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
