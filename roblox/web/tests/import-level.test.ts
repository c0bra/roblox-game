import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeChartBundle } from "@bands-battle/chart-pipeline";
import { CommandFailure, type CommandRunner } from "../scripts/ffmpeg-audio";
import { importWebLevel } from "../scripts/web-level-import";
import { levelCatalogSchema } from "../src/data/level-catalog";

const createFixture = async () => {
  const directory = await mkdtemp(join(tmpdir(), "web-level-test-"));
  const bundle = join(directory, "bundle");
  const sourceStems = {
    drums: join(directory, "drums.wav"),
    vocals: join(directory, "vocals.wav"),
    guitar: join(directory, "guitar.wav"),
    bass: join(directory, "bass.wav"),
  } as const;
  await Promise.all(
    Object.values(sourceStems).map((path) => Bun.write(path, "audio")),
  );
  await writeChartBundle({
    output: bundle,
    source: "fixture.wav",
    model: "fixture",
    clip: { start: 0, duration: 4 },
    maxSnapSeconds: 0.08,
    sourceStems,
    beatTimes: [0, 1, 2, 3, 4],
    events: {
      drums: [{ time: 1, pitch: 36, strength: 1, duration: 0, label: "kick" }],
      vocals: [{ time: 1, pitch: 60, strength: 1, duration: 1 }],
      guitar: [{ time: 2, pitch: 64, strength: 1, duration: 1 }],
      bass: [{ time: 3, pitch: 40, strength: 1, duration: 1 }],
    },
  });
  const levelsDirectory = join(directory, "public", "levels");
  const catalogFile = join(directory, "src", "data", "levels.json");
  await Promise.all([
    mkdir(levelsDirectory, { recursive: true }),
    mkdir(join(directory, "src", "data"), { recursive: true }),
  ]);
  await Bun.write(
    catalogFile,
    `${JSON.stringify(
      {
        defaultLevelId: "heavens-edge",
        levels: [{ id: "heavens-edge", title: "Heaven's Edge" }],
      },
      null,
      2,
    )}\n`,
  );
  return { directory, bundle, levelsDirectory, catalogFile };
};

describe("web level importer", () => {
  test("Given a neutral bundle, when imported, then it publishes twelve charts, eight audio files, and one catalog entry", async () => {
    const fixture = await createFixture();
    const commands: string[][] = [];
    const runner: CommandRunner = {
      run: async (command) => {
        commands.push([...command]);
        const output = command.at(-1);
        if (output) await Bun.write(output, "encoded audio");
      },
    };
    try {
      await importWebLevel({
        bundle: fixture.bundle,
        levelId: "blackened-crown",
        title: "Blackened Crown",
        levelsDirectory: fixture.levelsDirectory,
        catalogFile: fixture.catalogFile,
        runner,
      });

      const destination = join(fixture.levelsDirectory, "blackened-crown");
      expect(await readdir(join(destination, "charts"))).toHaveLength(12);
      expect(await readdir(join(destination, "audio"))).toHaveLength(8);
      expect(commands).toHaveLength(8);
      expect(
        levelCatalogSchema
          .parse(JSON.parse(await readFile(fixture.catalogFile, "utf8")))
          .levels.map((level) => String(level.id)),
      ).toEqual(["heavens-edge", "blackened-crown"]);
    } finally {
      await rm(fixture.directory, { recursive: true, force: true });
    }
  });

  test("Given ffmpeg fails, when importing, then no destination or catalog change is published", async () => {
    const fixture = await createFixture();
    const before = await readFile(fixture.catalogFile, "utf8");
    let commandCount = 0;
    const runner: CommandRunner = {
      run: async (command) => {
        commandCount += 1;
        if (commandCount === 3) throw new CommandFailure(command, 1, "failed");
        const output = command.at(-1);
        if (output) await Bun.write(output, "encoded audio");
      },
    };
    try {
      await expect(
        importWebLevel({
          bundle: fixture.bundle,
          levelId: "blackened-crown",
          title: "Blackened Crown",
          levelsDirectory: fixture.levelsDirectory,
          catalogFile: fixture.catalogFile,
          runner,
        }),
      ).rejects.toBeInstanceOf(CommandFailure);
      expect(await readdir(fixture.levelsDirectory)).toEqual([]);
      expect(await readFile(fixture.catalogFile, "utf8")).toBe(before);
    } finally {
      await rm(fixture.directory, { recursive: true, force: true });
    }
  });
});
