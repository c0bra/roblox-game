import { resolve } from "node:path";
import { z } from "zod";
import { systemCommandRunner } from "./ffmpeg-audio";
import { importWebLevel } from "./web-level-import";

const argsSchema = z.tuple([
  z.string().min(1),
  z.string().min(1),
  z.string().trim().min(1),
]);

const usage =
  'Usage: bun run level:import -- <bundle-directory> <level-id> "Display Title"';

const main = async (): Promise<void> => {
  const args = argsSchema.safeParse(Bun.argv.slice(2));
  if (!args.success) throw new Error(usage);
  const [bundle, levelId, title] = args.data;
  await importWebLevel({
    bundle: resolve(bundle),
    levelId,
    title,
    levelsDirectory: resolve(import.meta.dir, "../public/levels"),
    catalogFile: resolve(import.meta.dir, "../src/data/levels.json"),
    runner: systemCommandRunner,
  });
  console.log(`Imported ${title} as ${levelId}`);
};

try {
  await main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
