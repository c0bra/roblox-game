import { readdir } from "node:fs/promises";
import { extname, join } from "node:path";
import type { Instrument } from "./chart-format";
import { CommandFailure } from "./process-runner";
import { selectStemCandidate } from "./stem-selection";

const audioExtensions = new Set([".wav", ".mp3", ".flac", ".m4a", ".ogg"]);

export const findAudioFiles = async (
  directory: string,
): Promise<readonly string[]> => {
  const files: string[] = [];
  const visit = async (current: string): Promise<void> => {
    for (const entry of await readdir(current, { withFileTypes: true })) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (audioExtensions.has(extname(entry.name).toLowerCase()))
        files.push(path);
    }
  };
  await visit(directory);
  return files.sort((left, right) => left.localeCompare(right));
};

export const selectRequiredStems = (
  files: readonly string[],
): Record<Instrument, string> => {
  const select = (instrument: Instrument): string => {
    const selected = selectStemCandidate(files, instrument);
    if (!selected) {
      throw new CommandFailure(
        ["find-stem", instrument],
        1,
        `No ${instrument} stem found`,
      );
    }
    return selected;
  };
  return {
    drums: select("drums"),
    vocals: select("vocals"),
    guitar: select("guitar"),
    bass: select("bass"),
  };
};
