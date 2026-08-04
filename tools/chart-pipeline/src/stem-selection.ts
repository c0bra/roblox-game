import { basename, extname } from "node:path";
import type { Instrument } from "./chart-format";

const roleAliases = {
  drums: ["drum"],
  vocals: ["vocal"],
  guitar: ["guitar", "other"],
  bass: ["bass"],
} satisfies Record<Instrument, readonly string[]>;

const scoreCandidate = (file: string, aliases: readonly string[]): number => {
  const name = basename(file, extname(file)).toLowerCase();
  const exact = aliases.some(
    (alias) =>
      name === alias || name === `${alias}-stem` || name === `${alias}_stem`,
  );
  return (
    (exact ? 10 : 0) +
    (name.includes("htdemucs") ? 5 : 0) +
    (extname(file).toLowerCase() === ".wav" ? 1 : 0)
  );
};

export const selectStemCandidate = (
  files: readonly string[],
  instrument: Instrument,
): string | undefined => {
  const aliases = roleAliases[instrument];
  return files
    .filter((file) => {
      const name = basename(file).toLowerCase();
      return aliases.some((alias) => name.includes(alias));
    })
    .sort(
      (left, right) =>
        scoreCandidate(right, aliases) - scoreCandidate(left, aliases) ||
        left.localeCompare(right),
    )[0];
};
