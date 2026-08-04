import ky from "ky";
import type { BattleAudio } from "../audio/battle-audio";
import type { RunSelection } from "../game/run-selection";
import { chartSchema, type LevelChart } from "./level";
import {
  audioUrls,
  chartUrl,
  type LevelCatalog,
  resolveLevel,
} from "./level-catalog";
import { createQaChart } from "./qa-chart";

export class ChartSelectionMismatch extends Error {
  override readonly name = "ChartSelectionMismatch";

  constructor(readonly selection: RunSelection) {
    super(
      `Chart metadata does not match ${selection.instrument}/${selection.difficulty}`,
    );
  }
}

export type AudioPreparer = Pick<BattleAudio, "prepare">;

export type LoadRunAssetsInput = {
  readonly catalog: LevelCatalog;
  readonly selection: RunSelection;
  readonly json: (url: string) => Promise<unknown>;
  readonly audio: AudioPreparer;
  readonly qa: boolean;
};

export const loadRunAssets = async (
  input: LoadRunAssetsInput,
): Promise<LevelChart> => {
  const level = resolveLevel(input.catalog, input.selection.levelId);
  const chart = chartSchema.parse(
    await input.json(
      chartUrl(level, input.selection.instrument, input.selection.difficulty),
    ),
  );
  if (
    chart.instrument !== input.selection.instrument ||
    chart.difficulty !== input.selection.difficulty
  ) {
    throw new ChartSelectionMismatch(input.selection);
  }
  await input.audio.prepare(audioUrls(level, input.selection.instrument));
  return input.qa ? createQaChart(chart) : chart;
};

export type RunAssetLoader = (selection: RunSelection) => Promise<LevelChart>;

export const createBrowserRunLoader =
  (catalog: LevelCatalog, audio: AudioPreparer, qa: boolean): RunAssetLoader =>
  async (selection) =>
    loadRunAssets({
      catalog,
      selection,
      json: async (url) => ky.get(url).json(),
      audio,
      qa,
    });
