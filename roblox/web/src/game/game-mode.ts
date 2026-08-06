import { chartDifficulties, instruments } from "../data/level";
import { levelIdSchema } from "../data/level-catalog";
import type { RunSelection } from "./run-selection";

export type GameMode = "classic" | "arena";

export const arenaDemoSelection: RunSelection = {
  levelId: levelIdSchema.parse("heavens-edge"),
  instrument: "drums",
  difficulty: "easy",
};

export type ModeSelection = {
  readonly mode: GameMode;
  readonly selection: RunSelection;
};

export type ModeSelectionAction =
  | { readonly type: "select-mode"; readonly mode: GameMode }
  | { readonly type: "select-run"; readonly selection: RunSelection }
  | { readonly type: "use-arena-demo" };

export type ArenaSelectionSupport =
  | { readonly type: "supported"; readonly selection: RunSelection }
  | {
      readonly type: "unsupported";
      readonly selection: RunSelection;
      readonly demoSelection: RunSelection;
    };

class UnknownModeSelectionActionError extends Error {
  override readonly name = "UnknownModeSelectionActionError";

  constructor(readonly action: never) {
    super("Unknown mode selection action");
  }
}

export const modeFromSearchParams = (params: URLSearchParams): GameMode =>
  params.get("mode") === "arena" ? "arena" : "classic";

export const selectionFromSearchParams = (
  params: URLSearchParams,
  fallback: RunSelection,
): RunSelection => {
  const levelId = levelIdSchema.safeParse(params.get("level"));
  const instrument = instruments.find(
    (candidate) => candidate === params.get("instrument"),
  );
  const difficulty = chartDifficulties.find(
    (candidate) => candidate === params.get("difficulty"),
  );
  return levelId.success && instrument && difficulty
    ? { levelId: levelId.data, instrument, difficulty }
    : fallback;
};

export const reduceModeSelection = (
  state: ModeSelection,
  action: ModeSelectionAction,
): ModeSelection => {
  switch (action.type) {
    case "select-mode":
      return { ...state, mode: action.mode };
    case "select-run":
      return { ...state, selection: action.selection };
    case "use-arena-demo":
      return { mode: "arena", selection: arenaDemoSelection };
    default:
      throw new UnknownModeSelectionActionError(action);
  }
};

export const arenaSelectionSupport = (
  selection: RunSelection,
): ArenaSelectionSupport => {
  const supported =
    selection.levelId === arenaDemoSelection.levelId &&
    selection.instrument === arenaDemoSelection.instrument &&
    selection.difficulty === arenaDemoSelection.difficulty;
  return supported
    ? { type: "supported", selection }
    : { type: "unsupported", selection, demoSelection: arenaDemoSelection };
};
