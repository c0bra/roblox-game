import type {
  ChartDifficulty,
  Instrument,
} from "@bands-battle/chart-pipeline/format";
import type { LevelId } from "../data/level-catalog";

export type RunSelection = {
  readonly levelId: LevelId;
  readonly instrument: Instrument;
  readonly difficulty: ChartDifficulty;
};

export type RunSelectionAction =
  | { readonly type: "level"; readonly levelId: LevelId }
  | { readonly type: "instrument"; readonly instrument: Instrument }
  | {
      readonly type: "difficulty";
      readonly difficulty: ChartDifficulty;
    };

class UnknownRunSelectionActionError extends Error {
  override readonly name = "UnknownRunSelectionActionError";

  constructor(readonly action: never) {
    super("Unknown run selection action");
  }
}

export const reduceRunSelection = (
  selection: RunSelection,
  action: RunSelectionAction,
): RunSelection => {
  switch (action.type) {
    case "level":
      return { ...selection, levelId: action.levelId };
    case "instrument":
      return { ...selection, instrument: action.instrument };
    case "difficulty":
      return { ...selection, difficulty: action.difficulty };
    default:
      throw new UnknownRunSelectionActionError(action);
  }
};

export type SelectionMovement = {
  readonly current: number;
  readonly count: number;
  readonly step?: number;
  readonly edge?: "first" | "last";
};

export const moveSelectionIndex = (movement: SelectionMovement): number => {
  if (movement.edge === "first") return 0;
  if (movement.edge === "last") return movement.count - 1;
  return (
    (movement.current + (movement.step ?? 0) + movement.count) % movement.count
  );
};
