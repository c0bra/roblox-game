import {
  chartDifficulties,
  defaultDifficulty,
  instruments,
} from "../data/level";
import { type LevelCatalog, levelIdSchema } from "../data/level-catalog";
import {
  moveSelectionIndex,
  type RunSelection,
  type RunSelectionAction,
  reduceRunSelection,
} from "./run-selection";

type SelectionKind = "level" | "instrument" | "difficulty";

const selectors: Record<SelectionKind, string> = {
  level: "[data-level]",
  instrument: "[data-instrument]",
  difficulty: "[data-difficulty]",
};

export class SelectionControls {
  private selection: RunSelection;

  constructor(private readonly catalog: LevelCatalog) {
    this.selection = {
      levelId: catalog.defaultLevelId,
      instrument: "drums",
      difficulty: defaultDifficulty,
    };
  }

  get current(): RunSelection {
    return this.selection;
  }

  mount(): void {
    for (const kind of ["level", "instrument", "difficulty"] as const) {
      const buttons = Array.from(
        document.querySelectorAll<HTMLButtonElement>(selectors[kind]),
      );
      buttons.forEach((button) => {
        button.addEventListener("click", () => this.choose(kind, button));
        button.addEventListener("keydown", (event) => {
          const step =
            event.key === "ArrowLeft" || event.key === "ArrowUp"
              ? -1
              : event.key === "ArrowRight" || event.key === "ArrowDown"
                ? 1
                : undefined;
          const edge =
            event.key === "Home"
              ? "first"
              : event.key === "End"
                ? "last"
                : undefined;
          if (step === undefined && edge === undefined) return;
          event.preventDefault();
          const target =
            buttons[
              moveSelectionIndex({
                current: buttons.indexOf(button),
                count: buttons.length,
                ...(step === undefined ? {} : { step }),
                ...(edge === undefined ? {} : { edge }),
              })
            ];
          if (!target) return;
          this.choose(kind, target);
          target.focus();
        });
      });
    }
  }

  setDisabled(disabled: boolean): void {
    document
      .querySelectorAll<HTMLButtonElement>(
        "[data-level], [data-instrument], [data-difficulty]",
      )
      .forEach((button) => {
        button.disabled = disabled;
      });
  }

  private choose(kind: SelectionKind, button: HTMLButtonElement): void {
    const action = this.actionFrom(kind, button);
    if (!action) return;
    this.selection = reduceRunSelection(this.selection, action);
    document
      .querySelectorAll<HTMLButtonElement>(selectors[kind])
      .forEach((option) => {
        const chosen = option === button;
        option.classList.toggle("is-selected", chosen);
        option.setAttribute("aria-checked", String(chosen));
        option.tabIndex = chosen ? 0 : -1;
      });
  }

  private actionFrom(
    kind: SelectionKind,
    button: HTMLButtonElement,
  ): RunSelectionAction | undefined {
    if (kind === "level") {
      const levelId = levelIdSchema.safeParse(button.dataset.level);
      if (
        !levelId.success ||
        !this.catalog.levels.some((level) => level.id === levelId.data)
      )
        return undefined;
      return { type: "level", levelId: levelId.data };
    }
    if (kind === "instrument") {
      const instrument = instruments.find(
        (candidate) => candidate === button.dataset.instrument,
      );
      return instrument ? { type: "instrument", instrument } : undefined;
    }
    const difficulty = chartDifficulties.find(
      (candidate) => candidate === button.dataset.difficulty,
    );
    return difficulty ? { type: "difficulty", difficulty } : undefined;
  }
}
