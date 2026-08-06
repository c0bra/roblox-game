import { positionProfile } from "./combat-state";
import type {
  ArenaEffect,
  ArenaMoveAction,
  ArenaRunState,
  ArenaTransition,
} from "./combat-types";
import type { ArenaEncounter, ArenaPositionId } from "./encounter";
import { positionIdSchema } from "./encounter-schema";

const parsePosition = (
  encounter: ArenaEncounter,
  value: string,
): ArenaPositionId =>
  positionProfile(encounter, positionIdSchema.parse(value)).id;

export const moveArena = (
  encounter: ArenaEncounter,
  state: ArenaRunState,
  action: ArenaMoveAction,
): ArenaTransition => {
  if (state.phase !== "running") return { state, effects: [] };
  if (state.travel) {
    return {
      state,
      effects: [{ type: "move-unavailable", direction: action.direction }],
    };
  }
  const order = positionProfile(encounter, state.position).order;
  const boundary =
    (action.direction === "retreat" && order === 0) ||
    (action.direction === "advance" &&
      order === encounter.positions.length - 1);
  if (boundary) {
    return {
      state,
      effects: [{ type: "boundary", direction: action.direction }],
    };
  }
  const window = encounter.repositionWindows.find(
    ({ start, deadline }) => action.time >= start && action.time <= deadline,
  );
  const choice = window?.choices.find(
    ({ from, action: choiceAction }) =>
      from === state.position && choiceAction === action.direction,
  );
  if (!window || !choice) {
    return {
      state,
      effects: [{ type: "move-unavailable", direction: action.direction }],
    };
  }
  const to = parsePosition(encounter, choice.to);
  const end = action.time + window.travelDuration;
  const effects: readonly ArenaEffect[] = [
    { type: "input-ack", action: action.direction },
    {
      type: "move-start",
      direction: action.direction,
      from: state.position,
      to,
      end,
    },
  ];
  return {
    state: {
      ...state,
      songTime: action.time,
      travel: { from: state.position, to, start: action.time, end },
    },
    effects,
  };
};
