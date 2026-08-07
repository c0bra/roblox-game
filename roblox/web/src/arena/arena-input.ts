import type { ArenaAction } from "./encounter-schema";

export const arenaActionForCode = (code: string): ArenaAction | undefined => {
  if (code === "Space" || code === "KeyF") return "perform";
  if (code === "KeyW" || code === "ArrowLeft") return "retreat";
  if (code === "KeyD" || code === "ArrowRight") return "advance";
  return undefined;
};
