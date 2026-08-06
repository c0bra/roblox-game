import type { ChartNote, Lane } from "../data/level";

export type Grade = "perfect" | "great" | "good" | "miss";
export type AttackResult = "blocked" | "struck";
export type SustainResult = "holding" | "complete" | "broken";

export interface TapResult {
  grade: Grade;
  noteIndex?: number;
  offsetMs?: number;
}

export const gradeForOffsetMilliseconds = (offsetMs: number): Grade => {
  if (offsetMs <= 60) return "perfect";
  if (offsetMs <= 110) return "great";
  if (offsetMs <= 170) return "good";
  return "miss";
};

const sustainReleaseWindowSeconds = 0.08;

export const judgeTap = (
  notes: ChartNote[],
  judged: ReadonlySet<number>,
  songTime: number,
  lane: Lane,
): TapResult => {
  let closestIndex = -1;
  let closestOffset = Number.POSITIVE_INFINITY;
  for (const [index, note] of notes.entries()) {
    if (judged.has(index) || note.lane !== lane) continue;
    const offset = Math.abs(note.time - songTime) * 1_000;
    if (offset < closestOffset) {
      closestIndex = index;
      closestOffset = offset;
    }
    if (note.time > songTime + 0.17) break;
  }
  const grade = gradeForOffsetMilliseconds(closestOffset);
  return grade === "miss" || closestIndex < 0
    ? { grade: "miss" }
    : { grade, noteIndex: closestIndex, offsetMs: Math.round(closestOffset) };
};

export const resolveAttackWindow = (
  hits: number,
  total: number,
  threshold: number,
): AttackResult =>
  total > 0 && hits / total >= threshold ? "blocked" : "struck";

export const resolveSustain = (
  note: ChartNote,
  songTime: number,
  held: boolean,
): SustainResult => {
  if (songTime >= note.time + note.duration - sustainReleaseWindowSeconds)
    return "complete";
  return held ? "holding" : "broken";
};

export const scoreForGrade = (grade: Grade): number => {
  if (grade === "perfect") return 1_000;
  if (grade === "great") return 700;
  if (grade === "good") return 400;
  return 0;
};
