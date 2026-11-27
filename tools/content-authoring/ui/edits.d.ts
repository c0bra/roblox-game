type ChartNote = { timeUs: number; durationUs: number; lane: number };
export function secondsToUs(seconds: number): number;
export function clock(seconds: number, precise?: boolean): string;
export function restRange<T extends ChartNote>(notes: readonly T[], start: number, end: number): T[];
export function upsert<T extends ChartNote>(notes: readonly T[], old: T | null, note: T, duration: number): T[];
