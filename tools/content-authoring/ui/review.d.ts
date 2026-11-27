import type { Draft } from "../draft";
export function findings(draft: Draft): { at: number; text: string }[];
export function activity(
	draft: Draft,
	duration: number,
): Record<
	string,
	{
		startUs: number;
		endUs: number;
		noteStarts: number;
		crossingHold: boolean;
		nextNoteUs: number | null;
	}[]
>;
export function roleWorkload(charts: Draft["charts"], durationUs: number, beats: readonly number[]): Record<string, {noteCount:number; activeBeatFraction:number|null}>;
