import { z } from "zod";
import { compileDifficulties } from "../chart-pipeline/src/chart-compiler";
import {
	chartDifficulties,
	chartSchema,
} from "../chart-pipeline/src/chart-format";
import { importNotes } from "./intake";

const windowSchema = z
	.object({
		role: z.enum(["bass", "guitar", "vocals"]),
		start: z.number().nonnegative(),
		end: z.number().positive(),
		events: z
			.array(
				z.object({
					time: z.number().nonnegative(),
					pitch: z.number(),
					strength: z.number().nonnegative(),
					duration: z.number().nonnegative(),
				}),
			)
			.min(1),
	})
	.refine(
		(w) =>
			w.start < w.end &&
			w.events.every(
				(e) =>
					e.time >= w.start &&
					e.time < w.end &&
					e.time + e.duration <= w.end + 0.000001,
			),
		"Invalid transcription window",
	);
export const fillsSchema = z.object({
	sourceRevision: z.string(),
	windows: z.array(windowSchema),
});

export function fillEmptyWindows(
	charts: Record<string, ReturnType<typeof importNotes>>,
	fills: z.infer<typeof fillsSchema>,
	beatTimes: readonly number[],
	maxSnapSeconds: number,
	songEnd: number,
) {
	for (const w of fills.windows) {
		if (w.end > songEnd + 0.000001)
			throw Error("Transcription window extends beyond the song");
		const compiled = compileDifficulties({
			instrument: w.role,
			events: w.events,
			beatTimes,
			clip: { start: w.start, duration: w.end - w.start },
			maxSnapSeconds,
		});
		for (const d of chartDifficulties) {
			const label = { easy: "Easy", medium: "Normal", hard: "Hard" }[d];
			const key = w.role + "/" + label;
			const startUs = Math.round(w.start * 1e6),
				endUs = Math.round(w.end * 1e6);
			// A changed source transcription takes precedence, including any hold entering the window.
			if (
				charts[key].some(
					(n) =>
						n.timeUs < endUs &&
						(n.timeUs >= startUs || n.timeUs + n.durationUs > startUs),
				)
			)
				continue;
			const added = importNotes(
				chartSchema.parse({ ...compiled.charts[d], attacks: [] }),
				w.start,
			);
			charts[key] = [...charts[key], ...added].sort(
				(a, b) => a.timeUs - b.timeUs || a.lane - b.lane,
			);
		}
	}
}
