import { expect, test } from "bun:test";
import { fillEmptyWindows, fillsSchema } from "./transcription-fills";
import type { importNotes } from "./intake";

test("a fill targets an empty passage and preserves existing attacks and crossing holds", () => {
	const untouched = { timeUs: 100000, lane: 0 as const, durationUs: 0 };
	const hold = { ...untouched, durationUs: 1000000 };
	const attack = { timeUs: 1500000, lane: 2 as const, durationUs: 0 };
	const charts: Record<string, ReturnType<typeof importNotes>> = {
		"bass/Easy": [untouched],
		"bass/Normal": [hold],
		"bass/Hard": [attack],
		"guitar/Easy": [],
	};
	const fill = fillsSchema.parse({
		sourceRevision: "fixture",
		windows: [
			{
				role: "bass",
				start: 1,
				end: 2,
				events: [{ time: 1.25, strength: 1, pitch: 40, duration: 0 }],
			},
		],
	});
	fillEmptyWindows(charts, fill, [0, 0.5, 1, 1.5, 2, 2.5], 0.08, 3);
	expect(charts["bass/Easy"].map((n) => n.timeUs)).toEqual([100000, 1250000]);
	expect(charts["bass/Normal"]).toEqual([hold]);
	expect(charts["bass/Hard"]).toEqual([attack]);
	expect(charts["guitar/Easy"]).toEqual([]);
});
