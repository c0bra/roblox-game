import { expect, test } from "bun:test";
import { importNotes } from "./intake";

const clip = {
	instrument: "guitar" as const,
	difficulty: "medium" as const,
	duration: 2,
	notes: [{ time: 0.125, lane: 2 as const, duration: 0.375 }],
	attacks: [],
};

test("places a clipped note on the master clock and preserves its hold length", () => {
	expect(importNotes(clip, 60)).toEqual([
		{ timeUs: 60125000, lane: 2, durationUs: 375000 },
	]);
});
