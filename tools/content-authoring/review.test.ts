import { expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Conflict, digest, DraftStore, validated, type Draft } from "./draft";
import { activity, findings } from "./ui/review.js";
const baseline: Draft = {
	version: 1,
	sourceRevision: "fixture",
	notes: "",
	charts: { "piano/Normal": [{ timeUs: 2, lane: 1, durationUs: 8 }] },
	passages: [],
	beats: [1, 5, 10],
	meter: 4,
	rests: [],
	events: [],
};
test("activity accounts for sustained notes in the next beat", () => {
	const bins = activity(baseline, 20)["piano/Normal"]!;
	expect(bins.map((b) => [b.noteStarts, b.crossingHold])).toEqual([
		[0, false],
		[1, false],
		[0, true],
		[0, false],
	]);
});
test("candidate conflicts include both warnings and exclude adjoining endpoints", () => {
	const event = {
		id: "f",
		source: "all",
		kind: "fixed" as const,
		name: "Fixed",
		warningUs: 3,
		startUs: 10,
		endUs: 12,
		reviewed: true,
	};
	const candidate = {
		...event,
		id: "c",
		kind: "candidate" as const,
		name: "Opportunity",
		startUs: 8,
		endUs: 9,
		warningUs: 0,
	};
	expect(findings({ ...baseline, events: [event, candidate] })).toHaveLength(1);
	expect(
		findings({
			...baseline,
			events: [event, { ...candidate, startUs: 12, endUs: 13 }],
		}),
	).toHaveLength(0);
});
test("source changes and notes past the recording are rejected", () => {
	expect(() =>
		validated({ ...baseline, sourceRevision: "different" }, baseline, 20),
	).toThrow("Source changed");
	expect(() => validated(baseline, baseline, 6)).toThrow("outside");
});
test("reopening preserves the edit and a stale writer cannot replace it", async () => {
	const folder = await mkdtemp(join(tmpdir(), "authoring-store-"));
	try {
		const store = new DraftStore(folder, baseline, 20);
		const before = digest(await store.read());
		await store.save(before, { ...baseline, notes: "heard it" });
		const reopened = new DraftStore(folder, baseline, 20);
		expect((await reopened.read()).notes).toBe("heard it");
		expect((await reopened.previous()).notes).toBe("");
		await expect(reopened.save(before, baseline)).rejects.toBeInstanceOf(
			Conflict,
		);
		await expect(reopened.save("stale", baseline)).rejects.toBeInstanceOf(
			Conflict,
		);
	} finally {
		await rm(folder, { recursive: true, force: true });
	}
});

test("a failed processor preserves the last usable suggestion on disk", async () => {
	const { AnalysisJobs } = await import("./jobs");
	const folder = await mkdtemp(join(tmpdir(), "authoring-job-"));
	try {
		const jobs = new AnalysisJobs(
			folder,
			join(folder, "missing-bundle"),
			20,
			"fixture",
		);
		const previous = {
			sourceRevision: "fixture",
			beats: [1, 5],
			chartGrid: [1, 5],
			charts: {},
		};
		await Bun.write(jobs.file, JSON.stringify(previous));
		await jobs.restore();
		expect(jobs.start("beats", [])).toBe(true);
		const deadline = Date.now() + 4000;
		while (jobs.status.running && Date.now() < deadline) await Bun.sleep(20);
		expect(jobs.status.phase).toStartWith("Analysis failed.");
		expect(jobs.result).toEqual(previous);
		expect(await Bun.file(jobs.file).json()).toEqual(previous);
	} finally {
		await rm(folder, { recursive: true, force: true });
	}
});
