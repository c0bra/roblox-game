import { readFile, rename, writeFile, unlink } from "node:fs/promises";
import { join } from "node:path";
import { z } from "zod";
import type { openProject } from "./intake";

export type Project = Awaited<ReturnType<typeof openProject>>;
const us = z.int().nonnegative();
export const noteSchema = z.object({
	timeUs: us,
	durationUs: us,
	lane: z.int().min(0).max(2),
});
export const passageSchema = z.object({
	name: z.string().trim().min(1).max(100),
	startUs: us,
	endUs: us,
});
export const restsSchema = z.array(z.object({ source: z.string(), startUs: us, endUs: us })).max(2000);
const draftSchema = z.object({
	version: z.literal(1),
	sourceRevision: z.string(),
	notes: z.string().max(65536),
	charts: z.record(z.string(), z.array(noteSchema).max(100000)),
	passages: z.array(passageSchema).max(1000),
	beats: z.array(us).max(100000),
	rests: restsSchema.default([]),
	meter: z.int().min(1).max(16).default(4),
	events: z
		.array(
			z.object({
				id: z.string().min(1),
				name: z.string().trim().min(1).max(100),
				kind: z.enum(["fixed", "candidate"]),
				source: z.string(),
				startUs: us,
				endUs: us,
				warningUs: us,
				reviewed: z.boolean(),
			}),
		)
		.max(2000)
		.default([]),
});
export type Draft = z.infer<typeof draftSchema>;
export const digest = (value: unknown) =>
	new Bun.CryptoHasher("sha256").update(JSON.stringify(value)).digest("hex");
export function initialDraft(
	project: Project,
	notes: string,
	sourceRevision: string,
): Draft {
	return {
		version: 1,
		sourceRevision,
		notes,
		passages: project.passages,
		beats: [],
		meter: 4,
		rests: [],
		events: [],
		charts: Object.fromEntries(
			project.roles.flatMap((r) =>
				r.charts.map((c) => [`${r.source}/${c.difficulty}`, c.notes]),
			),
		),
	};
}
export function validated(
	raw: unknown,
	baseline: Draft,
	duration: number,
): Draft {
	const draft = draftSchema.parse(raw);
	if (draft.sourceRevision !== baseline.sourceRevision)
		throw Error(
			"Source changed. Open this draft with its original song and chart bundle.",
		);
	if (Object.keys(draft.charts).length !== Object.keys(baseline.charts).length)
		throw Error("Incomplete chart set.");
	for (const [id, notes] of Object.entries(draft.charts)) {
		if (!Object.hasOwn(baseline.charts, id)) throw Error("Unknown chart.");
		if (
			notes.some(
				(n) => n.timeUs >= duration || n.timeUs + n.durationUs > duration,
			)
		)
			throw Error("Note outside the recording.");
		notes.sort((a, b) => a.timeUs - b.timeUs || a.lane - b.lane);
		if (
			notes.some(
				(n, i) =>
					i > 0 &&
					notes[i - 1]?.timeUs === n.timeUs &&
					notes[i - 1]?.lane === n.lane,
			)
		)
			throw Error("Two notes occupy the same lane and time.");
	}
	if (draft.passages.some((p) => p.startUs >= p.endUs || p.endUs > duration))
		throw Error("Invalid passage bounds.");
	const sources = new Set(
		Object.keys(draft.charts).map((k) => k.split("/")[0]),
	);
	if (
		draft.rests.some(
			(r) =>
				!sources.has(r.source) || r.startUs >= r.endUs || r.endUs > duration,
		)
	)
		throw Error("Invalid instrument rest.");
	if (
		new Set(draft.events.map((e) => e.id)).size !== draft.events.length ||
		draft.events.some(
			(e) =>
				(e.source !== "all" && !sources.has(e.source)) ||
				e.startUs > e.endUs ||
				e.endUs > duration,
		)
	)
		throw Error("Invalid encounter event.");
	if (
		draft.beats.some(
			(t, i) => t >= duration || (i > 0 && t <= draft.beats[i - 1]!),
		)
	)
		throw Error("Beats must be ordered and within the recording.");
	return draft;
}
export async function atomicJson(file: string, value: unknown) {
	const temp = file + "." + crypto.randomUUID() + ".tmp";
	try {
		await writeFile(temp, JSON.stringify(value) + "\n", { flag: "wx" });
		await rename(temp, file);
	} finally {
		await unlink(temp).catch(() => {});
	}
}
export class Conflict extends Error {}
export class DraftStore {
	readonly file: string;
	constructor(
		folder: string,
		readonly baseline: Draft,
		readonly duration: number,
	) {
		this.file = join(folder, "authoring-draft.json");
	}
	async read() {
		try {
			return validated(
				JSON.parse(await readFile(this.file, "utf8")),
				this.baseline,
				this.duration,
			);
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT")
				return this.baseline;
			throw error;
		}
	}
	async previous() {
		const prior = Bun.file(this.file + ".previous");
		return (await prior.exists())
			? validated(await prior.json(), this.baseline, this.duration)
			: this.baseline;
	}
	async save(revision: string, raw: unknown) {
		const next = validated(raw, this.baseline, this.duration);
		const lock = this.file + ".lock";
		try {
			await writeFile(lock, String(process.pid), { flag: "wx" });
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "EEXIST")
				throw new Conflict("Another save is in progress.");
			throw error;
		}
		try {
			const current = await this.read();
			if (digest(current) !== revision)
				throw new Conflict(
					"A newer draft is on disk. Your edits are still here. Download a local copy before reloading.",
				);
			await atomicJson(this.file + ".previous", current);
			await atomicJson(this.file, next);
			return next;
		} finally {
			await unlink(lock);
		}
	}
}
