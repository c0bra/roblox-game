import { access, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { validateBundle } from "../chart-pipeline/src/bundle";
import {
	chartDifficulties,
	chartSchema,
	instruments,
} from "../chart-pipeline/src/chart-format";
import { decodedDurationSeconds } from "../chart-pipeline/src/audio-files";
import { systemProcessRunner } from "../chart-pipeline/src/process-runner";
import { z } from "zod";
import { passageSchema } from "./draft";

const layerSchema = z.object({
	source: z.enum(instruments),
	role: z.string().min(1).optional(),
	layer: z.string().min(1),
	review: z.string().min(1).optional(),
});
const configSchema = z.object({
	version: z.literal(1),
	title: z.string().min(1),
	master: z.string().min(1),
	legacyBundle: z.string().min(1),
	layers: z.array(layerSchema).min(1),
	notes: z.string(),
	passages: z.array(passageSchema).default([]),
});
const labels = { easy: "Easy", medium: "Normal", hard: "Hard" } as const;
const us = (seconds: number) => Math.round(seconds * 1_000_000);
export function importNotes(
	chart: z.infer<typeof chartSchema>,
	offsetSeconds: number,
) {
	const offset = us(z.number().nonnegative().parse(offsetSeconds));
	return chart.notes.map((n) => ({
		timeUs: offset + us(n.time),
		durationUs: us(n.duration),
		lane: n.lane,
	}));
}

export async function openProject(folder: string) {
	const config = configSchema.parse(
		JSON.parse(await readFile(join(folder, "project.json"), "utf8")),
	);
	const master = resolve(folder, config.master);
	const bundle = resolve(folder, config.legacyBundle);
	await access(master);
	const raw = JSON.parse(await readFile(join(bundle, "manifest.json"), "utf8"));
	if (raw?.timing?.sourceOffset == null)
		throw new Error(
			"Legacy source offset is missing. Verify the clip origin against the master before importing.",
		);
	const manifest = await validateBundle(bundle);
	const durationUs = us(
		await decodedDurationSeconds(master, systemProcessRunner),
	);
	const offset = manifest.timing.sourceOffset;
	if (us(offset + manifest.duration) > durationUs + 1000)
		throw new Error(
			"Legacy clip exceeds the decoded master. Check the selected edit and origin.",
		);
	if (new Set(config.layers.map((l) => l.source)).size !== config.layers.length)
		throw new Error("Duplicate source layer in project.json");
	const roles = await Promise.all(
		config.layers.map(async (layer) => {
			const charts = await Promise.all(
				chartDifficulties.map(async (difficulty) => {
					const file = manifest.charts[layer.source][difficulty];
					const chart = chartSchema.parse(
						JSON.parse(await readFile(join(bundle, file), "utf8")),
					);
					const notes = importNotes(chart, offset);
					if (notes.some((n) => n.timeUs + n.durationUs > durationUs + 1000))
						throw new Error(`Note exceeds master duration in ${file}`);
					return { difficulty: labels[difficulty], file, notes };
				}),
			);
			return {
				...layer,
				path: manifest.stems[layer.source],
				charts,
				review:
					layer.review ??
					(layer.role ? "" : "Choose a playable role for this layer."),
			};
		}),
	);
	return {
		title: config.title,
		master: config.master,
		legacyBundle: config.legacyBundle,
		durationUs,
		sourceOffsetUs: us(offset),
		roles,
		passages: config.passages,
		notes: config.notes,
	};
}
