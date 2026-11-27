// Child process so a cancelled run can stop its audio processors as a group.
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { z } from "zod";
import {
	instruments,
	chartDifficulties,
	chartSchema,
} from "../chart-pipeline/src/chart-format";
import { validateBundle } from "../chart-pipeline/src/bundle";
import { analyzeSong } from "../chart-pipeline/src/sonic-analysis";
import { analyzeDrums } from "../chart-pipeline/src/drum-onsets";
import { systemProcessRunner as runner } from "../chart-pipeline/src/process-runner";
import { compileDifficulties } from "../chart-pipeline/src/chart-compiler";
import { fillEmptyWindows, fillsSchema } from "./transcription-fills";
import { activeGaps } from "../chart-pipeline/src/compile/recovery";
import { restsSchema } from "./draft";
import { restRange } from "./ui/edits.js";
import { importNotes } from "./intake";
const [mode, bundle, limit] = Bun.argv.slice(2);
if (!bundle || !limit || !["beats", "charts"].includes(mode ?? ""))
	throw Error("Invalid analysis request.");
const manifest = await validateBundle(bundle);
const stems = { ...manifest.stems };
for (const role of instruments) stems[role] = join(bundle, stems[role]);
const withinSong = (values: readonly number[]) =>
	[
		...new Set(
			values
				.map((t) => Math.round(t * 1e6))
				.filter((t) => Number.isSafeInteger(t) && t >= 0 && t < Number(limit)),
		),
	].sort((a, b) => a - b);
if (mode === "beats") {
	console.error("Detecting beats on the drum stem");
	const csv = await runner.text([
		"sonic-annotator",
		"-d",
		"vamp:beatroot-vamp:beatroot:beats",
		"-w",
		"csv",
		"--csv-stdout",
		"--csv-omit-filename",
		"--force",
		stems.drums,
	]);
	const beats = withinSong(
		csv
			.trim()
			.split(/\r?\n/)
			.filter(Boolean)
			.map((line) => Number(line.split(",")[0])),
	);
	if (!beats.length) throw Error("The beat detector found no beats.");
	console.log(JSON.stringify({ beats, charts: {} }));
} else {
	const input = await Bun.stdin.text();
	const beatList = z.array(z.int().nonnegative());
	const { beats: grid, fills, rests } = z
		.union([
			beatList.transform((beats) => ({ beats, fills: undefined, rests: [] })),
			z.object({ beats: beatList, rests:restsSchema.default([]), fills: fillsSchema.optional() }),
		])
		.parse(input ? JSON.parse(input) : []);
	const report = z
		.object({
			maxSnapMs: z.number().positive().default(80),
			outroRoles: z.array(z.enum(instruments)).default([]),
		})
		.parse(
			JSON.parse(await readFile(join(bundle, manifest.validation), "utf8")),
		);
	console.error("Transcribing vocals, instruments and beats");
	const analysis = await analyzeSong(stems, runner);
	const clip = {
		start: manifest.timing.sourceOffset,
		duration: manifest.duration,
	};
	console.error("Finding drum onsets");
	const drums = await analyzeDrums({ audio: stems.drums, clip, runner });
	const beatTimes = grid.length ? grid.map((t) => t / 1e6) : analysis.beatTimes;
	const recovery = {...analysis.recovery,drums:drums.recovery};
	const charts: Record<string, ReturnType<typeof importNotes>> = {};
	for (const role of instruments) {
		const result = compileDifficulties({
			instrument: role,
			recovery: recovery[role],
			reserved: [...(fills?.windows.filter(w => w.role === role) ?? []), ...rests.filter(r=>r.source===role).map(r=>({start:r.startUs/1e6,end:r.endUs/1e6}))],
			events: role === "drums" ? drums.events : analysis[role],
			beatTimes,
			clip,
			maxSnapSeconds: report.maxSnapMs / 1000,
			continueFinalTempo: report.outroRoles.includes(role),
		});
		for (const d of chartDifficulties) {
			const label = { easy: "Easy", medium: "Normal", hard: "Hard" }[d];
			charts[role + "/" + label] = importNotes(
				chartSchema.parse({ ...result.charts[d], attacks: [] }),
				clip.start,
			);
		}
	}
	if (fills)
		fillEmptyWindows(
			charts,
			fills,
			beatTimes,
			report.maxSnapMs / 1000,
			Number(limit) / 1e6,
		);
	const gaps = Object.fromEntries(instruments.map(role=>{
		const marked = rests.filter(r=>r.source===role);
		for(const level of ["Easy","Normal","Hard"]) {
			const key=role+"/"+level;
			for(const rest of marked) charts[key]=restRange(charts[key],rest.startUs,rest.endUs);
		}
		const covered=charts[role+"/Normal"].map(n=>({start:n.timeUs/1e6,end:(n.timeUs+n.durationUs)/1e6}));
		return [role,activeGaps(recovery[role].active,clip,[...covered,...marked.map(r=>({start:r.startUs/1e6,end:r.endUs/1e6}))])];
	}));
	console.log(
		JSON.stringify({ beats: withinSong(analysis.beatTimes), charts, activeGaps: gaps }),
	);
}
