import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readdir } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { z } from "zod";
import { openProject } from "./intake";
import { studioModel } from "./studio";
import { AnalysisJobs } from "./jobs";
import { Conflict, digest, DraftStore, initialDraft, validated } from "./draft";
import { activity, findings, roleWorkload } from "./ui/review.js";
import { timelineSchema } from "../chart-pipeline/src/timeline-contract";

const arg = Bun.argv[2];
if (!arg)
	throw Error("Usage: bun tools/content-authoring/server.ts <project-folder>");
const folder = resolve(arg);
const folders = new Map([[basename(folder), folder]]);
for (const entry of await readdir(dirname(folder), { withFileTypes: true })) {
	const path = join(dirname(folder), entry.name);
	if (
		entry.isDirectory() &&
		(await Bun.file(join(path, "project.json")).exists())
	)
		folders.set(entry.name, path);
}
async function fileHash(path: string) {
	const hash = createHash("sha256");
	for await (const chunk of createReadStream(path)) hash.update(chunk);
	return hash.digest("hex");
}
async function load(path: string) {
	const project = await openProject(path);
	const bundle = resolve(path, project.legacyBundle);
	const media = new Map([
		["master", resolve(path, project.master)],
		...project.roles.map((r) => [r.source, resolve(bundle, r.path)] as const),
	]);
	const audioHashes = Object.fromEntries(
		await Promise.all(
			[...media].map(async ([key, file]) => [key, await fileHash(file)]),
		),
	);
	const sourceRevision = digest({
		audioHashes,
		origin: project.sourceOffsetUs,
		roles: project.roles.map((r) => ({
			source: r.source,
			role: r.role,
			charts: r.charts.map((c) => [c.difficulty, c.notes]),
		})),
	});
	const notesFile = Bun.file(join(path, "review-notes.txt"));
	const baseline = initialDraft(
		project,
		(await notesFile.exists()) ? await notesFile.text() : project.notes,
		sourceRevision,
	);
	validated(baseline, baseline, project.durationUs);
	const store = new DraftStore(path, baseline, project.durationUs);
	const jobs = new AnalysisJobs(
		path,
		bundle,
		project.durationUs,
		sourceRevision,
	);
	await jobs.restore();
	return { project, media, audioHashes, store, jobs };
}
const cache = new Map<string, ReturnType<typeof load>>();
function context(path: string) {
	let result = cache.get(path);
	if (!result) {
		result = load(path).catch((error) => {
			cache.delete(path);
			throw error;
		});
		cache.set(path, result);
	}
	return result;
}
await context(folder);
const assets = new Set([
	"index.html",
	"editor.css",
	"editor.js",
	"transport.js",
	"timeline.js",
	"edits.js",
	"review.js",
]);
const errorResponse = (msg: string, status: number) =>
	new Response(msg, { status });
function serveEditor(hostname: string, port: number, prefix = "") {
	const authoredUrl = (key: string) =>
		prefix + "/p/" + encodeURIComponent(key) + "/";
	const server = Bun.serve({
		hostname,
		port,
		maxRequestBodySize: 16 * 1024 * 1024,
		async fetch(req): Promise<Response> {
			const url = new URL(req.url);
			if (url.host !== server.url.host)
				return errorResponse("Unknown host", 403);
			if (
				req.method === "POST" &&
				req.headers.get("origin") !== server.url.origin
			)
				return errorResponse("Unknown origin", 403);
			try {
				if (prefix && !url.pathname.startsWith(prefix + "/"))
					return errorResponse("Unknown project", 404);
				const pathname = url.pathname.slice(prefix.length);
				if (pathname === "/" && req.method === "GET")
					return Response.redirect(
						new URL(authoredUrl(basename(folder)), server.url),
						302,
					);
				const match = /^\/p\/([^/]+)\/(.*)$/.exec(pathname);
				const id = match ? decodeURIComponent(match[1]!) : basename(folder);
				const path = folders.get(id);
				const route = match ? match[2]! : pathname.slice(1);
				if (prefix && (!match || id !== basename(folder)))
					return errorResponse("Unknown project", 404);
				if (!path) return errorResponse("Unknown project", 404);
				if (req.method === "GET" && assets.has(route || "index.html")) {
					return new Response(
						Bun.file(join(import.meta.dir, "ui", route || "index.html")),
						{
							headers: {
								"Cache-Control": "no-store",
								"Referrer-Policy": "no-referrer",
								"Content-Security-Policy":
									"default-src 'self'; connect-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'",
							},
						},
					);
				}
				const { project, media, audioHashes, store, jobs } =
					await context(path);
				if (req.method === "GET") {
					if (route.startsWith("audio/")) {
						const audio = media.get(route.slice(6));
						if (audio) {
							const file = Bun.file(audio);
							if (!(await file.exists()))
								return errorResponse("Recording missing from disk", 404);
							const range = req.headers.get("Range");
							const headers = {
								"Accept-Ranges": "bytes",
								"Content-Type": file.type,
							};
							if (!range) return new Response(file, { headers });
							const match = /^bytes=(\d+)-(\d*)$/.exec(range);
							if (!match) return new Response(file, { headers });
							const start = Number(match[1]);
							const end = Math.min(
								match[2] ? Number(match[2]) : file.size - 1,
								file.size - 1,
							);
							if (start > end || start >= file.size)
								return new Response(null, {
									status: 416,
									headers: { "Content-Range": `bytes */${file.size}` },
								});
							return new Response(file.slice(start, end + 1), {
								status: 206,
								headers: {
									...headers,
									"Content-Range": `bytes ${start}-${end}/${file.size}`,
								},
							});
						}
					}
					if (route === "job") return Response.json(jobs.status);
					if (route === "suggestions")
						return jobs.result
							? Response.json(jobs.result)
							: errorResponse("Run analysis first.", 404);
					const draft = await store.read();
					const revision = digest(draft);
					if (route === "session")
						return Response.json({
							title: project.title,
							durationUs: project.durationUs,
							roles: project.roles.map(({ charts, ...r }) => ({
								...r,
								difficulties: charts.map((c) => c.difficulty),
							})),
							draft,
							revision,
							projectId: id,
							projects: prefix ? [id] : [...folders.keys()],
						});
					if (route === "prior") return Response.json(await store.previous());
					if (route === "draft.json")
						return Response.json({
							...project,
							notes: draft.notes,
							roles: project.roles.map((r) => ({
								...r,
								charts: r.charts.map((c) => ({
									...c,
									notes: draft.charts[r.source + "/" + c.difficulty],
								})),
							})),
						});
					if (route === "export" || route === "studio") {
						if (
							url.searchParams.has("revision") &&
							url.searchParams.get("revision") !== revision
						)
							throw new Conflict(
								"Draft changed on disk. Reload before exporting.",
							);
						const roles = project.roles.filter((r) => r.role);
						const timelines = Object.fromEntries(
							["Easy", "Normal", "Hard"].map((d) => [
								d,
								timelineSchema.parse({
									schemaVersion: 1,
									contentRevision: revision,
									balanceRevision: "authoring-draft",
									roles: roles.map((r) => r.role),
									events: roles.flatMap((r, order) =>
										(draft.charts[r.source + "/" + d] ?? []).map((n) => ({
											eventId: [r.source, d, n.timeUs, n.lane].join("."),
											timeUs: n.timeUs,
											globalTieOrder: order * 3 + n.lane,
											roleId: r.role,
										})),
									),
								}),
							]),
						);
						if (route === "studio")
							return new Response(
								studioModel({
									Metadata: {
										title: project.title,
										revision,
										durationUs: project.durationUs,
										status: "authoring-draft",
										audioHashes,
									},
									...timelines,
									Charts: draft.charts,
									Clock: { beats: draft.beats, meter: draft.meter },
									Passages: draft.passages,
									Rests: draft.rests,
									Encounter: draft.events,
								}),
								{
									headers: {
										"Content-Type": "application/xml",
										"Content-Disposition":
											'attachment; filename="SongAuthoringDraft.rbxmx"',
									},
								},
							);
						return Response.json(
							{
								format: "song-authoring-draft-v1",
								title: project.title,
								durationUs: project.durationUs,
								revision,
								audioHashes,
								master: project.master,
								layers: project.roles.map(({ charts, ...r }) => r),
								...draft,
								timelines,
								activity: activity(draft, project.durationUs),
								workload: roleWorkload(draft.charts, project.durationUs, draft.beats.length ? draft.beats : jobs.result?.beats ?? []),
								findings: findings(draft),
							},
							{
								headers: {
									"Content-Disposition":
										'attachment; filename="song-draft.json"',
								},
							},
						);
					}
				}
				if (req.method === "POST") {
					if (route === "job/cancel") {
						jobs.cancel();
						return Response.json(jobs.status);
					}
					if (route === "job") {
						const { kind } = z
							.object({ kind: z.enum(["beats", "charts"]) })
							.parse(await req.json());
						const current = await store.read();
						if (kind === "charts" && current.beats.length === 1)
							throw Error(
								"Set at least two beats for chart derivation, or clear the draft grid to use detection.",
							);
						return jobs.start(kind, current.beats, current.rests)
							? Response.json(jobs.status)
							: errorResponse("Analysis already running", 409);
					}
					if (route === "save") {
						const { revision, draft } = z
							.object({ revision: z.string(), draft: z.unknown() })
							.parse(await req.json());
						return Response.json({
							revision: digest(await store.save(revision, draft)),
						});
					}
					if (route === "notes") {
						const data = await req.formData();
						const current = await store.read();
						if (
							createHash("sha256").update(current.notes).digest("hex") !==
							data.get("revision")
						)
							throw new Conflict("Notes changed on disk.");
						await store.save(digest(current), {
							...current,
							notes: z.string().parse(data.get("notes")),
						});
						return Response.redirect(new URL(authoredUrl(id), server.url), 303);
					}
				}
				return errorResponse("Unknown action", 404);
			} catch (error) {
				return errorResponse(
					error instanceof Error ? error.message : "Request failed",
					error instanceof Conflict ? 409 : 400,
				);
			}
		},
	});
	return server;
}
const port = Number(process.env.AUTHORING_PORT ?? 0);
const servers = [serveEditor("127.0.0.1", port)];
if (process.env.AUTHORING_WIFI_ADDRESS) {
	const address = z.ipv4().parse(process.env.AUTHORING_WIFI_ADDRESS);
	const key = z
		.string()
		.regex(/^[a-f0-9]{32,}$/)
		.parse(process.env.AUTHORING_WIFI_KEY);
	if (!/^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)/.test(address))
		throw Error("Choose a private Wi-Fi address.");
	servers.push(serveEditor(address, servers[0]!.port ?? port, "/" + key));
}
for (const signal of ["SIGINT", "SIGTERM"] as const)
	process.on(signal, async () => {
		for (const pending of cache.values()) (await pending).jobs.cancel();
		for (const server of servers) server.stop(true);
		process.exit();
	});
console.info("Song editor: " + servers.map((s) => s.url.origin).join(", "));
