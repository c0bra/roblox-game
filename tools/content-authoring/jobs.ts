import { spawn, type ChildProcess } from "node:child_process";
import { join } from "node:path";
import { fillsSchema } from "./transcription-fills";
import { atomicJson, digest, type Draft } from "./draft";

export type Suggestions = Pick<Draft, "beats" | "charts" | "sourceRevision"> & {
	chartGrid: number[];
	activeGaps?: Record<string, {start:number;end:number}[]>;
	fillRevision?: string;
};
export class AnalysisJobs {
	private child?: ChildProcess;
	private cancelling = false;
	private fills?: ReturnType<typeof fillsSchema.parse>;
	private fillRevision?: string;
	phase = "Idle";
	result?: Suggestions;
	constructor(
		readonly folder: string,
		readonly bundle: string,
		readonly duration: number,
		readonly sourceRevision: string,
	) {}
	get file() {
		return join(this.folder, "analysis-suggestions.json");
	}
	get status() {
		return {
			running: !!this.child,
			phase: this.phase,
			available: !!this.result,
		};
	}
	async restore() {
		const file = Bun.file(join(this.folder, "transcription-fills.json"));
		if (await file.exists()) {
			this.fills = fillsSchema.parse(await file.json());
			if (this.fills.sourceRevision !== this.sourceRevision)
				throw Error(
					"Transcription fills belong to a different source revision",
				);
			this.fillRevision = digest(this.fills);
		}
		if (await Bun.file(this.file).exists()) {
			const previous = await Bun.file(this.file).json();
			if (
				previous.sourceRevision === this.sourceRevision &&
				previous.fillRevision === this.fillRevision
			)
				this.result = previous;
		}
	}
	start(kind: "beats" | "charts", beats: number[], rests:Draft["rests"] = []) {
		if (this.child) return false;
		this.cancelling = false;
		this.phase = "Starting analysis";
		const child = spawn(
			process.execPath,
			[
				join(import.meta.dir, "analyze.ts"),
				kind,
				this.bundle,
				String(this.duration),
			],
			{ detached: true, stdio: ["pipe", "pipe", "pipe"] },
		);
		child.stdin.on("error", () => {});
		child.stdin.end(JSON.stringify({ beats, rests, fills: this.fills }));
		this.child = child;
		let output = "",
			log = "";
		child.stdout.on("data", (chunk) => {
			output += chunk;
		});
		child.stderr.on("data", (chunk) => {
			log = (log + chunk).slice(-4000);
			if (!this.cancelling)
				this.phase =
					log.trim().split("\n").at(-1)?.slice(0, 250) || "Analyzing";
		});
		child.on("error", (error) => {
			this.phase = error.message;
		});
		child.on("close", async (code) => {
			try {
				if (this.cancelling) {
					this.phase = "Cancelled. Previous suggestions retained.";
					return;
				}
				if (code !== 0) {
					this.phase = "Analysis failed. " + log.trim().slice(-400);
					return;
				}
				const data = JSON.parse(output) as Pick<
					Suggestions,
					"beats" | "charts" | "activeGaps"
				>;
				const result = {
					...data,
					activeGaps: kind === "beats" ? this.result?.activeGaps : data.activeGaps,
					charts: kind === "beats" ? (this.result?.charts ?? {}) : data.charts,
					sourceRevision: this.sourceRevision,
					fillRevision: this.fillRevision,
					chartGrid:
						kind === "beats"
							? (this.result?.chartGrid ?? [])
							: beats.length
								? beats
								: data.beats,
				};
				await atomicJson(this.file, result);
				this.result = result;
				this.phase =
					kind === "beats"
						? "Beat suggestions ready"
						: "Beat and chart suggestions ready";
			} catch (error) {
				this.phase = "Analysis failed. " + String(error);
			} finally {
				this.child = undefined;
			}
		});
		return true;
	}
	cancel() {
		if (!this.child?.pid) return;
		this.cancelling = true;
		this.phase = "Cancelling";
		try {
			process.kill(-this.child.pid, "SIGTERM");
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
		}
	}
}
