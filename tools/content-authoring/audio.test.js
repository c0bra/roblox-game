import { expect, test } from "bun:test";
import { loadRecording } from "./ui/transport.js";

test("recovers a rejected full audio response using verified byte ranges", async () => {
	const audio = new Uint8Array(1024 * 1024 + 5).fill(7);
	const requests = [];
	const result = await loadRecording("master", async (_url, init) => {
		const range = init.headers?.Range;
		requests.push(range);
		if (!range) return new Response(null, { status: 502 });
		const [, start, end] = range.match(/^bytes=(\d+)-(\d+)$/).map(Number);
		const last = Math.min(end, audio.length - 1);
		return new Response(audio.slice(start, last + 1), {
			status: 206,
			headers: { "Content-Range": `bytes ${start}-${last}/${audio.length}` },
		});
	});
	expect(requests.length).toBe(3);
	expect(new Uint8Array(result)).toEqual(audio);
});

test("a missing recording includes its name and HTTP status", async () => {
	await expect(
		loadRecording("master", async () => new Response(null, { status: 404 })),
	).rejects.toThrow(/master.*404/);
});

test("rejects gaps in a partial recording", async () => {
	const request = async (_url, init) =>
		init.headers?.Range
			? new Response(new Uint8Array([1]), {
					status: 206,
					headers: { "Content-Range": "bytes 3-3/5" },
				})
			: new Response(null, { status: 503 });
	await expect(loadRecording("master", request)).rejects.toThrow(/range/i);
});
