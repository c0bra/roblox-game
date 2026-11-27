export async function loadRecording(id, request = fetch) {
	const url = "audio/" + encodeURIComponent(id);
	try {
		const response = await request(url, {
			cache: "no-store",
			signal: AbortSignal.timeout(20000),
		});
		if (response.status === 200) return await response.arrayBuffer();
	} catch {
		// A connection can also drop after the response headers arrive.
	}
	const chunkSize = 1024 * 1024;
	let audio;
	let offset = 0;
	do {
		const last = offset + chunkSize - 1;
		const response = await request(url, {
			cache: "no-store",
			headers: { Range: `bytes=${offset}-${last}` },
			signal: AbortSignal.timeout(20000),
		});
		if (response.status === 200) return await response.arrayBuffer();
		if (response.status !== 206)
			throw Error(`Cannot open ${id} (HTTP ${response.status}).`);
		const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(
			response.headers.get("Content-Range") ?? "",
		);
		const bytes = new Uint8Array(await response.arrayBuffer());
		const total = Number(range?.[3]);
		if (
			!range ||
			!Number.isSafeInteger(total) ||
			total <= offset ||
			Number(range[1]) !== offset ||
			Number(range[2]) !== Math.min(last, total - 1) ||
			bytes.length !== Number(range[2]) - offset + 1 ||
			(audio && audio.length !== total)
		)
			throw Error(`Invalid audio range for ${id}.`);
		audio ??= new Uint8Array(total);
		audio.set(bytes, offset);
		offset += bytes.length;
	} while (offset < audio.length);
	return audio.buffer;
}

// All sources share a single sample clock, including seeks and loop wraps.
export class Transport {
	constructor(duration) {
		this.duration = duration;
		this.position = 0;
		this.playing = false;
		this.buffers = new Map();
		this.sources = [];
		this.master = true;
		this.solo = new Set();
		this.muted = new Set();
		this.loop = null;
		this.ready = false;
		this.ctx = new AudioContext({ sampleRate: 32000 });
	}
	async load(ids, progress) {
		const failures = [];
		for (const id of ids) {
			if (this.buffers.has(id)) continue;
			let buffer;
			try {
				buffer = await this.ctx.decodeAudioData(await loadRecording(id));
			} catch (error) {
				failures.push(id + ": " + error.message);
				continue;
			}
			this.buffers.set(id, buffer);
			if (this.playing) {
				const when = Math.max(this.ctx.currentTime + 0.025, this.started);
				this.connectBuffer(id, buffer, when, this.timeAt(when));
				this.mix();
			}
			if (id === "master") this.ready = true;
			progress(id, buffer);
		}
		this.ready = this.buffers.size > 0;
		return failures;
	}
	now() {
		return this.timeAt(this.ctx.currentTime);
	}
	timeAt(when) {
		if (!this.playing) return this.position;
		const pos = this.position + Math.max(0, when - this.started);
		if (this.loop && pos >= this.loop[1])
			return (
				this.loop[0] + ((pos - this.loop[1]) % (this.loop[1] - this.loop[0]))
			);
		return Math.min(this.duration, pos);
	}
	stop() {
		this.position = this.now();
		this.playing = false;
		for (const { source, gain } of this.sources) {
			source.stop();
			source.disconnect();
			gain.disconnect();
		}
		this.sources = [];
		this.stopClicks?.();
	}
	mix() {
		for (const { id, gain } of this.sources) {
			const active = this.master
				? this.buffers.has("master")
					? id === "master"
					: id !== "master"
				: id !== "master" &&
					!this.muted.has(id) &&
					(!this.solo.size || this.solo.has(id));
			gain.gain.setTargetAtTime(active ? 1 : 0, this.ctx.currentTime, 0.004);
		}
	}
	start() {
		if (!this.ready || this.playing) return;
		if (this.position >= this.duration) this.position = 0;
		if (
			this.loop &&
			(this.position < this.loop[0] || this.position >= this.loop[1])
		)
			this.position = this.loop[0];
		this.started = this.ctx.currentTime + 0.025;
		for (const [id, buffer] of this.buffers) {
			this.connectBuffer(id, buffer, this.started, this.position);
		}
		this.playing = true;
		this.scheduleClicks?.();
		this.mix();
	}
	connectBuffer(id, buffer, when, offset) {
		const source = this.ctx.createBufferSource(),
			gain = this.ctx.createGain();
		source.buffer = buffer;
		gain.gain.value = 0;
		if (this.loop) {
			source.loop = true;
			source.loopStart = this.loop[0];
			source.loopEnd = this.loop[1];
		}
		source.connect(gain).connect(this.ctx.destination);
		source.start(when, offset);
		this.sources.push({ id, source, gain });
	}
	seek(value) {
		const resume = this.playing;
		this.stop();
		this.position = Math.max(0, Math.min(this.duration, value));
		if (resume) this.start();
	}
	setLoop(range) {
		const resume = this.playing;
		this.stop();
		this.loop = range;
		if (range && (this.position < range[0] || this.position >= range[1]))
			this.position = range[0];
		if (resume) this.start();
	}
}
export function peaks(buffer, bins = 2400) {
	const samples = buffer.getChannelData(0),
		result = new Float32Array(bins);
	for (let i = 0; i < samples.length; i += 8) {
		const index = Math.min(bins - 1, Math.floor((i * bins) / samples.length));
		result[index] = Math.max(result[index], Math.abs(samples[i]));
	}
	return result;
}
