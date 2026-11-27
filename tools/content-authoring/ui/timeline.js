import { clock } from "./edits.js";

function surface(canvas) {
	const w = canvas.clientWidth,
		h = canvas.clientHeight,
		r = window.devicePixelRatio || 1;
	if (
		canvas.width !== Math.round(w * r) ||
		canvas.height !== Math.round(h * r)
	) {
		canvas.width = Math.round(w * r);
		canvas.height = Math.round(h * r);
	}
	const ctx = canvas.getContext("2d");
	ctx.setTransform(r, 0, 0, r, 0, 0);
	ctx.clearRect(0, 0, w, h);
	return { w, h, ctx };
}
function rect(ctx, color, x, y, width, height) {
	ctx.fillStyle = color;
	ctx.fillRect(x, y, width, height);
}
export function overview(canvas, { wave, range, loop, time, duration }) {
	const { w, h, ctx } = surface(canvas);
	rect(
		ctx,
		loop ? "#324733" : "#25322b",
		(w * range[0]) / duration,
		0,
		(w * (range[1] - range[0])) / duration,
		h,
	);
	if (wave)
		for (let x = 0; x < w; x += 2) {
			const peak =
				wave[Math.min(wave.length - 1, Math.floor((x / w) * wave.length))];
			const amp = Math.max(1, Math.min(1, peak) * h * 0.48);
			rect(
				ctx,
				x <= (time / duration) * w ? "#b4ef9e" : "#81a58c",
				x,
				h / 2 - amp,
				1.5,
				2 * amp,
			);
		}
	rect(ctx, "#ffcc78", Math.min(w - 2, (w * time) / duration), 0, 2, h);
}
export function pianoRoll(
	canvas,
	{
		notes,
		beats,
		time,
		zoom,
		duration,
		selected,
		range,
		loop,
		events,
		meter,
		suggested,
		sampleWave,
	},
) {
	const { w, h, ctx } = surface(canvas);
	const span = Math.min(zoom, duration),
		first = Math.max(0, Math.min(duration - span, time - span * 0.28)),
		last = first + span;
	const margin = 24,
		top = 22,
		row = (h - top) / 3;
	const x = (t) => margin + ((t - first) / span) * (w - margin);
	const posX = (px) => first + Math.max(0, (px - margin) / (w - margin)) * span;
	if (loop)
		rect(ctx, "#223028", x(range[0]), top, x(range[1]) - x(range[0]), h - top);
	ctx.font = "11px ui-monospace, monospace";
	ctx.fillStyle = "#b2bdb1";
	const step = span > 18 ? 4 : span > 9 ? 2 : 1;
	for (let t = Math.ceil(first / step) * step; t <= last; t += step) {
		rect(ctx, "#344039", x(t), top, 1, h - top);
		ctx.fillStyle = "#aeb8ad";
		ctx.fillText(clock(t), x(t) + 3, 14);
	}
	for (const b of beats) {
		const s = b / 1e6;
		if (s >= first && s < last) rect(ctx, "#4c564d", x(s), top, 1, 9);
	}
	for (const event of events) {
		if (event.endUs / 1e6 < first || event.startUs / 1e6 > last) continue;
		const left = Math.max(margin, x(event.startUs / 1e6)),
			right = Math.min(w, x(event.endUs / 1e6));
		rect(
			ctx,
			event.kind === "fixed" ? "#ffcc78" : "#c4b3ea",
			left,
			top - 4,
			Math.max(2, right - left),
			3,
		);
	}
	for (let i = 0; i < beats.length; i += meter) {
		const bx = x(beats[i] / 1e6);
		if (bx >= margin && bx <= w) rect(ctx, "#6f816e", bx, top, 1, h - top);
	}
	if (sampleWave)
		for (let px = margin; px < w; px += 2) {
			const peak =
				sampleWave[
					Math.min(
						sampleWave.length - 1,
						Math.floor((posX(px) / duration) * sampleWave.length),
					)
				];
			rect(
				ctx,
				"#344a39",
				px,
				top + row * 1.5 - peak * row * 0.4,
				1,
				Math.max(1, peak * row * 0.8),
			);
		}
	for (const n of suggested) {
		const t = n.timeUs / 1e6;
		if (t < first || t > last) continue;
		rect(
			ctx,
			"#dfb0df",
			x(t),
			top + (n.lane + 0.5) * row + 8,
			Math.max(3, (n.durationUs / 1e6 / span) * (w - margin)),
			2,
		);
	}
	const hits = [];
	for (let i = 0; i < 3; i++) {
		rect(ctx, "#344039", 0, top + i * row, w, 1);
		ctx.fillStyle = "#bac4b9";
		ctx.fillText(String(i + 1), 6, top + row * (i + 0.5) + 3);
	}
	for (const n of notes) {
		const start = n.timeUs / 1e6,
			end = start + n.durationUs / 1e6;
		if (end < first || start > last) continue;
		const left = Math.max(margin, x(start)),
			right = Math.min(w, Math.max(left + 6, x(end)));
		const cy = top + (n.lane + 0.5) * row;
		const color =
			n === selected ? "#ffcc78" : ["#b4ef9e", "#8cc9c5", "#c4b3ea"][n.lane];
		rect(ctx, color, left, cy - 2, Math.max(3, right - left), 5);
		rect(ctx, color, left, cy - 7, Math.min(6, Math.max(3, right - left)), 15);
		hits.push({ n, left, right, cy });
	}
	rect(ctx, "#ffcc78", Math.min(w - 2, x(time)), 0, 2, h);
	return {
		at: posX,
		hit: (px, py) =>
			hits
				.filter(
					(a) =>
						px >= a.left - 4 &&
						px <= a.right + 4 &&
						Math.abs(py - a.cy) < row * 0.45,
				)
				.sort((a, b) => Math.abs(a.left - px) - Math.abs(b.left - px))[0]?.n,
	};
}
