// Half-open spans throughout: a prompt on the end boundary belongs to the next passage.
export function findings(draft) {
	const issues = [];
	for (const e of draft.events) {
		if (e.warningUs > e.startUs)
			issues.push({
				at: e.startUs,
				text: e.name + " needs a warning before the song starts.",
			});
		if (!e.reviewed)
			issues.push({
				at: e.startUs,
				text: e.name + " awaits listening review.",
			});
		if (
			e.kind === "candidate" &&
			draft.events.some(
				(f) =>
					f.kind === "fixed" &&
					(f.source === "all" || e.source === "all" || f.source === e.source) &&
					f.startUs - f.warningUs < e.endUs &&
					f.endUs > e.startUs - e.warningUs,
			)
		)
			issues.push({
				at: e.startUs,
				text: e.name + " overlaps a fixed event or its warning.",
			});
	}
	for (const r of draft.rests) {
		for (const [id, notes] of Object.entries(draft.charts)) {
			if (
				id.startsWith(r.source + "/") &&
				notes.some(
					(n) =>
						n.timeUs < r.endUs &&
						(n.timeUs >= r.startUs || n.timeUs + n.durationUs > r.startUs),
				)
			)
				issues.push({
					at: r.startUs,
					text: id + " has notes inside a marked rest.",
				});
		}
	}
	if (!draft.beats.length)
		issues.push({ at: 0, text: "The draft has no reviewed beat grid." });
	return issues;
}
export function activity(draft, duration) {
	const cuts = [...new Set([0, ...draft.beats, duration])].sort(
		(a, b) => a - b,
	);
	return Object.fromEntries(
		Object.entries(draft.charts).map(([key, notes]) => {
			let index = 0,
				holdEnd = -1;
			return [
				key,
				cuts.slice(0, -1).map((startUs, i) => {
					const endUs = cuts[i + 1],
						crossingHold = holdEnd > startUs,
						first = index;
					const next = notes[index]?.timeUs;
					while (index < notes.length && notes[index].timeUs < endUs) {
						const n = notes[index++];
						holdEnd = Math.max(holdEnd, n.timeUs + n.durationUs);
					}
					return {
						startUs,
						endUs,
						noteStarts: index - first,
						crossingHold,
						nextNoteUs: next ?? null,
					};
				}),
			];
		}),
	);
}

// Fraction of the song clock occupied by beats containing a note start or hold.
// Extend the nearest observed tempo only for measurement outside the beat grid.
export function roleWorkload(charts, durationUs, beats) {
	const beatBounds = (time) => {
		let low = 0, high = beats.length;
		while (low < high) {
			const mid = Math.floor((low + high) / 2);
			if (beats[mid] <= time) low = mid + 1;
			else high = mid;
		}
		if (low > 0 && low < beats.length) return [beats[low - 1], beats[low]];
		const before = low === 0;
		const anchor = before ? beats[0] : beats.at(-1);
		const period = before ? beats[1] - anchor : anchor - beats.at(-2);
		const start = anchor + Math.floor((time - anchor) / period) * period;
		return [Math.max(0, start), Math.min(durationUs, start + period)];
	};
	return Object.fromEntries(Object.entries(charts).map(([id, notes]) => {
		let activeUs = 0, coveredUntil = 0;
		if (beats.length >= 2) {
			for (const n of notes) {
				const [start] = beatBounds(n.timeUs);
				const [, end] = beatBounds(n.timeUs + Math.max(0, n.durationUs - 1));
				activeUs += Math.max(0, end - Math.max(start, coveredUntil));
				coveredUntil = Math.max(coveredUntil, end);
			}
		}
		return [id, { noteCount: notes.length, activeBeatFraction: beats.length >= 2 ? activeUs / durationUs : null }];
	}));
}
