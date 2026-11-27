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
