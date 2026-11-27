export const secondsToUs = (s) => Math.round(s * 1e6);
export function clock(seconds, precise = false) {
	const ms = Math.round(seconds * 1000);
	const time =
		Math.floor(ms / 60000) +
		":" +
		String(Math.floor(ms / 1000) % 60).padStart(2, "0");
	return precise ? time + "." + String(ms % 1000).padStart(3, "0") : time;
}
export function restRange(notes, start, end) {
	return notes
		.filter((n) => n.timeUs < start || n.timeUs >= end)
		.map((n) =>
			n.timeUs < start && n.timeUs + n.durationUs > start
				? { ...n, durationUs: start - n.timeUs }
				: n,
		);
}
export function upsert(notes, old, note, duration) {
	if (
		!Number.isSafeInteger(note.timeUs) ||
		!Number.isSafeInteger(note.durationUs) ||
		note.timeUs < 0 ||
		note.durationUs < 0 ||
		note.timeUs >= duration ||
		note.timeUs + note.durationUs > duration
	)
		throw Error("The note must fit inside the recording.");
	if (
		notes.some(
			(n) => n !== old && n.lane === note.lane && n.timeUs === note.timeUs,
		)
	)
		throw Error("There is already a note here on this lane.");
	return [...notes.filter((n) => n !== old), note].sort(
		(a, b) => a.timeUs - b.timeUs || a.lane - b.lane,
	);
}
