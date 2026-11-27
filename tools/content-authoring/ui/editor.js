import { findings } from "./review.js";
import { Transport, peaks } from "./transport.js";
import { overview, pianoRoll } from "./timeline.js";
import { clock, restRange, secondsToUs, upsert } from "./edits.js";

const $ = (id) => document.getElementById(id);
async function api(path, body) {
	const response = await fetch(
		path,
		body === undefined
			? { cache: "no-store" }
			: {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify(body),
				},
	);
	if (!response.ok) throw Error(await response.text());
	return response.json();
}
let messageTimer;
const report = (message) => {
	clearTimeout(messageTimer);
	$("message").textContent = message;
	if (message === "Draft saved to the project folder.")
		messageTimer = setTimeout(() => {
			$("message").textContent = "";
		}, 3000);
};
const action = (id, fn) =>
	($(id).onclick = async () => {
		try {
			await fn();
		} catch (error) {
			report(error.message);
		}
	});
function option(select, name, value) {
	select.add(new Option(name, value));
}
function button(label, title, fn) {
	const b = document.createElement("button");
	b.textContent = label;
	b.title = title;
	b.setAttribute("aria-label", title);
	b.onclick = fn;
	return b;
}
async function init() {
	const session = await api("session");
	const { roles, durationUs } = session,
		duration = durationUs / 1e6;
	let { draft, revision } = session,
		source = roles[0].source,
		difficulty = "Normal",
		selected = null;
	let modified = false,
		saving = false,
		history = [],
		wave,
		hitMap,
		suggestions = null,
		jobTimer,
		lastDraw = -1,
		force = true;
	let prior = null;
	let generation = 0;
	const layerWaves = new Map();
	const player = new Transport(duration);
	let pendingPlay = false;
	let loadingAudio = false;
	const chartKey = () => source + "/" + difficulty,
		chart = () => draft.charts[chartKey()];
	function comparisonNotes() {
		if ($("overlay").value === "generated") return suggestions?.charts[chartKey()] ?? [];
		if ($("overlay").value === "prior") return prior?.charts[chartKey()] ?? [];
		return null;
	}
	const range = () => [Number($("start").value), Number($("end").value)];
	const validRange = () => {
		const [a, b] = range();
		if (
			!$("start").value ||
			!$("end").value ||
			a < 0 ||
			b - a < 0.05 ||
			b > duration
		)
			throw Error("Choose at least 0.05 s inside the song.");
		return [a, b];
	};
	$("title").textContent = session.title;
	document.title = session.title + " · Song editor";
	$("meta").textContent =
		roles.length +
		" layers  /  " +
		Object.keys(draft.charts).length +
		" charts";
	$("length").textContent = "/ " + clock(duration, true);
	$("scrub").max = duration;
	$("start").max = duration;
	$("end").max = duration;
	$("end").value = Math.min(8, duration);
	$("overviewTimes").replaceChildren(
		...Array.from({ length: 5 }, (_, i) => {
			const s = document.createElement("span");
			s.textContent = clock((duration * i) / 4);
			return s;
		}),
	);
	for (const id of session.projects) option($("project"), id, id);
	$("project").value = session.projectId;
	$("project").onchange = () => {
		location.assign("../" + encodeURIComponent($("project").value) + "/");
	};
	for (const name of roles[0].difficulties) option($("difficulty"), name, name);
	$("difficulty").value = difficulty;
	const layerButtons = new Map();
	for (const role of roles) {
		option($("eventLayer"), role.layer, role.source);
		const row = document.createElement("div");
		row.className = "layer";
		const name =
			role.layer +
			(role.role && role.role.toLowerCase() !== role.layer.toLowerCase()
				? " / " + role.role
				: "");
		const choose = button(name, "Edit " + name, () => {
			player.stopClicks?.();
			source = role.source;
			selected = null;
			refresh();
		});
		choose.className = "choose";
		if (role.review) choose.title = role.review;
		const mute = button("M", "Mute " + role.layer, () => {
			player.master = false;
			toggleSet(player.muted, role.source);
			player.mix();
			refresh();
		});
		const solo = button("S", "Solo " + role.layer, () => {
			player.master = false;
			toggleSet(player.solo, role.source);
			player.muted.delete(role.source);
			player.mix();
			refresh();
		});
		row.append(choose, mute, solo);
		$("layers").append(row);
		layerButtons.set(role.source, { choose, mute, solo });
	}
	function toggleSet(set, value) {
		if (set.has(value)) set.delete(value);
		else set.add(value);
	}
	function dirty() {
		report("");
		generation++;
		modified = true;
		force = true;
		refresh();
	}
	function remember() {
		history.push(structuredClone(draft));
		if (history.length > 32) history.shift();
	}
	function edit(fn) {
		const before = structuredClone(draft);
		fn();
		history.push(before);
		if (history.length > 32) history.shift();
		dirty();
	}
	function passages() {
		const select = $("passage");
		select.replaceChildren(new Option("Choose a passage", ""));
		draft.passages.forEach((p, i) => option(select, p.name, String(i)));
	}
	function showActivity() {
		const [a, b] = range().map(secondsToUs),
			table = document.createElement("table");
		const row = (values, head) => {
			const tr = table.insertRow();
			for (const value of values) {
				const cell = document.createElement(head ? "th" : "td");
				cell.textContent = value;
				tr.append(cell);
			}
		};
		row(["Layer", "Easy", "Normal", "Hard"], true);
		for (const r of roles)
			row([
				r.layer,
				...r.difficulties.map(
					(d) =>
						draft.charts[r.source + "/" + d].filter(
							(n) => n.timeUs >= a && n.timeUs < b,
						).length,
				),
			]);
		$("activity").replaceChildren(table);
	}
	function reviewTracks() {
		$("meter").value = draft.meter;
		const events = $("eventList"),
			id = events.value;
		events.replaceChildren(new Option("New event", ""));
		for (const e of draft.events) option(events, e.name, e.id);
		events.value = draft.events.some((e) => e.id === id) ? id : "";
		$("dropEvent").disabled = !events.value;
		const warnings = findings(draft);
		if ($("overlay").value === "generated" && difficulty === "Normal") {
			for (const gap of suggestions?.activeGaps?.[source] ?? []) {
				warnings.push({at: secondsToUs(gap.start), text: source + " stem activity with no suggested note through " + clock(gap.end, true) + "."});
			}
		}
		$("findings").replaceChildren(
			...warnings.map((f) =>
				button(clock(f.at / 1e6, true) + " · " + f.text, f.text, () =>
					seek(f.at / 1e6),
				),
			),
		);
		if (!warnings.length)
			$("findings").textContent =
				"No clock, marked-rest or encounter overlap findings.";
		$("rests").replaceChildren(
			...draft.rests
				.filter((r) => r.source === source)
				.map((r) =>
					button(
						"Rest " +
							clock(r.startUs / 1e6) +
							"–" +
							clock(r.endUs / 1e6) +
							" ×",
						"Remove this rest marker",
						() =>
							edit(() => {
								draft.rests = draft.rests.filter((v) => v !== r);
							}),
					),
				),
		);
	}
	function refresh() {
		force = true;
		$("saved").textContent = saving
			? "Saving…"
			: modified
				? "Unsaved changes"
				: "Saved";
		$("save").disabled = !modified || saving;
		$("undo").disabled = !history.length;
		$("master").setAttribute("aria-pressed", String(player.master));
		for (const [id, { choose, mute, solo }] of layerButtons) {
			choose.classList.toggle("selected", source === id);
			choose.setAttribute("aria-current", String(source === id));
			mute.setAttribute(
				"aria-pressed",
				String(!player.master && player.muted.has(id)),
			);
			solo.setAttribute(
				"aria-pressed",
				String(!player.master && player.solo.has(id)),
			);
		}
		const role = roles.find((r) => r.source === source);
		$("chartTitle").textContent =
			(role.role ?? role.layer).replace(/^./, (c) => c.toUpperCase()) +
			" chart";
		const comparison = comparisonNotes();
		$("chartCount").textContent = comparison === null
			? chart().length + " notes · " + difficulty
			: chart().length + " draft · " + comparison.length + " " + ($("overlay").value === "prior" ? "previous" : "suggested") + " · " + difficulty;
		$("delete").disabled = $("update").disabled = !selected;
		$("noteTime").disabled =
			$("hold").disabled =
			$("lane").disabled =
				!selected;
		$("noteTime").value = selected ? selected.timeUs / 1e6 : "";
		$("hold").value = selected ? selected.durationUs / 1e6 : "";
		$("lane").value = selected ? String(selected.lane) : "0";
		$("selection").textContent = selected
			? "Selected · " +
				clock(selected.timeUs / 1e6, true) +
				" · lane " +
				(selected.lane + 1)
			: "Click a note to edit it. Click empty space to seek.";
		$("beatCount").textContent =
			draft.beats.length +
			" beats in draft" +
			(suggestions
				? " · " +
					suggestions.beats.length +
					(draft.beats.length
						? " suggested"
						: " detected beats shown for review")
				: "");
		$("applyBeats").disabled = !suggestions?.beats.length;
		const activeGrid = draft.beats.length ? draft.beats : suggestions?.beats;
		const stale =
			suggestions?.charts[chartKey()] &&
			JSON.stringify(activeGrid) !== JSON.stringify(suggestions.chartGrid);
		$("applyChart").disabled = !suggestions?.charts[chartKey()] || !!stale;
		if (stale)
			$("beatCount").textContent +=
				" · Chart suggestions need a rerun for this clock.";
		$("loop").setAttribute("aria-pressed", String(!!player.loop));
		showActivity();
		reviewTracks();
	}
	function seek(t) {
		if (player.loop && (t < player.loop[0] || t >= player.loop[1]))
			player.setLoop(null);
		player.seek(t);
		force = true;
		$("loop").setAttribute("aria-pressed", String(!!player.loop));
	}
	function selectNote(n) {
		selected = n;
		seek(n.timeUs / 1e6);
		refresh();
	}
	function navigate(dir) {
		const notes = chart();
		if (!notes.length) return;
		const index = notes.indexOf(selected);
		if (index !== -1) {
			selectNote(notes[Math.max(0, Math.min(notes.length - 1, index + dir))]);
			return;
		}
		const t = secondsToUs(player.now());
		selectNote(
			dir > 0
				? (notes.find((n) => n.timeUs >= t) ?? notes.at(-1))
				: (notes.findLast((n) => n.timeUs <= t) ?? notes[0]),
		);
	}
	function draw(stamp) {
		const time = player.now();
		if (force || (player.playing && stamp - lastDraw > 22)) {
			if (player.playing && !player.loop && time >= duration) player.stop();
			$("time").textContent = clock(time, true);
			$("scrub").value = time;
			$("play").textContent = player.playing
				? "Ⅱ Pause"
				: pendingPlay
					? "Cancel playback request"
					: "▶ Play";
			const state = {
				wave,
				time,
				duration,
				range: range(),
				loop: !!player.loop,
			};
			overview($("wave"), state);
			hitMap = pianoRoll($("roll"), {
				...state,
				notes: chart(),
				beats: draft.beats.length ? draft.beats : (suggestions?.beats ?? []),
				selected,
				zoom: Number($("zoom").value),
				events: draft.events.filter(
					(e) => e.source === "all" || e.source === source,
				),
				meter: draft.meter,
				suggested: comparisonNotes() ?? [],
				sampleWave: layerWaves.get(source),
			});
			force = false;
			lastDraw = stamp;
		}
		requestAnimationFrame(draw);
	}
	// Schedule optional note taps ahead on the very same clock as the music.
	let scheduledTo = 0;
	const clicks = new Set();
	player.stopClicks = () => {
		scheduledTo = 0;
		for (const node of clicks) node.stop();
		clicks.clear();
	};
	player.scheduleClicks = () => {
		if (!player.playing || !$("clicks").checked) return;
		const now = player.ctx.currentTime,
			from = Math.max(now, player.started, scheduledTo),
			until = now + 0.13;
		if (from >= until) return;
		scheduledTo = until;
		const start = player.position,
			loop = player.loop,
			period = loop ? loop[1] - loop[0] : 0;
		for (const n of comparisonNotes() ?? chart()) {
			const t = n.timeUs / 1e6;
			if (loop && (t < loop[0] || t >= loop[1])) continue;
			let when = player.started + (t - start);
			if (period && when < from)
				when += Math.ceil((from - when) / period) * period;
			if (when < from) continue;
			while (when < until) {
				const osc = player.ctx.createOscillator(),
					gain = player.ctx.createGain();
				osc.frequency.value = [720, 960, 1200][n.lane];
				gain.gain.setValueAtTime(0.12, when);
				gain.gain.exponentialRampToValueAtTime(0.001, when + 0.035);
				osc.connect(gain).connect(player.ctx.destination);
				osc.start(when);
				osc.stop(when + 0.04);
				clicks.add(osc);
				osc.onended = () => {
					osc.disconnect();
					gain.disconnect();
					clicks.delete(osc);
				};
				if (!period) break;
				when += period;
			}
		}
	};
	setInterval(player.scheduleClicks, 50);
	$("clicks").onchange = () => {
		player.stopClicks();
	};
	const play = async () => {
		if (pendingPlay) {
			pendingPlay = false;
			force = true;
			return;
		}
		await player.ctx.resume();
		if (!player.ready) {
			pendingPlay = true;
			force = true;
			if (!loadingAudio) await loadAudio();
			return;
		}
		if (player.playing) player.stop();
		else player.start();
		force = true;
	};
	action("play", play);
	action("home", () => seek(0));
	action("master", () => {
		player.master = true;
		player.solo.clear();
		player.muted.clear();
		player.mix();
		refresh();
	});
	$("scrub").oninput = () => seek(Number($("scrub").value));
	$("roll").onclick = (e) => {
		if (!hitMap) return;
		const rect = $("roll").getBoundingClientRect();
		const px = e.clientX - rect.left,
			n = hitMap.hit(px, e.clientY - rect.top);
		if (n) selectNote(n);
		else {
			selected = null;
			seek(hitMap.at(px));
			refresh();
		}
	};
	$("difficulty").onchange = () => {
		player.stopClicks?.();
		difficulty = $("difficulty").value;
		selected = null;
		refresh();
	};
	$("zoom").onchange = () => {
		force = true;
	};
	function choosePassage(index) {
		const p = draft.passages[index];
		if (!p) return;
		$("start").value = p.startUs / 1e6;
		$("end").value = p.endUs / 1e6;
		player.setLoop([p.startUs / 1e6, p.endUs / 1e6]);
		seek(p.startUs / 1e6);
		refresh();
	}
	$("passage").onchange = () => {
		if ($("passage").value !== "") choosePassage(Number($("passage").value));
	};
	for (const id of ["start", "end"])
		$(id).onchange = () => {
			try {
				const bounds = validRange();
				if (player.loop) player.setLoop(bounds);
				$("passage").value = "";
				refresh();
			} catch (e) {
				player.setLoop(null);
				refresh();
				report(e.message);
			}
		};
	action("loop", () => {
		player.setLoop(player.loop ? null : validRange());
		refresh();
	});
	action("mark", () => {
		const [a, b] = validRange();
		const name = prompt("Name this passage");
		if (!name?.trim()) return;
		const p = {
			name: name.trim().slice(0, 100),
			startUs: secondsToUs(a),
			endUs: secondsToUs(b),
		};
		edit(() => {
			const i = draft.passages.findIndex((v) => v.name === p.name);
			if (i >= 0) draft.passages[i] = p;
			else draft.passages.push(p);
		});
		passages();
		$("passage").value = String(
			draft.passages.findIndex((v) => v.name === p.name),
		);
	});
	action("prev", () => navigate(-1));
	action("next", () => navigate(1));
	action("add", () => {
		const note = {
			timeUs: secondsToUs(player.now()),
			durationUs: 0,
			lane: selected?.lane ?? 0,
		};
		const notes = upsert(chart(), null, note, durationUs);
		selected = note;
		edit(() => {
			draft.charts[chartKey()] = notes;
		});
	});
	action("update", () => {
		if (!$("noteTime").value || !$("hold").value)
			throw Error("Enter a start time and hold length.");
		const n = {
			timeUs: secondsToUs(Number($("noteTime").value)),
			durationUs: secondsToUs(Number($("hold").value)),
			lane: Number($("lane").value),
		};
		const notes = upsert(chart(), selected, n, durationUs);
		selected = n;
		edit(() => {
			draft.charts[chartKey()] = notes;
		});
		seek(n.timeUs / 1e6);
	});
	action("delete", () => {
		edit(() => {
			draft.charts[chartKey()] = chart().filter((n) => n !== selected);
			selected = null;
		});
	});
	action("rest", () => {
		const [a, b] = validRange().map(secondsToUs);
		if (
			!confirm(
				"Remove " +
					source +
					" notes from " +
					clock(a / 1e6, true) +
					" to " +
					clock(b / 1e6, true) +
					" on all difficulties? Undo is available.",
			)
		)
			return;
		edit(() => {
			if (
				!draft.rests.some(
					(r) => r.source === source && r.startUs === a && r.endUs === b,
				)
			)
				draft.rests.push({ source, startUs: a, endUs: b });
			for (const d of roles.find((r) => r.source === source).difficulties) {
				const k = source + "/" + d;
				draft.charts[k] = restRange(draft.charts[k], a, b);
			}
			selected = null;
		});
	});
	action("undo", () => {
		if (history.length) {
			draft = history.pop();
			selected = null;
			$("notes").value = draft.notes;
			passages();
			dirty();
		}
	});
	$("notes").value = draft.notes;
	$("notes").onfocus = () => remember();
	$("notes").oninput = () => {
		draft.notes = $("notes").value;
		dirty();
	};
	async function save() {
		if (saving) throw Error("Save in progress.");
		if (!modified) return;
		saving = true;
		refresh();
		const at = generation;
		try {
			const result = await api("save", { draft, revision });
			revision = result.revision;
			if ($("overlay").value === "prior") prior = await api("prior");
			if (at === generation) modified = false;
			report("Draft saved to the project folder.");
		} catch (error) {
			$("backup").hidden = false;
			throw error;
		} finally {
			saving = false;
			refresh();
		}
	}
	action("save", save);
	async function download(route, filename) {
		await save();
		if (modified) throw Error("Save the latest edits before exporting.");
		const res = await fetch(route + "?revision=" + revision);
		if (!res.ok) throw Error(await res.text());
		const href = URL.createObjectURL(await res.blob()),
			a = document.createElement("a");
		a.href = href;
		a.download = filename;
		a.click();
		setTimeout(() => URL.revokeObjectURL(href), 3000);
	}
	action("export", () => download("export", "song-draft.json"));
	action("studio", () => download("studio", "SongAuthoringDraft.rbxmx"));
	action("backup", () => {
		const url = URL.createObjectURL(
			new Blob([JSON.stringify(draft)], { type: "application/json" }),
		);
		const a = document.createElement("a");
		a.href = url;
		a.download = "unsaved-song-draft.json";
		a.click();
		setTimeout(() => URL.revokeObjectURL(url), 3000);
	});
	async function checkJob() {
		clearTimeout(jobTimer);
		const job = await api("job");
		$("job").textContent = job.phase;
		$("cancel").disabled = !job.running;
		$("run").disabled = job.running;
		if (!job.running && job.available) {
			suggestions = await api("suggestions");
			refresh();
		}
		if (job.running)
			jobTimer = setTimeout(
				() => checkJob().catch((e) => report(e.message)),
				550,
			);
	}
	action("run", async () => {
		if ($("processor").value === "charts") await save();
		await api("job", { kind: $("processor").value });
		await checkJob();
	});
	action("cancel", async () => {
		await api("job/cancel", {});
		await checkJob();
	});
	$("overlay").onchange = async () => {
		player.stopClicks();
		try {
			if ($("overlay").value === "prior") prior = await api("prior");
		} catch (e) {
			report(e.message);
		}
		refresh();
	};
	$("meter").onchange = () => {
		const n = Number($("meter").value);
		if (Number.isInteger(n) && n >= 1 && n <= 16)
			edit(() => {
				draft.meter = n;
			});
		else refresh();
	};
	function changeBeat(kind) {
		const t = secondsToUs(player.now());
		if (t >= durationUs) throw Error("Seek inside the song first.");
		const beats = [...draft.beats],
			nearest = beats.reduce(
				(best, v, i) =>
					best < 0 || Math.abs(v - t) < Math.abs(beats[best] - t) ? i : best,
				-1,
			);
		if (kind !== "add") {
			if (nearest < 0) throw Error("No beat to edit.");
			beats.splice(nearest, 1);
		}
		if (kind !== "remove") {
			if (beats.includes(t)) throw Error("A beat already exists here.");
			beats.push(t);
		}
		edit(() => {
			draft.beats = beats.sort((a, b) => a - b);
		});
	}
	action("setBeat", () => changeBeat("add"));
	action("moveBeat", () => changeBeat("move"));
	action("removeBeat", () => changeBeat("remove"));
	$("eventList").onchange = () => {
		const e = draft.events.find((v) => v.id === $("eventList").value);
		$("eventName").value = e?.name ?? "";
		$("eventKind").value = e?.kind ?? "fixed";
		$("eventLayer").value = e?.source ?? "all";
		$("warning").value = (e?.warningUs ?? 0) / 1e6;
		$("reviewed").checked = !!e?.reviewed;
		if (e) {
			player.setLoop(null);
			$("start").value = e.startUs / 1e6;
			$("end").value = e.endUs / 1e6;
			$("passage").value = "";
			seek(e.startUs / 1e6);
		}
		refresh();
	};
	action("writeEvent", () => {
		const [a, b] = validRange(),
			name = $("eventName").value.trim(),
			warn = Number($("warning").value);
		if (!name || !Number.isFinite(warn) || warn < 0)
			throw Error("Enter an event name and a nonnegative warning.");
		const id =
			$("eventList").value ||
			Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
				byte.toString(16).padStart(2, "0"),
			).join("");
		const e = {
			id,
			name,
			kind: $("eventKind").value,
			source: $("eventLayer").value,
			startUs: secondsToUs(a),
			endUs: secondsToUs(b),
			warningUs: secondsToUs(warn),
			reviewed: $("reviewed").checked,
		};
		edit(() => {
			draft.events = [...draft.events.filter((v) => v.id !== id), e].sort(
				(x, y) => x.startUs - y.startUs,
			);
		});
		$("eventList").value = id;
		$("dropEvent").disabled = false;
	});
	action("dropEvent", () => {
		const id = $("eventList").value;
		edit(() => {
			draft.events = draft.events.filter((e) => e.id !== id);
		});
		$("eventList").onchange();
	});
	action("applyBeats", () => {
		edit(() => {
			draft.beats = [...suggestions.beats];
		});
		report("Suggested beats copied into the draft.");
	});
	action("applyChart", () => {
		if (
			!confirm(
				"Replace " +
					chartKey() +
					" with the generated suggestion? Undo is available.",
			)
		)
			return;
		edit(() => {
			let notes = structuredClone(suggestions.charts[chartKey()]);
			for (const r of draft.rests.filter((r) => r.source === source))
				notes = restRange(notes, r.startUs, r.endUs);
			draft.charts[chartKey()] = notes;
			selected = null;
		});
	});
	window.addEventListener("beforeunload", (e) => {
		if (modified) {
			e.preventDefault();
			e.returnValue = "";
		}
	});
	document.addEventListener("keydown", (e) => {
		if (e.target.matches("input,textarea,button,select")) return;
		if (e.code === "Space" && !$("play").disabled) {
			e.preventDefault();
			play().catch((e) => report(e.message));
		}
		if (e.key === "ArrowRight") navigate(1);
		if (e.key === "ArrowLeft") navigate(-1);
	});
	new ResizeObserver(() => {
		force = true;
	}).observe($("roll"));
	passages();
	refresh();
	const query = new URLSearchParams(location.search),
		audition = Number(query.get("passage"));
	if (query.get("compare") === "generated") {
		$("overlay").value = "generated";
		$("clicks").checked = true;
	}
	const at = Number(query.get("at"));
	if (query.has("at") && Number.isFinite(at) && at >= 0 && at < duration) seek(at);
	if (roles.some((r) => r.source === query.get("layer")))
		source = query.get("layer");
	if (
		query.has("passage") &&
		Number.isInteger(audition) &&
		draft.passages[audition]
	) {
		$("passage").value = String(audition);
		choosePassage(audition);
	} else refresh();
	requestAnimationFrame(draw);
	checkJob().catch((error) => {
		$("job").textContent = error.message;
	});
	const mismatches = [];
	function updatePlayback() {
		if (pendingPlay && player.ready) {
			pendingPlay = false;
			player.start();
		}
		for (const [id, { mute, solo }] of layerButtons) {
			mute.disabled = solo.disabled = !player.buffers.has(id);
		}
		force = true;
	}
	$("play").disabled = false;
	updatePlayback();
	async function loadAudio() {
		if (loadingAudio) return;
		loadingAudio = true;
		$("retryAudio").hidden = true;
		$("audio").textContent = "Loading audio…";
		try {
			const failures = await player.load(
				["master", ...roles.map((r) => r.source)],
				(id, buffer) => {
					if (id === "master") wave = peaks(buffer);
					else layerWaves.set(id, peaks(buffer, 24000));
					if (Math.abs(buffer.duration - duration) > 0.05) mismatches.push(id);
					$("audio").textContent =
						"Loaded " +
						player.buffers.size +
						" / " +
						(roles.length + 1) +
						" recordings";
					updatePlayback();
				},
			);
			$("audio").textContent = mismatches.length
				? "Check alignment · duration differs for " + mismatches.join(", ")
				: "Audio ready · M mute / S solo";
			if (failures.length) {
				const status = player.buffers.has("master")
					? "Master ready."
					: player.ready
						? "Playing mix available from stems."
						: "Audio unavailable.";
				$("audio").textContent = status + " " + failures.join(" · ");
				$("retryAudio").hidden = false;
			}
			if (!player.ready) pendingPlay = false;
			updatePlayback();
		} catch (error) {
			pendingPlay = false;
			force = true;
			$("audio").textContent = error.message;
			$("retryAudio").hidden = false;
		} finally {
			loadingAudio = false;
		}
	}
	$("retryAudio").onclick = loadAudio;
	await loadAudio();
}
init().catch((e) => {
	report(e.message);
	$("audio").textContent = "Could not finish opening the editor.";
});
