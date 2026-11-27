# Implementation Progress

The [implementation plan](IMPLEMENTATION_PLAN.md) defines the tasks and milestone checks. Update this file when a task changes state or a milestone passes. Record the task ID, revision, observed check and evidence link, blocker if any, and the next ready task.

| Task | State | Evidence or next action |
|---|---|---|
| T01 · Identify Blackened Crown inputs | In progress | Master and layers identified. Owner confirmed the vocals-only breakdown and guitar-only outro. Mark their exact boundaries against the master during authoring. |
| T02 · Reproduce the local build | Verified | Bun 1.3.12 frozen install, type/lint checks and 9 baseline tests passed. Rojo 7.7.0 built and synced the disposable place in Studio 0.748.0.7481526. Observations below. |
| T03 · Shared timing fixture | Verified | Bun and Studio read the same piano timeline and rejected six invalid inputs. The pipeline now passes 16 tests and the type/lint checks. |
| T04 · Open the song project | Verified | Opened Blackened Crown in the local browser app, inspected four layers and twelve charts, and saved the owner’s review notes. The Normal guitar import shows 51 notes. A regression check maps a note 0.125 s into a clip starting at 60 s to master time 60.125 s. [Run the app](tools/content-authoring/README.md) |
| T05 · Safe analysis reruns | Verified | Beat detection and full transcription run as cancellable local jobs. A new chart uses the saved grid, and generated suggestions stay separate from edits. |
| T06 · Synchronized listening and correction | Verified | Chromium played and looped the 2:35 sample with the vocal stem soloed. Edited a vocal note, saved, reopened, and verified its exact time in the exported chart. The editor flags a decoded layer length discrepancy. |
| T07–T08 · Charts, activity and encounter review | In progress | The editor derives three difficulties, displays per-layer activity, persists rests and encounter markers, and flags overlapping warnings. Full event lineage, contribution weights, candidate eligibility and roster coverage remain open. |
| T09–T15 | Waiting | The editor can export a local JSON or Studio draft for inspection. Approved staging content, audio bindings and in-game review follow the remaining gates. |
| Phases 3–6 | Planned | Expand each phase into individual tasks after the preceding milestone supplies its evidence. |

Use **Ready** when prerequisites are satisfied, **In progress** while someone owns the work, **Blocked** for a specific missing input or failed dependency, and **Verified** after the task's acceptance check passes. Use **Waiting** for ordinary task dependencies. Split grouped rows as tasks become ready. The initial entries describe the starting queue as of 2026-11-23.

## T01 · Blackened Crown input inventory

Inspected 2026-11-23 at source revision `bfab495dec32`. All times below use seconds from the first decoded master sample. The reusable inputs for T04 are the root mix and the matching local WAVs.

| Material | Finding and use |
|---|---|
| [Blackened Crown - New.mp3](Blackened%20Crown%20-%20New.mp3) | Working master, stereo 48 kHz. FFprobe reports a 199.920 s MP3 stream. Decoding with FFmpeg yields 199.880 s of audio. SHA-256 `e907ac5cfe33636831538ecd4f2168f753a9b1009a7706174223091d25480ffd` matches the input copy in the local build. |
| `build/blackened-crown-new/audio/stems/{drums,vocals,bass,guitar}.wav` | Matching full-length layers, stereo 44.1 kHz, 199.880 s each. All four hashes match the corresponding HTDemucs exports under `analysis/separated/`. The WAV named guitar matches the separator's Other output byte for byte. Preserve that distinction pending a listening decision. These generated files live under ignored `build/` on this machine. |
| `build/blackened-crown-new/manifest.json` and `charts/validation.json` | Legacy schema v1, HTDemucs, source offset 0, chart clip 188.710 s. Local command `./chart validate build/blackened-crown-new` passed. The 12 role/difficulty JSON files match the tracked [demo charts](roblox/web/public/levels/blackened-crown/charts/) byte for byte. Reuse as editable starting references. Convert medium to Normal explicitly and preserve the full master clock. |
| [Browser audio](roblox/web/public/levels/blackened-crown/audio/) | Eight 199.880 s stereo AAC files, one backing and one stem per legacy role. Playback references. Use the WAVs for authoring. |
| [Older processing folder](audio/Blackened%20Crown/) | The four labeled MP3 stems report 215.040 s. Treat these and their filtered WAV, MIDI and onset derivatives as historical references. The full-length WAVs above already match the selected master. |

**Alignment check.** FFmpeg decoded the master and four WAVs to mono at 8 kHz. Cross-correlating the summed WAVs with the master in three-second windows starting at 5, 95 and 190 s gave a best lag of zero samples within a ±1 s search at each point (correlations 0.9995, 0.9996 and 0.9999). Use a zero origin for this set.

**Source and gap check (2026-11-23).** The new chart draft reused the four WAVs in the local August 3 build. Their SHA-256 hashes match both the stored separator outputs and the WAVs in the new draft. The saved manifest names the root MP3 and HTDemucs model. The staged input matches the root MP3 byte for byte. The exact historical separator image digest and invocation were not recorded in that manifest. The 2025 MP3 stems under `audio/Blackened Crown/` belong to a separate set and were not used in this draft.

After decoding to stereo at 16 kHz, the master and each WAV had 3,198,080 frames. The summed stems correlated with the master at 0.99953 across the song and 0.99958 during the chart gap (133.526–190.131 s). The Other layer averaged −21.7 dBFS RMS during the gap. Aubio reported 20 candidate onsets there while pYIN produced no new pitched notes. Those measurements establish substantial recorded signal and detector disagreement. Identifying guitar, chords, other instruments, or separation bleed within that signal requires listening or a matching isolated instrument export.

**Chart extent.** Across Easy, medium and Hard, note spans including holds are 3.116–188.609 s for drums, 4.520–188.710 s for vocals, 4.720–133.526 s for legacy guitar, and 8.235–146.920 s for bass. The full mix continues **11.170 s** beyond the chart boundary. The first second after that boundary measures about −19 dBFS RMS. Review the ending for playable notes or an intentional outro. No reviewed musical clock, editable correction history, or approved Activity Map was identified in the recovered legacy bundle.

**Cause of the short chart.** On 2026-11-23, rerunning the pipeline's BeatRoot analysis on the matching drum WAV returned 465 beats, the last at 188.710 s. The saved validation report has the same count and cutoff. With no explicit clip duration, the [legacy generator](tools/chart-pipeline/src/song-pipeline.ts) ends the chart at the last detected beat. The full-song authoring timeline should use the decoded master duration of 199.880 s. Existing events keep their master-relative times.

**Full-song draft.** The generator now measures decoded audio length for a full-song build. A fresh stem-based build in `build/blackened-crown-full.7ImDduEK/` passed bundle validation with all 12 chart durations equal to 199.880 s. The existing demo exports remain the legacy reference.

**Guitar coverage defect.** Rerunning pYIN yielded 51 candidate notes. The last retained hold ends at 133.526 s in the exported chart. The detector reported no further attacks until 190.131 s, then reported 18 candidates ending at 195.082 s. The drum detector's last beat was at 188.710 s. The quantizer rejected all 18 outro candidates against its 80 ms snap limit. In an in-memory comparison, extending the beat grid at its preceding 0.400 s interval was the only change. Accepted candidates rose from 33 to 51, including all 18 in the outro. This establishes the cause of the missing outro. The middle gap still needs listening to distinguish a real rest from missed transcription. For authoring, retain the raw outro candidates, review a master-clock grid across the drum dropout, and check for detected material lost outside grid coverage. The published draft still needs that correction.

**Outro authoring.** The owner identified a drumless guitar-only outro. The 11.170 s figure measures the distance from the detector's last beat to the decoded end. Mark the actual guitar attacks, holds and decay by ear against the master. All roles share the end time. During a real dropout, a stationary player may opt into sparse universal beat actions via **Join In**, or use the rest to move or listen. Validate the required Finishing Cadence for solo and same-role groups as well as mixed bands. [Game Design GD-05](GAME_DESIGN.md#gd-05-phrase-preview-and-performance-cues) · [Content Authoring §9](CONTENT_AUTHORING.md#9-encounter-timeline-coverage-and-conflicts)

**Corrected guitar draft.** The owner does not expect to have an isolated guitar export for this mix. Use the master and existing Other layer. Building with `--outro-role guitar` produced the validated local draft `build/blackened-crown-guitar-outro.xS4bDwKo/`. All 51 guitar candidates survived alignment. The outro contributes 13 notes on Easy and 18 on medium/Hard. All nine bass, vocal and drum chart files matched the prior full-length draft byte for byte. The final tempo continuation is a draft timing assumption for listening review. The middle passage still needs musical identification. For a quick comparison, `middle-mix-then-other.m4a` in that draft plays 2:35–2:43 first from the mix, then from Other.

**Owner listening decision (2026-11-25).** The comparison at 2:35–2:43 is part of a vocals-only breakdown. Chart the vocal phrase and mark guitar, bass and drums as resting for the authored breakdown. Preserve the separate guitar-only outro and the full 199.880 s timeline. In the comparison window the current medium draft has 15 vocal prompts, 26 drum prompts, zero bass prompts and zero guitar prompts (counts by note start). The drum suggestions conflict with the arrangement. When authoring, mark the breakdown boundaries against the master, then suppress instrument candidates across that region on every difficulty and check holds at its edges. The long gap between detected guitar phrases gives no exact start or end for the breakdown.

## T02 and T03 · Studio verification, 2026-11-23


In edit mode in `T02-Disposable.rbxl`, created `ReplicatedStorage.T02StudioMarker`
with a distinguishing attribute, then accepted the Rojo sync for BandsBattle.
The preview included the server script addition. After sync its source matched
the repository, the marker retained its attribute, and the shared RhythmChart
module contained 9,360 bytes. The subsequent addition of the timeline reader
also reached the same place over live sync.

The [Bun check](tools/chart-pipeline/tests/timeline-contract.test.ts) and
[Studio check](roblox/tests/timeline_contract.luau) consumed the same [fixture](roblox/tests/fixtures/instrumental-timeline.json).
Both preserved the revision references, piano role, negative pickup time and
authored order of simultaneous events. Both rejected an unsupported version,
empty revision, unresolved role, duplicate role, duplicate event and fractional time.


## Local listening editor · 2026-11-25

The [song editor](tools/content-authoring/README.md) supports synchronized master and layer playback, note taps, beat and note edits, range loops, passage counts, fixed and conditional encounter entries, previous-save comparison, and separate analysis suggestions.

In the confirmed 155–163 s sample, the saved project removes 19 Easy, 26 Normal and 26 Hard drum prompts. All three instrumental layers have explicit rests. The vocal charts retain 12, 15 and 15 note starts respectively. The source bundle still contains the original suggestions. The guitar outro remains available through the full 3:19.880 song clock. The actual beginning and end of the wider breakdown, and the timing of the final guitar phrase, await listening review.

**Drum rest correction (2026-11-26).** Listening review identified audible drums within the previously suppressed 155–163 s window. The separated drum stem has a strike at 159.909 s, decays to about −70 dBFS by 160.450 s, and returns at 163.108 s. The revised draft restores 13 Easy, 19 Normal and 19 Hard drum prompts and marks 160.450–163.108 s as resting. The listening bookmark extends through 165 s to include the re-entry.


**Guitar and bass correction (2026-11-26).** The owner identified both parts in the breakdown and a shorter rest. Removed the 155–163 s rest markers for these two parts. Audited the 155–165 s listening loop against the matching stems. Both layers drop near 159.95 s. Other fades back after about 162.8 s and bass returns near 163.3 s. The revised Normal charts place their first returning attacks at 163.110 and 163.310 s respectively. These are draft gameplay prompts; the guitar source contains the separator’s combined Other material.

[Supplemental transcription](content/projects/blackened-crown/transcription-fills.json) stores input measurements in master seconds. QM polyphonic transcription ran on 150–170 s mono stem excerpts. Simultaneous detections were reduced to one playable candidate using harmonic support, excluding low-level frames and implausible clusters. For bass, octave-raised, low-passed pYIN measurements were converted back to source timing and pitch and preferred where both methods detected the same attack. Only events in the sounding portions of the review loop were used. Two editor reruns produced identical suggestions. No other charts or passages changed.

## Blackened Crown · saved multiplayer chart workload

On November 27 the original editor project was saved with all twelve generated charts. The previously saved Normal charts had 65 bass and 82 guitar prompts. The saved revision now has 395 bass, 432 guitar, 281 vocal and 656 drum prompts. Existing notes, holds and marked rests survived on all difficulties. A fresh mobile WebKit session opened the base project URL with Current draft selected. The JSON and Studio exports matched the saved charts note for note.

Time on beats containing a note or hold is 66% for bass, 68% for guitar, 77% for vocals and 82% for drums on Normal, using the detected beat grid. Compare the saved chart counts and this measure when reviewing role workload. The design targets roughly 65–80% meaningful participation and calls for musical exceptions. Listening and multiplayer playtests still determine whether the roles feel comparably engaging. [Balance framework](BALANCE_FRAMEWORK.md#participation-and-re-entry)
