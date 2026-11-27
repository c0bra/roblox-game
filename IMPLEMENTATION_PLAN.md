# Implementation Plan

Baseline adopted 2026-11-23 from the twelve resolved [interview decisions](IMPLEMENTATION_PLAN_WORKING.md). Start with **T01, identify the Blackened Crown source set**. The first playable build should answer a practical question. Can two people start the same song together, hear the layers stay aligned, and get timely feedback while playing its Normal chart on desktop and phone?

The roadmap covers the first public release. Tasks T01 through T15 take the project through its first playable test. Before starting each subsequent phase, write its tasks using the interfaces and measurements established by the preceding phase. Track execution in [Implementation Progress](IMPLEMENTATION_PROGRESS.md).

## Starting materials

- The repository contains a candidate mix named `Blackened Crown - New.mp3`. The song workspace contains drum, bass, vocal and “Other” stems, processing scripts, MIDI and onset data. The browser demo contains twelve example charts and per-role backing/layer audio. Establish which recordings belong to the same edit and where their timelines begin. The “Other” stem and the demo's guitar part need a musical check before mapping them together. [Song workspace](audio/Blackened%20Crown/) · [Demo material](roblox/web/public/levels/blackened-crown/)
- The Bun chart pipeline already has a compiler, process runner, stem selection and tests. Inspect these before adding import or analysis code. The local authoring app owns edits and approvals. The exporter consumes an immutable package. [Pipeline](tools/chart-pipeline/README.md) · [Content contract](CONTENT_AUTHORING.md)
- Rojo maps the Luau source tree. The repository includes a Studio place, rhythm judge, input and presentation prototypes. Preserve the Studio-authored scene when building a private test place. Use the browser demo for comparison. [Roblox workflow](roblox/README.md) · [Source ownership and target layout, §§4–5](TECHNICAL_ARCHITECTURE.md)

## Milestones and acceptance

| Phase | Playable or usable outcome | What allows dependent work to begin |
|---|---|---|
| 0. Foundation | A contributor can identify the chosen song inputs, run local checks and build the test source tree. | Source inventory, pinned tools, Studio test shell and a shared contract fixture are reproducible. Resolve missing musical inputs before authoring the real package. |
| 1. Author Blackened Crown | Open the folder in the local web app, listen, make a correction, reopen it, rerun analysis and export the approved revision. | The owner reviews the music. Structural, musical, compatibility and export checks pass against the same revision. |
| 2. Play the song | Play the full Normal chart in a readable test scene, with immediate feedback and a synchronized two-client start. | Run in private published staging on desktop and a physical phone. Record timing, drift, controls and recovery. Fix any failure before starting dependent gameplay. |
| 3. Run an authoritative encounter | Apply validated player actions to a frozen attempt through the server scheduler. Project combat and encounter facts to clients. | Replay the same inputs twice and compare outcomes. In Studio, exercise duplicate, late and out-of-order messages and reconnect snapshots. |
| 4. Finish and save one battle | Enter one distinct boss level with a basic Loadout. Finish solo and in co-op. Rejoin a running fight and recover the exact saved reward on return. | Complete architecture §18.3 in private staging with test profiles. Verify one-time admission, server loss/No Contest, interrupted saves, and safe retry without duplicate rewards. Owner tries the saved battle. |
| 5. Complete three bosses | Add boss two, then boss three, each with its own authored song, level and mechanics. Integrate hub, onboarding, builds, equipment, progression, abilities and accessible presentation in playable increments. | Playtest each boss when playable. Complete the first-release catalogs, asset and localization passes. Check difficulty, solo support, one-to-six-player coverage and supported devices against the design and playtest specs. |
| 6. Prepare the public release | Add purchases after demonstrating safe fulfillment. Finish operational diagnostics, backups, runbooks, release assembly and publication review. | In staging, retry an interrupted purchase receipt and observe one grant. Rehearse rollback and forward data repair. Obtain explicit owner approval for the exact release manifest, then smoke-test a newly started production server. |

## Executable tasks through the first playable test

Each row owns one deliverable. “After” identifies the required handoff. Directory names below identify existing or planned ownership from architecture §4.3. Changes are limited to the named behavior and its checks. Treat song inputs as read-only during discovery. Run transformations into a fresh draft or revision, keep the last good output on failure, and require explicit approval to promote a revision. Runtime tasks use an isolated test place and release allowlist. Stop and unload a failed slice or switch new staging runs back to the preceding verified revision.

For each task, put the build or content revision, exact check, observed result and any evidence link in the progress file under its task ID. The implementer runs the check and hands the verified output to the next consumer. Keep large media and device traces outside the progress file.

| ID / after | Deliverable and editing boundary | Acceptance and failure check |
|---|---|---|
| T01 / ready | **Identify the song inputs.** Inspect the candidate mix, song folder and demo material listed above. Own only the inventory entry under T01 in the progress file. Record paths, durations, roles, time origins, existing edits and gaps. Content Authoring §§4–6. | Inspect audio metadata and chart extents, then spot-listen near the start and end against the proposed master. Classify each input as usable, reference-only or needing authoring. Flag any uncertain version or offset for owner listening. Hand exact source references to T04. |
| T02 / ready | **Create a repeatable local test build.** Own tool pins, test place setup and contributor commands in the pipeline and Roblox READMEs. Architecture §§4, 18.2. | From the documented setup run pipeline type checks and tests, build the Rojo source tree, and load it into a disposable Studio place. Verify Studio-owned instances survive sync. Record versions and commands. |
| T03 / T02 | **Read the same contract fixture in Bun and Luau.** Own pipeline schemas, shared Roblox contracts and test fixtures. Encode a role, a master-origin time in integer microseconds, ordered simultaneous events and revision references. Architecture §§5, 9, 18.1. | Both readers produce the expected event order, IDs and times. Include an instrumental fixture with a nonstandard role. Reject an unsupported schema or missing reference. |
| T04 / T01–T03 | **Open Blackened Crown as an editable local project.** Own project intake in `tools/content-authoring/` and the song draft. Import known source roles and legacy charts against the verified origin. Content Authoring §§4–6, 10, 12. | Open the folder, identify its master and layers, explicitly convert legacy medium to Normal and verify an offset event against master time. Report an ambiguous role or missing offset for review. |
| T05 / T04 | **Rerun one selected processing stage safely.** Own the app's local job adapter around existing pipeline commands. Write generated suggestions separately from accepted edits. Architecture §5.1. | Start a job, view progress, cancel it, then rerun. A failed or repeated job preserves the last good revision and manual edits. Show dependent artifacts needing review. |
| T06 / T04 | **Listen and correct the musical source.** Own the editor transport, clock and transcription editing. Present synchronized stems and master-time positions. Content Authoring §§6, 10. | Listen, seek, correct a beat or transcribed note, save, close and reopen. Replay at the edit and verify its time and role. Surface a layer that fails alignment review. |
| T07 / T05, T06 | **Derive playable charts.** Own pipeline chart derivation and the editor's chart review. Use approved transcription, phrases and density to produce Easy, Normal and Hard per eligible role. Content Authoring §7. | Run identical reviewed inputs twice and compare event lineage, ordering and contribution weights. Play through a dense passage. Check a saved correction survives reprocessing. |
| T08 / T07 | **Review what can happen during the song.** Own Activity Map generation and encounter timeline editing. Use chart events and authored sections to place candidates and resolve conflicts. Content Authoring §§8–10, 14. | Inspect one busy and one quiet passage for each declared role. Exercise a conflicting fixed event and a conditional recovery candidate. Validate legal coverage and persist reviewed edits. |
| T09 / T08 | **Approve a complete song package.** Own project validation, the review action and immutable package output. Bind the clock, all charts/maps, timeline, role audio, arena and presentation references, dependencies, approval and integrity manifest. Architecture §5.2 and Content Authoring §§11–14. | Owner walkthrough 1: correct, reopen, review and approve Blackened Crown. Reject stale artifacts or incompatible references. Editing after approval creates a new draft. Record the exact approved revision and remaining in-game review obligations. |
| T10 / T09 | **Export the approved revision for Roblox.** Own `tools/roblox-export/`, generated content and staging asset bindings. Architecture §§5.3, 6. | Compare IDs, microsecond times, chart weights, order and hashes across the package and generated export. Reject altered output or inaccessible required assets. Deliver an allowlisted staging manifest. |
| T11 / T10 | **Load exactly that export into the test scene.** Own the server package reader and approved client projection. Architecture §§5.3, 6, 10. | Start with the selected revision in Studio. Observe the expected Normal chart and asset references. A mismatched revision or missing critical asset prevents start and clears partial state. |
| T12 / T11 | **Play aligned layers against one clock.** Own the client audio adapter and musical-clock mapping. Architecture §10 and Audio Presentation §§3–5. | Compare start, mid-song and end against master landmarks. Seek, stop and rebuild the graph. Check neutral mix, drift, cleanup and reconstruction. Record device results in T15. |
| T13 / T12 | **Play the Normal chart with immediate feedback.** Own semantic input, local judgment and readable cues in the test scene. Rhythm Gameplay §§4–7, 14–16 and UI/UX §§6–8, 14. | Exercise keyboard and touch, taps and holds, including a press before any eligible note. Check the miss and the later note independently. Compare judgments with deterministic fixtures and verify protected cues remain readable. |
| T14 / T12, T13 | **Start two clients on the same full-song timeline.** Own staging start coordination, readiness and timing recovery for the slice. Architecture §§8, 10. | Run two Studio clients with delayed readiness and network jitter. Compare their musical positions and verify unsafe timing suspends judgment and allows fair re-entry at a safe point. |
| T15 / T14 | **Prove the slice on real devices.** Own private staging acceptance and the measurement record. Architecture §§16, 18, 21. | Owner walkthrough 2: play Blackened Crown on a physical phone alongside a desktop client. Record device/build/package, start skew, full-song drift, input feel, measured offset, recovery, frame/load/memory and audio behavior. Compare to the approved criteria or document a measured candidate where a budget is still open. Retry failed checks on the affected device. |

T01 and T02 can proceed independently. T05 and T06 can proceed once the project import interface is stable. On missing source audio, device access, staging permissions or human review, record the specific missing input and the work it holds up. Retain Blackened Crown as the selected song.

## First-release coverage

| Owning specification | Milestone where its behavior is checked |
|---|---|
| [Content Authoring](CONTENT_AUTHORING.md) | 1 produces the first immutable package. 5 repeats the workflow for both remaining bosses and completes remaining tool and catalog requirements. |
| [Rhythm Gameplay](RHYTHM_GAMEPLAY.md) | 2 proves immediate judgment and timing. 3 validates shared consequences. 5 checks all charts, difficulty settings and recovery cases. |
| [Combat](COMBAT.md) | 3 exercises authoritative damage and survival rules. 4 proves one battle. 5 covers the full combat and recovery rules. |
| [Boss Encounters](BOSS_ENCOUNTERS.md) | 3 proves lifecycle and ordering. 4 loads one distinct level. 5 checks all three bosses, positions, mechanics and outcomes. |
| [Progression](PROGRESSION.md) | 4 reloads earned progress. 5 checks restoration, rank, unlocks and mastery across bosses. |
| [Items and Equipment](ITEMS_AND_EQUIPMENT.md) | 4 equips a valid starter Loadout. 5 exercises inventory, supplies, cosmetics and gear changes in the hub. |
| [Abilities and Cooperative Actions](ABILITIES_AND_COOPERATIVE_ACTIONS.md) | 4 exercises the solo and co-op minimum. 5 verifies abilities, cooperative actions and solo support across party sizes. |
| [Multiplayer](MULTIPLAYER.md) | 2 tests a two-client start. 4 verifies parties, admission, rejoin and recovery. 5 covers communication, safety and the full roster range. |
| [Rewards and Economy](REWARDS_AND_ECONOMY.md) | 4 confirms saved battle rewards. 5 completes economy content and tuning. 6 proves retry-safe paid fulfillment. |
| [Builds and Specialization](BUILDS_AND_SPECIALIZATION.md) | 4 freezes a valid configuration. 5 tests saved Loadouts, hub switching and locked abilities/equipment during a battle. |
| [UI/UX](UI_UX.md) | 2 checks cues and input, 4 Results and recovery messages, and 5 the hub, onboarding, localization and accessible navigation. |
| [Audio Presentation](AUDIO_PRESENTATION.md) | 2 checks role layers, timing and feedback. 5 verifies protected cues, fallback, device mix and full presentation. |
| [Player Data](PLAYER_DATA.md) | 4 checks single-writer profiles, atomic rewards, ambiguous saves and reload. 6 exercises receipts, backups, migrations and forward repair. |

Use the [systems map](SYSTEMS_MAP.md) for ownership and the [system tracker](SYSTEM_SPECIFICATIONS_TRACKER.md) for the design baseline. Apply [art direction](ART_DIRECTION.md) to levels and cues, [naming and tone](NAMING_AND_TONE.md) to visible text, [balance](BALANCE_FRAMEWORK.md) to tuning hypotheses, and [playtest and analytics](PLAYTEST_AND_ANALYTICS.md) to evidence and readiness throughout. Owner walkthrough 3 in phase 4 ends by reopening the profile and checking the earned reward.

## Architecture proof and release checklist

Each item corresponds to the same numbered item in [Technical Architecture §21](TECHNICAL_ARCHITECTURE.md). Record observations against the relevant task or phase as implementation proceeds.

| §21 | Evidence owner and timing |
|---|---|
| 1 | T12/T15 measure aligned layered audio start, seek, stop, full-song drift and reconstruction on devices. |
| 2 | T14/T15 measure server-time mapping, clock confidence, judgment envelope and recovery under realistic delay. |
| 3 | Phase 2 starts device, memory, audio, package and load budgets. Phases 3–5 extend measurements to server steps, remotes, platform services and real profiles, then validate under representative load. |
| 4 | Phase 4 loads the exact boss scene version, validates anchors and verifies cleanup. Evaluate the architecture's dedicated-place fallback if the measured budgets fail. Repeat in phase 5 for each boss. |
| 5 | Phase 4 uses private published reserved servers to test admission once, rejoin grace, server loss and No Contest. |
| 6 | Pin and audit Wally, Jest Lua, React Lua and ProfileStore as each enters use in its owning task. Record lockfiles, compatibility and repeatable installation before production use. |
| 7 | Phase 4 tests profile capacity, lease handoff, migrations and ambiguous saves under injected failures. Phase 6 exercises receipt retries, backup recovery and forward repair. |
| 8 | Start a reproducible Studio/Rojo staging assembly in phase 2. Phase 6 rehearses assembly and rollback against the exact authored place, commit, assets and manifest before automating promotion. |
| 9 | Phase 6 reviews then-current platform requirements for privacy, child safety, commerce, retention, deletion, asset permissions and publication. |
| 10 | Owner selects the public game title before publication in phase 6. Use the working project identity during development. |

## Handoffs and changes

At the end of phase 2, use the measured timing and package contracts to detail phase 3. Expand phases 4, 5 and 6 at their respective entry points. For each new task apply architecture §20.1: assign an ID, one observable result, prerequisites, owning spec sections and files, precise input/output revisions, excluded work, failure and cleanup behavior, targeted checks, evidence location and a way to revert or disable the change safely.

The task owner verifies the result through its normal entry point. The milestone integrator runs the exit scenario on the combined revision and records the next ready task. For a failed check, capture the smallest reproducible case, identify dependent tasks, correct the owning component and repeat the check. Route a change to gameplay behavior or authority to the owning design or architecture document before revising this roadmap.
