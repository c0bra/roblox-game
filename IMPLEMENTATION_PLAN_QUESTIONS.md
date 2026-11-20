# Implementation Plan Questions

- **Status:** In progress (7 of 12 questions resolved)
- **Started:** 2026-09-10
- **Current checkpoint:** C - From a song test to a complete game
- **Architecture authority:** [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md), especially sections 18 through 21
- **Design inputs:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **System coverage:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)
- **Working record:** [`IMPLEMENTATION_PLAN_WORKING.md`](IMPLEMENTATION_PLAN_WORKING.md)
- **Planned result:** `IMPLEMENTATION_PLAN.md`, to be written from resolved decisions

## 1. Interview method

Use four checkpoints of three related questions. Ask one question or a small
group at a time, explain unfamiliar terms, and offer a concrete recommendation.
The owner can accept a group together or change any part. Keep follow-ups limited
to choices that materially change scope, priorities, or the owner's workload.

Persist answers, accepted recommendations, and consequences in the working
record at meaningful checkpoints. Update resolved counts and project references
alongside it. Recommendations below are proposals, not approved decisions.

The assistant supplies task decomposition, dependencies, specification links,
and appropriate engineering checks. The owner need not select libraries or
approve individual field names. This interview establishes an execution plan;
actual implementation, testing, content approval, and release remain separate
work with their own evidence.

## 2. Settled inputs

- The thirteen system specs and supporting design documents are approved.
  Implementation sequencing cannot silently remove their first-release scope.
- Technical Architecture section 20 supplies phases 0 through 6. Refine those
  phases into milestones and tasks while preserving their dependency gates.
- Content Authoring runs locally outside Roblox and builds on the existing
  pipeline. Usable, approved song data precedes dependent gameplay work.
- Phase 0 establishes only the foundations needed to support that sequence.
  Later dependencies are pinned and verified before their production use.
- The first Roblox proof uses a real song, aligned audio layers, a Normal chart,
  immediate rhythm feedback, and two-client start, with desktop and physical
  phone evidence. Broader gameplay depends on that proof.
- Roblox is the playable product. The browser demo remains a reference and
  experiment surface. Existing tools, code, and assets require inspection before
  counting them as production-ready or deciding to replace them.
- Bosses have distinct levels loaded through the shared battle runtime. The
  approved first-release target includes three bosses and one to six humans.
- Runtime authority, persistence, accessibility, release approval, and recovery
  rules remain as specified. Smaller interim demonstrations may omit features
  only when their limits are explicit and later required work remains scheduled.
- Architecture section 21 lists selected mechanisms still needing proof. Those
  become implementation tasks and milestone gates, not completed capabilities.
- Final public naming remains deferred until publication. `Bands Battle` is an
  internal project name.

## 3. Question plan

### Checkpoint A - First playable target

#### IP-01 - How much of the roadmap should be detailed now?

- **Status:** Resolved 2026-10-06.
- **Decision:** Map the whole first release at the milestone level. Fully detail
  tasks through the first real-song Roblox proof, then expand later tasks using
  evidence from the earlier work.
- **Question:** Should we map the whole first release, with the most detailed
  tasks focused on getting the first real song working in Roblox?
- **Recommendation:** Yes. Give every architecture phase an outcome,
  dependencies, and completion checks. Fully detail the work through the first
  real-song proof, then expand later tasks as earlier results become available.
- **Must establish:** Plan horizon, initial task detail, when later tasks are
  expanded, and the distinction between a first playable test and public release.
  Avoid speculative dates or hundreds of tickets based on untested assumptions.

#### IP-02 - Which song should we use first?

- **Status:** Resolved 2026-10-06.
- **Decision:** Use Blackened Crown for the first Roblox test. Inspect its source
  audio, stems, editable data, and existing example charts before treating any
  of them as production-ready inputs.
- **Question:** Which song do you want to hear and play in the first Roblox test?
- **Recommendation:** Start with Blackened Crown, subject to checking its source
  audio and editable inputs. Its existing example charts provide a useful
  reference. Heaven's Edge or another owner-selected song can serve instead.
- **Must establish:** Song choice, source readiness inspection, full-song scope,
  representative layers and chart, and required authoring work. Existing generated
  charts alone do not constitute an approved source package. Raise missing inputs
  before substituting a different song.

#### IP-03 - What should the first playable test feel like?

- **Status:** Resolved 2026-10-06.
- **Decision:** Use a simple, readable scene to prove Blackened Crown's aligned
  audio layers, Normal chart, clear controls, immediate hit feedback, and
  synchronized two-client start on desktop and a physical phone. Full boss
  presentation, combat, rewards, and progression follow in later milestones.
- **Question:** Is a simple, readable test scene enough while we prove the song,
  controls, and timing, before building out the boss spectacle?
- **Recommendation:** Use a modest test scene with the required music, Normal
  chart, clear input and feedback, and synchronized two-client start. Include
  desktop and physical phone checks. Add full boss presentation and progression
  in their later milestones.
- **Must establish:** What the owner will actually try, acceptable temporary
  visuals, essential protected cues and accessibility, and clear omissions.
  This milestone demonstrates rhythm/audio feasibility, not a finished battle.

### Checkpoint B - Work before broader gameplay

#### IP-04 - What must the first authoring tool let you do?

- **Status:** Resolved; recorded 2026-11-19.
- **Decision:** The first authoring milestone covers the complete Blackened Crown
  folder-to-export workflow: load stems, review and correct timing/charts/activity
  mappings, rerun processing while preserving edits, save/reopen, validate, and
  approve/export usable song data.
- **Question:** Should the first tool focus on taking the chosen song all the
  way from opening its folder to approving and exporting a usable package?
- **Recommendation:** Yes. Prioritize loading stems, synchronized playback,
  reviewing and correcting the clock/charts/activity and encounter timelines,
  safe pipeline reruns, saved edits, validation, approval, and export. Schedule
  remaining authoring requirements explicitly after this minimum complete path.
- **Must establish:** A concrete owner walkthrough, supported first-song inputs,
  reuse inspection, required edit/reopen behavior, and which convenience features
  can wait. Extensible instruments and protection of human edits remain required.

#### IP-05 - How should we handle the early technical unknowns?

- **Status:** Resolved 2026-11-19.
- **Decision:** After producing usable song data, prove Blackened Crown's playback,
  timing, and controls on desktop and a physical phone before broader gameplay
  depends on them. Resolve failed assumptions before continuing dependent work.
- **Question:** Should we prove the important audio and timing behavior before
  investing in the larger gameplay systems?
- **Recommendation:** Follow the approved content-first gates. Prepare only
  necessary contracts and checks, produce the song package, then measure Roblox
  playback, drift, input timing, and recovery on desktop and a phone. Stop work
  that depends on a failed assumption until it is resolved.
- **Must establish:** Named evidence tasks, prerequisite order, measurement
  outputs, a bounded investigation path for failures, and which independent
  preparation can continue. Preserve architecture sections 20 and 21.

#### IP-06 - What should you review along the way?

- **Status:** Resolved 2026-11-20.
- **Decision:** Give the owner a few things to try when the authoring tool can
  export the song, Blackened Crown is playable on a phone, and a completed
  battle saves its rewards. Include technical checks in the implementation tasks.
- **Question:** Would short hands-on reviews at useful milestones work, with
  technical checks handled as part of implementation?
- **Recommendation:** Ask the owner to try the authoring/export path, the first
  song on a phone, and the first complete saved battle loop. Provide a short
  description of what to try and record feedback using the existing lightweight
  playtest process.
- **Must establish:** Owner involvement, access to a representative physical
  phone and a second client, and how unavailable hardware or human feedback is
  recorded. Do not equate a missing human check with a pass.

### Checkpoint C - From a song test to a complete game

#### IP-07 - What should the first complete battle include?

- **Status:** Resolved 2026-11-20.
- **Decision:** Follow the Blackened Crown rhythm test with one complete boss
  battle in its own level. Check solo and co-op play, rejoining after a
  disconnect, and whether the earned reward survives leaving and returning.
- **Question:** After the music test works, should the next major target be one
  boss battle that you can enter, finish, earn rewards from, and return to later?
- **Recommendation:** Yes. First establish the authoritative encounter runtime,
  then add one distinct boss scene, deployment/rejoin, a small valid Loadout,
  outcome handling, confirmed rewards, and profile reload. Retain explicit tasks
  for all remaining first-release features.
- **Must establish:** Minimum battle content, phase 3 versus phase 4 boundaries,
  solo/co-op coverage, recovery paths, and persisted-results acceptance.

#### IP-08 - How should we expand to the full first-release scope?

- **Status:** Open.
- **Question:** Once the first complete battle works, should we expand the game
  around that working example and bring the other two bosses through the tools?
- **Recommendation:** Complete the required combat, abilities, progression,
  equipment, builds, hub, UI/accessibility, and remaining content in playable
  increments. Use the additional bosses to exercise content reuse and distinct
  level design, with balance and playtest reviews as each becomes playable.
- **Must establish:** Coverage for all thirteen specs, catalog/art/audio work,
  dependencies between feature increments, and the three-boss readiness target.
  A feature may be staged, but an approved release requirement cannot disappear.

#### IP-09 - What can wait until the game loop is reliable?

- **Status:** Open.
- **Question:** Should paid purchases and final publication work come after the
  battle, rewards, and save/recovery paths have been demonstrated?
- **Recommendation:** Yes, following phase 6. Include basic diagnostics and
  test hooks when their systems are built; finish commerce, operational checks,
  release assembly, and publication readiness after the required proofs.
- **Must establish:** Later work versus excluded work, receipt/persistence
  dependencies, title and policy gates, release evidence, and explicit human
  production approval. Do not delay all observability until the last milestone.

### Checkpoint D - Tasks, evidence, and plan completion

#### IP-10 - How should implementation tasks be packaged?

- **Status:** Open.
- **Question:** Should each task deliver one small behavior that can be checked
  independently, with a clear handoff to the next task?
- **Recommendation:** Yes. Use stable task IDs, specification sections, owned
  files/interfaces, prerequisites, outputs, exclusions, failure behavior,
  automated and hands-on checks, evidence location, and safe-disable/rollback
  notes, as required by architecture section 20.1.
- **Must establish:** Task template, sizing rule, integration responsibility,
  and how a blocked task reports a missing decision. Describe safe opportunities
  for parallel work without authorizing agents or creating new tasks in the app.

#### IP-11 - How should progress and changes be tracked?

- **Status:** Open.
- **Question:** Would a simple checked-in progress record, updated as tasks and
  milestones finish, be enough for you to see what works and what comes next?
- **Recommendation:** Use the implementation plan as the roadmap and one compact
  execution record for task state and evidence links. Record relevant revisions,
  checks, observed behavior, blockers, and the next ready task. Expand later task
  detail before starting it.
- **Must establish:** Record location, status meanings, update cadence,
  integration checks, and when scope changes need an owning design amendment.
  Keep planning approval, implemented code, verified behavior, and release
  approval distinguishable. Avoid duplicate tracking systems by default.

#### IP-12 - What makes the plan ready to use?

- **Status:** Open.
- **Question:** Should we finish by checking every required system has a place
  in the roadmap and identifying the exact first implementation task?
- **Recommendation:** Yes. Reconcile the plan against architecture phases and
  unverified gates, the thirteen specs, and supporting content, visual, balance,
  and playtest requirements. Publish `IMPLEMENTATION_PLAN.md` with the first
  ready task and clearly identified owner inputs or external dependencies.
- **Must establish:** Completion audit, unresolved dependency handling,
  downstream references in project indexes, and initial task readiness.
  Approving a plan does not mark its implementation or evidence complete.

## 4. Expected final document

`IMPLEMENTATION_PLAN.md` should contain the accepted target and constraints,
current repository baseline, milestones in dependency order, initial executable
tasks, later task-expansion points, specification coverage, evidence gates,
owner/content/device inputs, progress/change rules, and the first ready task.

The final audit must account for all architecture phases and every selected but
unverified gate. It must identify both software work and authored content work.
Create the canonical document from the interview's resolved decisions; keep this
question plan and the working record as history.
