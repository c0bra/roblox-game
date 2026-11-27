# Implementation Plan Working Record

- **Status:** Complete (12 of 12 questions resolved)
- **Started:** 2026-09-10
- **Completed:** 2026-11-23
- **Question plan:** [`IMPLEMENTATION_PLAN_QUESTIONS.md`](IMPLEMENTATION_PLAN_QUESTIONS.md)
- **Architecture authority:** [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md)
- **Design inputs:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **Canonical result:** [Implementation Plan](IMPLEMENTATION_PLAN.md)

## 1. Purpose

Preserve owner answers and accepted implementation-planning decisions at
meaningful checkpoints. Recommendations in the question plan remain proposals
until accepted. Inherited design and architecture requirements remain binding.

For each resolved question, record the owner answer, decision, rationale,
milestone/task consequences, and remaining inputs or evidence needs. Keep actual
implementation progress separate from interview progress.

## 2. Planning baseline

Inspected 2026-09-10:

- The architecture interview is complete at 24 of 24. Its canonical section 20
  defines seven ordered phases, beginning with foundations and local content
  authoring, then the real-song Roblox proof.
- Architecture sections 18 and 21 define verification layers and remaining
  feasibility gates. Architecture approval does not mean these have passed.
- The system tracker lists thirteen approved design specifications. They and
  their supporting documents supply scope for the implementation coverage audit.
- Existing architecture records identify the Roblox prototype, browser demo,
  and separate chart pipeline. Their production reuse needs task-level inspection.
- The owner selected Blackened Crown for the first Roblox test. Its example
  charts are useful references, but its source audio, stems, editable data, and
  existing outputs still need a task-level readiness inspection.
- No canonical implementation plan or implementation interview files existed
  before this setup. This work creates the question plan and answer record only.

## 3. Checkpoint progress

| Checkpoint | Questions | State | Resolved |
|---|---|---|---:|
| A - First playable target | IP-01 through IP-03 | Approved 2026-10-06 | 3 of 3 |
| B - Work before broader gameplay | IP-04 through IP-06 | Approved 2026-11-20 | 3 of 3 |
| C - From a song test to a complete game | IP-07 through IP-09 | Approved 2026-11-20 | 3 of 3 |
| D - Tasks, evidence, and plan completion | IP-10 through IP-12 | Approved 2026-11-23 | 3 of 3 |

## 4. Decision record

### Checkpoint A - First playable target

#### IP-01 - Roadmap horizon and detail

- **Status:** Resolved 2026-10-06.
- **Owner answer:** Yes, use the recommended level of planning.
- **Decision:** Map the complete first release through architecture phase 6 at
  the milestone level. Write executable task detail through the first real-song
  Roblox proof. Expand later task groups before implementation, using evidence
  and interface decisions from completed milestones.
- **Rationale:** This makes the complete destination and dependencies visible
  without pretending that untested assumptions support a fixed backlog.
- **Plan consequences:** The canonical plan will clearly distinguish the first
  playable rhythm test, the first complete saved battle loop, content-complete
  first release, and public production approval. It will not invent delivery
  dates during this interview.

#### IP-02 - First real song

- **Status:** Resolved 2026-10-06.
- **Owner answer:** Use Blackened Crown.
- **Decision:** Blackened Crown is the first song carried through the authoring,
  export, and Roblox rhythm-proof milestones.
- **Rationale:** Existing example charts give the work a concrete reference and
  make it possible to test the complete content path against real material.
- **Plan consequences:** An early task must inventory and validate Blackened
  Crown's source audio, stems, editable inputs, charts, and timing origin.
  Generated examples are not promoted
  to canonical source data without review. Missing inputs become explicit
  authoring work rather than an unrecorded switch to another song.
- **Remaining evidence:** Source readiness and the eventual human-approved
  content package have not yet been demonstrated.

#### IP-03 - First playable-test scope

- **Status:** Resolved 2026-10-06.
- **Owner answer:** "Sure do it" in acceptance of the proposed test scope.
- **Decision:** Use a simple, readable scene with Blackened Crown's aligned audio
  layers and Normal chart, clear controls, immediate hit feedback, synchronized
  two-client start, and desktop plus physical-phone testing.
- **Rationale:** The first playable milestone establishes whether the music,
  input, and timing work reliably before broader gameplay depends on them.
- **Plan consequences:** Temporary visuals are acceptable. Preserve clear,
  protected gameplay cues and applicable input/accessibility requirements. Full
  boss environment and combat presentation, rewards, and progression remain
  scheduled for later milestones. This approval resolves planning scope; no
  playable implementation or device evidence is claimed.
- **Remaining evidence:** The approved content package and observed Roblox
  audio/rhythm behavior are still required before this milestone can pass.

### Checkpoint B - Work before broader gameplay

#### IP-04 - First authoring milestone

- **Status:** Resolved; recorded 2026-11-19 from the preceding owner answer.
- **Owner answer:** "Sure" to the complete folder-to-export workflow.
- **Decision:** Open Blackened Crown's folder, load its stems, listen while
  reviewing and correcting timing, notes, and activity mappings, rerun existing
  processing tools while preserving human edits, then save/reopen, validate,
  approve, and export usable song data for Roblox.
- **Rationale:** One complete song path makes the tool useful and supplies the
  real content required by the next milestone.
- **Plan consequences:** Build on existing processors and retain the approved
  extensible-instrument and encounter-timeline contracts. Include reopen and
  rerun acceptance checks. Schedule remaining authoring conveniences explicitly
  after this minimum complete path.
- **Remaining evidence:** Actual source readiness, the working authoring flow,
  and the approved/exported package remain implementation deliverables.

#### IP-05 - Early technical proof

- **Status:** Resolved 2026-11-19.
- **Owner answer:** "Yea go for it" to proving playback, timing, and controls
  before the larger gameplay systems.
- **Decision:** Produce usable Blackened Crown data first, then test Roblox
  playback, timing, and controls on desktop and a physical phone. Resolve any
  failed assumption before building features that depend on it.
- **Rationale:** Later gameplay relies on reliable music and input timing.
- **Plan consequences:** Assign measured audio alignment/drift, input timing,
  and recovery checks to the first real-song proof. Record failures and their
  dependent tasks, investigate the affected boundary, and rerun the failed check.
  Independent preparation may continue under architecture section 20; design
  changes still return to their owning specification.
- **Remaining evidence:** Device access and observed results are not established
  by this approval. IP-06 will address the owner's hands-on review process.

#### IP-06 - Hands-on reviews

The owner approved three short reviews on 2026-11-20. Give the owner a few
things to try at each milestone and record their feedback.

1. Open Blackened Crown in the authoring tool, make an edit, and export the song.
2. Play the song on a physical phone and judge whether the notes feel in time.
3. Finish a boss battle, return to the game, and check that the reward is still there.

Implementation tasks own the technical checks. Confirm access to a phone and
second client when arranging the device test. If either is unavailable, record
which check is pending.

### Checkpoint C - From a song test to a complete game

#### IP-07 - First complete boss battle

After the Blackened Crown rhythm test, the owner wants one boss battle playable
from entry through saved rewards. Phase 3 establishes how the server runs and
resolves the encounter. Phase 4 connects that behavior to the boss's level,
deployment, rejoining, and player saves.

To check the milestone, enter with a basic Loadout and finish the fight in
both solo and co-op play. In a co-op attempt, disconnect and rejoin the
ongoing fight. After earning a reward, leave the game, return, and verify the
same reward appears on the reloaded profile.

#### IP-08 - Bringing the remaining bosses into the game

Once the first complete battle works, use the same song tools to bring in the
other two bosses one at a time. Give each a distinct level and mechanics. As
each fight becomes playable, playtest it alongside the hub, abilities,
equipment, and progression features available at that point. This lets the
owner judge how those features work together during a real session.

#### IP-09 - Timing of purchases and public release work

The owner approved scheduling paid purchases and final release preparations
after a player can finish a battle, keep the reward across sessions, and
recover from an interrupted save. Build the diagnostics needed to observe
those outcomes alongside the systems they test.

Phase 6 adds purchase-specific checks. Retry the same receipt after an
interruption and verify that the player receives the item exactly once.
Final publication follows the release checks and owner approval in
Technical Architecture section 19.

### Checkpoint D - Tasks, evidence, and plan completion

The owner accepted the final three proposals together on 2026-11-23.

- IP-10. Size a task around one result another contributor can check. The plan's task table pairs each deliverable with its prerequisites, scope and acceptance check.
- IP-11. Keep a single [progress file](IMPLEMENTATION_PROGRESS.md). Update it as work finishes or becomes blocked, with the observed result and next ready task.
- IP-12. Map all thirteen specs to milestones and identify the first ready task. The [completed plan](IMPLEMENTATION_PLAN.md) also assigns all seven architecture phases and ten outstanding architecture checks. T01 inventories the selected song inputs.

## 5. Change log

- **2026-09-10:** Created the twelve-question interview and working record.
  Linked the active interview from project entry points. Progress is 0 of 12.
- **2026-10-06:** Approved the whole-release planning horizon with detailed work
  through the first real-song proof, and selected Blackened Crown as that song.
  Progress is 2 of 12; checkpoint A remains open for IP-03.
- **2026-10-06:** Approved IP-03, completing checkpoint A. Opened checkpoint B
  on authoring workflow, early technical proof, and owner reviews. Progress is
  3 of 12. Kept the source-readiness task focused on usable song inputs, consistent
  with the owner's earlier decision not to add provenance work for their own music.
- **2026-11-19:** Recorded IP-04's accepted authoring workflow and IP-05's early
  desktop/phone proof requirement. Progress is 5 of 12; IP-06 remains open.
- **2026-11-20:** Recorded the three hands-on reviews and completed checkpoint B.
  Progress is 6 of 12. Checkpoint C begins with the first complete boss battle.
- **2026-11-20:** Approved IP-07. Progress is 7 of 12.
- **2026-11-20:** Approved IP-08. Progress is 8 of 12.
- **2026-11-20:** Approved IP-09 and completed checkpoint C. Progress is 9 of 12.
- **2026-11-23:** Completed IP-10 through IP-12 and created the canonical plan and execution record.
