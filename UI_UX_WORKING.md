# Bands Battle UI/UX Working Record

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-30
- **Question plan:** [`UI_UX_QUESTIONS.md`](UI_UX_QUESTIONS.md)
- **Parent systems:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#7-experience-shell)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Canonical result:** [`UI_UX.md`](UI_UX.md)

## 1. Role of this record

This file preserves the approved answers, refinements, inherited constraints,
and cross-system handoffs from the completed UI/UX interview. It is evidence
for the canonical specification, not the final authority.

## 2. Inherited boundary

UI/UX owns experience composition, navigation, focus, responsive presentation,
semantic action mapping, settings/calibration definitions, onboarding
orchestration, results presentation, and accessibility acceptance. It presents
and routes semantic facts from owning systems; it does not recalculate gameplay,
resolve transactions, change rewards, repair loadouts, invent progression, or
implement persistence.

The complete inherited decision set is recorded in
[`UI_UX_QUESTIONS.md`](UI_UX_QUESTIONS.md#2-fixed-inherited-decisions).

## 3. Decision record

### Checkpoint A - Experience architecture, hub, and navigation

#### UX-01 - Player jobs, experience modes, and navigation hierarchy

- **Status:** Resolved 2026-08-30.
- The global safe-menu hierarchy has four primary task destinations: **Play**,
  **Prepare**, **Progress**, and **Band**. These are functional labels pending
  the later naming/tone pass. Settings, controls, accessibility, and help remain
  globally available utilities. The voluntary store is never a primary
  destination or recommended next action.
- Play owns discovery/selection routes into unlocked shards and the current
  story/replay choice. Prepare composes loadout, inventory, upgrades, builds,
  abilities, consumables, and full spec presets. Progress composes campaign,
  mastery, records, collection, and archive evidence. Band composes party,
  matchmaking, safe communication, and compatible social actions. Domain
  systems remain authoritative for every fact and mutation behind those routes.
- The shell distinguishes first-run, safe hub, queued-safe hub/menu, encounter
  card, staging, practice, active encounter, solo pause, results, and blocking
  recovery states. Current mode, current task, queue/party state when relevant,
  and one truthful next action remain apparent. A higher-priority mode may
  constrain navigation but never disguise where the player is or strand them.
- Global navigation is available in safe hub/menu and compatible queued states.
  Encounter card, staging, practice, and results retain explicit task exits.
  Active battle exposes only the permitted pause/settings/control-reference/
  leave surface; cooperative play does not present pause as available.
- Navigation is at most two levels below a primary destination. A focused item
  detail may replace the second level rather than create deeper nesting. No
  primary task depends on hidden overflow, physical traversal, search, or a
  remembered shortcut.
- Back returns to the prior valid surface and restores draft, selection,
  filters, sort, scroll, and focus. It never returns to an expired encounter or
  transaction. Leaving a queue, committed Ready state, purchase confirmation,
  or meaningful unsaved edit requires a clear consequence-specific confirmation;
  ordinary reversible navigation does not.

#### UX-02 - Physical hub wayfinding and fast-access routes

- **Status:** Resolved 2026-08-30.
- First-time arrival uses a short, unmistakable route toward accessibility/
  practice setup and the first unlocked shard. Returning players appear on the
  stable central route or their highest unlocked landing, within a few seconds
  of playable content and essential anchors.
- The shard ascent, practice, workshop, archive/progress, social commons, and
  voluntary store use stable silhouettes, interaction footing, labels, and
  landmarks. Campaign restoration may enrich structure, population, lighting,
  music, and portals but cannot relocate a learned anchor or invalidate the
  route vocabulary.
- Each shard communicates boss identity, campaign tier, availability, and exact
  lock reason through label/icon/shape plus color. A newly unlocked state may
  receive one restrained, dismissible emphasis. It cannot resemble a Robux
  gate. Proximity only reveals the action; a deliberate input opens the same
  encounter card used by every other route.
- **Fast Play** from the global Play destination exposes Continue Story, the
  last played unlocked shard, and appropriate recent/replay choices. Choosing
  one opens the normal encounter card rather than silently deploying. Unlocked
  landing shortcuts may also move the avatar near a shard without making repeat
  traversal mandatory.
- World-space landmarks and a compact optional objective/landmark guide provide
  orientation without turning the hub into a mission-board overlay. The guide
  can identify the selected destination and accessible route, then dismiss when
  the player arrives. Critical anchors remain discoverable without it.
- Every required route avoids precision jumping, forced camera control, narrow
  collision challenges, and materially longer accessibility detours. Optional
  lore, NPC dialogue, emotes, and non-scored music never block preparation,
  progression, or play.
- Matchmaking remains visible as a compact persistent status while the player
  moves and uses compatible menus. An action that conflicts with the locked
  staging/loadout state explains the conflict and asks whether to leave the
  queue; it never cancels matchmaking silently.

#### UX-03 - Responsive shell, safe areas, and device navigation

- **Status:** Resolved 2026-08-30.
- Touch landscape is the first-release phone baseline. Tablet, desktop, and
  gamepad layouts preserve the same destinations, terms, ordering, information
  architecture, and task state. Portrait entry gives a simple accessible rotate
  instruction and access to required pre-play settings rather than attempting a
  compressed encounter layout.
- Safe touch menus use a labeled four-destination bottom bar; wide desktop uses
  a compact labeled rail; gamepad uses the same four focusable destinations and
  ordering. Queue/party context and global utilities occupy a consistent header
  or utility surface. A breakpoint changes composition, never destination order
  or meaning.
- General touch actions are at least 48 by 48 logical pixels with separation;
  timing pads and other frequent encounter actions are substantially larger.
  Platform safe areas, notches, system gestures, aspect ratio, and the player's
  chosen interface scale cannot cover a critical action or status.
- Primary navigation is never hidden in a generic overflow. A destination may
  use one clear subnavigation level. Contextual details use inline expansion,
  a side panel on wide layouts, or a bottom sheet/drawer on touch. Tabs are not
  nested. A modal is reserved for a genuinely blocking decision, consent, or
  irreversible/externally committed action.
- Keyboard navigation and gamepad focus cover every action with visible focus,
  deterministic spatial order, safe initial focus, and focus restoration after
  close/back. Focus does not jump because content loads, and opening a panel
  constrains focus until it closes. Touch never requires hover; desktop hover
  is supplementary rather than the only explanation.
- Context return preserves the prior surface, focus, selected item, filters,
  sort, scroll, and uncommitted draft when legal. Layout reflow preserves the
  same state across supported device/aspect changes.
- Components allow at least roughly 30% text expansion, multi-line labels where
  safe, localized number/time formats, and future right-to-left mirroring. Core
  meaning is never embedded only in an image, clipped, or converted to an
  unexplained icon to fit a smaller screen.

### Checkpoint B - Encounter HUD, controls, and readable action

#### UX-04 - Persistent encounter hierarchy and contextual surfaces

- **Status:** Resolved 2026-08-31.
- Encounter composition protects a central gameplay field for the boss, avatar,
  arena geometry, positions, and physical telegraphs. Persistent interface
  regions frame rather than cover that field and remain stable through normal
  play, device-glyph changes, difficulty, and roster size.
- A compact upper boss region shows identity when needed, current song function/
  encounter phase, current Resolve layer and progress, future locked layers, and
  Momentum only while it has a legal destination. It never implies pressure on
  an unopened layer or early victory.
- The lower performance region contains the right-to-left staff, fixed strike
  line, immediate compact judgment, and three fixed rhythm controls. Ward/
  survival and effective/queued intent stay adjacent to their relevant control
  clusters. Hype state/progress lives on Special; Band Call readiness/progress
  lives on its own control. No duplicate persistent meters are added.
- A compact band strip exposes only necessary safe state: identity/appearance,
  role, active/downed/recovering/return-protected, connection availability, and
  required group readiness. Healthy entries stay quiet; relevant target,
  recovery, disconnection, acolyte suppression, or return state gains restrained
  emphasis. It shows no public performance, damage, accessibility, gear, or
  blame data.
- Phrase preview, position choices/routes, attack response, consumables,
  Signature commitment, group invitations/windows, revival/recovery, pings, and
  contextual teaching appear only when actionable or necessary. Their cue
  anchors identify the related world source, target, control, and musical
  countdown rather than becoming detached notification text.
- The contextual priority is: unsafe synchronization/control or terminal state;
  imminent committed threat and urgent recovery; required authored group event;
  accepted/pending player group action; movement and item opportunity; then
  tutorial, ping, and informational feedback. Lower priority never obscures,
  delays, or impersonates a protected automatic cue. Conflict rules prevent
  Band Call and Crescendo performance windows from competing.
- One dominant contextual cue receives full treatment. Other simultaneously
  valid facts reduce to ordered source/target chips or world markers until they
  become primary. Time-critical cues use shape/icon/label and beat/time progress,
  not explanatory sentences, and never cover playable notes or the boss.
- Rhythm suspension withdraws unresolved notes without showing Miss, labels the
  real cause, and communicates fair return readiness/countdown when available.
  Immediate grades remain near the staff; detailed distributions wait for
  Results. A player should identify their survival and intent, the current boss
  layer/phase, the nearest committed danger, and the current rhythm action in
  roughly a one-second glance during representative phone testing.

#### UX-05 - Semantic controls, remapping, and active-device transitions

- **Status:** Resolved 2026-08-31.
- The GD-06 baseline semantic action map is binding. Three rhythm actions remain
  fixed and never acquire another meaning. Movement remains WASD/left stick or
  hub joystick/encounter destination tap; Attack/Defend/Special, Join In, Band
  Call, and two consumables retain their approved device families.
- Every action presents owner-supplied states such as available, unavailable
  with reason, pressed, queued, effective, committed, cooling/recovering,
  reserved, or spent. A legal queued intent/action responds immediately at its
  control and identifies the boundary at which it will become effective. UI
  never guesses legality or makes an unavailable action look queued.
- One physical event can emit at most one semantic action in its active context.
  Different dedicated controls may be used simultaneously; the owning gameplay
  systems resolve their identified boundary order without UI dropping or
  duplicating input. An active encounter does not open a focus-stealing modal.
- Remapping is available only from a safe settings/control surface. The editor
  prevents reserved/platform conflicts and duplicate bindings within the same
  simultaneously active context, displays every conflict together, previews
  device glyphs, and applies or resets a whole profile atomically. The same
  physical binding may be reused only across provably exclusive contexts.
- Active-device prompts change after deliberate input, not passive device noise,
  and use a short stability delay to prevent glyph flicker. Changing active
  device updates labels/reference while preserving HUD geometry, current
  semantic states, held-contact truth, and queued actions. It never synthesizes
  a press/release or rearranges timing controls during a phrase.
- Touch layout editing occurs only in a safe menu or solo pause. A preview canvas
  permits handedness, size, spacing, position, and opacity changes while enforcing
  safe areas, minimum target/separation, no critical overlap, and retained access
  to all actions. Editing emits no gameplay input and provides atomic Apply,
  Cancel, and Reset to Default.
- If an active device disconnects but another valid profile receives deliberate
  input, play changes prompts without penalty. If no usable input remains, solo
  enters exact pause automatically. Cooperative play cannot pause the song, so
  the affected chart enters an identified input-unavailable suspension: no
  artificial Misses and no contribution until a usable profile returns and
  Rhythm performs fair re-entry. Coverage records the lost opportunity.
- A device-specific control reference is available before play, from every safe
  menu, during practice, and from solo pause. Cooperative battle offers a
  nonblocking compact reference that cannot cover active notes or protected cues.

#### UX-06 - Telegraphs, camera, position, and cue arbitration

- **Status:** Resolved 2026-08-31.
- Boss pose/motion, arena geometry, and world-space source/target treatment are
  primary. Compact interface reinforcement repeats the same identified attack,
  response category, affected player/location/route, and countdown; it never
  invents a safer recommendation than the committed geometry supports.
- Telegraph, Commit, Impact, and Recovery have distinct consistent states.
  Telegraph reveals source, shape, target class, response, and validated lead.
  Commit visibly locks the exact targets, unsafe geometry, impact boundary, and
  effect expectation. Impact confirms the identified resolution. Recovery
  closes the threat and exposes any authored earned-advantage opportunity.
- A committed cue cannot drift, retarget, or change countdown. Advisory response
  language is visibly distinct from locked target/geometry. A safe cancellation
  visibly dissipates all related world/UI cues and never substitutes another
  impact.
- Multiple legal threats use one ordered compact threat stack keyed by impact
  boundary plus simultaneous world geometry. The earliest threat affecting the
  player receives dominant reinforcement, matching Combat's automatic Defend
  focus. Equal-time order is stable. Required group/recovery and attack cues
  follow authored reservation priority; player pings remain visually and
  aurally secondary to protected automatic cues.
- World position markers show legal destinations, risk tier, cover/hazard state,
  current location, movement charge/recovery, and visible route. A farther touch
  destination previews every paid edge and arrival expectation. Disabled or
  unsafe destinations explain why; unavailable movement rejects with restrained
  feedback and never appears queued or silently reroutes.
- Off-screen/source reinforcement points toward the authored source or affected
  edge using direction plus icon/label. Shape, boundary/hatching, placement, and
  text distinguish safe/unsafe, target/non-target, committed/advisory, and
  cover/hazard states without requiring color, motion, sound, or haptics alone.
- The default camera remains stable behind/above the performer and follows legal
  movement smoothly. Directed arrival, break, group, and outcome moments occur
  only outside an active phrase or reaction window and never hide current
  geometry. Reduced-camera-motion replaces them with stable framing and compact
  state emphasis while preserving all information.
- Authoring/validation must reject impossible combinations before publication.
  If required critical cue/geometry identity is missing or contradictory at
  runtime, the encounter cancels the affected event safely when still possible
  or invokes the established outcome-critical No Contest path. Presentation
  never guesses, compresses the warning, or allows a surprise impact.

### Checkpoint C - Learning, preparation, and post-battle flow

#### UX-07 - Onboarding, practice, and contextual teaching

- **Status:** Resolved 2026-08-31.
- First arrival begins with one short explanation of the play fantasy and clear
  **Begin Practice**, **Accessibility and Comfort**, and **Skip Practice** routes.
  Accessibility/settings are usable before any lesson. Skip first presents the
  active device's complete control reference and requires explicit confirmation;
  it never implies reduced rewards or inability.
- Setup chooses one unlocked, song-supported starter instrument from clear
  options, previews its identity, shows current-device controls, and offers
  guided calibration. It asks no role/class questionnaire and does not expose
  advanced loadout/build decisions before the core interaction is understood.
- Practice uses the real Rhythm, Combat, Survival, and Positioning interfaces in
  a safe authored environment. Its six action-oriented modules are Setup,
  Perform, Attack, Defend, Move, and Combine. The interface shows the named
  sequence, current module, and saved progress, but only one instruction and one
  immediate success criterion at a time.
- Module completion verifies that the player attempted the required semantic
  interactions and reached the safe exercise endpoint; it does not require a
  minimum judgment, personal rating, Perfect, survival pressure, or repeated
  mastery test. A mistake gives normal nonshaming feedback and repeats only the
  relevant exercise. Repeated difficulty offers demonstration, control reference,
  calibration/settings, and retry rather than secretly changing controls or
  grading.
- Completion saves after every module. Return resumes at the next incomplete
  module with prior settings/instrument choice intact. Players may replay one
  module or the full sequence later from Practice/Help. Practice progress is not
  farmable account progress and no lesson becomes inaccessible after completion
  or skip.
- Completing or explicitly skipping the sequence unlocks public matchmaking.
  Store eligibility remains separate and additionally requires one completed
  encounter. These gates use owner-supplied state and never infer completion
  from closing a surface.
- First-boss teaching uses the approved real encounter script: ordinary Attack
  at Arrival; movement, Defend, and Ward in First Clash; Hype/Signature during
  Escalation; the guaranteed Crescendo; first relevant solo/co-op recovery; then
  Band Call and consumable prompts on first genuine availability.
- Each contextual lesson has unseen, presented, action-observed, dismissed, and
  disabled history. It previews before the relevant decision, anchors to the
  real control/world cue, contains minimal current-device text/glyphs, and never
  pauses, rewinds, covers notes, or outranks protected gameplay cues. Dismissal
  prevents repetition in that moment; action observation records understanding.
  Missed/dismissed teaching remains available through control reference,
  practice, and Help.
- Contextual teaching may be disabled globally or by category from a safe menu.
  Critical automatic cues remain even when teaching is disabled. Returning
  players see only genuinely new or still-unseen concepts, never a repeated
  first-run tour without consent.

#### UX-08 - Staging, loadout, inventory, builds, upgrades, and store surfaces

- **Status:** Resolved 2026-08-31.
- Prepare opens on one **Current Setup** summary containing the selected role,
  three gear slots, Signature, Band Call, build configuration, two consumable
  types/charges, active full spec preset when applicable, encounter compatibility,
  and every current validation finding. Functional labels remain subject to the
  naming/tone pass.
- The summary provides the primary **Apply/Ready** path plus three visible full
  spec presets when unlocked. Selecting a slot/configuration opens one item/
  option browser in a wide side panel or touch bottom/full sheet, preserving the
  setup behind it. A focused detail replaces that panel rather than adding a
  third navigation level or nested tabs.
- Browsers group and filter by owned type/slot, compatible role/encounter,
  functional emphasis, tier/rank, and availability; deterministic sort and
  search appear when catalog size requires them. Active filters and result count
  stay visible with Clear All. Incompatible, locked, retired, disabled, empty,
  and missing choices remain discoverable with exact reasons instead of silently
  disappearing.
- Beginner summaries lead with role/behavior, primary stat, fixed trait, main
  tradeoff, and compatibility. Expandable exact detail exposes current-versus-
  candidate values, sources, caps, trigger/target/fallback, upgrade state/cost,
  synergies/conflicts, and affected preset/loadout references. Critical cost,
  incompatibility, or drawback is never hidden in a tooltip or collapsed detail.
- Editing produces an explicit recoverable draft. Local constraints respond
  immediately; final owner-domain/server validation reports all issues together.
  Apply is atomic: success confirms the exact current configuration; failure
  preserves both prior applied state and the repairable draft. It never silently
  drops, substitutes, weakens, spends, or changes role.
- Hub queue edits remain legal until staging. Editing the active staging loadout
  clears that player's Ready with a visible explanation; editing a nonactive
  saved preset does not. Final lock turns all snapshot-affecting controls read-
  only and identifies the locked encounter. Deployment rollback restores
  unlocked staging rather than losing the draft.
- Upgrade, uplift, craft, salvage, and purchase review shows exact input/cost,
  output, pre/post value, compatibility, references affected, and balance before
  commitment. Reversible navigation needs no confirmation. Resource spends,
  referenced-instance salvage, preset overwrite, and Robux purchase receive one
  consequence-specific confirmation; duplicate confirmations and destructive
  action plus redundant Undo are avoided.
- Empty/zero/full states explain what is absent and what legitimate action can
  change it. Zero consumable quantity keeps the remembered type visibly Empty.
  No compatible alternative preserves the current setup and offers a useful
  filter reset or route, not an automatic substitute. First release has no
  inventory-full, mailbox, or paid-storage state.
- The store is a separate voluntary hub destination, available only after its
  eligibility gate. It clearly separates Robux and earned paths, shows exact
  price/function/tier/equivalent/ownership before purchase, prevents accidental
  duplicates, and restores receipts through Commerce. No Prepare recommendation,
  validation error, empty slot, queue, staging warning, or power comparison turns
  into an unsolicited store prompt.

#### UX-09 - Results summary, evidence, and next actions

- **Status:** Resolved 2026-08-31.
- A brief Victory, Defeat, or Invalid / No Contest presentation may establish
  tone and is always skippable. The immediate phone-first summary then shows in
  order: outcome; canonical exact reason; personal rating explicitly separate
  from outcome; most important already-granted rewards/unlocks; and one large
  valid primary next action.
- Exact reason follows Boss Encounters' frozen priority, including all humans
  down, all humans departed, all humans inactive/departed, Resolve remaining,
  finishing threshold missed, and outcome-critical Invalid / No Contest. UI does
  not choose a more flattering or dramatic reason. No Contest is system/content
  invalidity, not player Defeat or weak performance.
- Primary action uses available authoritative state: first eligible Victory
  favors Continue Story; repeat Victory favors Retry Same Shard or an available
  Stay with Band flow; Defeat favors Retry Same Shard; No Contest favors safe
  retry when available or Return to Hub. Loadout and Upgrades plus Return to Hub
  remain visible secondary choices. An unavailable/stale route is removed with
  its reason rather than presented optimistically.
- Four peer detail sections remain optional and state-preserving:
  **Performance**, **Combat**, **Band**, and **Progress**. They use labeled tabs
  where they fit and the same ordered one-open-section pattern when touch scale/
  localization cannot. There are no nested tabs.
- Performance shows private judgments, early/late, holds, authentic participation
  coverage/absence, and instrument/difficulty personal best. Combat shows private
  intent effects, Resolve/Momentum, Ward/survival, threats, position/risk, and
  consumable evidence. Band shows personal Call/Crescendo/revival/Cohesion share
  plus collective result without ranking. Progress shows exact resource/item/
  mastery/campaign/unlock/restoration/crafting-path results.
- At most two private suggestions use sufficient identified evidence and compare
  the player only with their own history. Each explains the observed pattern and
  a relevant voluntary action such as calibration, practice, control reference,
  loadout review, or telegraph replay. Weak/ambiguous evidence produces no
  suggestion. Suggestions can be dismissed and never disclose accessibility,
  spending, hidden skill, or another player's performance.
- Rewards and progression are already committed before Results. Animations may
  celebrate but remain skippable, never gate access, and never delay Retry. New
  unlock/reward details route to their legitimate owner surface while preserving
  Results context; there is no per-item claim loop.
- Retry, Stay with Band, Continue Story, Loadout and Upgrades, and Return to Hub
  are individual choices. UI reflects accepted/waiting/declined/expired/refill
  state without presenting a binding vote. Follow-up timeout affects only that
  player and defaults to Hub under Multiplayer.
- Results contains no store/Robux offer, rescue, public rank, damage leaderboard,
  public suggestion, blame label, or statement that one performer caused defeat.
  Acolyte support is explicitly fixed NPC support rather than human performance.

### Checkpoint D - Accessibility, system states, and implementation contract

#### UX-10 - Settings, calibration, accessibility, and saved profiles

- **Status:** Resolved 2026-08-31.
- Settings use five task groups: **Input and Calibration**, **Interface and
  Readability**, **Comfort and Motion**, **Audio and Captions**, and **Teaching
  and Language**. These are functional labels pending the naming/tone pass.
  Each setting declares identity, label/help keys, type/range/options, safe
  default, current value, scope, apply boundary, preview/reset behavior,
  dependencies, and every consumer that must preserve critical information.
- Account-wide preference intent includes Hold Assist, captions/subtitles,
  language, teaching, high-contrast/color-vision treatment, and reduced motion/
  flashing/effects. Compatible values sync by default without overriding an
  explicit device exception. Other players never see these facts.
- Per-device/control profiles own calibration, keyboard/gamepad bindings, touch
  handedness/layout, staff/note/interface scale, visual scroll speed, haptics,
  output-specific levels/dynamic-range choice, and caption size/background when
  screen/output needs differ. The active profile and any override are clearly
  identified; changing device never silently resets explicit choices.
- Presentation-only changes preview and apply immediately when they can preserve
  every cue. Calibration applies only to future practice/attempt playback and
  never reinterprets active/resolved timing. Safe-menu input/touch edits apply to
  the next activity; a solo-pause edit may apply to the frozen attempt only after
  successful validation and the full visible/audible resume countdown. Active
  cooperative play does not expose a focus-stealing editor.
- Every group supports readable current/default state, immediate preview where
  safe, atomic Apply/Cancel for grouped edits, and Reset for the explicit scope.
  Reset shows the affected values before commitment and never resets unrelated
  account/device profiles. A failed save preserves the applied profile and the
  repairable draft.
- Safe defaults already restrain full-screen flashing, bloom, particles, camera
  motion/shake, impact zoom, and haptics. Independent controls may reduce each
  further without removing targeting, timing, position, state, or response cues.
  No setting requires restart for presentation behavior.
- Safe menus remain usable at 200% interface/text scaling through reflow and
  scrolling. Normal text meets at least 4.5:1 contrast; large text, icons,
  controls, boundaries, and focus indicators meet at least 3:1. Encounter staff,
  notes, interface, touch controls, and captions scale independently within
  validated nonoverlap limits.
- Hold Assist, calibration, remapping, touch layout, high contrast, color-vision
  treatments, reducible effects, captions/source labels, language, replayable
  teaching, and solo pause never alter difficulty, maximum contribution,
  rewards, matchmaking, public identity, or result treatment.

#### UX-11 - Feedback, focus, loading, failure, and recovery states

- **Status:** Resolved 2026-08-31.
- Every applicable component specifies default, hover, pressed, focus, selected,
  queued, effective/committed, disabled, unavailable-with-reason, read-only,
  loading/pending, success, warning, error, stale, expired, empty, and maximum-
  content behavior. States use semantic label/icon/shape plus color where useful.
- Primary actions respond immediately and accept one identified request while
  pending. They disable or expose owner-approved cancel behavior until success,
  rejection, expiry, or retry. Double tap/click, repeated delivery, and stale
  response cannot produce another transaction, consent, Ready, spend, or effect.
- Known waits use truthful determinate progress/countdown. Unknown short waits
  use a restrained busy state; structured loading uses a matching skeleton.
  Longer waits explain the task and provide owner-approved cancellation or safe
  background continuation. UI never invents percentages or lets animation imply
  authoritative completion.
- Field/choice validation is adjacent, specific, nonblaming, announced, and
  retained until repaired. A summary links to every issue for an atomic task.
  Low-priority completed facts may use brief notices; an ongoing degraded state
  uses one persistent banner; a blocking consent/irreversible decision uses one
  dialog. Critical errors never auto-dismiss or hide behind a notice queue.
- First-use, no-match, zero-quantity, unavailable, retired, missing, cleared,
  failed-load, one-item, long localized text, large catalog, and maximum-roster/
  maximum-cue states explain why the state exists and the legitimate next action.
  No surface is blank or fabricates content to fill space.
- Navigation and failure preserve legal drafts, selection, filters, sort, scroll,
  focus, and last confirmed read-only facts. Stale facts are labeled and cannot
  authorize a mutation. UI does not speculatively queue purchase, economy,
  inventory, loadout Apply, Ready, reward, progression, or consent changes while
  their authority is unavailable.
- Durable edits expose Draft/Changes Pending, Saving, Saved, Not Saved with Retry,
  and Save Unsafe/Read-Only as applicable. A success state appears only after
  Player Data or the owning transaction confirms it. Retry reuses the same
  idempotency identity and never overwrites a newer confirmed revision.
- Network/session degradation distinguishes local connection, queue/staging,
  deployment, active-attempt grace, input-unavailable Rhythm suspension, and
  global critical failure. It shows remaining authoritative grace/countdown and
  valid alternatives without predicting rejoin. Known Results may render while
  a reward section truthfully says it is still syncing; it never claims a grant
  is lost or complete before confirmation.
- Opening/closing/recovery restores deterministic focus. Dynamic status changes
  use priority-aware accessible announcements where Roblox exposes them, without
  repeatedly interrupting timing play. Player-facing messages use plain language,
  explain consequence and recovery, and keep technical codes in private evidence.

#### UX-12 - Design system, semantic outputs, localization, and acceptance

- **Status:** Resolved 2026-08-31.
- The implementation design system defines versioned tokens for spacing/grid,
  typography, color/contrast/pattern, borders/shapes, iconography, elevation,
  motion/duration/easing, safe areas/breakpoints, touch/click target sizes, focus,
  and semantic feedback. Brand/art direction selects final appearance within
  these functional and accessibility constraints.
- The component catalog covers the responsive shell, bottom navigation/rail,
  headers/status, buttons/semantic controls, staff/lanes/pads, meters/progress,
  band status, threat stack/world markers, cards/lists/details/comparison,
  filters/search, tabs/accordions, drawers/sheets/dialogs, forms/settings,
  notifications/banners/skeletons/empty states, captions/subtitles, onboarding,
  staging, transaction review, and Results. Every applicable state from UX-11
  is documented with focus, input, scaling, and content behavior.
- A presentation registry maps stable owner-domain semantic fact/cue keys to
  allowed component, priority, grouping/deduplication, focus/announcement,
  localized text, icon/shape/pattern, caption/source, optional haptic request,
  and Audio handoff. Identity and revision prevent stale or duplicate display.
  UI never infers a gameplay transition from animation or raw totals.
- Critical cue priority follows UX-04/UX-06. Lower facts coalesce or wait rather
  than cover protected state. If a required semantic mapping or alternative
  modality is missing, validation blocks publication; runtime uses the owner-
  defined safe cancellation/No Contest path rather than a generic guessed cue.
- All player text uses external stable keys with parameters, plural/number/time
  support, complete sentences rather than concatenated fragments, and explicit
  internal-versus-player-name fields. Layout supports Unicode, at least roughly
  30% expansion, long words/names, future right-to-left text/safe-menu navigation
  mirroring, and real localized testing. Rhythm lane/timeline order, encounter
  left/right, world directions, attack geometry, and arena graphs never mirror
  as a localization side effect. Flags do not represent language and images do
  not contain the only copy.
- Privacy allowlists restrict every surface and announcement. Other players see
  only safe identity/appearance, role, Ready/availability, survival/group state,
  and protected cues. Settings/accessibility/calibration, inventory/quantities/
  purchases, private build details, performance/history/recommendations,
  moderation/report state, and transaction failures remain private.
- Performance profiles cap decorative concurrent elements, animation/particles,
  update rates, textures, and replication for representative phones/tablets/
  desktop. Degradation removes or simplifies decorative treatment first and
  never drops a note, input target, committed threat, position, recovery, group,
  state, caption, or accessibility alternative. Exact engine budgets belong to
  technical architecture and must pass stable 30/60-fps device profiles without
  changing semantic timing.
- Acceptance covers touch phones/tablets, keyboard/mouse, and gamepad; supported
  aspect/safe-area/device changes; solo and one-to-six humans; all difficulties;
  practice/hub/staging/encounter/Results; empty/minimum/maximum/long/localized/
  right-to-left content; 200% safe-menu scale; keyboard/gamepad focus; platform-
  exposed screen-reader/announcement support; contrast/color-vision/reduced-
  effect/caption/Hold Assist combinations; and loading/network/save/transaction/
  content failure.
- Required evidence includes annotated responsive wireframes, interactive phone
  prototypes of critical flows, token/component/state catalogs, semantic mapping
  and copy catalogs, representative-device/accessibility manual scripts, and the
  observable GDD gates. At minimum, testing retains the GDD targets for
  onboarding/control/telegraph understanding and for 85% of players identifying
  outcome, reason, reward, and next action within ten seconds of Results.

## 4. Content/configuration reconciliation register

- No new authoring requirements have been approved yet.
- The completed reconciliation distinguishes runtime UI configuration, localized
  content, authored encounter cue metadata, and facts owned by other systems.
- `CONTENT_AUTHORING.md` gains only song/encounter-owned cue and validation
  requirements, not general UI layout, input mapping, settings, or hub behavior.
- Encounter packages must expose the semantic cue identities, alternative
  modality keys, attack-stage/source/target/geometry/countdown facts, position
  marker data, and phone-scale overlap evidence approved in UX-04/UX-06.
- Authored practice content must cover every UX-07 module and starter role with
  real legal mechanics, safe retry endpoints, stable instruction/cue keys, and
  first-boss contextual trigger/cue coverage without embedding tutorial logic in
  chart notes.

## 5. Confirmed architecture handoffs

- Gameplay systems own semantic state, timings, legality, and results; UI owns
  their composition, labels, focus, announcements, and device presentation.
- Multiplayer owns queue/party/rematch membership and communication safety; UI
  owns visible state, consent surfaces, focus, mute/block/report access, and
  recovery routing.
- Items, Builds, Progression, Rewards, and Commerce own domain validation and
  mutation; UI owns task flow, exact disclosure, confirmation, and failure
  presentation.
- Audio Presentation owns mix and audible cue output; UI/UX owns caption and
  subtitle rendering plus visual/haptic reinforcement requirements.
- Player Data owns durable profile/configuration guarantees; UI/UX owns setting
  definitions and player-visible save-unavailable or unsafe-save treatment.
- Rhythm and Multiplayer must recognize the UX-05 input-unavailable suspension
  and fair-return contract. It creates no Misses or contribution, records lost
  coverage, and cannot pause cooperative song time.
- Onboarding supplies module/skip/prompt-history facts to Player Data and exact
  eligibility facts to Multiplayer/Commerce. It never owns their durable commit
  or gate evaluation.
- Results consumes only frozen outcomes, transactions, progress, participation,
  and follow-up states. Owner domains must supply exact reason and evidence;
  Results never reconstructs them from displayed totals.
- Audio Presentation owns bus/mix/cue execution and caption/source metadata;
  UI/UX owns settings composition, visual/haptic reinforcement requirements,
  caption rendering, and the semantic presentation registry.
- Player Data must preserve account-wide intent, per-device/control profiles,
  explicit overrides, drafts, onboarding/prompt history, and confirmed revision/
  save state without taking ownership of setting meaning.

## 6. Change log

- **2026-08-30:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-30:** Approved UX-01 through UX-03. The four-destination task
  hierarchy, state-preserving navigation, physical and Fast Play hub routes,
  touch-landscape baseline, responsive composition, focus behavior, and
  accessibility constraints are resolved. Progress is 3 of 12 questions.
- **2026-08-31:** Approved UX-04 through UX-06. Persistent/contextual HUD
  hierarchy, protected cue priority, semantic controls, safe remapping/device
  changes, input-unavailable suspension, attack-stage/position presentation,
  camera limits, and critical-cue failure behavior are resolved. Progress is 6
  of 12 questions.
- **2026-08-31:** Approved UX-07 through UX-09. Replayable/skippable real-mechanic
  onboarding, first-boss contextual teaching, task-based atomic preparation,
  voluntary store separation, immediate/evidence-on-demand Results, private
  suggestions, and individual follow-up are resolved. Progress is 9 of 12
  questions.
- **2026-08-31:** Approved UX-10 through UX-12. Settings/profile scope and apply
  boundaries, accessibility baselines, complete component/loading/failure/save
  states, tokenized design-system and semantic mapping contracts, localization,
  privacy, performance degradation, and acceptance evidence are resolved. All
  twelve answers were reconciled into canonical `UI_UX.md`.
