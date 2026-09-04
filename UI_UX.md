# Bands Battle UI/UX

- **Status:** Approved
- **Approved:** 2026-08-31
- **Parent systems:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#7-experience-shell)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Multiplayer dependency:** [`MULTIPLAYER.md`](MULTIPLAYER.md)
- **Items/preset dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Decision source:** [`UI_UX_WORKING.md`](UI_UX_WORKING.md)
- **Interview plan:** [`UI_UX_QUESTIONS.md`](UI_UX_QUESTIONS.md)

## 1. Role and authority

This document defines the task structure, navigation, responsive presentation,
semantic controls, settings/calibration surfaces, onboarding, Results, feedback,
and accessibility contract for every first-release player-facing experience.

UI/UX owns:

- experience modes, task hierarchy, navigation, focus, and context return;
- hub wayfinding and fast-access composition;
- responsive HUD/screen layout and component behavior;
- mapping supported physical inputs into semantic player actions;
- setting definitions, profile scope, preview/apply/reset behavior;
- onboarding/practice flow and contextual-teaching history;
- Results composition and follow-up routing;
- accessible visual/caption/focus/announcement treatment;
- presentation priority, deduplication, and semantic cue registry; and
- complete ideal, loading, failure, recovery, and content-scale states.

UI/UX does not own gameplay state or legality, charts/timing/judgments, encounter
selection, loadout/inventory mutation, rewards/progression, multiplayer consent,
audio mixing, or durable storage. It presents identified semantic facts and
routes identified requests to their owning systems. It never derives or repairs
authoritative state from visible totals or animations.

## 2. Governing invariants

1. **Phone-first parity:** touch, keyboard/mouse, and gamepad offer the same
   gameplay opportunity, information, rewards, and consent.
2. **Boss and music remain primary:** interface frames the boss, performer,
   arena, positions, and staff instead of covering or replacing them.
3. **Timing controls stay fixed:** three rhythm controls never move during play
   or acquire unrelated meanings.
4. **One semantic truth:** every displayed state/action references an identified
   owner-domain fact; presentation never creates a second gameplay timeline.
5. **Simple first, exact on demand:** primary tasks and consequences are clear
   immediately while optional evidence remains reachable without hiding risk,
   cost, incompatibility, or failure.
6. **No silent change:** UI never silently substitutes, spends, equips, applies,
   readies, queues, consents, purchases, retries, or follows another player.
7. **Back preserves work:** legal draft, selection, filters, sort, scroll, and
   focus survive navigation and recoverable failure.
8. **Critical meaning is multimodal:** color, sound, motion, or haptics alone are
   never required for a note, action, target, position, state, or outcome.
9. **Accessibility is private and value-neutral:** assists never affect
   difficulty, maximum contribution, rewards, progression, matchmaking, public
   identity, or Results treatment.
10. **Failure is truthful:** stale, pending, unavailable, unsafe, canceled,
    rejected, and invalid states remain distinguishable from success.
11. **No speculative mutation:** unavailable authority blocks durable or
    economic mutation rather than accepting an unprovable local success.
12. **No public blame:** UI contains no public damage rank, performance rank,
    accessibility label, defeat culprit, purchase history, or private suggestion.
13. **No coercive store:** Commerce appears only through a voluntary eligible
    destination and never through combat, Results, rescue, or validation repair.
14. **Names are governed:** internal keys and placeholder system labels cannot
    ship before the required naming/tone/localization pass.
15. **Decorative degradation first:** device/performance reduction never drops a
    note, control, threat, position, recovery, group cue, caption, or alternative.

## 3. Player jobs, modes, and global hierarchy

The safe experience shell has four primary task destinations. Their current
labels are functional placeholders:

- **Play:** Continue Story, unlocked shard discovery/selection, recent/replay,
  mode, difficulty, and encounter-card entry;
- **Prepare:** current loadout, inventory, upgrades, builds, abilities,
  consumables, and three full spec presets;
- **Progress:** campaign, mastery, records, collection, restored fragments,
  archive, and hub-restoration evidence; and
- **Band:** party, matchmaking, safe communication, and compatible social state.

Settings, Controls, Accessibility, and Help are global utilities. The store is
not primary navigation or a recommended next action.

The shell distinguishes these modes:

- first arrival and resumable practice;
- safe hub exploration/menu;
- queued-safe hub/menu;
- encounter card and party/match proposal;
- unlocked staging and final loadout lock;
- practice or active encounter;
- solo pause;
- immutable Results/follow-up; and
- blocking load, recovery, save-unsafe, or critical-failure state.

Current mode, current task, relevant queue/party state, and one truthful next
action remain apparent. Safe modes expose global navigation. Task modes have
explicit exits. Active battle exposes only permitted pause/settings/reference/
leave behavior; cooperative play never presents pause as available.

Navigation is at most two levels below a primary destination. A focused detail
replaces its browser panel rather than nesting another navigation hierarchy.
Back restores the prior valid surface and its legal state. It never returns to
an expired encounter/transaction. Leaving a queue, Ready commitment, purchase,
or meaningful unsaved edit uses one consequence-specific confirmation; ordinary
reversible navigation does not.

## 4. Order hub and fast access

First arrival uses a short unmistakable route toward accessibility/practice and
the first unlocked shard. Returning players appear on the stable central path or
highest unlocked landing, within a few seconds of playable content and essential
functions.

The tiered shard ascent remains the dominant physical landmark. Practice,
workshop/preparation, archive/progress, social commons, and voluntary store use
stable silhouettes, labels, interaction footing, and landmarks. Campaign
restoration may enrich architecture, population, lighting, music, and portals
but cannot relocate a learned anchor or invalidate route vocabulary.

Every shard communicates boss identity, campaign tier, availability, and exact
lock reason with label/icon/shape plus color. A newly unlocked shard may receive
one restrained dismissible emphasis. Locked content never resembles a Robux
gate. Proximity reveals the action; a deliberate input opens the normal
encounter card. It never deploys by accidental proximity.

The Play destination offers Fast Play for Continue Story, last played unlocked
shard, and appropriate recent/replay choices. Each opens the same encounter card
as the physical shard. An unlocked landing shortcut may move the avatar near a
shard without making repeat traversal mandatory.

World landmarks and a compact optional objective/landmark guide provide
orientation without turning the hub into a flat mission-board overlay. Required
routes avoid precision jumping, forced camera control, narrow collision tests,
and materially longer accessibility detours. Optional lore, NPCs, emotes, and
non-scored music never block play, preparation, or progression.

Public queue state remains visible during compatible movement/menu use. A
conflicting action explains why and asks whether to leave; it never cancels the
queue silently.

## 5. Responsive shell and navigation controls

Touch landscape is the first-release phone baseline. Portrait entry provides an
accessible rotate instruction plus pre-play settings rather than compressing an
encounter. Tablet, desktop, and gamepad preserve destination names, order,
information architecture, and task state.

- Safe touch menus use a labeled four-destination bottom bar.
- Wide desktop uses a compact labeled rail.
- Gamepad exposes the same four destinations in the same order with deterministic
  focus navigation.
- Queue/party context and utilities occupy a consistent header/utility surface.

A breakpoint changes composition, not meaning. Primary destinations never hide
inside generic overflow. Context details use inline expansion, a wide side panel,
or a touch sheet/drawer. Tabs are not nested. Modals exist only for blocking
consent, irreversible/externally committed action, or a decision that cannot
continue safely in context.

General touch actions are at least 48 by 48 logical pixels with separation;
timing pads and frequent battle actions are larger. Safe areas, notches, system
gestures, aspect ratio, and chosen UI scale cannot cover a critical action/state.

Keyboard/gamepad cover every action with visible focus, logical/spatial order,
safe initial focus, modal focus containment, and focus restoration on close/back.
Loading never steals focus. Touch never requires hover; desktop hover is
supplementary. Reflow/device change preserves the same draft, selection, filters,
sort, scroll, and focus when legal.

## 6. Encounter composition and persistent HUD

A central gameplay field is reserved for the boss, avatar, arena geometry,
positions, and physical telegraphs. Persistent UI frames that field and remains
stable across ordinary play, device glyph changes, difficulty, and roster size.

The upper boss region shows only necessary identity, song function/phase,
current Resolve layer/progress, future locked layers, and Momentum while it has
a legal destination. It never implies pressure on an unopened layer or victory
before Finishing Cadence.

The lower performance region contains the right-to-left staff, fixed strike
line, compact immediate judgment, and three fixed rhythm actions. Ward/survival
and effective/queued intent remain near their controls. Hype lives on Special;
Band Call readiness lives on its equipped control. UI adds no redundant
persistent resource meter.

A compact band strip exposes only safe necessary state: identity/appearance,
role, active/downed/recovering/return-protected state, connection availability,
and required group readiness. Healthy members remain quiet. Relevant target,
recovery, disconnect, acolyte suppression, or return state receives restrained
emphasis. No public performance, damage, accessibility, gear, or blame appears.

The normal one-second-glance target is recognition of:

- personal survival and effective/queued intent;
- current boss layer and phase;
- nearest committed danger affecting the player; and
- current rhythm action/strike timing.

Representative phone testing must validate that target without requiring
sentence reading.

## 7. Contextual cue priority and rhythm feedback

Phrase previews, position routes, attack response, consumables, Signature
commitment, group invitations/windows, revival/recovery, pings, and teaching
appear only when actionable or necessary. Each cue identifies its world source,
target/control, and musical countdown.

Priority is:

1. unsafe synchronization/control or terminal state;
2. imminent committed threat and urgent recovery;
3. required authored group event;
4. accepted/pending player group action;
5. movement and prepared-item opportunity; then
6. tutorial, ping, and informational feedback.

One dominant cue receives full treatment. Other valid facts reduce to ordered
source/target chips or world markers until primary. Lower priority cannot cover,
delay, or impersonate a protected cue. Band Call and Crescendo performance
windows remain mutually exclusive under canonical reservation priority.

Time-critical cues use shape/icon/label and beat/time progress, not explanatory
sentences. Rhythm suspension clears unresolved notes without Miss, labels the
actual cause, and presents fair-return readiness/countdown when available.
Immediate grades remain near the staff; detailed timing distributions wait for
Results.

## 8. Semantic actions, remapping, and device transitions

The complete GD-06 action map is binding. Rhythm uses three fixed actions;
movement uses WASD/left stick or hub joystick/encounter destination; intent,
Join In, Band Call, and two consumables retain their approved device families.

Every action presents owner-supplied states such as available, unavailable with
reason, pressed, queued, effective, committed, cooling/recovering, reserved, or
spent. A legal queue responds immediately and identifies its effective musical
boundary. UI never guesses legality or makes rejection look queued.

One physical event produces at most one semantic action in its active context.
Different dedicated controls may act simultaneously; owning systems resolve
identified boundary order without UI dropping or duplicating input.

Remapping occurs only in a safe surface. It prevents reserved/platform conflicts
and duplicates within a simultaneously active context, reports all conflicts,
previews glyphs, and applies/resets a profile atomically. Reuse is legal only in
provably exclusive contexts.

Active-device prompts change after deliberate input with stability delay to
avoid flicker. They update labels/reference without moving the HUD, changing
semantic state, synthesizing press/release, or rearranging timing controls.

Touch layout editing provides handedness, size, spacing, position, and opacity
on a safe preview canvas. It enforces safe areas, minimum targets/separation, no
critical overlap, and all-action reachability. Editing emits no gameplay input
and offers Apply, Cancel, and Reset.

If another valid profile receives input after device loss, labels change without
penalty. If none remains, solo pauses. Cooperative play uses the identified
input-unavailable Rhythm suspension: no Misses or contribution, shared song
continues, lost coverage records, and fair re-entry follows restoration.

A current-device control reference is available before play, in safe menus,
practice, and solo pause. Cooperative battle may show a nonblocking compact
reference that cannot cover notes or protected cues.

## 9. Telegraphs, positions, and camera

Boss pose/motion, arena geometry, and world source/target treatment are primary.
Compact UI repeats the same identified attack, response class, affected player/
location/route, and countdown. It never invents a safer recommendation than the
committed geometry supports.

Attack presentation distinguishes:

- **Telegraph:** source, shape, target class, response, and validated lead;
- **Commit:** visibly locked targets, unsafe geometry, impact boundary, and
  effect expectation;
- **Impact:** the identified ordered resolution; and
- **Recovery:** closed threat plus any earned-advantage opportunity.

Committed cues cannot drift, retarget, or change countdown. Advisory response is
distinct from locked target/geometry. Safe cancellation visibly dissipates every
related cue and never substitutes a surprise impact.

Multiple threats use one impact-ordered compact stack plus simultaneous world
geometry. The earliest affecting threat is dominant, matching automatic Defend
focus. Equal-time order is stable. Player pings remain secondary to automatic
cues.

World position markers show current/legal destination, risk tier, cover/hazard,
movement readiness, and route. A farther touch destination previews paid edges
and arrival. Disabled/unsafe destinations explain why. Unavailable movement
rejects without queuing or silent reroute.

Off-screen/source indicators use direction plus icon/label. Shape, hatching,
placement, and text distinguish safe/unsafe, target/non-target, Commit/advisory,
and cover/hazard without relying on color, sound, motion, or haptics.

The default camera stays stable behind/above the performer and follows legal
movement smoothly. Directed arrival/break/group/outcome moments occur only
outside active phrases/reaction windows. Reduced-camera-motion uses stable
framing and compact state emphasis.

Authoring rejects impossible combinations. Missing/contradictory critical cue
identity causes safe event cancellation when possible or the established
outcome-critical No Contest path. UI never guesses or compresses warning time.

## 10. Onboarding, practice, and contextual teaching

First arrival briefly establishes the play fantasy and exposes **Begin
Practice**, **Accessibility and Comfort**, and **Skip Practice**. Labels remain
subject to naming. Skip first shows the complete active-device control reference
and requires explicit confirmation without shame or reward warning.

Setup chooses one unlocked supported starter instrument, previews its identity,
shows controls, and offers guided calibration. It asks no class questionnaire or
advanced build choice.

Practice uses real canonical mechanics in safe authored content and exposes six
named modules:

1. Setup;
2. Perform;
3. Attack;
4. Defend;
5. Move; and
6. Combine.

The sequence/current module/saved progress are visible, but only one instruction
and one success criterion appear at a time. Completion verifies attempted
required interactions and exercise endpoint, not a minimum grade, Perfect,
survival pressure, or mastery test. Mistakes repeat only the relevant safe
exercise. Repeated difficulty offers demonstration, controls, calibration/
settings, and retry without secret control/difficulty changes.

Progress saves after every module. Return resumes at the next incomplete module.
One module or the whole practice remains replayable. Practice cannot farm account
progress. Completion or explicit skip unlocks public matchmaking; store access
separately also requires a completed encounter.

The first boss teaches Attack, movement/Defend/Ward, Hype/Signature, guaranteed
Crescendo, first relevant recovery, then Band Call/consumables when genuinely
available. Each lesson tracks unseen, presented, action-observed, dismissed, and
disabled history. It previews in context, anchors to real controls/cues, never
pauses/rewinds/covers play, and yields to protected cue priority.

Dismissal prevents repetition at that moment; observed action records
understanding. Missed/dismissed teaching remains in Help/practice/reference.
Teaching can be disabled globally/by category, but protected automatic cues
remain. Returning players see only genuinely new/unseen concepts.

## 11. Preparation, inventory, presets, and transactions

Prepare opens on a Current Setup summary: role, three gear slots, Signature,
Band Call, build, two consumable types/charges, active full spec preset where
applicable, encounter compatibility, and all validation findings.

The summary exposes Apply/Ready plus three presets when unlocked. A slot opens
one browser panel/sheet and a focused detail replaces it rather than adding a
third level.

Browsers filter/group by owned type/slot, role/encounter compatibility,
functional emphasis, tier/rank, and availability. Deterministic sort/search
appear as the catalog grows. Active filters and result count remain visible with
Clear All. Incompatible, locked, retired, disabled, empty, and missing choices
stay discoverable with exact reasons.

Beginner summaries show role/behavior, primary stat, trait, tradeoff, and
compatibility. Exact detail shows current/candidate values, sources, caps,
trigger/target/fallback, upgrade state/cost, synergies/conflicts, and affected
references. Cost, incompatibility, and drawback cannot hide in tooltip/collapse.

Editing produces a recoverable draft. Local constraints respond immediately;
final owner validation reports all findings. Apply is atomic: success confirms
exact configuration, while failure preserves prior applied state plus draft.
Nothing is dropped, substituted, weakened, spent, or role-changed silently.

Queued hub edits remain legal until staging. Active staging edits clear Ready
with explanation; inactive saved-preset edits do not. Final lock makes snapshot
controls read-only and names the encounter. Deployment rollback returns to
unlocked staging without losing legal draft state.

Upgrade/uplift/craft/salvage/purchase review shows exact input/cost, output,
pre/post value, compatibility, affected references, and balance. Resource spend,
referenced salvage, preset overwrite, and Robux purchase use one confirmation.
Reversible navigation does not.

Empty/zero/full states explain cause and legitimate next action. Zero consumable
quantity preserves Empty type. No compatible alternative preserves current
setup and offers filter reset/route, never substitution. First release has no
inventory-full, mailbox, or paid-storage state.

The voluntary eligible store clearly separates Robux/earned paths, exact price/
function/tier/equivalent/ownership, prevents duplicates, and supports receipt
restoration. Prepare, validation, queue, staging, power comparison, and empty
slots never turn into a store prompt.

## 12. Results and voluntary follow-up

A brief outcome presentation is skippable. The immediate summary shows, in
order:

1. Victory, Defeat, or Invalid / No Contest;
2. canonical exact reason;
3. personal rating clearly separate from outcome;
4. most important already-granted rewards/unlocks; and
5. one large valid primary next action.

Reason follows Boss Encounters' frozen priority, including down/departed/
inactive, Resolve, finishing, and No Contest cases. No Contest is system/content
invalidity, not weak player performance.

First eligible Victory favors Continue Story; repeat Victory favors Retry or an
available Stay with Band; Defeat favors Retry; No Contest favors safe Retry or
Hub. Loadout and Upgrades plus Return to Hub remain secondary. Stale/unavailable
routes are omitted with reason rather than shown optimistically.

Four optional peer detail sections preserve state:

- **Performance:** judgments, early/late, holds, authentic coverage/absence,
  personal best by instrument/difficulty;
- **Combat:** personal intent effects, Resolve/Momentum, Ward/survival, threats,
  position/risk, and consumables;
- **Band:** personal Call/Crescendo/revival/Cohesion share plus collective result
  without rank; and
- **Progress:** exact resources, items, mastery, campaign, unlock/restoration,
  and crafting-path results.

Tabs are used when labels fit; an ordered one-open-section pattern adapts under
touch scale/localization. Tabs never nest.

At most two private suggestions require sufficient evidence, compare only with
personal history, explain the pattern, and offer a voluntary relevant route.
Ambiguous evidence produces no suggestion. Suggestions are dismissible and
never disclose accessibility, spending, hidden skill, or another player.

Rewards/progression are committed before Results. Skippable animations cannot
gate access or Retry. New details may route to owner surfaces while preserving
Results context; no claim loop exists.

Follow-up choices are individual. UI shows accepted/waiting/declined/expired/
refill state without binding vote. Results contains no Store/Robux offer, rescue,
public rank/leaderboard, public suggestion, or blame. Acolytes are fixed NPC
support, not human performance.

## 13. Settings and profile model

Settings use five functional groups:

- Input and Calibration;
- Interface and Readability;
- Comfort and Motion;
- Audio and Captions; and
- Teaching and Language.

Every setting declares stable identity, label/help keys, type/range/options,
safe default, current value, scope, apply boundary, preview/reset behavior,
dependencies, and consumers responsible for critical-cue preservation.

Account-wide intent includes Hold Assist, captions/subtitles, language,
teaching, high-contrast/color-vision treatment, and reduced motion/flashing/
effects. Compatible values sync while respecting explicit device exceptions.

Per-device/control profiles own calibration, bindings, touch layout, staff/note/
interface scale, visual scroll speed, haptics, output levels/dynamic range, and
caption size/background where screen/output differ. Active profile/override is
visible and device change cannot reset explicit choices.

Safe presentation changes preview/apply immediately when cues remain complete.
Calibration applies only to future practice/attempt timing. Safe-menu input/
touch changes apply next activity. A solo-pause edit may affect the frozen
attempt only after validation and full beat-counted resume. Cooperative play
offers no focus-stealing active editor.

Groups provide current/default state, safe preview, atomic Apply/Cancel, and
scope-specific Reset. Reset previews affected values and leaves unrelated
profiles untouched. Failed save preserves applied profile plus repairable draft.

## 14. Accessibility and comfort acceptance

Safe defaults restrain full-screen flashing, bloom, particles, camera shake/
motion, impact zoom, and haptics. Players reduce them independently without
losing timing, targeting, position, state, response, or outcome cues.

Safe menus remain usable at 200% interface/text scale through reflow/scrolling.
Normal text meets at least 4.5:1 contrast. Large text, icons, components,
boundaries, and focus meet at least 3:1. Staff, notes, interface, touch controls,
and captions scale independently within validated nonoverlap ranges.

Every note, judgment, intent, threat, target, position, cover/hazard, survival,
group, queue, transaction, and outcome uses shape/label/placement/pattern or
another channel in addition to color. Critical audio has visual or optional
haptic reinforcement; critical visuals have audible/caption or appropriate
alternative reinforcement.

Hold Assist, calibration, remapping, touch layout, contrast/color-vision,
effects reduction, captions/source labels, language, teaching, and solo pause
never alter difficulty, maximum contribution, reward, matchmaking, public
identity, or Results.

All safe-menu actions support keyboard/gamepad focus and platform-exposed
accessible names, descriptions, values, error/status announcements, and focus
restoration. Screen-reader/assistive behavior is tested on each supported Roblox
surface; unavailable platform capability requires an equivalent documented
route rather than a false claim.

## 15. Component states, loading, and feedback

Every applicable component specifies:

- default, hover, pressed, focus, and selected;
- queued, effective/committed, reserved, and spent;
- disabled and unavailable with reason;
- read-only and locked;
- loading/pending, success, warning, and error;
- stale and expired; and
- empty, minimum, maximum, and long-content behavior.

Primary actions respond immediately and accept one identified request while
pending. They disable or expose owner-approved cancellation until resolved.
Repeated input/delivery/stale response cannot duplicate transaction, consent,
Ready, spend, or effect.

Known waits use truthful progress/countdown. Unknown short waits use a restrained
busy state; structured loading uses matching skeletons. Longer waits explain
the task and offer allowed cancellation/background continuation. No fake
percentage or success animation precedes owner confirmation.

Validation is adjacent, specific, nonblaming, announced, and persistent until
repaired. A summary links to all issues. Brief notices serve low-priority
completion; one banner serves ongoing degradation; one dialog serves blocking
consent/irreversible decision. Critical errors never auto-dismiss.

First-use, no-match, zero, unavailable, retired, missing, cleared, failed-load,
one-item, long localized text, large catalog, maximum roster, and maximum cue
states explain both cause and legitimate next action. No surface is blank or
fabricates content.

## 16. Failure, persistence, and recovery presentation

Navigation/failure preserve legal drafts, selection, filters, sort, scroll,
focus, and last confirmed read-only facts. Stale data is labeled and cannot
authorize mutation.

UI does not queue purchase, economy, inventory, loadout Apply, Ready, reward,
progression, or consent mutations while authority is unavailable. Durable edits
show Draft/Changes Pending, Saving, Saved, Not Saved with Retry, or Save Unsafe/
Read-Only. Success requires Player Data/transaction confirmation. Retry reuses
the idempotency identity and cannot overwrite a newer revision.

Network degradation distinguishes local connection, queue/staging, deployment,
active-attempt grace, input-unavailable suspension, and global critical failure.
It displays authoritative remaining grace/countdown and valid alternatives
without predicting rejoin.

Known Results may render while a reward section says it is still syncing; UI
never claims a grant is lost or complete before confirmation. Focus restoration
is deterministic. Priority-aware status announcements do not repeatedly
interrupt timing play. Messages explain consequence/recovery in plain language;
technical codes remain private evidence.

## 17. Design-system and semantic presentation contract

The implementation design system contains versioned tokens for:

- spacing/grid and responsive breakpoints/safe areas;
- typography and content density;
- color, contrast, pattern, border, and shape;
- iconography and gameplay/state symbols;
- elevation/layer priority;
- motion, duration, easing, and reduced-motion alternatives;
- touch/click target sizes and separation; and
- focus, feedback, and semantic status.

Final brand/art direction selects appearance within these constraints.

The component catalog includes shell navigation/status, semantic controls,
staff/lanes/pads, meters/progress, band status, threat stack/world markers,
cards/lists/details/comparison, filters/search, tabs/accordions, sheets/dialogs,
forms/settings, notices/banners/skeletons/empty states, captions, onboarding,
staging, transaction review, and Results. Each documents applicable states,
focus, input, scaling, and content behavior.

A presentation registry maps stable semantic fact/cue keys to:

- allowed component and placement;
- priority, grouping, and deduplication;
- focus/announcement treatment;
- localized copy and icon/shape/pattern;
- caption/subtitle/source treatment;
- optional haptic request; and
- Audio Presentation handoff.

Identity/revision prevent stale/duplicate display. UI never infers gameplay from
animation/raw totals. Missing required mapping/alternative blocks publication;
runtime follows owner-defined safe cancellation/No Contest, not a guessed cue.

## 18. Localization, naming, and privacy

All player text uses external stable keys with parameters, plural/number/time
support, and complete sentences rather than concatenated fragments. Layout
supports Unicode, roughly 30% expansion, long words/names, future RTL text and
safe-menu navigation mirroring, and real localization tests. Rhythm timeline/
lane order, encounter left/right, world directions, attack geometry, and arena
graphs do not mirror merely because the language is RTL. Flags do not represent
language; an image never contains the only copy.

Internal system/category/option/event keys never leak before the naming/tone
pass. Age-appropriate language avoids shame, accusation, technical jargon,
religious preaching, or coercive spending language.

Privacy allowlists permit others to see only safe identity/appearance, role,
Ready/availability, survival/group state, and protected cues. Settings,
accessibility, calibration, inventory/quantities/purchases, exact builds,
performance/history/suggestions, moderation/report state, and transaction/save
failures remain private.

## 19. Cross-system and Content Authoring contracts

UI/UX requires identified presentation-neutral semantic facts from every owner:

- Rhythm supplies chart/judgment/participation/suspension/re-entry facts;
- Combat/Survival supply intent/effect/Ward/down/recovery facts;
- Boss/Positioning supply lifecycle/Resolve/attack/geometry/movement/outcome;
- Abilities supply Hype/Signature/Call/Crescendo/acolyte state;
- Multiplayer supplies party/queue/staging/connection/ping/follow-up state;
- Items/Builds supply draft/validation/preset/snapshot/migration facts;
- Rewards/Progression/Commerce supply frozen transactions/unlocks/eligibility;
- Player Data supplies profile/draft/history/revision/save safety; and
- Audio Presentation supplies mix/cue and caption/source metadata.

Consumers receive only required privacy-safe facts. UI request identities let
owners deduplicate Apply, Ready, consent, purchase, spend, queue, and follow-up.

The completed 2026-09-02 Content Authoring reconciliation includes:

- stable UI/Audio cue keys and alternative-modality requirements for authored
  attacks, event stages, positions, graph changes, group/recovery windows, and
  story/function transitions;
- attack source/target/geometry/countdown and phone-scale overlap evidence;
- real safe practice content covering all six modules and starter roles;
- stable first-boss teaching trigger/cue coverage; and
- publication validation that every required fact/mapping/alternative survives
  export/load for every role, difficulty, roster, and supported device profile.

Practice packages contain authored content and cue identities, not hidden
tutorial logic. General layout/input/settings/hub behavior never belongs in a
song package.

## 20. Verification and observable acceptance

Required design/QA evidence includes:

- annotated responsive wireframes for every major mode;
- interactive representative-phone prototypes of practice, battle, preparation,
  settings, Results, and failure recovery;
- versioned token, component, state, semantic mapping, and copy catalogs;
- keyboard/gamepad focus and platform-assistive-technology audits;
- representative device/performance/accessibility manual scenarios; and
- observation plus privacy-reviewed analytics for the GDD readiness gates.

The matrix covers phones/tablets, keyboard/mouse/gamepad, supported aspect/safe-
area/device changes, solo and one-to-six humans, all difficulties, empty/minimum/
maximum/long/localized/RTL content, 200% safe-menu scale, contrast/color vision,
reduced motion/effects/haptics, captions, Hold Assist, and every loading/network/
save/transaction/content failure.

At minimum, testing retains the GDD targets for:

- at least 80% completing or deliberately skipping onboarding without coaching;
- at least 80% understanding/using rhythm, intent, and movement basics;
- at least 75% recognizing and answering a major telegraph;
- at least 75% understanding optional group-action invitations;
- accessibility combinations preserving every essential cue; and
- at least 85% identifying outcome, exact reason, important reward, and next
  action within ten seconds of Results.

Performance profiles cap decorative concurrent elements, animation/particles,
update rate, textures, and replication. Exact engine budgets belong to technical
architecture, but representative stable 30/60-fps profiles must preserve
semantic timing and every critical input/cue.

## 21. Deferred visual, tuning, and technical work

Behavior is complete; these remain downstream design-system, prototype,
playtest, content, balance, policy, or architecture work:

- final hub/system/menu/control/setting names and localized copy;
- final brand palette, type, icon, shape, texture, elevation, and motion style;
- exact responsive measurements, breakpoints, scale ranges, and component skin;
- exact calibration, binding, touch-layout, comfort, caption, and audio ranges;
- prompt copy/timing, practice assets, and first-boss teaching script;
- rating/suggestion formulae and exact Results ordering/animation;
- platform screen-reader/remapping/communication capability validation;
- engine performance/input latency/replication budgets and technical authority;
- localization languages/RTL schedule and policy compliance; and
- representative target-age/device/accessibility usability iterations.

None may introduce moving/reused rhythm controls, hidden critical information,
deeper task hierarchy, silent mutation, fake progress, speculative transaction,
public blame/accessibility labels, inaccessible critical cue, unsolicited Store,
or platform capability claims without evidence.

## 22. Approval and change control

The owner interview resolved UX-01 through UX-12 on 2026-08-31. This document is
the canonical UI/UX, Input/Settings/Calibration, Onboarding, Results, Order Hub,
and Accessibility design specification.

A material change to the four-destination hierarchy, hub/Fast Play relationship,
responsive/device parity, encounter HUD/cue priority, semantic controls,
telegraph/camera treatment, onboarding/gates, preparation atomicity, Results/
follow-up, profile scope/apply timing, accessibility neutrality, failure/save
behavior, or semantic design-system contract requires an explicit amendment
citing the superseded rule. Visual/numeric tuning within these boundaries creates
a new revision and cannot change an active encounter snapshot or confirmed
transaction.
