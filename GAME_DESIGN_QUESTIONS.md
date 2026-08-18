# Bands Battle Game Design Interview Plan

- **Status:** Interview in progress; 31 of 34 questions resolved
- **Created:** 2026-08-14
- **Parent:** [`GAME_VISION.md`](GAME_VISION.md)
- **Working record:** [`GAME_DESIGN_WORKING.md`](GAME_DESIGN_WORKING.md)
- **Final destination:** [`GAME_DESIGN.md`](GAME_DESIGN.md)

## Purpose

This is the finite reference plan for turning the settled game vision into an
implementable game design. It controls the order and scope of the owner
interview; answers and resulting decisions are recorded in
`GAME_DESIGN_WORKING.md` as the interview proceeds.

The current `GAME_DESIGN.md` predates the Vision v1 baseline. It remains useful
source material, but it is not authoritative where it conflicts with
`GAME_VISION.md`. The working record will eventually replace or comprehensively
revise it after the interview is complete and internally consistent.

## Interview protocol

- Ask exactly one numbered top-level question at a time.
- Use the indented prompts only to clarify the current question. They are not
  separate required questions.
- After each material answer, record the owner's words or a faithful summary,
  the design decision derived from it, consequences, and any explicitly
  deferred details in `GAME_DESIGN_WORKING.md`.
- At the start of each new question, update the progress count and mark the
  preceding question `[x]`, `[~]` if explicitly deferred, or `[>]` if resolved by
  another answer. Never silently skip a question.
- If one answer resolves later questions, record cross-references instead of
  asking for the same decision again.
- Follow-up questions do not receive new identifiers unless they materially
  expand the bounded design scope.
- Keep technical architecture, schemas, file layout, asset-production steps,
  and implementation tasks out of this interview unless a player-facing design
  choice depends on them. Route those matters to an OpenSpec change or a
  dedicated technical specification.
- `GAME_VISION.md` is the higher authority. A proposed design that changes a
  settled vision boundary must be identified as a vision revision rather than
  quietly recorded here.

## Completion rule

The interview is complete when every `GD-` question is answered, explicitly
deferred with a destination, or resolved by another recorded answer; the working
document contains no contradictory decisions; and the resulting design is
specific enough to define the first shippable product without inventing core
player behavior during implementation.

## Phase 1: Experience framework

- [x] **GD-01: What is the player's complete repeatable game loop, from entering
  the experience through choosing what to do after a boss attempt?**
  - Where does the player begin, and what choices can they make before battle?
  - How do solo, a preformed band, and public matchmaking enter the same flow?
  - What preparation is required, optional, or intentionally unavailable?
  - What happens after victory, defeat, rewards, and retry?
  - Decision: players enter the Order hub and approach a stylized encounter area
    where phasing shards poke through into existence. Each shard is labeled and
    represents a boss encounter. Activating one begins that boss's entry flow;
    the player then chooses solo, public matchmaking, or their current party and
    confirms difficulty and preparation before transport. The encounter runs to
    the song's final cadence, followed by results, rewards, and immediate choices
    to retry, change difficulty, return to the hub, improve equipment, continue
    the story, or remain with the band. The hub includes item shopping. Crafting
    belongs there when crafting is added. First-time onboarding may lead directly
    into the first mission, while returning players should have a fast route back
    into play. The hub should be meaningful without making traversal or repeated
    preparation compulsory busywork.

- [x] **GD-02: How is a normal one-song boss encounter divided into playable
  phases?**
  - What does the player do during opening, pressure, recovery, climax, and
    resolution passages?
  - Which phase changes are authored by the song, boss resistance, or both?
  - What information carries across phases?
  - Decision: a normal encounter uses five flexible phases shaped around the
    specific song. **Arrival** handles transport, boss reveal, an approachable
    opening phrase, and position readability. **First Clash** establishes the
    core rhythm, the simplest boss attack, and the first resistance layer.
    **Escalation** adds resistance layers, stronger patterns, meaningful
    movement, and possible Band Call or Crescendo windows. **Climax** uses the
    song's most intense section for the final resistance layer and highest
    pressure. **Finishing Cadence** presents a clearly previewed final phrase
    that determines destruction or survival. Quiet passages provide breathing
    room, recovery, repositioning, or story moments wherever the song naturally
    supports them rather than forming a mandatory fixed phase. The song controls
    phase timing; breaking resistance changes combat state but never skips,
    pauses, or rewinds the music. Early breaks bank Momentum and late breaks
    leave less time. Player survival state, special resources, prepared
    consumables, and accumulated Momentum persist across phases.

- [x] **GD-03: What does the player see and control on the main battle surface?**
  - What camera relationship keeps the avatar, instrument, boss, positions, and
    rhythm cues readable on a phone?
  - How much direct avatar control remains during battle?
  - Which information must always be visible, and which is contextual?
  - Decision: battle uses a director-assisted third-person camera behind and
    slightly above the player's avatar. The performer and instrument occupy the
    lower foreground, the boss commands the upper center, and tactical positions
    form a readable arc facing it. Players directly control rhythm, combat
    intent, and position selection rather than free locomotion or routine camera
    rotation. Selecting a destination causes a dash and smooth camera follow.
    Attack paths and safe areas appear in the arena with compact interface
    reinforcement. Short directed shots may emphasize arrival, resistance
    breaks, Crescendos, and victory, but never interrupt an active phrase or
    reaction window. Other band members remain visible when practical,
    especially during coordinated actions. Survival state, selected intent,
    personal special meter, boss resistance and phase, and the three rhythm pads
    remain visible. Phrase previews, position choices, attack geometry, Band
    Call invitations, Join In, consumables, and recovery cues appear only when
    relevant.

## Phase 2: Rhythm interaction

- [x] **GD-04: What is the playable phrase grammar?**
  - Which combinations of taps, holds, repeated beats, rests, simultaneous
    inputs, or other actions belong in the first release?
  - What makes a phrase musically recognizable without becoming a scrolling
    note highway?
  - What are the shortest and longest useful phrase shapes?
  - Decision: the first-release grammar contains single-pad taps, holds,
    repeated strikes on one pad, alternation among pads, and intentional rests.
    A hold judges its initial press and continued duration; precise release is
    not a separate judgment. A phrase normally covers one or two measures and
    contains roughly four to eight actions as one readable scoring group within
    the moving staff. Easier
    charts preserve the strongest musical accents while simplifying patterns;
    harder charts add subdivisions, syncopation, and denser alternation. Two- or
    three-pad chords, swipes, flicks, and dragging between pads are excluded from
    the first release.

- [x] **GD-05: How are upcoming phrases previewed and performed without pulling
  attention away from the battle?**
  - How early does a preview appear?
  - How does the player read order, timing, holds, rests, and phrase completion?
  - How do interface cues and in-world cues reinforce each other?
  - Decision: phrases organize judgments and combat meaning but do not become
    arbitrary permission windows. When a settled player has authored
    notes available for the selected instrument, those notes should generally be
    playable. One- or two-measure phrases may chain into sustained performance
    passages. Breaks should follow musical rests, player movement, boss
    knockback, transitions, recovery, or another meaningful encounter event
    rather than occurring automatically after every phrase. During a performance
    passage, a compact staff moves notes right to left toward a fixed strike
    line. The staff recedes when no playable notes or a meaningful encounter
    event creates a break. Desktop retains labeled Z/X/C targets. Mobile uses
    the same timing model with three generous fixed touch pads. Lanes and pads
    use label and shape as well as color. As an initial tuning hypothesis,
    roughly 65 to 80 percent of a normal encounter should offer meaningful
    rhythmic participation, adjusted per song, instrument, and playtest results;
    inactivity must still contain something meaningful to observe, decide, or
    do. This decision explicitly amends the static-cue language in Vision v1.

- [x] **GD-06: What are the exact player input actions on touch, keyboard and
  mouse, and gamepad?**
  - How are the three fixed rhythm pads operated?
  - How are Attack, Defend, Special, movement, Join In, Band Calls, and
    consumables invoked without crowding a phone screen?
  - Which actions can be remapped?
  - Settled constraint: `W`, `A`, `S`, and `D` are reserved exclusively for
    movement. They must not be assigned to Attack, Defend, Special, or another
    non-movement action. WASD provides ordinary free movement in the Order hub.
    During boss encounters, `W` advances toward the boss, `S` retreats, and `A`
    or `D` dashes among neighboring authored tactical positions. Boss combat
    does not include continuous free locomotion.
  - Decision: keyboard uses `Z`/`X`/`C` for rhythm, `1`/`2`/`3` for
    Attack/Defend/Special, `Space` for Join In or accepting a cooperative
    invitation, `4` for the equipped Band Call, and `5`/`6` for prepared
    consumables. Touch uses three large fixed rhythm pads, a normal hub joystick,
    direct encounter-position markers, three persistent intent buttons, one
    contextual cooperative-action button, one equipped Band Call button, and
    two prepared-consumable buttons. Gamepad uses the left, bottom, and right
    face buttons for rhythm; the left stick for movement; D-pad left/up/right for
    Attack/Defend/Special; the top face button for the contextual cooperative
    action; right trigger for the Band Call; and the bumpers for consumables.
    Device-appropriate labels appear automatically, and keyboard/gamepad
    bindings are remappable. Touch permits handedness adjustments for secondary
    controls. Frequently timed rhythm inputs remain stable and separate from
    occasional queued actions.

- [x] **GD-07: How are timing accuracy, latency, and performance feedback
  judged?**
  - Which judgment grades exist?
  - Are early and late inputs distinguished?
  - How do calibration and device latency affect the judgment clock?
  - What feedback teaches improvement without obscuring combat?
  - Decision: note timing uses Perfect, Great, Good, and Miss. Initial
    Normal-difficulty test windows are ±60 ms, ±110 ms, and ±170 ms, with values
    subject to playtesting. Notes resolve immediately at the strike line; the
    relevant pad and instrument react, and a compact grade appears near the
    staff. Great and Good add a restrained early/late direction; Perfect does
    not need one. Miss feedback is clear without shaming or covering the boss.
    Per-phrase summaries remain small, while detailed timing trends belong on
    results. Touch/gamepad may use restrained haptics, and individual notes do
    not cause major camera shake. A hold's initial press receives a timing grade;
    maintained duration earns contribution, and early release stops future
    contribution without creating a separate release judgment. Onboarding
    offers a skippable calibration that aligns an audio/visual pulse, samples
    roughly 12–16 taps, rejects obvious outliers, suggests an offset, and allows
    a test plus manual adjustment. Calibration is saved per device/control
    profile, remains easy to reopen, may be recommended after a consistent
    early/late trend or audio-device change, and never changes difficulty or
    rewards.

- [x] **GD-08: How do difficulty levels transform the same authored song and
  encounter?**
  - What changes in phrase density, timing, boss pressure, coordination, and
    recovery?
  - What must remain identical across difficulties?
  - How does the game recommend or unlock a difficulty?
  - Decision: the first release uses Easy, Normal, and Hard. Easy emphasizes the
    strongest musical accents, uses starting timing windows of ±90/150/230 ms,
    lengthens telegraphs, simplifies attack combinations, lowers resistance and
    incoming damage, strengthens recovery, makes the positive Cohesion Bonus
    easier to earn, and may
    provide one additional recovery Crescendo. Normal uses the intended chart,
    the ±60/110/170 ms baseline, and intended encounter balance. Hard approaches
    the full authored detail, uses starting windows of ±45/85/135 ms, combines
    more dangerous but fair attacks, raises resistance and incoming pressure,
    limits recovery, and uses stricter—but still fair—Cohesion Bonus thresholds.
    Every
    difficulty normalizes maximum combat contribution per musical passage:
    reducing the number of charted inputs never reduces the damage, defense,
    healing, or utility available from a fully performed passage. Difficulty
    does not change song speed or length, story, boss identity, arena, controls,
    phases, or accessibility rights. All levels advance the campaign and use the
    same boss-themed loot pool. Higher levels may improve reward quantity or
    roll quality and grant mastery cosmetics, but essential power is never
    exclusive to Hard. Easy and Normal begin unlocked; Hard unlocks per boss
    after a Normal victory. Recommendations may respond privately to player
    performance but never change difficulty automatically. A fourth Master tier
    is deferred until player evidence demonstrates a need.

- [x] **GD-09: How does player performance change the audible song and combat
  soundscape?**
  - What happens to the selected instrument on hits, misses, movement, downing,
    and recovery?
  - How are judgment, boss telegraphs, group invitations, and musical clarity
    mixed on small speakers?
  - Decision: every player hears a stable complete backing mix. The selected
    instrument receives local performance emphasis: Perfect adds the clearest
    attack and brief lift, Great a confident normal accent, Good a softer accent,
    and Miss a brief duck or filtered stumble without complete silence.
    Movement returns the instrument to backing level without a failure sound;
    downing makes it muffled and distant; recovery restores it on-beat. These
    changes are primarily local, so one player's misses do not spoil other
    players' music. Teammates hear meaningful combat effects and group
    contributions, while crowd, arena, and combat layers respond to aggregate
    band performance. Band Calls and Crescendos widen and strengthen the shared
    ensemble. Solo retains the full mix without pretending that acolytes perform
    notes. Mix priority is critical boss telegraphs and timing cues, then the
    local instrument and judgments, then the core song, then other combat,
    crowd, and ambience. Critical cues use distinct rhythm, pitch range, and
    sound shape rather than volume alone; nonessential effects may duck while
    the musical pulse remains audible. Phone-critical cues use strong midrange
    transients, and repeated hits reinforce the instrument rather than layering
    a noisy unrelated effect on every note.

- [x] **GD-10: What authoring and review workflow turns a song into playable
  phrases and encounter timing?**
  - What may analysis or AI suggest automatically?
  - Which decisions require human authorship and approval?
  - What preview, edit, validation, and difficulty-generation capabilities are
    necessary for repeatable content production?
  - Decision: content moves through seven stages: ingest the approved master,
    stems, rights/provenance, lyrics, duration, and arrangement context; automate
    suggestions for tempo, beats, onsets, holds, rests, dropouts, energy, and
    phrase boundaries; human-edit the beat grid, playable instrument events,
    three-input mapping, holds, rests, phrases, and chained passages; derive
    Normal and Easy suggestions from the detailed source chart with normalized
    passage output and human review; author the five encounter functions,
    resistance windows, boss events, movement/recovery moments, group windows,
    and finishing performance; automatically validate musical-clock alignment,
    chart authenticity, passage availability, event fairness, difficulty
    normalization, activity density, and a valid ending; then review every
    instrument and difficulty in Roblox across solo, representative co-op, phone,
    desktop, and gamepad. Release requires explicit musical, design, and
    technical approval. The eventual authoring surface provides waveform and
    stem views, beat grid, three lanes, difficulty and event tracks, loop,
    scrub, drag editing, validation, and direct test export. Automation and AI
    may propose but never approve or publish. It also generates an Activity Map
    for every instrument and difficulty plus ensemble eligibility data. The map
    describes playable density around each beat or measure boundary, rests,
    crossing holds, instrument entries, exits and solos, musical energy,
    conflicting boss or phase events, finisher proximity, available reaction
    time, and whether the following passage sustains activity. Event-specific
    validators must prove that required solo-recovery, revive, boss-attack, Band
    Call, and Crescendo windows occur within their allowed delays for supported
    rosters; missing required windows fail the chart. The first three bosses may
    use a lightweight internal timeline tool rather than a polished creator
    product.

## Phase 3: Combat rules

- [x] **GD-11: What player survival resources exist, and how are damage and
  protection represented?**
  - Is the primary resource health, ward, both, or something else?
  - What causes loss or restoration?
  - Which states need clear breakpoints or warnings?
  - Decision: Ward is the first release's single player-survival resource; there
    is no separate health bar. Direct boss impacts, hazards, and failed defensive
    responses reduce Ward. An ordinary missed instrument input does not directly
    damage it. There is no separate defense chart or defense passage: during the
    normal available instrument material, selecting Defend routes successful
    performance into mitigation or Ward reinforcement for the telegraphed threat.
    Dangerous positions increase outgoing damage and potential rewards while
    also increasing exposure or incoming damage. Defend performance, support,
    authored recovery moments, and prepared consumables can restore or reinforce
    Ward; it does not automatically refill completely between phases. Stable,
    below-half, critical-below-quarter, and empty states use meter, Ward geometry,
    cracks, animation, and sound rather than color alone. Empty Ward shatters and
    downs the player. Damage is tuned so downing follows accumulated failure or a
    clearly telegraphed major attack rather than one ordinary mistake.

- [x] **GD-12: How do Attack, Defend, and Special intent work during a phrase?**
  - What does each intent convert accurate inputs into?
  - What is the default intent?
  - How does queuing or switching at a musical boundary work?
  - What prevents one intent from dominating every situation?
  - Decision: all three intents route the same available instrument performance;
    they never substitute a separate chart. Attack is the default and converts
    contribution into resistance damage, with valid excess becoming Momentum.
    Defend converts contribution into temporary mitigation for the next
    telegraphed impact and modest Ward reinforcement. Special redirects
    contribution into the equipped personal ability. When it is not ready,
    successful inputs accelerate its charge; filling it returns the player to
    the previous Attack or Defend intent without firing. Selecting a ready
    Special arms it for resolution after the next scoring group at a clean
    musical boundary. Pressing an intent immediately confirms a queued state and
    applies it at the next playable beat or note. Earlier inputs
    retain their original intent; switching never reinterprets notes or creates
    misses. While moving, a change applies to the next playable material. An
    unavailable Special explains why and does not change intent. Boss telegraphs
    allow a full-phrase Defend choice, and baseline success never requires
    mid-phrase switching. Attack advances victory, Defend preserves survival,
    and Special trades immediate output for a build-specific effect.

- [x] **GD-13: How do boss resistance layers, banked Momentum, and the final
  finishing performance resolve?**
  - How many layers are useful for a normal encounter?
  - What happens when a layer breaks early or late?
  - How is progress communicated without using a conventional health fantasy?
  - What exactly determines victory at the final cadence?
  - Decision: **Resolve** is the working name for a normal boss's three visible
    sequential resistance layers, aligned with First Clash, Escalation, and
    Climax. The count stays consistent across the three difficulties while
    normalized requirements change. Only the current layer takes damage; later
    layers have authored opening points. An early break converts further Attack
    contribution into visible Momentum. When the next layer opens, Momentum
    applies immediately, with an initial cap equivalent to roughly 20 percent so
    it cannot skip the layer. A late break exposes the next layer immediately if
    its opening has passed, leaving less song time. After the final early break,
    excess Momentum improves result tier, reward potential, and finishing
    spectacle but cannot replace the ending performance. Victory requires all
    three layers broken plus success on the clearly previewed, difficulty-scaled
    finishing phrase. Failure of either condition means defeat; randomness never
    reverses it. All three meter segments are visible from the start, only the
    active one is illuminated, and every break visibly shatters part of the
    boss's guard, armor, halo, or supernatural structure.

- [x] **GD-14: What rules govern boss attacks and their relationship to the
  musical clock?**
  - What attack families should the core system support?
  - How are preparation, targeting, reaction windows, impact, and recovery
    authored?
  - How do attacks interact with phrases without creating unfair conflicts?
  - Decision: every boss attack progresses through Telegraph, Commit, Impact,
    and Recovery. Telegraph combines pose, sound, arena geometry, and compact
    warning; Commit locks targets and unsafe areas; Impact resolves on a
    musically significant beat; Recovery creates a brief earned opportunity.
    The core families are lateral sweep, range attack, targeted strike,
    arena-wide pulse, persistent hazard, and rare major band attack, with each
    boss using a focused subset. Warnings begin on readable boundaries: Easy
    generally gives about two measures, Normal one to two, and Hard at least one
    clear measure for major attacks. Required information uses multiple channels
    rather than color alone. Movement-required impacts do not arrive
    unexpectedly during committed performance, targets never change after
    commitment, and runtime variation chooses only pre-authored candidates that
    fit the song. Major attacks cannot form impossible overlaps. Co-op targeting
    is explicit; player count may change target count but not cue reliability.
    Difficulty may shorten or combine learned patterns without hiding necessary
    information. Failure may damage Ward, knock players back, displace them, or
    create hazards, but never scrambles controls or fabricates rhythm misses.
    Successfully answering a major attack creates a brief advantage. Multi-part
    attacks that test dash budgeting announce the sequence before its first
    commitment. Every impact is validated against known movement recovery and
    preserves a readable response through a legal route, Defend with playable
    material, cover, a prepared ability, or knowingly accepted Ward damage;
    surprise retargeting cannot manufacture an impossible cooldown state.

- [x] **GD-15: How do arena positions and movement create risk and reward?**
  - How many positions should a typical arena offer?
  - What differs between safe, neutral, and dangerous locations?
  - How long does movement take, what can interrupt it, and how is destination
    selection controlled?
  - How do cover and attack geometry remain readable?
  - Settled constraint from GD-11: dangerous positions increase outgoing combat
    damage and potential encounter rewards as well as exposure or incoming
    damage. The reward bonus must come from successful performance at risk, not
    merely standing in the dangerous location. Exact values and banking rules
    remain open for GD-15.
  - Decision: nine locations arranged as three lateral lanes across Near,
    Middle, and Rear risk tiers are the baseline, not a universal grid. An arena
    may use more, fewer, or an irregular graph when its boss mechanics justify
    it. Directional neighbors preserve `W`/`S` advance and retreat plus `A`/`D`
    lateral dashes; touch may select a farther marker but traverses the same graph
    and travel time. A baseline dash is roughly 0.75 seconds followed by the
    established settling period, grants no invulnerability, and leaves a player
    exposed while traveling unless the route avoids the attack. Tactical
    locations are shared zones with formation offsets, not exclusive slots.
    Humans cannot body-block one another, acolytes arrange around players without
    consuming capacity, and all occupants share the location's danger. Starting
    Near values are +25% Attack, +30% incoming danger, and +25% reward potential;
    Middle is baseline; Rear is −20% Attack, −25% incoming danger, and no risk
    bonus. These are tuning hypotheses. Risk Bonus comes only from successful
    exposed performance, banks when a phrase completes, and loses only the
    current unbanked amount on movement or downing. Cover protects against
    specified attack shapes rather than everything. Bosses may alter locations,
    but encounter validation must preserve readable, valid response paths.
  - Amendment approved, 2026-08-17: each voluntary dash consumes one visible
    movement charge. It refreshes two beats after landing, clamped initially to
    roughly 0.75–1.25 seconds. This recovery is separate from the shorter rhythm-
    settling time, so the player may perform and Defend before another dash is
    available. Beat pips on the current position and destination markers show
    recovery, a restrained cue confirms readiness, and unavailable dashes never
    auto-queue. Spending movement may knowingly leave Defend, cover, an ability,
    or Ward absorption as the remaining response. Involuntary displacement
    neither consumes a ready dash nor restarts recovery. Gear and difficulty do
    not modify this rule, and multi-edge travel honors recovery at each edge.

- [x] **GD-16: What happens when a player is hurt, disabled, downed, revived, or
  given a solo last chance?**
  - Which penalties accumulate before downing?
  - How does a co-op revive phrase divert effort?
  - What makes the solo recovery challenge brief, difficult, and fair?
  - What state does the player return with after recovery?
  - Decision: empty Ward downs the player and pauses their chart.
    Ordinary targeting ignores a downed player; all humans down means co-op
    defeat. Bandmates may route normal performance into a roughly two-measure
    revive, one player can complete it, and more participants accelerate or
    strengthen it. A revived player returns near 35% Ward, potentially up to 60%
    with added participation, plus about two beats of re-entry protection. Solo
    receives one difficulty-scaled, one- or two-measure emergency challenge per
    encounter using the familiar controls; success returns about 35% Ward and
    failure or a second down ends the attempt. No paid bypass is permitted.
    The authoring pipeline generates Activity Maps for every instrument and
    difficulty plus the ensemble, recording density before and after candidate
    boundaries, longest rests, crossing holds, entries, exits, solos, energy,
    phase and event conflicts, finisher proximity, reaction time, and following
    passage activity. Each event type applies its own eligibility rules at
    runtime for the current instrument, difficulty, roster, phase, and active
    events, then selects the earliest eligible boundary within its allowed wait.
    Urgent recovery with no suitable instrument passage uses a clearly labeled
    universal beat challenge derived from the song's BPM and familiar controls;
    it does not pretend a silent instrument is playing. Nonurgent events wait or
    skip. Songs never pause, and authoring validation fails when a required event
    has no candidate inside its maximum delay for a supported configuration.

- [x] **GD-17: What personal specials, resources, and equipped ability choices
  belong in the core combat loop?**
  - How is the resource earned and spent?
  - How many choices are equipped before battle?
  - How are effects timed to clean musical boundaries?
  - How do abilities express different builds without replacing rhythm skill?
  - Decision: each player equips one personal **Signature Special** before
    entering a shard, separate from their equipped Band Call. A one-charge
    **Hype** meter fills slowly through successful normal performance. Selecting
    Special before it is ready redirects contribution away from Attack or
    Defend to charge it much faster. Reaching full Hype never fires
    automatically; the player returns to their previous intent and must select
    Special again to arm it. The next scoring group becomes its activation
    performance, and the effect resolves at a valid musical boundary. The base
    effect always occurs, while execution quality scales strength or duration,
    so one mistake cannot waste an entire charge. Signature Specials may focus
    on offense, Ward, support, or positional utility, and any instrument may
    equip them. Hype survives downing and revival, resets between encounters,
    has no separate cooldown, and cannot hold a second charge.

- [x] **GD-18: How do player-initiated Band Calls work from activation through
  resolution?**
  - Who may initiate, when, and at what cost?
  - How does the invitation appear and how does another player join or decline?
  - How are participant timing and additive contributions resolved?
  - Decision: each player equips one Band Call and earns one initiation per
    encounter through meaningful successful performance. Any active player with
    a ready Call may initiate, subject to a shared lockout initially targeted at
    roughly eight measures. The Activity Map schedules the earliest valid
    ensemble window; if none occurs within the allowed delay, the request
    cancels without spending the charge. The invitation identifies the player,
    ability, effect, and musical countdown, and bandmates accept with Join In.
    Acceptance is provisional until the boundary, so normal play continues and
    moving, withdrawing, or becoming downed simply removes that participant.
    The initiator's charge is spent only when the performance begins. For about
    one or two measures, valid participants perform their own ordinary chart
    material and temporarily route its contribution into the Call. The initiator
    guarantees the base effect; each participant adds an accuracy-scaled share,
    and weak play never cancels stronger contributions. In solo, active acolytes
    add a small, predictable fixed bonus and join the presentation without
    receiving invented rhythm scores.

- [x] **GD-19: How do song-authored Crescendo opportunities work from preview
  through resolution?**
  - How many candidate and guaranteed windows are appropriate?
  - What qualifies a song section as a valid window?
  - How does difficulty change recovery opportunities and collective stakes?
  - Decision: a standard encounter authors two to four valid Crescendo
    candidates and guarantees exactly one activation on every difficulty. Easy
    may activate one additional unused candidate when the band falls
    substantially behind; Normal and Hard retain the authored budget. A valid
    candidate provides current-roster instrument coverage, roughly two measures
    of reaction time, sustained activity afterward, and separation from major
    attacks, recovery, silence, and the finishing cadence. The Activity Map
    chooses among candidates. If one becomes invalid, it selects a later
    candidate; Crescendos never use a universal beat fallback. Players receive a
    prominent two-measure preview and may opt in with Join In at no resource
    cost. Participants then perform roughly two measures of their own chart,
    temporarily routing normal intent contribution into the Crescendo.
    Instrument- and difficulty-normalized individual results combine additively
    into **Echo**, **Crescendo**, or **Full Crescendo** tiers; one weak player
    never reduces another's contribution. The encounter previews and authors
    the resulting effect, normally a major Resolve burst with some Ward
    reinforcement. Solo acolytes provide their established predictable fixed
    contribution without fabricated performance scores.

## Phase 4: Solo and cooperative play

- [x] **GD-20: What exact support do Order acolytes provide in solo encounters?**
  - Which passive pressure and authored abilities are predictable?
  - How do acolytes share locations through formation offsets and react when a
    human joins their location?
  - How can boss attacks temporarily suppress them?
  - How is their group-ability contribution represented without fake rhythm
    scores?
  - Decision: solo uses a fixed squad of three readable support acolytes.
    **Vanguard** adds small Resolve pressure after each successfully completed
    player scoring group but can never break a layer. **Warden** supplies modest
    Ward reinforcement on a visible authored cadence, initially about every
    eight measures. **Herald** slightly improves Band Call readiness and provides
    the squad's small, capped, fixed contribution during Band Calls and
    Crescendos. Acolytes never play charts, receive judgments, create combos, or
    earn performance credit. They occupy authored locations, reposition
    automatically at musical boundaries, and share locations through formation
    offsets without blocking, swapping, consuming capacity, or granting their
    position's risk/reward modifiers to support. Clearly telegraphed attacks can
    suppress an affected acolyte for roughly four measures; a portrait and
    countdown identify the lost function. Acolytes recover automatically and
    cannot be permanently downed, revived, equipped, or commanded. Suppression
    never creates an escort objective. During solo recovery, they add
    presentation but no score or mechanical assistance.

- [x] **GD-21: How do encounters scale from one human to six, including duplicate
  instruments and uneven skill?**
  - What scales in boss pressure, resistance, targeting, rewards, and group
    thresholds?
  - How are individual contributions and the positive Cohesion Bonus combined?
  - What prevents either solo or a full band from becoming the obviously easier
    optimal path?
  - Decision: each additional human adds about 75% of the solo Resolve
    requirement, producing initial one-through-six-human targets of 1.0, 1.75,
    2.5, 3.25, 4.0, and 4.75 player equivalents. The musical timeline, chart
    density, timing windows, and individual incoming damage do not scale with
    roster size. Additional boss pressure comes from broader target coverage,
    initially capped at about half the roster and never more than three
    simultaneous individual targets, with fair distribution among equivalent
    targets. Ward remains individual; greater Resolve and coverage offset the
    revival advantage of larger groups. Duplicate instruments are unrestricted
    and independently charted, scored, and normalized by instrument, available
    material, and difficulty. Weak or absent play supplies less positive output
    but never subtracts another player's contribution. Broad successful
    participation can instead earn a difficulty-adjusted **Cohesion Bonus**
    capped initially around 15%. Group-event tiers scale against the eligible
    roster, so one expert earns meaningful value but cannot represent an entire
    inactive six-player band. Victory drops are individual rather than split,
    party size does not alter the core reward pool, and solo versus co-op is
    tuned toward comparable success at equal skill.

- [x] **GD-22: What is the complete cooperative session flow?**
  - How are private bands formed and public matches found?
  - When are boss, difficulty, instrument, build, and readiness locked?
  - What happens when instrument choices duplicate?
  - How does the group return, retry, or disband after results?
  - Decision: activating a shard opens its boss card, shared encounter
    difficulty, reward preview, and Solo, Current Party, or Public Band choice.
    Public matchmaking is server-owned, matches the chosen boss and difficulty,
    targets three to six humans, and supports two-player launches. After roughly
    45 seconds with only two matches, both may explicitly launch, keep waiting,
    or leave for solo. Current parties support two to six; the leader proposes
    the shard but every member confirms. Staging permits changes to instrument,
    equipment, Signature Special, Band Call, and consumables, and explicitly
    treats duplicate instruments as valid. Boss and difficulty lock when
    matching begins; loadouts and roster lock at the final three-second
    deployment countdown. Public staging uses a short ready timer, never drags
    unready players into combat, and may replace them. Encounters have no
    ordinary join-in-progress. Results grant individual rewards immediately and
    show personal performance, positional-risk earnings, mastery, and the band
    result. Retry Same Shard, Stay with Band, and Return to Hub are individual
    choices rather than a binding majority vote. Public players choosing Retry
    or Stay form a rematch group whose open places may refill; current parties
    persist unless members leave. Retry returns to staging for loadout changes.

- [x] **GD-23: How does the design handle communication and multiplayer failure
  cases?**
  - Which pings, preset messages, and battle cues are necessary?
  - What happens on disconnect, rejoin, AFK behavior, deliberate nonparticipation,
    or host departure?
  - Which protections are needed against griefing without punishing ordinary
    mistakes?
  - Decision: core coordination never requires chat. Automatic cues cover
    attacks, downing, revival, group events, and repositioning; rate-limited
    pings whose senders may be muted cover Move, Defend, Join Call, Revive, and
    Ready/Thanks. A disconnect preserves the last server-confirmed Ward,
    position, Hype, Band Call charge, consumables, and downed state; already
    committed attacks still resolve. The performer then becomes untargetable,
    supplies no contribution, and is excluded from group thresholds. Absent
    material grants neither misses nor contribution and cannot improve the
    performance rating. A roughly 45-second rejoin grace returns the player at a
    safe musical boundary, at the prior valid or nearest Middle location, with
    normal settling protection unless still downed. After grace, the current
    Resolve layer stays unchanged, unopened layers rescale at the next layer
    boundary, and prior meaningful contribution retains appropriate reward
    eligibility. AFK detection counts ignored playable material rather than
    silence, warns privately after roughly two scoring groups, and permits one
    safe-boundary resume before repeated inactivity removes participation-based
    reward eligibility. There is no combat vote-kick. Server rules prevent
    friendly fire, blocking, shared-resource theft, forced votes, and negative
    contribution. Public sessions have no host; current-party leadership
    transfers on departure. Normal Roblox block/report remains available, and
    low accuracy alone never triggers punishment.

## Phase 5: Builds, rewards, and progression

- [x] **GD-24: What is the player's pre-battle loadout, and which stats may alter
  combat?**
  - Which slots exist for instrument, equipment, abilities, and consumables?
  - Which stats can change damage, ward, support, resources, or recovery?
  - Which changes must remain cosmetic or accessibility-only?
  - Decision: the readable first-release loadout has three power-bearing gear
    slots: **Instrument**, providing musical identity plus a modest combat
    emphasis and one trait; **Ward Core**, affecting Ward, Defend, reinforcement,
    or bounded recovery; and **Resonator**, affecting Attack conversion, Hype,
    Signature potency, support, or Band Call readiness. Every instrument category
    must offer variants for multiple roles. Action slots hold one Signature
    Special, one Band Call, and two prepared consumables with limited encounter
    charges. Stagewear, instrument finishes, auras, effects, and titles use
    separate cosmetic slots. Gear may scale the combat consequence of successful
    performance: Resolve damage, Ward and defense, restoration and support,
    Hype and Call readiness, and bounded ability or consumable potency. It may
    not change judgment windows, calibration, song speed, charts, grades, boss
    telegraphs, movement timing or invulnerability, recovery counts, automatic
    note correction, mechanic immunity, positional ratios, or reward eligibility.
    Scoring resolves before gear modifies its combat result. First-release items
    expose one primary stat and one readable trait. The three gear slots are a
    simple starting surface, not a permanent complexity ceiling: future
    techniques, traits, sidegrades, sockets, or advanced configuration may
    deepen builds while preserving the same combat controls and fairness rules.

- [x] **GD-25: How do role specialization and long-term character development
  create distinct builds for any instrument?**
  - What skill-tree or equivalent structure supports offense, defense, healing,
    utility, and hybrids?
  - How are choices unlocked, combined, and respecced?
  - Which differences should change play style rather than only increase numbers?
  - Decision: every instrument uses the same four universal functional option
    categories: offense and Momentum, Ward and protection, group support, and
    Hype and utility. The working system terms are **Disciplines**, **Build
    Core**, and **Techniques**, but none of these or the individual option names
    are approved player-facing language. A build equips one major behavior-
    changing rule and three smaller supporting rules, freely mixed across the
    four categories. These modify results of existing choices such as intent,
    position, movement, Specials, and group actions without changing charts,
    judgments, or controls. New players use clear role presets; an advanced
    editor later exposes individual combinations. General progression and boss
    mastery unlock options, no instrument owns a category, respeccing is free
    outside combat, and players initially receive three saved presets. Gear
    carries most raw power while these choices emphasize play-style differences
    and hybrids. Synergy is capped to prevent a mandatory multiplicative build.
    Long-term depth comes from adding more major rules, supporting rules, traits,
    and interactions rather than more active buttons. A dedicated naming and
    tone pass is required before these systems or choices receive final names.

- [x] **GD-26: What is the reward, item, material, upgrade, and crafting economy?**
  - What does victory award that failure does not?
  - How do boss identity, tier, difficulty, and performance affect rewards?
  - What are the primary currencies and resource sinks?
  - How do deterministic crafting and random drops complement each other?
  - Decision: the economy uses only one general earned resource plus
    boss-specific materials; all names remain placeholders pending the required
    naming pass. Victory guarantees both resources and mastery progress, may drop
    a complete fixed-stat boss item or cosmetic, and grants story progression
    and a Shattered Song fragment on the first clear. Failure grants modest
    mastery and general resources for meaningful participation but no signature
    boss material. Every difficulty shares the boss's item pool; higher
    difficulty may improve quantities, drop chances, or starting upgrade rank.
    Performance and banked positional risk add bounded bonuses. Every victory
    advances a visible deterministic path, initially targeting roughly four to
    six clears to craft one chosen standard boss item, while random complete-item
    drops provide surprise. Items have fixed stats and traits rather than random
    affix ranges. Duplicates salvage into useful boss material while preserving
    unlocked appearances. Items initially have about three guaranteed upgrade
    ranks using both resource types. Crafting and upgrades show exact results and
    never fail, destroy items, lower stats, or require timers. Consumables are a
    repeatable general-resource sink. There are no repair or death taxes, paid
    luck, respec fees, energy limits, daily earning caps, or first-release player
    trading.

- [x] **GD-27: How are mastery, power progression, campaign access, and replay
  connected?**
  - What unlocks bosses, difficulties, abilities, and regions?
  - What permanent progress comes from practicing a failed encounter?
  - How do old bosses remain useful without out-rewarding current-tier bosses?
  - Decision: progression has three distinct tracks. A first victory on any
    difficulty restores the boss's Shattered Song fragment and unlocks the next
    campaign destination; meaningful cooperative participation earns the clear
    even if the player ends downed. Easy and Normal begin available, and Normal
    victory unlocks Hard for that boss. General player progression comes from
    meaningful victory or failure and unlocks systems, saved builds,
    specialization options, and broader equipment choices while supplying only
    limited raw power. Each boss has about ten initial mastery ranks shared
    across instruments, while personal bests remain separate by instrument and
    difficulty. Victory and failure grant participation-scaled mastery, with
    victory substantially more efficient. Mastery awards lore, cosmetics,
    recipes, specialization options, titles, and deterministic milestones;
    signature items, boss materials, campaign fragments, and first-clear access
    still require victory. Current-tier bosses remain the best power source. Old
    bosses retain unique traits, cosmetics, mastery, recipes, and materials; an
    old item may reach the current tier only through a recipe requiring mostly
    current-tier resources plus its original material. Recommended power is not
    a mandatory gear gate. Failure preserves personal bests, mastery, and modest
    general resources. No daily streak, energy, expiring progression, or
    exclusive rotating reward is required.

- [x] **GD-28: How is the vision's monetization policy expressed in actual game
  surfaces and item balance?**
  - Where may players browse or buy cosmetics and permanent equipment?
  - How is an earnable equivalent communicated and obtained?
  - What comparisons or safeguards prove paid equipment stays within the normal
    tier ceiling?
  - Decision: purchases appear only in a clearly identified hub shop or a
    voluntarily opened menu, unlocked after onboarding and at least one completed
    encounter. No prompt appears during battle, downing, recovery, defeat,
    results, or immediate retry, and closing or declining stops repeated prompts.
    Every stat-bearing paid item has an exact functional equivalent earnable at
    the same campaign tier; only appearance may be exclusive. Its page shows an
    Earn Through Play route, exact Robux price, tier, stats, trait, appearance,
    upgrade state, and equivalent before confirmation. Purchase grants the item
    at the player's currently unlocked tier, never auto-advances future tiers,
    and uses normal earned upgrade materials thereafter. Duplicate purchases,
    fake discounts, false scarcity, resetting countdowns, and ambiguous rarity
    are prohibited. Launch sells only deterministic cosmetics and permanent
    equipment: no random products, rescue purchases, consumables, temporary
    boosts, resources, materials, drop modifiers, content access, subscriptions,
    convenience, or progression skips. Every paid-equipment record identifies
    its earnable equivalent; automated validation rejects excess stat budgets or
    paid-only traits, and design and economy reviewers compare both versions
    through the same balance tests before publication. Future monetization
    categories require separate explicit approval.

## Phase 6: Product flow and launch content

- [x] **GD-29: What does onboarding teach, in what order, before a player is
  trusted with a full boss encounter?**
  - How are rhythm pads, intent, telegraphs, movement, positions, survival, and
    rewards introduced?
  - What can be learned inside the first real boss instead of a separate lesson?
  - What may an experienced player skip?
  - Decision: onboarding begins with a four-to-six-minute, checkpointed,
    replayable Order practice. Setup chooses an instrument, shows device-specific
    controls and comfort settings, and offers skippable calibration. Two short
    musical phrases teach the moving staff, strike line, three inputs, taps,
    holds, repeats, rests, and judgments. Safe modules then demonstrate Attack
    damaging Resolve, Defend reinforcing Ward against a harmless telegraph,
    directional position dashes and risk tiers, and one combined sequence. Each
    module presents one low-text instruction, repeats without death or a minimum
    grade, and never requires Perfect. Completing or explicitly skipping
    practice unlocks public matchmaking. The first boss contextually reinforces
    Attack, movement, Defend, and Ward; later phases introduce Hype and the
    Signature Special, a guaranteed Crescendo teaches Join In, and the first
    relevant down teaches recovery or revival. Band Calls and consumables prompt
    only when first available. Contextual teaching never pauses or rewinds the
    song. Experienced players may skip after confirming controls, while
    calibration, references, replayable practice, and prompt controls remain
    accessible. The store stays locked until onboarding and one encounter are
    complete.

- [x] **GD-30: What functions does the Order hub provide, and how does the player
  move among them?**
  - Where are mission choice, matchmaking, practice, loadouts, upgrades, story,
    social activity, and the store located conceptually?
  - Which functions need physical spaces and which can be menus?
  - What changes in the hub as the campaign advances?
  - Decision: the phasing-shard field is the hub's dominant physical landmark
    and uses a visually tiered, stair-stepped or terraced ascent. Higher campaign
    levels remain visible but physically blocked until reached. Shards are
    glowing broken-glass forms piercing through swirling portals, generally but
    not universally oriented toward the hub center; they may rise from floors,
    descend from above, lean through walls, or occupy suspended fractures. Their
    boss and arena identities use distinct color, portal behavior, particles,
    silhouette, sound, and labels, creating controlled “beautiful chaos” without
    sacrificing activation readability. Every shard retains a stable labeled
    interaction point and locked states use more than color. Practice, workshop,
    story and mastery, social gathering, and the voluntary store use distinct
    physical anchors around the shard structure and open focused phone-friendly
    menus. Loadout, inventory, party, mastery, settings, and queue status remain
    available from a compact menu anywhere. Returning players have fast access
    to unlocked shard tiers, and result-screen Retry bypasses hub traversal.
    Optional dialogue, social spaces, and non-scored musical play never become
    errands. Campaign progress changes restored fragments, available levels,
    portal activity, music, light, NPCs, and architecture while core landmarks
    and paths stay familiar. Functional area labels remain placeholders.

- [x] **GD-31: What are the concrete content specifications for the first three
  bosses and songs?**
  - What does each encounter teach, test, and add?
  - Which instruments, positions, attack families, rewards, region, and story
    beat does each require?
  - How do the three encounters demonstrate both repeatability and progression?
  - Owner direction, 2026-08-17: do not assume Heaven's Edge or Blackened Crown
    will be launch songs. Their current versions lack sufficient intensity and
    dynamic movement for the intended encounters. New song candidates can be
    generated readily, so encounter requirements should drive song generation
    and selection rather than forcing existing assets into the launch lineup.
    The existing songs may remain processing fixtures or be reconsidered only
    after substantial musical revision. GD-31 remains open.
  - Approved production process, 2026-08-17: define the three encounter and
    musical briefs before selecting songs; generate at least two or three new
    full-stem candidates per brief; score them for intensity, dynamics,
    five-function structure, instrument coverage, solos and rests, Crescendo and
    recovery windows, and finishing cadence; then design the final boss, arena,
    chart, and events around the selected song's actual structure. GD-31 remains
    open until all three briefs are approved.
  - First encounter brief approved, 2026-08-17: generate a roughly 3¼-to-4¼-
    minute full-stem song with immediate dark cinematic intensity, strong dynamic
    contrast across all five encounter functions, active solos and rests, two to
    four Crescendo candidates, multiple recovery windows, and a decisive
    non-fade ending. Use the normal nine locations and a focused lateral-sweep,
    targeted-strike, and arena-pulse grammar to reinforce the core systems
    without later-boss combinations. The boss is a visually singular spiritual
    monster empowered by a fragment; destroying it proves fragments can restore
    a damaged region but does not reveal the conspiracy. First victory guarantees
    a useful starter choice across Instrument, Ward Core, or Resonator functions.
    Reject flat intensity, dead quiet sections, weak stem coverage, or unusable
    peak boundaries. GD-31 remains open for the second and third briefs.
  - Second encounter brief approved, 2026-08-17: generate a roughly 3½-to-5-
    minute full-stem song with more aggressive rhythmic tension, stronger dynamic
    volatility, a meaningful active breakdown or arrangement shift, two linked
    Climax pressure peaks, and a hard finishing cadence. Retain nine familiar
    locations while corrupting or disabling only one or two at a time. Persistent
    hazards, cover, and clearly announced two-part attacks test dash budgeting,
    Defend, abilities, and deliberate Ward absorption; Band Calls gain tactical
    value and one Crescendo remains guaranteed. The fragment holder embodies an
    existing obsession with command or hierarchy amplified into monstrosity,
    raising suspicion without revealing the conspiracy. Rewards emphasize Ward,
    dangerous positions, Band Calls, and tactical hybrids while never modifying
    movement recovery. Reject candidates without linked pressure peaks, dynamic
    contrast, fair two-part boundaries, or complete instrument activity. GD-31
    remains open for the third brief.
  - Third encounter brief approved, 2026-08-17: generate a roughly 4-to-5½-
    minute full-stem song with the launch lineup's greatest structured intensity,
    broadest dynamic range, multiple escalating Climax peaks, strongest finishing
    cadence, genuine instrument handoffs, and complete group and event windows.
    Begin with the familiar location language, then transform the graph at
    authored phase boundaries while combining the learned attack grammar,
    movement recovery, hazards, cover, Specials, group actions, revival, and
    builds without adding a new core control. The boss or arena is directly tied
    to the ancient betrayal and yields the first credible evidence that official
    Order history is false without revealing the mastermind or resolving the
    novice mystery. Rewards complete the launch's Hype, Signature, Band Call,
    group-support, and hybrid possibilities, and victory visibly opens the next
    hub tier. Once separately implemented, the deferred song pipeline must
    qualify this encounter without extensive one-off exceptions. Reject
    repetitive peaks, inadequate dynamics, incomplete roster coverage, weak
    endings, or music that fights the mechanics. With all three briefs approved,
    GD-31 is resolved; exact songs, names, and asset production remain downstream
    selection work.

- [ ] **GD-32: What must the result, reward, retry, and post-battle screens tell
  the player?**
  - How are musical accuracy, combat contribution, teamwork, mastery, drops, and
    improvement opportunities explained?
  - How does the next action stay obvious after victory or defeat?

- [ ] **GD-33: Which accessibility, comfort, and age-appropriate safety options
  must ship with the first product?**
  - Which calibration, contrast, non-color cues, reduced motion/flashing, camera,
    remapping, subtitle, and audio controls are required?
  - Which assists are independent of difficulty and rewards?
  - What presentation and social defaults protect the target audience?

- [ ] **GD-34: What observable playtest results prove the game design is ready
  for implementation and later release?**
  - Which tasks should first-time target-age players understand without coaching?
  - What should be measured for rhythm readability, boss attention, fairness,
    replay desire, solo completeness, and co-op coordination?
  - What failure patterns require redesign rather than numeric tuning?

## Deferred technical specifications

These documents may be created after their player-facing dependencies are
settled. They do not add questions to this interview by themselves.

- Rhythm chart schema, authoring-tool architecture, validators, and export format.
- Song-processing pipeline upgrade to generate or export structural section
  markers, five-function encounter alignment data, Crescendo candidate windows,
  per-instrument and per-difficulty Activity Maps, ensemble coverage, dynamic-
  intensity features, and validation metadata required by GD-10, GD-16, GD-19,
  and GD-31. This task is flagged only and is not part of the current interview.
- Roblox client/server authority, networking, anti-cheat, persistence, and
  analytics event schema.
- UI component specification, responsive layouts, safe-area measurements, and
  input maps.
- Economy tables, item catalogs, numeric balance sheets, and drop-rate tables.
- Boss/song content briefs, narrative bible, production schedule, and asset
  manifests.

## Plan change log

- **2026-08-14:** Created the 34-question baseline from `GAME_VISION.md`, its
  downstream backlog, and the pre-vision `GAME_DESIGN.md`. No owner questions
  have yet been added or removed.
- **2026-08-17:** GD-31 rejected Heaven's Edge and Blackened Crown as presumed
  launch songs because their current intensity and dynamics are insufficient.
  Added a deferred pipeline-upgrade specification item; no implementation was
  authorized or performed.
