# Bands Battle Game Design

- **Status:** Approved game-design baseline
- **Approved:** 2026-08-18
- **Parent vision:** [`GAME_VISION.md`](GAME_VISION.md)
- **Visual authority:** [`ART_DIRECTION.md`](ART_DIRECTION.md)
- **System ownership:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)
- **Decision source:** [`GAME_DESIGN_WORKING.md`](GAME_DESIGN_WORKING.md)
- **Interview plan:** [`GAME_DESIGN_QUESTIONS.md`](GAME_DESIGN_QUESTIONS.md)

## 1. Role and authority

This document is the canonical player-facing game-design specification for Bands
Battle. It translates the product vision into encounter, rhythm, combat,
multiplayer, progression, content, accessibility, and validation rules.

`GAME_VISION.md` remains authoritative for product purpose, world foundation,
audience, platform, tone, and scope. `ART_DIRECTION.md` remains authoritative
for visual language. This document governs how those constraints become a game.
`SYSTEMS_MAP.md` is the downstream authority for system responsibilities,
dependencies, and required detailed specifications; it does not override the
player-facing behavior defined here. The working record and question plan
preserve interview history but do not override this approved specification.

Numeric values labeled as starting points, hypotheses, targets, or approximate
values require playtesting. Changing a settled behavioral rule requires an
explicit design amendment, not silent implementation drift.

## 2. Product definition

Bands Battle is a native Roblox, touch-first rhythm boss-combat game for a
primary audience around ages 10 to 14. Players perform instrument parts from
full-length, high-intensity songs, route accurate musical play into combat
intent, reposition among authored tactical locations, survive readable boss
attacks, recover fragments of the Shattered Song, and develop flexible builds
alone or with a band.

### Product pillars

1. **Music is combat:** rhythm performance directly creates damage, protection,
   support, recovery, resource generation, and group effects.
2. **The boss remains the focus:** the compact staff supports performance without
   turning the encounter into a permanent note highway.
3. **The whole song shapes the encounter:** authored musical dynamics govern
   phases, attacks, breathing room, climaxes, and the finishing cadence.
4. **Touch-first clarity:** three fixed rhythm inputs, stable controls, readable
   telegraphs, and phone-scale information hierarchy define the baseline.
5. **Tactical commitment:** intent, position, dash recovery, Ward, and group
   opportunities create choices beyond note accuracy.
6. **Solo is complete and co-op is expressive:** Order acolytes support solo
   without simulated scores; human bands add revival and coordinated spectacle.
7. **Skill and builds both matter:** equipment and specialization change combat
   consequences and strategy without changing musical judgments.
8. **Progress respects the player:** deterministic earning paths, useful failure
   progress, accessible options, and low-pressure monetization avoid coercion.

## 3. Fixed vision constraints

These product-level constraints are settled:

- Bands Battle is a native Roblox, touch-first supernatural rhythm-combat game
  for a primary audience around ages 10 to 14.
- Rhythm directly controls combat; the shipping game has no permanent song-wide
  note highway, but active passages may use a compact moving staff.
- Mobile performance uses three large fixed pads and short, glanceable phrases.
- The song is the encounter's master clock and a normal encounter uses one full
  three-to-seven-minute song.
- Players choose Attack, Defend, and Special intent separately from rhythm
  execution.
- Tactical arena positions create risk and reward; moving suspends phrase
  participation without creating artificial misses.
- Boss progress uses sequential resistance layers, banked value for early
  breaks, and a final authored finishing performance rather than a conventional
  biological-health fantasy.
- Both complete solo play and three-to-six-player co-op are core. Solo uses
  lightweight Order acolytes; co-op supports duplicate instruments and public or
  preformed bands.
- Instruments are identities, not locked combat classes. Equipment, abilities,
  and role development create builds.
- Failure grants modest practice progress and ordinary materials; victory is
  required for story progress, fragments, and signature drops.
- The first shippable product includes an Order hub, onboarding, three replayable
  bosses, solo, co-op, and a basic equipment/reward/progression loop.
- PvP, user-authored songs, free-roaming worlds, deep crafting, paid recovery,
  paid randomness, and multi-song raids are outside the first release.

## 4. Detailed game design

### GD-01: Complete repeatable game loop

- **Approved direction:** The proposed hub-centered loop is accepted with one major
  presentation change: encounter selection is not a mission board. A stylized
  area in the hub contains phasing shards that poke through into existence. The
  shards have labels, and activating a shard leads into its boss encounter flow.
  Solo, public matchmaking, or current-party play is selected after activating
  the shard. The hub can contain item shopping and, when crafting is eventually
  added, crafting.
- **Specification:** Players begin in the Order hub. First-time players are
  guided into onboarding, while returning players have a fast route back to
  play. Boss selection happens through labeled, phasing shards in a dedicated
  in-world encounter area. After activating a shard, the player chooses solo,
  public matchmaking, or their current party, then confirms difficulty and
  preparation before being transported to the boss. After the full-song
  encounter, results present performance, progression, and rewards, followed by
  immediate choices to retry, change difficulty, return to the hub, improve
  equipment, continue to the next story encounter, or remain with the band.
- **Consequences:** The shards are both worldbuilding and the primary encounter
  navigation. The hub must support shopping and loadout improvement, but repeat
  play must not require unnecessary traversal. Boss labels and activation states
  must remain clear enough for the target audience even though selection is
  presented in-world rather than as a conventional mission board.
- **Deferred:** The shard area's visual composition and activation treatment
  await the owner's future visualization plus later world/UI design. The exact
  fast-play shortcut is deferred to GD-30. Crafting functionality is outside the
  first release and should appear in the hub only when that system is added.

### GD-02: Normal encounter phases

- **Approved direction:** The proposed five-phase, song-shaped encounter structure is
  approved as presented.
- **Specification:** A normal encounter has five flexible functions rather
  than five equal-duration blocks:
  1. **Arrival:** transport into the arena, boss reveal, an approachable opening
     phrase, and time to read the available positions.
  2. **First Clash:** establish the core rhythm interaction, introduce the
     boss's simplest attack, and open the first resistance layer.
  3. **Escalation:** introduce additional resistance layers, stronger attack
     patterns, meaningful repositioning, and possible Band Call or Crescendo
     windows.
  4. **Climax:** use the song's most intense section for the final resistance
     layer and the encounter's highest pressure.
  5. **Finishing Cadence:** clearly preview a final authored phrase that
     determines whether the weakened boss is destroyed or survives.

  Quiet passages may provide breathing room, recovery, repositioning, or story
  moments wherever the song naturally supports them. They are not required to
  form a standalone phase. The song controls the phase timeline. A resistance
  break changes combat state but never skips, pauses, or rewinds the music.
  Breaking early banks Momentum; breaking late leaves less time for later
  layers. Player survival state, personal special resources, prepared
  consumables, and accumulated Momentum carry across phase boundaries.
- **Consequences:** Encounter authors must map these functions onto each song's
  actual structure instead of forcing equal timestamps or a universal template.
  Boss logic and presentation must react to resistance progress without changing
  the audio timeline. The finishing phrase must be telegraphed before the final
  cadence, and persistent combat resources must remain legible throughout the
  encounter.
- **Deferred:** Encounter-specific attack placement, resource values, and final
  numeric thresholds require content design and balance testing.

### GD-03: Main battle surface

- **Approved direction:** The proposed director-assisted third-person battle surface
  is approved as presented.
- **Specification:** The battle camera sits behind and slightly above the
  player's avatar. The performer and instrument occupy the lower foreground,
  while the boss commands the upper center of the frame. Tactical positions
  form a readable arc facing the boss. The player directly controls rhythm,
  combat intent, and position selection; ordinary battle play does not require
  free locomotion or manual camera rotation. Selecting a position initiates a
  dash, and the camera follows the relocation smoothly.

  Boss attack paths and safe areas are represented physically in the arena and
  reinforced through compact interface warnings. Short directed camera moments
  may emphasize arrival, resistance breaks, Crescendos, and victory, but they
  must not interrupt an active phrase or reaction window. Other band members
  remain visible where practical and receive greater framing emphasis during
  coordinated actions.

  The persistent information layer contains the player's survival state,
  selected combat intent, personal special meter, the boss's current resistance
  layer and encounter phase, and the three fixed rhythm pads. Phrase previews,
  position choices, attack geometry, Band Call invitations, Join In,
  consumables, revive, and recovery cues appear contextually.
- **Consequences:** Encounters can be composed around known player positions and
  predictable framing. Boss scale, pose telegraphs, ground targeting, and
  performer actions must read from the normal camera at phone size. UI design
  must protect the boss view and avoid duplicating arena information more than
  necessary. Camera emphasis is subordinate to gameplay timing and must provide
  a reduced-motion treatment.
- **Deferred:** Exact camera distances, fields of view, screen proportions,
  transition curves, position count, and responsive UI measurements belong to
  prototyping, GD-15, and the later UI specification.

### GD-04: Playable phrase grammar

- **Approved direction:** The proposed small first-release phrase grammar and the
  exclusions are approved as presented.
- **Specification:** Playable phrases use five elements:
  - **Tap:** press one indicated pad on its beat.
  - **Hold:** press on the starting beat and maintain through the shown
    duration. The initial press and continued hold are judged; precise release
    timing is not a separate judgment in the first release.
  - **Repeat:** strike the same pad multiple times to express repeated musical
    notes.
  - **Alternate:** move among pads to express a changing musical pattern.
  - **Rest:** leave intentional empty timing spaces so phrases can express rests
    and syncopation.

  A phrase normally spans one or two measures and contains roughly four to eight
  actions as one readable scoring and combat group within the moving staff.
  Easier charts retain a phrase's strongest musical accents while reducing its
  pattern. Harder charts may add subdivisions, syncopation, and denser
  alternation. The first release excludes two-pad and three-pad chords, swipes,
  flicks, and dragging between pads.
- **Consequences:** All launch songs and instruments must be expressible through
  this shared grammar. Difficulty authoring should preserve musical identity
  instead of merely slowing the chart. Input detection can prioritize clear
  discrete contacts and sustained contacts without gesture recognition or
  simultaneous-touch chord rules.
- **Deferred:** Exact density ceilings, subdivision limits, hold leniency, and
  any post-launch mechanic additions require playtesting and the timing design
  in GD-07 and GD-08.

### GD-05: Phrase preview and performance cues

- **Approved direction:** The game should not arbitrarily decide that a player
  may perform only a small selection of isolated phrases. When a player is
  settled at a playable position and their selected instrument has notes in the
  arrangement, the player should generally be able to perform those available
  notes. Natural breathing room is welcome when the music rests, the boss has
  been knocked back, the encounter moves to a different scenario, or another
  meaningful event creates a break. The player should not mash continuously for
  a full three-to-seven-minute song, but the opposite extreme of two measures of
  play followed repeatedly by inactivity is also unacceptable. The encounter
  needs a deliberate mix with a substantial amount of action.
- **Specification:** The selected instrument's authored chart is
  the source of playable material. A **phrase** groups notes for readability,
  judgments, combat contribution, and intent boundaries; it is not an arbitrary
  permission window. Multiple one- or two-measure phrases may chain without a
  gap to form a longer **performance passage**. Breaks occur for musically or
  dramatically legible reasons: an actual rest or dropout, movement chosen by
  the player, boss knockback or phase transition, recovery, repositioning, or
  another authored encounter event. Reaching the end of a phrase alone does not
  force downtime.
- **Consequences:** The compact moving staff may remain active across chained
  phrases during sustained song sections. Encounter authors control when combat
  activity should yield to observation or movement, but they do not discard
  playable instrument notes merely to impose a repeating play/rest cadence.
  Instrument dropouts continue to respect the real arrangement; optional Join
  In beat actions may provide sparse participation where the vision already
  permits them. As an initial tuning hypothesis, roughly 65 to 80 percent of a
  normal encounter should offer meaningful rhythmic participation, adjusted per
  song, instrument, difficulty, and playtest evidence. Inactive time must still
  contain something meaningful to observe, decide, or do. Desktop retains the
  compact right-to-left staff, fixed strike line, and labeled Z/X/C targets.
  Mobile uses the same time-to-impact model with three generous fixed touch pads.
  All lanes and pads use shape and label as well as color.
- **Deferred:** Exact phrase-boundary visuals, staff animation and collapse
  behavior, mobile layout, and final density values require prototyping and
  playtesting. This decision explicitly amends the static-cue language in
  `GAME_VISION.md` while preserving its prohibition against a permanent
  song-wide note highway.

### GD-06: Cross-device input actions

- **Approved direction:** On keyboard, `W`, `A`, `S`, and `D` are always
  movement keys. Assigning `A`, `S`, or `D` to combat intent is unacceptable.
  Free movement is useful in the hub, while boss combat should use dashes from
  position to position; continuous free combat movement does not add a clear
  benefit.
- **Specification:** Reserve the entire WASD cluster exclusively for
  movement. In the Order hub, WASD provides ordinary continuous movement. In a
  boss encounter, `W` dashes to an available position closer to the boss, `S`
  retreats to an available safer position, and `A` or `D` moves to a neighboring
  authored position on the corresponding side. Boss combat has no continuous
  free locomotion. Rhythm may remain on `Z`, `X`, and `C`. Attack, Defend,
  Special, and other non-movement actions require different keys.
- **Consequences:** Hub exploration preserves familiar Roblox movement. Boss
  arenas retain readable cover, risk tiers, camera composition, attack geometry,
  mobile parity, and the approved position system without requiring a virtual
  movement joystick during performance.

  The complete baseline mapping is:

  | Action | Keyboard | Touch | Gamepad |
  |---|---|---|---|
  | Rhythm | `Z`, `X`, `C` | Three large fixed pads | Left, bottom, and right face buttons |
  | Movement | WASD | Hub joystick; tap an encounter position | Left stick |
  | Attack / Defend / Special | `1`, `2`, `3` | Three persistent intent buttons | D-pad left, up, and right |
  | Join In / accept invitation | `Space` | Contextual cooperative-action button | Top face button |
  | Initiate Band Call | `4` | Equipped Band Call button | Right trigger |
  | Consumables | `5`, `6` | Two prepared-item buttons | Left and right bumpers |

  Frequently timed rhythm controls never move or acquire unrelated meanings.
  Occasional actions may queue to a musical boundary instead of requiring a
  simultaneous precision input. Prompts show the active device's labels or
  glyphs. Keyboard and gamepad bindings are remappable where Roblox permits it.
  Touch layouts may swap secondary-control clusters for handedness and maintain
  generous target sizes. Critical actions never rely on color alone.
- **Deferred:** If later design changes the number of equipped Band Calls or
  consumable slots, their secondary bindings may be revised without changing the
  movement, rhythm, or intent-control principles.

### GD-07: Timing accuracy, latency, and feedback

- **Approved direction:** The proposed timing judgments, feedback hierarchy, hold
  treatment, and calibration flow are approved as presented.
- **Specification:** Each timed input receives one of four judgments:

  | Judgment | Initial Normal test window |
  |---|---:|
  | Perfect | ±60 ms |
  | Great | ±110 ms |
  | Good | ±170 ms |
  | Miss | Beyond ±170 ms |

  These values are starting points for playtesting, not immutable release
  constants. A note resolves immediately when judged at the strike line. Its pad
  flashes, the instrument visibly responds, and a compact grade appears near the
  staff instead of covering the boss. Great and Good include a small early/late
  arrow. Perfect does not need directional feedback. Misses remain unmistakable
  but avoid shaming language or excessive visual punishment. Individual note
  feedback never causes major camera shake. Touch and gamepad may provide
  restrained haptics. A small phrase summary communicates local performance;
  detailed early/late distributions and improvement guidance belong on the
  result screen.

  A hold's initial press receives the normal timing grade. Maintained hold time
  continues earning contribution. Releasing early stops additional contribution
  but does not create a second release-timing judgment or an extra miss.

  Onboarding offers a skippable guided calibration:
  1. The player aligns a visual pulse with an audible beat.
  2. The player taps along for approximately 12–16 beats.
  3. The game rejects obvious outliers and suggests an offset.
  4. The player tests a short sample and may adjust the result manually.
  5. The game saves calibration by device/control profile and keeps
     recalibration easy to access.

  A consistent early/late trend or audio-device change may trigger a private
  recalibration suggestion. Calibration changes timing alignment only; it never
  changes difficulty, progression, or rewards.
- **Consequences:** Timing feedback remains immediate enough to teach while the
  boss stays visually primary. Aggregate results can distinguish execution
  problems from calibration problems. Difficulty may adjust judgment windows in
  GD-08, but equipment never does. Exact offsets must be applied consistently to
  the same musical clock used by notes and encounter events.
- **Deferred:** Final window values, haptic patterns, calibration statistics, and
  device-profile persistence details require cross-device playtesting and a
  later technical specification.

### GD-08: Difficulty transformation

- **Approved direction:** The proposed Easy, Normal, and Hard model is approved after
  making explicit that reducing note count on Easy cannot reduce the player's
  available combat output.
- **Specification:** The first release uses three difficulties:

  | Area | Easy | Normal | Hard |
  |---|---|---|---|
  | Rhythm chart | Strong accents and simpler patterns | Intended core arrangement | Dense authored detail, syncopation, and faster alternation |
  | Perfect | ±90 ms | ±60 ms | ±45 ms |
  | Great | ±150 ms | ±110 ms | ±85 ms |
  | Good | ±230 ms | ±170 ms | ±135 ms |
  | Boss attacks | Longer telegraphs and simpler combinations | Intended encounter | Tighter but fair telegraphs and more dangerous combinations |
  | Resistance | Forgiving targets | Intended targets | Higher performance requirements |
  | Survival | Lower incoming damage and stronger recovery | Intended balance | Greater incoming pressure and limited recovery |
  | Group cohesion | Positive bonus with forgiving thresholds | Positive bonus with intended thresholds | Same capped bonus with stricter thresholds |
  | Recovery Crescendo | At most one additional opportunity when substantially behind | Authored guaranteed opportunity only | Authored guaranteed opportunity only |

  Maximum combat contribution is normalized per musical passage on every
  difficulty. If Easy contains 8 inputs and Hard contains 16, a fully performed
  passage produces the same maximum damage, defense, healing, or utility before
  the encounter's difficulty-specific resistance and pressure are applied.
  Removing chart inputs never makes Easy weaker. Wider timing windows, lower
  resistance, lower incoming damage, and stronger recovery make it genuinely
  easier.

  Difficulty never changes song speed or duration, story outcome, boss identity,
  arena, core phases, controls, calibration, or access to independent
  accessibility assists. All difficulties can advance the campaign and release
  the Shattered Song fragment. They share the same boss-themed reward pool.
  Normal and Hard may improve reward quantity or high-quality roll chances, and
  Hard may grant mastery cosmetics, titles, or badges; essential combat power is
  never exclusive to Hard.

  Easy and Normal begin unlocked. Hard unlocks separately for each boss after a
  Normal victory. The game may privately recommend moving up after consistently
  strong performance or moving down after repeated struggle, but it never
  changes the selected difficulty automatically.
- **Consequences:** Chart reduction must preserve musical identity and active
  passages rather than generating long Easy-mode gaps. Combat scoring consumes
  normalized passage performance instead of raw note counts. Boss and co-op
  balance can therefore change by difficulty without penalizing a simplified
  chart's lower input count.
- **Deferred:** All numeric timing windows and balance relationships remain
  playtest starting points. A fourth Master difficulty is deferred until player
  evidence demonstrates demand.

### GD-09: Performance-responsive audio

- **Approved direction:** The proposed personal instrument response, local-versus-shared
  co-op mix, and audio-priority model are approved as presented.
- **Specification:** Every player hears the complete song at a stable backing
  level. The selected instrument receives additional local emphasis based on
  performance:
  - **Perfect:** crisp attack accent, brief clarity or level lift, and the
    strongest combat response.
  - **Great:** confident normal instrument accent.
  - **Good:** softer accent with less emphasis.
  - **Miss:** brief duck or filtered stumble, never complete silence.
  - **Movement:** return to ordinary backing level without a miss sound.
  - **Downed:** become muffled and distant while the full song continues.
  - **Recovery:** return through an on-beat swell.

  These responsive changes are primarily local. One player's weak performance
  does not damage every teammate's song mix. Teammates hear meaningful combat
  effects and group contributions rather than every local miss. The shared
  crowd, arena, and combat layers respond to aggregate band performance. Band
  Calls and Crescendos temporarily widen and strengthen the ensemble. Solo keeps
  the complete backing mix even though acolytes do not simulate instrument
  phrases or generate artificial performance scores.

  The mix priority is:
  1. Critical boss telegraphs and timing cues.
  2. The local player's selected instrument and judgment response.
  3. The core song mix.
  4. Other combat effects, crowd, and ambience.

  Critical cues use distinct rhythm, pitch range, and sound shape instead of
  depending only on greater volume. Nonessential effects may duck briefly for a
  critical telegraph, but the song's pulse remains audible. Phone-critical cues
  use strong midrange transients and remain meaningful without deep bass or
  stereo separation. Repeated note hits strengthen the performed instrument
  rather than adding an unrelated noisy sound on every input.
- **Consequences:** The content pipeline must preserve usable instrument stems
  or equivalent controllable layers. Each client can express its local player's
  performance without synchronizing every mix change across the network. Shared
  group audio responds to aggregate events, making ensemble success audible
  without allowing one weak performer to spoil the song.
- **Deferred:** Exact gain, filter, ducking, haptic, and dynamic-range values;
  device mix profiles; and any substitute for songs without usable stems require
  audio implementation and cross-device mixing tests.

### GD-10: Song and encounter authoring workflow

- **Approved direction:** The proposed human-directed, automation-assisted authoring
  and approval workflow is approved as presented.
- **Specification:** Each song and encounter passes through seven stages:
  1. **Ingest:** collect the final master, instrument stems, rights and generation
     provenance, lyrics, duration, intended difficulty, and available arrangement
     notes.
  2. **Automated musical analysis:** propose a tempo map, beats, downbeats,
     instrument onsets, pitches, holds, rests, dropouts, energy changes, section
     boundaries, and one- or two-measure phrase groupings.
  3. **Human chart editing:** correct the beat grid first; approve playable
     events; map them to the three inputs; correct holds, rests, and phrase
     boundaries; chain phrases into passages; and verify that every chart
     reflects the audible instrument.
  4. **Difficulty generation:** treat the detailed chart as the source, derive
     Normal and Easy suggestions by preserving important accents, normalize
     combat contribution per passage, require human review, and prevent Easy
     from becoming inactive.
  5. **Encounter authoring:** map Arrival, First Clash, Escalation, Climax, and
     Finishing Cadence onto the song; place resistance windows, attacks,
     knockbacks, recovery, repositioning, Band Call and Crescendo candidates;
     author the finishing performance; and review activity and breathing room
     for each instrument.
  6. **Automatic validation:** reject unfair phrase/impact conflicts, fake notes
     during dropouts, arbitrary gaps between available phrases, unnormalized
     difficulty output, activity-density violations, invalid final windows, and
     events that do not align to the shared musical clock.
  7. **In-Roblox review:** play every instrument and difficulty; test solo and
     representative co-op sizes; test phone first, then desktop and gamepad; and
     explicitly approve musical quality, design, and technical behavior before
     release.

  The eventual authoring surface provides waveform and stem views, a beat grid,
  three note lanes, difficulty layers, encounter-event tracks, looping,
  scrubbing, drag editing, validation, and direct test export. Automation and AI
  may suggest content but never approve or publish it. Production of the first
  three bosses may begin with a lightweight internal timeline tool rather than a
  polished external creator product.

  The pipeline also generates an **Activity Map** for every instrument and
  difficulty plus ensemble eligibility data. At each beat or measure boundary,
  it records playable density before and after the boundary, longest rest,
  crossing holds, instrument entries, exits and solos, musical energy or
  quietness, boss and phase conflicts, distance from the finisher, available
  reaction time, and whether the following passage sustains activity. Separate
  rules evaluate candidates for solo recovery, cooperative revival, boss
  attacks, Band Calls, and Crescendos. Validation must prove that every required
  event type has a candidate within its maximum delay for supported instruments,
  difficulties, and rosters; otherwise the chart cannot ship. Human reviewers
  approve the resulting coverage rather than manually placing every possible
  runtime activation.
- **Consequences:** The beat grid becomes foundational shared data for charts,
  boss events, group opportunities, and the finishing cadence. Generated output
  is always treated as a proposal. A chart cannot ship merely because it passes
  structural validation; it must work on the real Roblox surface and receive
  human musical judgment.
- **Deferred:** Editor architecture, file schemas, analysis models, validators,
  export formats, versioning, and approval metadata belong in a dedicated
  technical specification.

### GD-11: Player survival resources

- **Approved direction:** Ward as the single survival resource is approved. Dangerous
  positions must increase outgoing damage and potential rewards in exchange for
  greater danger. The design must not introduce an unexplained separate
  “defense passage.”
- **Specification:** Ward is the only player-survival bar in the first release;
  there is no separate conventional health resource. Every player begins an
  encounter with full Ward. Direct boss impacts, hazards, and failed defensive
  responses reduce it. An ordinary missed instrument input does not directly
  remove Ward; weak general performance instead produces less offense, resource
  generation, and protection.

  Defense uses the same instrument notes already available to the player. There
  is no separate defense chart or defense passage. When a boss telegraphs an
  attack, selecting Defend routes subsequent successful performance into
  mitigation or Ward reinforcement for that threat. Poor execution leaves more
  of the attack unblocked at impact. Defend performance, support abilities,
  authored recovery moments, and prepared consumables may restore or reinforce
  Ward. Ward does not automatically refill completely between phases.

  Dangerous positions increase outgoing combat damage and potential encounter
  rewards while also increasing exposure or incoming damage. A reward bonus is
  earned through successful performance while exposed, not by merely standing
  at the dangerous location.

  Ward has four readable presentation states:

  | State | Presentation |
  |---|---|
  | Safe | Solid cyan form and stable meter |
  | Below 50% | Visible hairline fractures |
  | Below 25% | Strong cracks, restrained warning sound, and urgent outline |
  | Empty | Ward shatters and the player is downed |

  These states use geometry, animation, sound, and meter changes rather than
  color alone. Damage tuning ensures that downing follows accumulated failures
  or a clearly telegraphed major attack, not one ordinary mistake.
- **Consequences:** A single survival resource keeps the phone HUD and combat
  decisions legible. General rhythm misses still matter without creating
  arbitrary chip damage. Defensive play becomes an intent applied to familiar
  instrument performance, and the position system must balance offensive and
  reward upside against survival risk.
- **Deferred:** Ward values, damage amounts, restoration rates, dangerous-position
  multipliers, reward-bonus banking, revive strength, and solo last-chance
  recovery require numeric tuning and encounter testing.

### GD-12: Attack, Defend, and Special intent

- **Approved direction:** The proposed three-intent routing and beat-boundary switching
  model are approved as presented.
- **Specification:** Attack, Defend, and Special route the same available
  instrument performance into different combat outcomes. They never substitute
  a separate note chart.
  - **Attack** is the default. Successful performance damages the active boss
    resistance layer. Valid contribution after an early break becomes Momentum.
  - **Defend** converts successful performance into temporary mitigation against
    the next telegraphed impact and may modestly reinforce Ward.
  - **Special** redirects performance into the equipped personal ability. If the
    ability is charging, successful inputs fill it. Reaching full charge returns
    the player to the previous Attack or Defend intent without automatically
    firing. Selecting a ready Special arms it; the next scoring group determines
    its potency, and it resolves at the following clean musical boundary before
    returning to the previous intent.

  Pressing an intent immediately highlights it as queued. It takes effect on the
  next playable beat or note. Already performed inputs retain their previous
  intent, and switching never reinterprets notes or creates misses. While moving,
  a new intent applies to the next playable material. If Special is unavailable,
  its control explains why and does not change intent. Boss telegraphs provide
  enough warning to select Defend before impact. Mid-phrase switching remains an
  advanced optimization; baseline encounter success never requires it.
- **Consequences:** Attack advances victory, Defend preserves survival, and
  Special trades immediate output for a build-specific effect. The interface
  must distinguish selected, queued, unavailable, charging, and ready states
  without moving the controls. Scoring must split contribution at the exact
  beat-boundary where intent changes.
- **Deferred:** Exact conversion ratios, Ward reinforcement, ability charge,
  activation units, and build modifiers require numeric tuning.

### GD-13: Boss Resolve, Momentum, and finishing cadence

- **Approved direction:** The proposed three-layer Resolve model, capped Momentum,
  world-visible breaks, and two-condition finishing cadence are approved as
  presented.
- **Specification:** **Resolve** is the working name for the boss resistance
  system. A normal encounter contains three sequential layers associated with
  First Clash, Escalation, and Climax. Easy, Normal, and Hard retain three layers
  while changing the normalized contribution required to break them.

  Only the current layer can receive damage. Every later layer has an authored
  opening point on the song timeline. Breaking a layer never skips, pauses, or
  rewinds the song. After an early break, further Attack contribution becomes
  visible Momentum. When the next layer opens, that Momentum applies as initial
  damage. Its starting cap is equivalent to roughly 20 percent of the next layer,
  ensuring that strong early performance matters without allowing a layer to be
  skipped. If a layer breaks after the next layer's scheduled opening, the next
  becomes vulnerable immediately and the band simply has less remaining time.

  Resolve requirements scale by difficulty and human player count using
  normalized passage contribution rather than raw note counts. After an early
  break of the third layer, continued successful Attack performance improves
  result tier, reward potential, and finishing spectacle. It cannot satisfy or
  replace the required final performance.

  Victory requires both:
  1. All three Resolve layers are broken before the song ending.
  2. The band meets the selected difficulty's threshold on the clearly previewed
     finishing phrase.

  Failure of either condition means that the boss survives and the encounter
  ends in defeat. Random effects cannot reverse the result. All three Resolve
  segments are visible from encounter start, with future layers locked and only
  the active layer illuminated. Each break visibly shatters a part of the boss's
  guard, armor, halo, or surrounding supernatural structure so progress exists
  in the world as well as the interface.
- **Consequences:** Encounter charts require three authored openings and one
  final performance window. The runtime must preserve progress against a late
  layer without shifting the song. Momentum and finishing rewards keep early
  breaks valuable while the required final phrase preserves the musical climax.
- **Deferred:** Resolve thresholds, player-count scaling, exact Momentum caps,
  result-tier bonuses, and finishing thresholds remain numeric tuning decisions.
  The working name Resolve may receive a later worldbuilding review without
  changing the mechanic.

### GD-14: Boss attack grammar and timing

- **Approved direction:** The proposed attack families, four-stage timing structure,
  and fairness constraints are approved as presented.
- **Specification:** Every boss attack follows four authored stages:
  1. **Telegraph:** boss pose, sound motif, arena geometry, and compact warning
     identify the threat and likely response.
  2. **Commit:** targeted players, positions, and unsafe areas lock. The attack
     cannot retarget at the last instant.
  3. **Impact:** damage and other consequences resolve on a musically significant
     beat.
  4. **Recovery:** the boss completes the action and creates a short offensive,
     recovery, or repositioning opportunity.

  The first-release grammar supports lateral sweeps, range attacks, targeted
  strikes, arena-wide pulses, persistent hazards, and rare major band attacks.
  Individual bosses use a focused subset rather than every family.

  Telegraphs begin on readable beat or measure boundaries. Easy generally gives
  around two measures of warning, Normal gives one to two measures, and Hard
  still gives at least one clear measure for major attacks. Required responses
  use pose, geometry, sound, shape, and text or icon reinforcement rather than
  color alone. Movement-required impacts do not arrive unexpectedly during a
  committed performance. Targets never change after Commit. Runtime variation
  chooses only among pre-authored candidates that fit the current song window.
  Major attacks cannot overlap into an impossible survival requirement.

  In co-op, each targeted player and affected position is explicit. Player-count
  scaling may change how many targets are selected but never degrades telegraph
  reliability. Difficulty may shorten warnings or combine already established
  patterns without hiding necessary information. Failure may cause Ward damage,
  knockback, position loss, or a temporary hazard; it never scrambles controls
  or fabricates rhythm misses. Successfully avoiding or defending a major attack
  creates a brief earned advantage against the boss.
- **Consequences:** Boss authors need reusable attack families plus boss-specific
  poses and audio identities. Event validation must cover target locking,
  committed-performance conflicts, candidate-window validity, co-op clarity, and
  impossible overlaps. Recovery is a designed combat beat rather than unused
  animation time.
- **Deferred:** Exact warning measures for minor attacks, damage, target counts,
  recovery duration, hazard persistence, and earned-advantage values require
  boss-specific design and playtesting.

### GD-15: Arena positions and movement

- **Approved direction:** Nine tactical locations are a good arena baseline, but boss
  and arena mechanics may justify more or fewer. Players and acolytes may share a
  location using formation offsets; if sharing works for multiple humans, there
  is no reason to force an acolyte to vacate. The remaining proposed movement,
  risk, reward, and cover rules are approved.
- **Specification:** A baseline arena uses nine tactical locations: Near,
  Middle, and Rear risk tiers, each with left, center, and right locations. This
  is an authoring default rather than a universal grid. An encounter may use
  more, fewer, or an irregular location graph when its boss mechanics justify
  the change and the result remains readable.

  Every location exposes directional neighbors. On keyboard, `W` advances,
  `S` retreats, and `A` or `D` moves laterally. Gamepad uses the left stick.
  Touch players tap a visible destination; a farther destination follows the
  same graph and cumulative travel time rather than teleporting. A baseline dash
  takes roughly 0.75 seconds and is followed by the established settling period
  before rhythm participation resumes. Dashing grants no invulnerability. A
  player caught traveling at impact remains exposed unless the actual route
  avoids the attack.

  Each voluntary dash consumes one visible movement charge. The charge refreshes
  two beats after landing, clamped initially to roughly 0.75–1.25 seconds so very
  slow or fast songs preserve a usable movement cadence. This movement recovery
  is separate from the shorter post-landing rhythm-settling time: once settled,
  the player may perform, select Defend, use cover, or activate an available
  ability even while another dash remains unavailable.

  Beat pips on the current location and destination markers communicate recovery,
  and a restrained cue confirms readiness. An unavailable dash never auto-queues.
  Every edge of a longer touch-selected route honors the same recovery, allowing
  the route and arrival time to remain honest. Involuntary displacement neither
  consumes a ready charge nor restarts an existing recovery. Gear and difficulty
  cannot modify movement recovery.

  Multi-part boss attacks that test dash budgeting announce their sequence before
  the first commitment. Attack validation evaluates every impact against the
  player's known charge state and preserves a readable response through a legal
  route, Defend with playable material, cover, a prepared ability, or knowingly
  accepted Ward damage. A player who chose to spend movement may therefore face
  a meaningful Defend-or-absorb decision, but surprise targeting cannot create
  an impossible cooldown state.

  Tactical locations are shared gameplay zones, not exclusive slots. Multiple
  human players use formation offsets and cannot body-block one another. Order
  acolytes use the same formation system, automatically arrange around human
  performers, consume no gameplay capacity, and never block a movement or
  risk/reward choice. Everyone sharing a location remains subject to its attack
  geometry and danger.

  Starting risk-tier hypotheses are:

  | Tier | Attack output | Incoming danger | Reward potential |
  |---|---:|---:|---:|
  | Near | +25% | +30% | +25% |
  | Middle | Baseline | Baseline | Baseline |
  | Rear | −20% | −25% | No risk bonus |

  Risk Bonus is earned only through successful performance while exposed. A
  completed phrase banks its bonus. Moving or being downed before completion
  loses only the current phrase's unbanked bonus; previously banked value remains.
  Cover belongs to specific locations and protects against specified attack
  shapes rather than granting universal immunity. Bosses may temporarily damage,
  corrupt, disable, add, or remove locations when authored mechanics preserve
  readable valid response paths.
- **Consequences:** The location system supports a consistent movement language
  without forcing every arena into identical geometry. Shared formation offsets
  eliminate blocking and simplify solo population while making clustered players
  share positional risk. Reward value reflects successful exposure rather than
  passive occupancy.
- **Deferred:** Final location counts per boss, graph layouts, dash, settling,
  recovery, and clamp times, multipliers, bonus caps, formation spacing, route
  evaluation, recovery indicators, and cover behavior require encounter
  prototypes and tuning. This decision explicitly amends the older acolyte-
  swapping language in `GAME_VISION.md`.

### GD-16: Downing, revival, and solo recovery

- **Approved direction:** The proposed downing, cooperative revival, and solo
  last-chance behavior is approved together with a generated, instrument-aware
  Activity Map and automatic runtime window selector. Different instruments may
  peak, solo, rest, or become quiet at different times; an event cannot create a
  difficulty spike at an unsuitable boundary merely because that boundary is
  mathematically convenient.
- **Specification:** Empty Ward downs the player at their
  current location, pauses their chart and combat contribution, and removes them
  from ordinary targeting. All humans down simultaneously ends a co-op attempt.
  A downed player may still share in a later victory according to prior
  contribution.

  Any active bandmate may route ordinary instrument performance into revival.
  One competent participant can complete it in roughly two measures; more
  participants accelerate completion and may raise returned Ward from a starting
  target of 35% toward roughly 60%. Participants sacrifice their normal combat
  contribution while helping. A revived player returns at their prior location,
  or the nearest valid Middle location when necessary, and receives about two
  beats of protection and settling before targeting and rhythm resume.

  Solo receives one emergency recovery challenge per encounter at a clean
  musical boundary. It uses the familiar three inputs for roughly one or two
  measures and scales by difficulty. Success returns about 35% Ward plus brief
  re-entry protection. Failure or a second down ends the attempt. Acolytes add
  presentation but no score. Robux cannot purchase or bypass recovery.

  Recovery, boss attacks, Band Calls, Crescendos, and other dynamic events use
  the Activity Map generated by the GD-10 authoring pipeline. Every event type
  has its own eligibility rules. At runtime, the selector filters candidates for
  the current instrument, difficulty, roster, encounter phase, and active events,
  then chooses the earliest eligible musical boundary within that event's
  allowed wait. The song never pauses to accommodate an event.

  If urgent recovery has no suitable instrument passage soon enough, it uses a
  clearly labeled universal beat challenge generated from the song's BPM and
  played with the familiar controls; it does not invent notes for a silent
  instrument. A nonurgent event waits for a valid candidate or is skipped.
  Authoring validation fails when a required event cannot find a candidate
  within its maximum delay for any supported configuration, and human review
  confirms that the generated candidates are musically credible.
- **Deferred:** Exact returned-Ward scaling, revive acceleration, protection
  duration, candidate-scoring thresholds, maximum event delays, universal beat
  patterns, and Activity Map implementation require prototyping and tuning.

### GD-17: Personal specials and resources

- **Approved direction:** The proposed one-slot Hype and Signature Special model is
  approved as presented.
- **Specification:** Each player equips one personal **Signature Special**
  before entering a shard. This slot is separate from the equipped Band Call.
  Signature Specials may emphasize offense, Ward, support, or positional
  utility, and any instrument can equip any special unless a later ability has a
  clear thematic restriction that does not create a required party composition.

  A one-charge **Hype** meter fills slowly through successful normal instrument
  performance. Selecting Special before it is ready redirects successful
  performance away from Attack or Defend and into much faster Hype generation,
  creating a deliberate short-term tradeoff. When Hype becomes full, the player
  returns to their prior Attack or Defend intent and receives an unmistakable
  Ready state. The ability never fires automatically.

  Selecting Special while ready arms the Signature Special. Its next ordinary
  scoring group becomes the activation performance, and the effect resolves at
  a valid musical boundary after that group. Every activation produces its
  reliable base effect; timing quality scales additional strength, duration, or
  utility. A single miss therefore cannot erase the whole charge. Resolution
  consumes all Hype and returns the player to their prior intent.

  Hype persists through downing and revival but resets between encounters. It
  has no separate cooldown and cannot store a second charge. This keeps the HUD
  and decision model compact while preventing hoarding multiple bursts.
- **Consequences:** Players can build around a meaningful personal ability
  without replacing instrument skill or adding another note grammar. Explicit
  arming preserves control over timing, while guaranteed base value prevents a
  hard-earned charge from feeling wasted after one ordinary error. The Special
  intent remains a real tradeoff because accelerated charging sacrifices current
  offense or protection.
- **Deferred:** Exact passive and redirected Hype rates, performance scaling,
  ability values, effect catalog, VFX, and balance across encounter lengths
  require ability prototyping and numeric tuning.

### GD-18: Player-initiated Band Calls

- **Approved direction:** The proposed earned, once-per-player Band Call model is
  approved as presented.
- **Specification:** Each player equips one Band Call before entering a shard,
  separate from their Signature Special. Meaningful successful performance fills
  readiness on the Band Call control itself rather than adding another
  persistent HUD meter. Each player may earn and initiate at most one Call per
  encounter. A shared band-wide lockout, initially targeted at roughly eight
  measures after a Call begins, prevents a larger group from chaining Calls
  continuously.

  Any active, non-downed player with a ready Call may request it when no other
  group action or recovery state blocks initiation. The Activity Map finds the
  earliest valid ensemble window within the Call's allowed delay. If no valid
  window exists, or the initiator becomes invalid before it begins, the request
  cancels and retains the charge.

  While the Call queues, every eligible bandmate receives a prominent invitation
  in the reserved cooperative-action area. It identifies the initiator, ability
  name, effect, and a beat-based countdown. The stable Join In control accepts.
  Acceptance is provisional: players continue their ordinary performance until
  the scheduled boundary and may withdraw. Moving, becoming downed, or otherwise
  becoming ineligible before the boundary removes only that participant without
  punishing the initiator or the rest of the band.

  At the boundary, the initiator's charge is spent and all remaining participants
  commit their own ordinary instrument material for roughly one or two measures.
  Their successful performance temporarily contributes to the Band Call instead
  of their individual Attack, Defend, or Special output. The initiator guarantees
  the Call's base effect. Each participant then adds an independently scored,
  accuracy-scaled share, so weak execution reduces only that share and never
  erases stronger contributions.

  In solo, active Order acolytes join the presentation and provide a small,
  predictable, capped fixed contribution. They never receive fabricated rhythm
  judgments or appear to outperform the human player.
- **Consequences:** Band Calls create deliberate cooperative moments while
  remaining useful when nobody accepts. Individual charges prevent strangers
  from spending a shared resource, and the shared lockout prevents full bands
  from overwhelming the encounter through sequential activations. Provisional
  acceptance keeps the invitation from disrupting active rhythm play.
- **Deferred:** Readiness thresholds, the eight-measure lockout, invitation lead
  time, active duration, participant scaling, acolyte contribution, cancellation
  edge cases, and the Band Call catalog require prototyping and numeric tuning.

### GD-19: Song-authored Crescendos

- **Approved direction:** The proposed candidate-window and three-tier Crescendo model
  is approved as presented.
- **Specification:** A standard encounter contains two to four authored,
  validated Crescendo candidate windows and guarantees exactly one activation on
  Easy, Normal, and Hard. The runtime may choose among candidates for musical
  fit and encounter variation. Easy may activate at most one additional unused
  candidate as a recovery opportunity when the band is substantially behind;
  Normal and Hard retain the normal authored budget.

  A candidate must provide strong playable coverage for the current roster,
  roughly two measures of readable reaction time, sustained instrument activity
  through the performance, and adequate separation from major boss attacks,
  recovery, silence or quiet transitions, and the finishing cadence. The
  Activity Map validates these constraints for each supported instrument,
  difficulty, and roster. If the selected candidate becomes invalid at runtime,
  the system chooses a later authored candidate. Crescendos are nonurgent and
  never use the universal beat fallback; authoring validation guarantees that a
  viable window remains available.

  A prominent musical and visual preview begins roughly two measures before the
  window and clearly identifies the authored effect. Participation costs no
  resource and is optional through the stable Join In control. Declining leaves
  ordinary play unchanged. At the boundary, each valid participant performs
  roughly two measures of their own instrument chart and temporarily routes
  normal Attack, Defend, or Special contribution into the Crescendo.

  Each result is normalized for that player's instrument and difficulty before
  the band total is assembled. Additive contribution produces three readable
  outcome tiers: **Echo**, **Crescendo**, and **Full Crescendo**. One weak player
  cannot reduce another player's earned share. Near-zero collective performance
  may fail to rise above Echo, but no ordinary individual mistake cancels the
  group result.

  Each candidate previews an encounter-authored effect. The normal starting
  pattern is a major Resolve burst paired with modest Ward reinforcement, while
  specific bosses may substitute a clearly communicated attack, defense,
  recovery, or positional effect. In solo, active Order acolytes join the
  presentation and add their established predictable fixed contribution without
  fabricated rhythm scores.
- **Consequences:** Crescendos become rare song-level ensemble spectacles rather
  than player-triggered Band Calls with a different name. Retaining one
  guaranteed activation across difficulties preserves the feature and the song's
  dramatic arc, while Easy's conditional extra opportunity supplies recovery
  without guaranteeing victory.
- **Deferred:** Candidate count exceptions for unusually short or long songs,
  exact lead time and duration, behind-state detection, outcome thresholds,
  effect values, acolyte contribution, and VFX require authoring trials and
  numeric tuning.

### GD-20: Solo Order acolytes

- **Approved direction:** The proposed Vanguard, Warden, and Herald solo-support model
  is approved as presented.
- **Specification:** Solo uses a fixed squad of three mechanically simple,
  visually distinct Order acolytes:
  - **Vanguard** adds a small amount of Resolve pressure after each successfully
    completed player scoring group. Its contribution stops short of breaking a
    layer; the human must supply the decisive successful Attack performance.
  - **Warden** provides modest Ward reinforcement on a visible authored cadence,
    initially targeted at once every eight measures. The pulse is previewed so
    it feels predictable rather than like hidden rescue AI.
  - **Herald** modestly improves the rate at which the human earns Band Call
    readiness and supplies the squad's small, capped, fixed contribution during
    Band Calls and Crescendos.

  Acolytes never receive instrument charts, timing judgments, combo counts, or
  performance and reward credit. Their support resolves on the shared musical
  clock. They occupy encounter-authored tactical locations and reposition
  automatically at musical boundaries. When a human enters an occupied
  location, formation offsets arrange everyone without swapping, blocking, or
  consuming gameplay capacity. Acolyte positions do not apply player
  risk/reward multipliers to their support output.

  Clearly telegraphed boss attacks may suppress an acolyte whose location is
  affected for roughly four measures. A compact portrait state and beat-based
  countdown identify which support function is temporarily unavailable. The
  acolyte recovers automatically at a clean boundary. Acolytes cannot be
  permanently downed, revived, individually equipped, or commanded, and the
  player never abandons the boss interaction for an escort objective.

  During the solo emergency recovery challenge, acolytes contribute presentation
  only. They cannot complete, score, shorten, or rescue the human's challenge.
- **Consequences:** The fixed trio makes solo feel populated and supported while
  leaving rhythm execution, intent, positioning, survival, layer breaks, and
  recovery under human control. Distinct functions make suppression legible
  without introducing companion management or simulated performers.
- **Deferred:** Successful-group threshold, pressure amount, Ward cadence and
  value, Herald readiness bonus, group-event contribution, suppression duration,
  automatic choreography, portraits, and effects require solo balance and
  readability tests.

### GD-21: Scaling from one to six humans

- **Approved direction:** The proposed scaling model is approved after restating
  “sublinear Resolve” in plain language: each additional human adds about 75% of
  the solo Resolve requirement rather than another full 100%.
- **Specification:** Initial Resolve targets by active human count are:

  | Humans | Resolve target in solo-player equivalents |
  |---:|---:|
  | 1 | 1.00 |
  | 2 | 1.75 |
  | 3 | 2.50 |
  | 4 | 3.25 |
  | 5 | 4.00 |
  | 6 | 4.75 |

  These are playtest starting relationships, not final values. A larger band
  faces more total Resolve, but each additional player creates some breathing
  room for movement, revival, uneven skill, and coordination. The song timeline,
  chart density, judgment windows, and individual incoming damage remain tied to
  the selected difficulty and never accelerate with population.

  Additional boss pressure comes through broader positional targeting rather
  than faster attacks. The starting cap for simultaneous individually targeted
  players is roughly half the active roster, never exceeding three. When several
  equivalent targets are valid, selection avoids repeatedly focusing the same
  player. Global attack shapes remain authored encounter events. Ward stays
  individual; the increased Resolve and target coverage offset the revival and
  coordination advantages available to larger bands.

  Duplicate instruments are unrestricted. Every performer receives their own
  chart for the selected instrument and difficulty, hears their own responsive
  layer, and is judged independently. Contribution is normalized by instrument,
  available musical material, and difficulty before being added to boss or group
  results. No party-composition requirement or duplicate penalty is permitted.

  Uneven performance never subtracts another player's earned contribution. The
  earlier placeholder negative cohesion modifier is replaced with a positive
  **Cohesion Bonus**, initially capped around 15%, for broad successful
  participation. Easy uses forgiving eligibility thresholds, Normal uses the
  intended thresholds, and Hard may require stronger participation without
  raising the cap. A weak or inactive performer provides little or no share and
  may leave some bonus unearned, but never applies negative output.

  Band Call and Crescendo tiers scale against the eligible roster. One expert can
  still produce meaningful group value, but cannot achieve the full-band top
  tier on behalf of five inactive players. This is missing positive contribution,
  not a punishment applied to the expert's result.

  Victory rewards are granted per player rather than divided from a shared pool.
  Party size does not change the core drop table. Selected difficulty, personal
  performance, and individually banked positional risk may affect quantity or
  quality. Solo acolytes and multiplayer scaling are tuned toward comparable
  completion rates at equivalent human skill, while co-op remains more
  expressive through revival and coordination.
- **Consequences:** Adding an average player makes the encounter more demanding
  overall but helpful to the band. Public players cannot damage teammates merely
  by struggling, duplicate instruments remain socially safe, and neither solo
  nor a full roster becomes the mandatory reward strategy.
- **Deferred:** Final Resolve curve, target caps and distribution, active-roster
  change handling, Cohesion Bonus thresholds, group-tier formulas, reward
  quantities, and comparative success-rate targets require population testing.

### GD-22: Cooperative session flow

- **Approved direction:** The proposed shard-to-staging-to-results cooperative flow is
  approved as presented.
- **Specification:** Activating a labeled phasing shard opens its encounter
  card with boss identity, shared encounter difficulty choices, reward preview,
  and **Solo**, **Current Party**, and **Public Band** options. Boss and
  difficulty are chosen before public matchmaking so nobody is placed into an
  encounter or difficulty they did not request.

  Public matchmaking is server-owned and has no player host. It matches by boss,
  difficulty, and appropriate connection region, targets three to six humans,
  and supports two-human encounters. If only two players are available after an
  initial target of roughly 45 seconds, both receive an explicit choice to
  launch together, continue waiting, or leave for solo without penalty.

  A Current Party may contain two to six humans. Its leader proposes the shard
  and difficulty, but every member confirms participation before staging. The
  leader cannot force a member into an encounter.

  The staging room allows each player to change instrument, equipment,
  Signature Special, Band Call, and prepared consumables. Duplicate instruments
  are explicitly marked as allowed rather than shown as a conflict. Boss and
  difficulty lock when public matching begins or the current party accepts the
  proposal. Individual loadouts and the roster lock at the final three-second
  deployment countdown. Before that lock, leaving carries no penalty.

  Once a viable public roster is present, a short ready timer begins. Unready
  players are never dragged into battle and may be returned to the hub or
  replaced when the timer expires. An ordinary encounter has no
  join-in-progress; the deployment roster establishes its initial population
  scaling. Disconnect and rejoin behavior is defined by GD-23.

  After victory or defeat, individual rewards are granted immediately. Results
  show personal rhythm performance, positional-risk earnings, mastery progress,
  and the band's collective result. **Retry Same Shard**, **Stay with Band**,
  and **Return to Hub** are individual choices. A majority cannot force another
  player into a follow-up encounter.

  Public players who choose Retry or Stay form the next rematch group, and
  matchmaking may refill empty places. Current parties remain together unless
  members leave. Retry returns the group to staging so loadouts may change before
  the next lock; it never bypasses readiness or force-starts combat.
- **Consequences:** The shard remains the clear encounter entry point, public
  players avoid host authority and binding votes, duplicate instruments remain
  socially safe, and rematching is quick without trapping anyone in a longer
  session. Lock timing is late enough for experimentation but early enough for
  stable population scaling and server setup.
- **Deferred:** Matchmaking regions and skill inputs, exact queue and ready
  timers, two-player consent presentation, party ownership details, staging-room
  layout, rematch refill timing, and service failure behavior require multiplayer
  prototyping and operational testing.

### GD-23: Communication and multiplayer failure cases

- **Approved direction:** The proposed ping, rejoin-grace, AFK, and structural
  anti-grief model is approved as presented.
- **Specification:** Core coordination never depends on free-form text or
  voice chat. Automatic visual, audio, text, and shape cues communicate boss
  attacks, targeting, movement, downing, revival progress, Band Calls,
  Crescendos, and phase changes. Players also receive rate-limited preset pings
  for **Move**, **Defend**, **Join Call**, **Revive**, and **Ready/Thanks**.
  Individual ping sources may be muted without hiding automatic encounter cues.

  When a connection disappears, the server preserves the last confirmed Ward,
  tactical location, Hype, Band Call charge, consumables, and downed state.
  Already committed boss impacts still resolve against that state so disconnects
  cannot dodge damage. The performer then becomes untargetable and contributes
  nothing until returning. Chart material during absence creates neither timing
  misses nor contribution; participation coverage records the absence so it
  cannot improve the player's performance rating.

  The player has an initial rejoin grace of roughly 45 seconds. A successful
  rejoin enters at the next safe musical boundary, using the previous location
  when valid or the nearest valid Middle location otherwise. A standing player
  receives the normal brief settling protection. Someone who disconnected while
  downed returns downed and must use the established revival rules. Spent
  resources remain spent.

  During grace, the missing player is excluded from boss targeting and current
  group-event eligibility. If grace expires, they leave the active roster. The
  current Resolve layer remains unchanged, preventing a departure exploit;
  unopened layers rescale for the smaller roster at the next layer boundary.
  Meaningful contribution already completed retains appropriate result and
  reward eligibility, so a late network failure does not erase legitimate play.

  AFK detection measures ignored eligible gameplay rather than wall-clock time
  during a rest or quiet passage. A private warning appears after roughly two
  consecutive ignored scoring groups. Continued nonparticipation marks the
  player inactive and removes them from targeting and group thresholds. A
  connected player may request one safe-boundary resume. Repeated inactivity can
  remove participation-based reward eligibility, but weak accuracy, choosing a
  safer position, declining an optional group action, or struggling with a boss
  never counts as griefing.

  Combat has no vote-kick. Server-owned rules already prohibit friendly fire,
  body blocking, spending another player's resources, forced follow-up votes,
  and negative contribution. Public sessions continue after any departure
  because they have no host. If a Current Party leader leaves, leadership
  transfers without ending combat. Standard Roblox block and report surfaces
  remain available after results, and individual pings can be muted.
- **Consequences:** A mobile connection failure has a fair recovery path without
  becoming an invulnerability technique. Participation rules address deliberate
  idling without equating poor performance with misconduct, while structural
  protections remove the most damaging grief vectors instead of relying on
  player policing.
- **Deferred:** Ping cooldown and localization, exact grace and AFK thresholds,
  reconnect transport, persistent server-state schema, active-roster rescaling,
  reward eligibility thresholds, telemetry, and moderation integration require
  network simulation and multiplayer testing.

### GD-24: Pre-battle loadout and combat stats

- **Approved direction:** The proposed first-release loadout is approved as a good
  starting point, with the explicit expectation that later progression can offer
  more build complexity.
- **Specification:** The first-release power-bearing loadout contains three
  readable gear slots:
  - **Instrument** establishes the performed instrument and provides a modest
    primary combat emphasis plus one distinctive trait. Every instrument
    category must offer offensive, defensive, and support-oriented variants so
    instrument choice never becomes a class lock.
  - **Ward Core** affects maximum Ward, Defend conversion, Ward reinforcement,
    or bounded recovery received.
  - **Resonator** affects Attack conversion, Hype generation, Signature potency,
    support effects, or Band Call readiness.

  Separate action slots contain one Signature Special, one Band Call, and two
  prepared consumables with limited encounter charges. The full inventory stays
  inaccessible during combat. Separate appearance slots contain stagewear,
  instrument finishes or skins, auras or performance effects, titles, and other
  profile presentation; those cosmetic choices carry no combat stats.

  Equipment may modify the combat consequence produced after normalized rhythm
  scoring, including Resolve damage, maximum Ward, defensive conversion, Ward
  restoration, support potency, Hype and Band Call readiness, Signature and Band
  Call effect strength, consumable potency, and small bounded recovery bonuses
  that never add another recovery attempt.

  Equipment must never modify:
  - judgment windows, calibration, or Perfect, Great, Good, and Miss definitions;
  - song speed, note density, chart contents, or phrase availability;
  - boss telegraph duration or authored reaction time;
  - dash timing, distance, settling, or invulnerability;
  - revive count or solo recovery attempts;
  - automatic note correction, autoplay, or immunity to encounter mechanics; or
  - positional risk/reward ratios or reward eligibility.

  Rhythm accuracy is therefore determined before gear modifies its combat
  result. A first-release item exposes one primary stat and one readable trait
  rather than a large collection of small random modifiers.

  The three gear slots define the starting player-facing surface, not a permanent
  ceiling. The underlying system must leave deliberate extension points for
  additional techniques, traits, sidegrades, sockets, set interactions, or
  advanced configuration after the core game is proven. Later depth should
  create more combinations and behavioral choices without adding mandatory
  combat buttons, changing rhythm judgments, or invalidating early equipment.
- **Consequences:** The launch loadout is compact enough to understand and edit
  on a phone, while the stat boundary protects musical skill and encounter
  readability. Separating action, power, and cosmetic slots makes future depth
  possible without turning the initial inventory into a wall of modifiers.
- **Deferred:** Stat names and ranges, item tiers, trait catalog, consumable
  charges, advanced slots, socket or set systems, power budgets, presets, and the
  schedule for post-launch complexity require specialization, economy, and
  progression testing.

### GD-25: Role specialization and long-term builds

- **Approved direction:** The underlying build structure sounds good, but the proposed
  Build Core and Technique names are unacceptable and must be replaced with
  substantially stronger, cooler player-facing names in a dedicated later pass.
- **Specification:** Every instrument draws from the same four universal
  functional categories:
  - offense, Momentum, and dangerous-position play;
  - Ward, Defend, mitigation, and revival support;
  - teammate support, Band Calls, Crescendos, and Cohesion interactions; and
  - Hype, Signature Specials, movement-triggered utility, and hybrids.

  **Discipline**, **Build Core**, and **Technique** remain internal working terms
  only. A build equips one major behavior-changing rule and three smaller
  supporting rules. Players may mix all four categories, so an instrument never
  becomes a class and a build is not forced into one pure role.

  The major rule changes how existing decisions resolve, such as rewarding
  Attack from dangerous positions, turning Defend into shared protection,
  altering the balance of a Band Call, or changing how a Signature Special uses
  Hype. Supporting rules create smaller conditional interactions with intent,
  positioning, movement completion, group actions, or resource generation. None
  of these rules changes note charts, judgments, timing windows, or the combat
  control set.

  New players initially choose clear role presets rather than confronting the
  full editor. Continued progression reveals the advanced combination surface.
  General progression and boss mastery unlock additional options; no option is
  exclusive to an instrument category. Respeccing is free outside active combat,
  and players initially receive three saved build presets for experimentation.

  Gear carries most direct power growth. Specialization choices primarily create
  play-style differences, conditional strengths, tradeoffs, and hybrids. Synergy
  effects use caps and a shared power budget so multiplicative combinations do
  not create one mandatory build. Long-term updates deepen the option library,
  traits, and interactions rather than adding more required battle buttons.
- **Consequences:** Beginners can start from recognizable roles while experienced
  players gain substantial combinatorial depth using the same instruments and
  controls. Free respec and cross-category mixing encourage experimentation
  instead of trapping players in an early choice.
- **Deferred:** All player-facing system labels and individual option names are
  explicitly unapproved. Final category count, major and supporting rule counts,
  unlock pace, presets, power budget, synergy caps, effect library, and editor
  presentation require prototyping, balance tests, and the dedicated naming pass.

### GD-26: Reward, item, upgrade, and crafting economy

- **Approved direction:** The proposed two-resource, fixed-item,
  random-drop-plus-deterministic-crafting economy is approved as presented.
- **Specification:** The economy contains only two broad earned resource types:
  - one general resource used for ordinary upgrades, basic crafting,
    consumables, and normal shop purchases; and
  - boss-specific materials used for that boss's Instruments, Ward Cores,
    Resonators, traits, and cosmetics.

  These descriptions are functional; their player-facing names remain subject
  to the required naming and tone pass.

  Victory always grants both resource types and boss mastery progress. It also
  provides a chance at a complete fixed-stat item or cosmetic from that boss's
  pool. The player's first victory supplies the associated campaign progression
  and Shattered Song fragment. Failure grants modest boss mastery and general
  resources according to meaningful participation, but does not grant signature
  boss material or the first-clear progression reward.

  Easy, Normal, and Hard share the same boss-themed combat-item pool. Normal and
  Hard may improve resource quantities, complete-item chances, or the starting
  upgrade rank of a dropped item rather than making an essential build exclusive
  to Hard. Personal performance and banked positional risk add bounded bonuses;
  they never turn the reward into all or nothing.

  Complete boss items may drop randomly, but every victory also advances a
  visible deterministic path. The starting target is roughly four to six
  victories to earn enough boss material to craft one chosen standard combat
  item from that boss. Item definitions use fixed primary stats and traits,
  never random affix ranges or endless rerolls. Duplicate items may be salvaged
  into useful boss material while their appearance remains permanently unlocked.

  Each item initially supports about three guaranteed upgrade ranks within its
  tier. An upgrade consumes the general resource plus relevant boss material.
  Crafting and upgrading show the exact result before confirmation and never
  fail, destroy an item, lower its stats, or impose a completion timer.
  Consumables provide a repeatable low-cost use for the general resource.

  The economy has no repair tax, death fee, paid luck, punitive respec cost,
  energy limit, or daily earning cap. Player trading remains outside the first
  release to reduce scams, alternate-account exploitation, and economy
  instability.
- **Consequences:** Random complete drops remain exciting without controlling
  access to a desired build. Every successful clear makes visible progress
  toward a chosen item, duplicate rewards retain value, and the small resource
  set stays understandable on a phone. Failure acknowledges genuine practice
  without becoming a more efficient farming method than victory.
- **Deferred:** Resource and item names, exact rewards, craft-clear target,
  upgrade ranks and costs, salvage return, inventory limits, boss-pool sizes,
  difficulty bonuses, performance caps, consumable prices, and trading's
  long-term status require economy simulation and exploit testing.

### GD-27: Mastery, power progression, campaign access, and replay

- **Approved direction:** The proposed campaign, general progression, and shared
  boss-mastery model is approved as presented.
- **Specification:** Long-term progress uses three distinct tracks whose final
  player-facing names remain open:

  1. **Campaign progression:** A player's first victory against a boss on any
     difficulty restores its Shattered Song fragment and unlocks the next
     campaign destination. Meaningful participation in a cooperative victory
     grants campaign credit even if the player is downed when the encounter
     ends. Easy and Normal start available; a Normal victory unlocks Hard for
     that boss.
  2. **General player progression:** Meaningful victories and failed attempts
     grant broad progress that unlocks game systems, saved builds,
     specialization options, and wider equipment choices. It supplies only
     limited direct statistical power so repeatedly losing cannot replace gear
     acquisition and successful boss clears. No choice is irreversible, and
     respeccing remains free outside combat.
  3. **Boss mastery:** Each boss begins with approximately ten visible mastery
     ranks. The track is shared across instruments so changing musical identity
     does not restart progress, while personal-best records remain separate for
     each instrument and difficulty. Victory and failure both grant
     participation-scaled mastery, but victory is substantially more efficient.

  Boss mastery may award lore, cosmetics, crafting recipes, specialization
  options, titles, and deterministic reward milestones. Signature combat items,
  boss-specific materials, campaign fragments, and first-clear progression still
  require victory. Completing mastery does not create an endless power ladder;
  replay continues through personal records and the boss's normal rewards.

  Current-tier bosses remain the best source of current-tier power. Older bosses
  retain unique traits, cosmetics, mastery rewards, recipes, and materials. Once
  the player reaches a later tier, an older item may be raised to that tier only
  through a recipe consuming mostly current-tier resources plus material from
  its original boss. Replaying old content can therefore preserve a favorite
  identity or sidegrade but cannot replace progress against current bosses.

  Recommended power may appear before an encounter, but it is never a mandatory
  gear-score gate. A skilled under-equipped player may attempt harder content.
  Failed attempts preserve personal-best improvements, boss mastery, and modest
  general resources. No required daily streak, energy system, expiring
  progression, or exclusive rotating reward is part of this model.
- **Consequences:** Story access remains understandable, practice is never empty,
  and instrument experimentation does not fragment mastery. Victory still owns
  the strongest progression rewards, current bosses remain relevant, and older
  encounters preserve identity and collection value without becoming the best
  power farm.
- **Deferred:** Progression-track names, rank counts, experience curves, unlock
  levels, mastery reward tables, meaningful-participation thresholds, personal-
  best categories, recommended-power calculation, item tier-raising costs, and
  post-mastery replay incentives require progression and economy testing.

### GD-28: Store surfaces and paid-item safeguards

- **Approved direction:** The proposed exact-equivalent, voluntary-store,
  no-rescue-purchases model is approved as presented.
- **Specification:** Purchases appear only through a clearly identified
  physical shop in the Order hub or a store menu the player voluntarily opens.
  The store becomes available only after onboarding and at least one completed
  encounter. It never presents purchase prompts during combat, downing,
  recovery, defeat, results, or immediate retry. A hub surface may indicate
  genuinely new stock once, but closing or declining the store stops repeated
  prompts.

  Every stat-bearing paid item has an exact functional equivalent earnable
  through normal play at the same campaign tier. A paid version may use an
  exclusive appearance, but never an exclusive trait, effect, or larger stat
  budget. Its page includes an **Earn Through Play** route identifying the boss,
  recipe, mastery reward, or progression source for that equivalent.

  Purchase grants the item at the player's currently unlocked tier. It never
  jumps ahead of campaign progression or scales automatically through future
  tiers. Raising it later consumes the same earned resources and follows the
  same upgrade rules as non-paid equipment. Paid items cannot alter rhythm
  judgments, chart content, telegraphs, movement fairness, recovery counts,
  rewards, or matchmaking desirability.

  Before confirmation, the store shows the exact Robux price, item tier, primary
  stat, trait, appearance, upgrade state, and earnable functional equivalent. It
  prevents accidental duplicate purchases and makes owned items and purchase
  history easy to inspect. Fake discounts, false scarcity, restarting
  countdowns, and ambiguous rarity claims are prohibited. Genuine seasonal
  cosmetics may show truthful end dates.

  Launch monetization sells only direct, deterministic cosmetics and permanent
  equipment. It excludes paid loot boxes, random bundles, gacha, prize wheels,
  paid luck, random upgrades, revives, recovery attempts, Ward refills, Hype,
  Band Call charges, consumables, temporary boosts, resources, boss materials,
  drop modifiers, bosses, songs, campaign access, subscriptions, battle passes,
  convenience products, and progression skips.

  Every paid-equipment record must identify its earnable equivalent. An
  automated power-budget validator rejects any paid item exceeding the normal
  same-tier budget or using a paid-only functional trait. Design and economy
  reviewers compare the paid and earned builds before publication, and both run
  through the same combat-balance tests. A future monetization category requires
  a new explicit review rather than inheriting approval from this launch model.
- **Consequences:** Purchases remain deterministic, inspectable, and separated
  from moments of vulnerability. Paid equipment may accelerate access to a
  preferred current-tier sidegrade or appearance, but it cannot buy timing skill,
  campaign advancement, rescue, exclusive mechanics, or superior tier power.
- **Deferred:** Shop and button names, layout, item preview, exact-equivalent data
  schema, purchase receipt and restoration flows, Roblox platform compliance,
  seasonal presentation, validator thresholds, review ownership, and any
  post-launch category require product, economy, legal, and safety review.

### GD-29: First-time onboarding

- **Approved direction:** The proposed short practice followed by contextual first-boss
  teaching is approved as presented.
- **Specification:** Before the first full encounter, the player receives a
  four-to-six-minute, checkpointed, replayable Order practice:
  1. **Setup:** choose an unlocked starter instrument, show device-specific
     controls and comfort settings, and offer guided but skippable calibration.
  2. **Perform:** use two short musical phrases to teach the right-to-left staff,
     strike line, three rhythm inputs, taps, holds, repeats, rests, and the four
     judgments.
  3. **Attack:** route successful performance through Attack and visibly damage
     a practice Resolve layer.
  4. **Defend:** answer a harmless telegraphed attack with Defend and see Ward
     absorb the result.
  5. **Move:** use directional dashes among positions, avoid one clear attack,
     and preview the Near, Middle, and Rear risk relationship.
  6. **Combine:** complete a short sequence containing performance, one intent
     decision, and one reposition.

  Each module shows one instruction at a time with current-device glyphs and
  minimal text. A failed instruction repeats safely without death, shame, or a
  minimum grade; Perfect is never required. Progress saves after every section.
  Completing or explicitly skipping practice unlocks public matchmaking, and
  practice remains accessible from the hub.

  The first boss teaches advanced systems in their real context. Arrival
  reinforces ordinary Attack performance. First Clash reinforces movement,
  Defend, and Ward. Escalation introduces Hype and the Signature Special. A
  guaranteed Crescendo teaches Join In and group contribution. The first
  relevant down teaches solo recovery or cooperative revival with generous
  warning. Band Calls and consumables receive short prompts when first
  available rather than additional front-loaded lessons. These prompts respect
  the musical timeline and never pause or rewind the song.

  Experienced players may skip practice after confirming the control reference.
  Calibration remains independently accessible, contextual teaching may be
  disabled, and no important instruction becomes permanently missable. Under
  GD-28, the store remains unavailable until onboarding and one encounter are
  complete.
- **Consequences:** Players reach the real boss quickly after demonstrating the
  minimum interaction vocabulary. Advanced systems gain meaning inside combat
  instead of becoming a long pre-play lecture, while experienced rhythm players
  retain a fast route into the game.
- **Deferred:** Exact duration, training music, instrument-choice presentation,
  module success signals, skip wording, public-unlock handling, first-boss cue
  script, prompt suppression, and practice access require target-age usability
  testing.

### GD-30: Order hub functions and navigation

- **Approved direction:** The proposed physical anchors and optional fast menus are
  approved, but the shard area is not a flat circular selection ring. It should
  be a tiered, stair-stepped environment whose locked campaign levels are visible
  but blocked. Glowing broken pieces of glass pierce swirling portals in varied
  orientations, colors, and effects, forming a kind of beautiful chaos.
- **Specification:** The phasing-shard structure is the Order hub's dominant
  physical landmark. A broad staircase, terraces, broken ascents, or a comparable
  vertical progression language arranges shards by campaign tier. Higher levels
  remain dramatically visible from below but are physically blocked until the
  player reaches them. Unlocking a campaign tier opens its ascent and provides a
  fast repeat route to that landing so hierarchy does not become repeated travel
  friction.

  Each encounter shard is a glowing, broken-glass form piercing reality through
  a swirling portal. Many shards angle toward the hub center, but variation is
  intentional: forms may stick upward from the floor, descend from above, lean
  through walls, hang inside suspended fractures, or break the general radial
  pattern. Boss and arena identity determines color, portal motion, particles,
  distortion, silhouette, and sound. The overall composition seeks controlled
  **beautiful chaos** rather than a tidy mission board.

  Readability constrains that spectacle. Every shard has a stable interaction
  footing or reach point, persistent label, clear unlocked or locked shape, and
  sufficient quiet space to identify its boss, difficulty access, and state.
  Color is never the only distinction. Locked shards cannot be mistaken for
  Robux gates, and interacting with an unlocked shard deliberately opens the
  GD-22 encounter card rather than triggering through accidental proximity.

  Essential supporting functions use distinct in-world anchors around or beneath
  the shard ascent:
  - a practice area for onboarding replay, calibration, controls, instrument
    preview, and song practice;
  - a workshop for loadouts, upgrades, salvage, and eventual crafting;
  - an archive or story area for restored fragments, campaign history, boss
    mastery, lore, records, and earned rewards;
  - a social commons for parties, emotes, and non-scored musical interaction; and
  - the voluntary store for cosmetics and permanent equipment under GD-28.

  These are functional descriptions, not approved final names. Each physical
  anchor opens a focused phone-friendly menu rather than requiring detailed
  inventory manipulation in the world. Loadout, inventory, party controls,
  mastery, settings, and queue state remain available through a compact menu
  from anywhere. The store never becomes an unsolicited shortcut or prompt.

  First-time arrival guides the player toward practice and the first unlocked
  shard. Returning players appear on the fast central route or their highest
  unlocked landing, within a few seconds of an available shard and essential
  functions. Result-screen Retry continues to bypass the hub entirely. Public
  matchmaking may continue while the player moves through the hub and uses menus
  that do not conflict with the locked encounter setup.

  NPC dialogue, lore, social gathering, and non-scored musical interactions are
  optional. No upgrade or story step requires carrying materials among NPCs or
  repeating errands. The workshop may leave believable physical room for future
  crafting without presenting a useless locked interface at launch.

  Campaign progress visibly restores the hub. New shard tiers open, recovered
  fragments alter the central structure, and architecture, music, lighting,
  portal activity, NPC population, and signs of repair evolve. Core paths and
  landmarks remain stable so growth increases wonder without damaging navigation.
- **Consequences:** The shard field becomes a memorable expression of campaign
  scale and supernatural instability rather than a decorated menu. Physical
  progression and varied portals provide spectacle, while fixed activation
  footing, labels, shortcuts, and stable supporting anchors preserve usability.
- **Deferred:** Final hub and area names, shard visualization, exact tier
  geometry, locked barriers, ceiling and wall use, shortcut treatment, portal
  VFX and audio, accessibility routes, station layout, menu entry, NPC density,
  performance budgets, and campaign-state variants require the owner's future
  visualization plus environment and usability prototypes.

### GD-31: First three bosses and songs

- **Approved direction:** Heaven's Edge and Blackened Crown are processing fixtures
  or substantial-revision candidates, not presumed launch songs. Their current
  versions lack the required intensity and dynamic movement, and new song
  candidates can be generated readily.
- **Specification:** Define all three encounter and musical briefs
  before selecting songs. Generate at least two or three new full-stem candidates
  per brief, score them against the brief, and design the final boss, arena,
  chart, and event placement around the selected song's actual structure.

#### Approved brief 1: Core-combat revelation

- **Music:** Target roughly 3¼ to 4¼ minutes. Establish dark, dangerous,
  cinematic EDM and K-pop energy immediately rather than spending a long time in
  atmosphere. Use genuine dynamic contrast to define an approachable but exciting
  Arrival, an unmistakable First Clash hook or drop, a contrasting Escalation,
  the largest sustained peak during Climax, and a distinct chartable Finishing
  Cadence rather than a fade-out.
- **Arrangement and data:** Supply clean source stems for every eventual launch
  instrument. Include active but varied passages, real solos and rests, two to
  four Crescendo candidates, and multiple recovery and event windows that test
  Activity Map generation without creating dead play.
- **Gameplay:** Use the regular nine-location layout. Reinforce Attack, Defend,
  Ward, movement, risk tiers, Hype, the Signature Special, recovery, and one
  guaranteed Crescendo. Focus on lateral sweeps, targeted strikes, and one
  arena-wide pulse. Reserve position destruction, persistent hazard
  combinations, and demanding multi-part dash traps for later encounters.
- **Boss and story:** Present a visually singular, enormous spiritual monster
  empowered by a fragment. Its dissonance visibly damages the region. Destroying
  it releases the first fragment and proves that recovery is possible without
  revealing the ancient conspiracy yet. Final names and appearance follow song
  selection and the required naming process.
- **Rewards:** First victory guarantees a useful starter choice representing the
  Instrument, Ward Core, or Resonator function. The broader pool supports
  balanced early builds, and mastery introduces the first boss-specific
  cosmetic, lore, and specialization options.
- **Rejection rules:** Reject candidates with flat intensity, quiet sections that
  become dead gameplay, weak launch-instrument coverage, a climax that cannot
  support gameplay, or an ending without a decisive final performance.

#### Approved brief 2: Tactical commitment

- **Music:** Target roughly 3½ to 5 minutes. Increase rhythmic aggression,
  tension, and dynamic volatility beyond the first encounter. Use a controlled
  but threatening Arrival, a forceful recurring First Clash hook, and an active
  Escalation shift such as half-time pressure, stripped percussion, or a tense
  breakdown. Build Climax around two linked pressure peaks separated by a short
  deceptive release, then end with a hard, unmistakable Finishing Cadence.
- **Arrangement and data:** Supply full clean launch-instrument stems, active
  solos and rests, two to four Crescendo candidates, and deliberate Band Call,
  recovery, and two-part attack windows.
- **Gameplay:** Retain the familiar nine-location structure while temporarily
  corrupting, endangering, or disabling only one or two locations at a time.
  Introduce persistent hazards and specific cover interactions. Clearly
  announced two-part sequences test moving immediately versus preserving the
  dash charge, with Defend, cover, prepared abilities, or knowingly accepted
  Ward damage remaining valid alternatives. Band Calls gain more tactical value,
  and one Crescendo remains guaranteed. Reserve larger arena transformation for
  the third encounter.
- **Boss and story:** Present a fragment holder whose existing obsession with
  command, hierarchy, or control has been monstrously amplified. Strict
  ceremonial arena geometry cracks, blackens, and becomes disordered over the
  song. The holder may be a regional ruler or spiritual tyrant rather than an
  Order conspirator. Victory shows that fragments amplify specific existing
  discord and raises suspicion without revealing the false history.
- **Rewards:** Emphasize Ward, Defend, dangerous-position, Band Call, and
  tactical-hybrid sidegrades. Traits may respond to safe arrival, defending after
  movement, holding an exposed location, or coordinating a Call, but never alter
  movement recovery, dash charges, or invulnerability.
- **Rejection rules:** Reject candidates without clear linked pressure peaks,
  strong dynamic contrast, fair two-part attack boundaries, full instrument
  coverage, or active musical material during tactical sections.

#### Approved brief 3: Full-system revelation

- **Music:** Target roughly 4 to 5½ minutes. Deliver the launch lineup's greatest
  but still structured intensity and widest dynamic range. Arrival immediately
  distinguishes the threat, First Clash establishes a central motif, Escalation
  substantially transforms arrangement or rhythmic pressure, and Climax builds
  through multiple connected peaks rather than repeating one unchanged chorus.
  End with the strongest and most decisive Finishing Cadence of the three songs.
- **Arrangement and data:** Supply clean stems for every launch instrument,
  genuine solos, rests, handoffs, ensemble peaks, two to four Crescendo
  candidates, Band Call opportunities, and urgent and nonurgent event windows.
- **Gameplay:** Begin with the familiar nine-location language, then add, remove,
  reconnect, elevate, or corrupt positions at authored phase boundaries while
  preserving readable routes. Combine learned sweeps, strikes, pulses,
  persistent hazards, cover, and announced multi-part dash decisions. Attack,
  Defend, movement recovery, dangerous positions, Signature Specials, Band
  Calls, Crescendos, revival, and role builds all remain useful. Add no new core
  control; difficulty comes from combining mastered systems under strict
  validation against impossible movement, silence, holds, and event overlap.
- **Boss and story:** Use a damaged former Order performance site or another
  arena directly linked to the Shattering. The boss is the first fragment holder
  personally connected to the ancient betrayal, potentially a surviving
  conspirator transformed by prolonged fragment use. Victory reveals credible
  evidence that celebrated Order history is false but does not reveal the
  mastermind's full identity or resolve the vanished novice mystery. Releasing
  the fragment opens the next visible shard tier and substantially restores the
  hub.
- **Rewards:** Complete the launch's Hype, Signature, Band Call, group-support,
  and hybrid-build possibilities. Include the first advanced cross-category
  options, a major mastery cosmetic, and recovered historical evidence while
  preserving normal tier ceilings and deterministic acquisition.
- **Production requirement:** After the separately deferred pipeline upgrade is
  implemented, this encounter is its first full production qualification.
  Section detection, Activity Maps, Crescendo discovery, ensemble coverage,
  intensity analysis, and validators must support it without extensive one-off
  exceptions. No pipeline implementation is part of this interview decision.
- **Rejection rules:** Reject repetitive peaks, inadequate dynamic contrast,
  incomplete roster coverage, weak endings, or musical structures that force
  mechanics against the song.

- **Consequences:** Launch content is selected against purpose-built briefs rather
  than sunk assets. The three encounters progress from core-combat revelation,
  through tactical commitment, to complete system and story integration, while
  every song must earn its place through intensity, dynamics, coverage, and
  usable structure.
- **Deferred:** Selected songs, final launch instrument roster, exact boss and
  region designs, names, generation prompts, candidate score thresholds,
  pipeline implementation, and production approval remain downstream content
  and technical work.

### GD-32: Results, rewards, retry, and post-battle flow

- **Approved direction:** The proposed immediate-summary plus optional-detail results
  model is approved as presented.
- **Specification:** After a brief, skippable victory or defeat presentation,
  a phone-first immediate summary shows:
  - **Outcome:** Victory or Defeat;
  - **exact reason:** fragment recovered, all humans down, Ward broken, Resolve
    remaining at the ending, or Finishing Cadence missed;
  - a **personal performance rating** distinct from the binary outcome;
  - the most important already-granted rewards and unlocks; and
  - one large, obvious next action.

  The primary action adapts to context. A first victory prioritizes **Continue
  Story**. A repeat victory prioritizes **Retry Same Shard** or **Stay with
  Band**. Defeat prioritizes **Retry Same Shard**. **Loadout and Upgrades** and
  **Return to Hub** remain visible secondary choices. Public rematch decisions
  remain individual and never become a binding vote.

  Optional phone-friendly tabs or expanding sections provide deeper evidence:
  1. **Performance:** Perfect, Great, Good, and Miss distribution; early-versus-
     late tendency; hold completion; participation coverage, including
     connection absence; and personal-best comparison by instrument and
     difficulty.
  2. **Combat:** Attack, Defend, and Special contribution; Resolve damage and
     Momentum; Ward loss, reinforcement, and restoration; attacks avoided,
     defended, or absorbed; position use; dangerous-position performance; and
     banked Risk Bonus.
  3. **Band:** Band Call and Crescendo participation, revival help, personal
     group contribution, earned Cohesion Bonus, and the collective result without
     ranking public players. Solo acolyte output is identified as fixed NPC
     support rather than performance.
  4. **Progress:** General and boss-specific resources, item drops, unlocked
     appearances, boss mastery, campaign restoration, deterministic crafting
     progress, and newly available builds, recipes, difficulties, story, or hub
     tiers.

  The system offers at most two private, evidence-based improvement suggestions,
  such as a consistent late timing trend, one repeatedly missed telegraph, or
  unbanked risk lost through frequent movement. It compares the player with their
  own previous results rather than strangers. It never shames misses, publishes a
  damage leaderboard, or labels one performer as the cause of defeat.

  Rewards are granted without requiring a separate claim for each item. Reward
  and progress animations are skippable and cannot delay Retry. No store offer,
  Robux button, paid-equipment comparison, or monetized rescue appears anywhere
  in victory, defeat, rewards, or immediate retry.
- **Consequences:** The first screen answers what happened, what was earned, and
  what to do next without overwhelming the player. Optional evidence supports
  learning and build decisions while avoiding public comparison systems that
  encourage blame.
- **Deferred:** Presentation duration, performance-rating formula, tab layout,
  exact statistics, reason priority, suggestion rules, personal-best comparison,
  animation timing, action ordering, rematch status, and data instrumentation
  require mobile usability and multiplayer testing.

### GD-33: Accessibility, comfort, and age-appropriate safety

- **Approved direction:** The proposed launch accessibility and safety baseline is
  approved as presented.
- **Specification:** Required input and rhythm access includes:
  - guided and manual audio/visual calibration saved per device profile;
  - keyboard and gamepad remapping where the platform supports it;
  - touch handedness plus pad size, spacing, position, and opacity controls;
  - adjustable staff, note, and interface scale;
  - adjustable visual scroll speed without changing musical timestamps;
  - a persistent device-specific control reference and replayable practice; and
  - **Hold Assist**, which still judges the authored initial press but removes
    the need to physically maintain the control until the endpoint.

  Visual access and comfort requires every note, intent, attack, position, and
  state to use shape, label, placement, or motion in addition to color. UI-scale,
  high-contrast, and color-vision options preserve semantic distinctions.
  Flashing, bloom, particles, camera shake, camera motion, impact zoom, and
  haptics reduce independently, and no essential cue disappears when they do.
  Default presentation is already restrained against rapid full-screen flashing
  and excessive motion rather than depending on players to discover a safety
  option.

  Audio and language access includes independent levels for master, song, local
  instrument, timing and boss cues, voices, combat effects, crowd, and ambience;
  dynamic-range presets including a quieter compressed mode; mono-compatible
  critical cues; captions and subtitles with speaker or sound-source identity;
  adjustable text size and background; and clear age-appropriate language.
  Critical information always combines audio with visual or haptic reinforcement.

  Tutorials and contextual help remain replayable. Prompt duration outside fixed
  musical timing may be adjusted. Calibration and improvement suggestions are
  private. Solo pause freezes the encounter and song, then resumes with a visible
  and audible beat countdown. Cooperative play cannot pause the shared song and
  explains that limitation before entry. Accessibility settings are available
  before onboarding and from every safe menu.

  Core social play requires neither voice nor unrestricted text. Safe preset
  pings remain the default coordination tool, with obvious individual mute,
  block, and report access. Any platform communication uses the platform's
  filtering and age controls. The game publishes no accessibility label, public
  damage rank, or defeat blame.

  Bosses and story presentation avoid gore, realistic suffering, profanity,
  sexual content, nightmare-oriented horror, and direct real-world religious
  preaching. Existing low-pressure purchase safeguards remain binding.

  Accessibility assists are independent of Easy, Normal, and Hard. Players may
  combine them freely without reduced rewards, mastery, campaign credit,
  matchmaking access, or public identification.
- **Consequences:** Players can adapt input, sensory intensity, audio, language,
  and teaching without being pushed into an easier encounter or penalized for
  access needs. Safe defaults protect players who never open settings, while
  multimodal cues preserve gameplay when individual effects are reduced.
- **Deferred:** Final option names, control ranges, Hold Assist scoring detail,
  supported remapping surface, calibration persistence, contrast palettes,
  flash and motion budgets, caption format, audio mix ranges, pause networking,
  localization, platform-policy compliance, and accessibility testing require
  dedicated specifications and representative user testing.

### GD-34: Observable playtest readiness gates

- **Approved direction:** The proposed implementation-commitment, content-complete
  release, and structural-redesign gates are approved as presented.
- **Specification:** Before committing to full native production, the smallest
  complete encounter prototype passes at least two rounds of new target-age
  testing. Each round initially contains roughly 8–12 players ages 10–14, with
  most play occurring on representative phones and tablets. Evidence must show:
  - at least 80% complete or deliberately skip onboarding without coaching;
  - at least 80% can explain and use the three rhythm inputs, Attack, Defend, and
    positional movement;
  - at least 75% correctly identify a major boss telegraph and choose a viable
    response;
  - players look away from the staff often enough to describe boss behavior;
  - players understand that musical performance produces combat outcomes;
  - at least half voluntarily retry, continue, change difficulty, or experiment
    with a build; and
  - no observed attack is impossible because of chart activity, movement
    recovery, position state, or unreadable targeting.

  Failure at this stage changes the core design before larger content production.
  These percentages are initial evidence thresholds subject to formal research
  planning, not substitutes for observing why players succeed or fail.

  Content-complete release testing includes at least 30 new target-age players
  plus returning players, segmented across phone, tablet, desktop, gamepad, Easy,
  Normal, Hard, every launch instrument, solo, and two-, three-, and six-human
  groups. Initial design-release gates require:
  - at least 85% identify outcome, exact reason, important reward, and next action
    within ten seconds of results appearing;
  - at least 80% recognize their instrument's responsive audio and critical boss
    cues on ordinary phone speakers or headphones;
  - at least 80% respond correctly to established major telegraphs after seeing
    each pattern once;
  - zero validated impossible attack combinations;
  - no launch instrument lacks required playable activity or event windows;
  - Easy is observably easier without reducing maximum combat contribution;
  - solo and similarly skilled co-op completion rates remain within roughly 15
    percentage points;
  - at least 75% of invited players understand how to accept or decline group
    actions without voice chat;
  - at least half voluntarily choose another meaningful action after an encounter;
    and
  - accessibility combinations preserve every essential cue and never alter
    rewards or public status.

  Evidence combines direct observation, a player's short explanation, input and
  event telemetry, and voluntary next actions rather than survey satisfaction
  alone. Results remain segmented by device, difficulty, instrument, solo or
  co-op, and population so an overall average cannot hide a failing surface.

  The following patterns mandate structural redesign rather than small numeric
  tuning:
  - players watch only the staff and cannot describe the boss;
  - damage repeatedly feels unavoidable or unexplained;
  - movement recovery surprises players instead of creating deliberate choice;
  - Defend is ignored because its value is unclear;
  - quiet passages feel empty;
  - Easy feels weaker because it has fewer notes;
  - any instrument repeatedly lacks meaningful activity;
  - weak players feel blamed or stronger players feel punished by teammates;
  - group invitations are routinely missed;
  - players cannot explain why they won or lost; or
  - the third encounter requires numerous one-off pipeline exceptions.

  A material design change must pass two consecutive testing rounds before its
  evidence is considered stable. Technical performance, networking, persistence,
  security, anti-cheat, data safety, and platform-compliance gates remain
  separately required before release.
- **Consequences:** Readiness depends on observed comprehension, fairness,
  attention, coordination, and voluntary behavior rather than a green build or a
  favorable average score. Structural problems return to design, while numeric
  tuning remains appropriate only when players understand the system and its
  outcomes.
- **Deferred:** Formal research protocol, recruitment, consent and safeguarding,
  device matrix, instrumentation, statistical confidence, exact thresholds,
  technical release gates, issue severity, and sign-off ownership require a
  dedicated playtest and release-readiness plan.

## 5. Required follow-up and deferred specifications

### Required product follow-up

- **Required naming and tone pass:** Replace the working Discipline, Build Core,
  and Technique terminology and every individual option name with a coherent,
  memorable music-warrior vocabulary. The mechanics may be prototyped under
  functional internal labels, but the current names must not ship.
- **Deferred song-processing pipeline upgrade:** Extend the maintained pipeline
  to generate or export structural section markers, candidate mappings for the
  five encounter functions, Crescendo windows, per-instrument and per-difficulty
  Activity Maps, current-roster ensemble coverage, dynamic-intensity features,
  and validation metadata. This is a flagged future technical task; the owner
  explicitly does not want it implemented during this interview turn.

### Deferred technical specifications

- Rhythm chart schema, authoring-tool architecture, validators, and export format.
- Roblox client/server authority, networking, anti-cheat, persistence, and
  analytics event schema.
- UI component specification, responsive layouts, safe-area measurements, and
  input maps.
- Economy tables, item catalogs, numeric balance sheets, and drop-rate tables.
- Final boss/song production specifications, narrative bible, production
  schedule, and asset manifests.

## 6. Approval, validation, and change control

The bounded owner interview resolved GD-01 through GD-34 on 2026-08-18. The
question plan is finite and complete. Deferred technical documents may refine
schemas, values, implementation, production, and test procedures, but they do
not reopen settled player-facing behavior by themselves.

Every implementation handoff must distinguish:

- approved behavioral rules in this document;
- numeric hypotheses that require tuning;
- explicitly deferred systems or technical specifications; and
- working terminology that still requires the mandated naming and tone pass.

A material change to rhythm interaction, combat fairness, player-count scaling,
accessibility treatment, progression integrity, monetization safeguards, or
content readiness must update this document and cite the superseded decision.
