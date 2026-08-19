# Bands Battle Game Vision Interview Checklist

- **Status:** Vision v1 interview complete
- **Baseline:** 2026-08-14
- **Parent document:** [`GAME_VISION.md`](GAME_VISION.md)

## Purpose

This checklist records the owner decisions behind the first stable version of
`GAME_VISION.md` and governs any future additions to the bounded vision scope. It
exists to keep vision discovery thorough without making it endless.

The checklist records product-vision decisions only. Detailed mechanics,
numerical tuning, interface layouts, technical architecture, asset production,
and content implementation belong in `GAME_DESIGN.md`, `ART_DIRECTION.md`,
a future Roblox-native design document, or an OpenSpec change. Files under
`roblox/web/` describe a retired prototype and are not active design authority.

## How to use this checklist

- `[x]` means the owner answered the question and the decision is captured in
  `GAME_VISION.md`.
- `[ ]` means owner input is still needed.
- Ask one top-level unchecked question at a time. Use only the subquestions that
  help clarify that answer; they are prompts, not separate mandatory interviews.
- A question may be closed by an explicit decision to defer it. Record the
  destination and reason beside the question, capture that boundary in
  `GAME_VISION.md`, and then mark it complete.
- The vision interview is complete when every question in **Required to close
  Vision v1** is checked or explicitly deferred. The downstream backlog does not
  block completion.

At the 2026-08-14 baseline, 13 foundation questions were resolved and 12 vision
questions remained. The bounded scope contains 25 `GV-` questions in total.
Current progress is 25 resolved and 0 remaining.

## Rules for adding a new question

A new required question may be added only when its answer could materially
change at least one of the following:

- the target player or core player promise;
- the central battle loop or solo/co-op model;
- the intended platform or product boundary;
- progression, monetization, or content strategy;
- the world's tone, themes, or age appropriateness;
- a development priority that would be expensive to reverse later.

Before adding one, check that it is not a subquestion of an existing item. If it
is about values, timing windows, schemas, control placement, balance, assets, or
edge-case behavior, route it to the downstream backlog instead. Any genuinely
new vision question must receive the next `GV-` identifier and a one-sentence
rationale in the change log at the end of this file.

## Resolved foundation

- [x] **GV-01: What game are we making, and what should playing it feel like?**
  A supernatural rhythm-controlled boss-combat game blended with a musical
  action RPG. Musical performance visibly and audibly causes combat.
- [x] **GV-02: Who is it for, and how challenging may it become?** Primary
  audience is approximately ages 10 to 14. Multiple difficulty levels support
  approachable play through moderately hard mastery.
- [x] **GV-03: Where should player attention live during combat?** On the boss,
  performers, arena, and compact rhythm cues. The intended product has no
  permanent song-wide note highway, but active performance passages may use a
  compact right-to-left moving staff with a fixed strike line.
- [x] **GV-04: What authors encounter timing?** The song is the master clock and
  structure for phrases, boss behavior, movement windows, breathing room, and
  cooperative moments.
- [x] **GV-05: What tactical choice does arena position create?** Players dash
  among distinct performance positions. Closer positions are more dangerous and
  more rewarding; cover and active defense answer major attacks.
- [x] **GV-06: How are rhythm phrases assigned in the initial multiplayer
  version?** The song chart and selected instrument determine all available
  playable notes and their phrase grouping. Phrases organize performance rather
  than grant isolated permission to play. Duplicate instruments are
  unrestricted, and dropouts do not generate fake instrument parts.
- [x] **GV-07: How does rhythm play work during instrument dropouts?** Stationary
  players may press **Join In** for sparse BPM-derived beat actions. Moving exits
  that state. Normal instrument performance passages enroll players
  automatically with about two seconds of advance warning, and consecutive
  phrases may chain without forced downtime.
- [x] **GV-08: What is the high-level co-op model?** Solo and co-op are both core,
  with co-op as the fullest expression. Bands are approximately three to six
  players, support friends or public matchmaking, allow duplicate instruments,
  and lock the roster when an encounter begins.
- [x] **GV-09: How forgiving are group performances?** Each player's execution
  primarily changes that player's contribution. Broad successful participation
  may earn a capped, difficulty-adjusted positive Cohesion Bonus, while weak or
  absent play never subtracts another player's earned value. One expert cannot
  supply a full-band top tier for an otherwise inactive roster.
- [x] **GV-10: How do avatars, instruments, and roles relate?** Players keep their
  Roblox avatar and layer Order clothing, instruments, equipment, and effects on
  it. Instruments do not lock roles; equipment and skill trees create flexible
  builds.
- [x] **GV-11: What is the high-level failure philosophy?** Failure accumulates
  through repeated rhythm mistakes and combat damage. Co-op supports downing and
  revival; solo receives one limited last-chance opportunity before defeat.
- [x] **GV-12: What sustains long-term play?** Power growth, new abilities, song
  and boss mastery, boss drops, lightweight crafting, story unlocks, replayable
  bosses, and occasional scheduled or semi-random encounters.
- [x] **GV-13: What is the world and story foundation?** New recruits rebuild a
  diminished musical Order, destroy mostly spiritual or demon-like bosses, and
  restore a Shattered Song while uncovering an ancient infiltration, coup, and
  cover-up. Themes are Christian-inspired, original, and broadly approachable.

## Required to close Vision v1

- [x] **GV-14: How does a boss encounter reach victory or defeat in relation to
  the song?**
  Normal encounters are finite and use sequential resistance layers tied to
  scheduled beat- or time-based windows; they do not require automatic detection
  of named song sections. Layers must be broken in order, and a late break leaves
  less time for later layers. Early overflow becomes visible Momentum or bonus
  stacks that strengthen the next layer, so successful play is never wasted. At
  the final cadence, breaking all required layers and succeeding at the existing
  designated final phrase or phrases destroys the boss; otherwise the boss
  withstands the attempt and the run fails. Randomness may create bounded flavor
  or bonus variation but does not secretly decide the core result. Special
  bosses may use finite multi-song raids. A generated near-miss coda is deferred
  beyond the MVP.

- [x] **GV-15: What makes solo play structurally complete?** Solo uses the same
  arena geometry and positions as co-op. Order acolytes share tactical locations
  through formation offsets, automatically arrange around the human, consume no
  gameplay capacity, and never block the player's movement or risk/reward choice.
  They provide predictable passive pressure and authored support but do not play
  rhythm phrases, receive timing judgments, or break resistance without the
  player's performance. For MVP they cannot be permanently downed, though boss
  attacks may knock them away or temporarily disable support. The player alone
  performs solo group-ability phrases while acolytes join the presentation and
  supply a fixed contribution. Numeric support and boss scaling are downstream
  tuning decisions; the human remains decisive.

- [x] **GV-16: How should movement hand off to an automatically enrolled
  instrument phrase?** Players may reposition at any time, including during an
  active phrase. Leaving suspends phrase judgments immediately; travel produces
  no misses or direct penalty, but the player earns no progress from unavailable
  beats. After reaching a valid position, the player receives approximately one
  to one-and-a-half seconds of grace before the phrase cue appears and judgments
  activate. They then join the remaining phrase at the next playable beat or
  step. Earlier and grace-period beats remain unscored. Once joined, normal
  rewards and failure consequences apply.

- [x] **GV-17: How does a band initiate and accept a group ability?** Group
  abilities have two entry paths. Any eligible player may start an equipped
  **Band Call** at the next clean musical boundary; it works at base strength
  alone, while invited bandmates optionally add stacking or multiplier effects.
  Declining, moving, or being downed merely removes that player's contribution.
  Larger **Crescendos** are offered at spaced, pre-authored candidate windows,
  with a guaranteed opportunity budget defined by boss and difficulty. Easier
  difficulties may provide at most one clearly presented extra recovery
  opportunity when the band falls behind; it never guarantees victory and is
  reduced or absent at higher difficulties. Exact counts and timings are
  downstream tuning decisions.

- [x] **GV-18: What should recovery, defeat, and retry feel like?**
  - Settled so far: the limited solo last chance is brief, frenetic, and
    meaningfully difficult, followed by a strong feeling of relief when earned.
    The player earns it through a short emergency rhythm challenge tied to the
    current song pulse and using the normal rhythm controls, not a disconnected
    puzzle or random result. Exact inputs and timing are downstream design.
  - Settled so far: in co-op, bandmates voluntarily divert musical effort into a
    short revive phrase. One bandmate can complete it, while additional
    participants accelerate or strengthen the revival. Exact inputs and values
    are downstream design.
  - Settled so far: failed encounters award modest song and boss mastery progress
    plus ordinary crafting materials based on performance. Story progression,
    Shattered Song fragments, and signature boss drops require victory.
  - Active-encounter recovery is earned through the solo challenge or co-op
    revive performance. Robux or another paid currency cannot purchase or bypass
    it.

- [x] **GV-19: What is the song and music-content strategy?**
  - Settled so far: the initial catalog uses original, creator-directed,
    AI-assisted songs with human review and documented usage rights and
    provenance. Licensed, commissioned, and community-created music may be
    considered later but is not a launch dependency.
  - Settled so far: the musical identity is dark, dangerous EDM with cinematic
    wall-of-sound scale and relentless K-pop energy. The catalog should preserve
    this identity rather than becoming a general assortment of unrelated genres.
  - Lyrics and vocals carry the same supernatural, Christian-inspired themes as
    the world through original, broadly approachable mythology. They may be dark
    and intense but exclude profanity, sexual content, graphic gore, and direct
    real-world religious preaching.

- [x] **GV-20: What is the intended encounter and play-session shape?**
  - Settled so far: normal boss songs typically run from three to seven minutes.
  - Settled so far: a normal boss attempt uses one complete song from beginning
    to end, and the track's full length determines the encounter duration.
  - Settled so far: a comfortable normal session supports two to four complete
    encounters, while completing only one remains worthwhile and does not break
    a required reward chain.
  - Exceptional raids use two or three songs, target roughly ten to twenty
    minutes total, and must not exceed about twenty-five minutes including
    transitions.

- [x] **GV-21: What are the boundaries of persistent power progression?**
  - Settled so far: stronger gear makes older bosses noticeably easier, while
    current-tier challenges still require rhythm skill, positioning,
    cooperation, and boss knowledge.
  - Settled so far: boss tier determines base loot quality. Difficulty and
    stronger performance may increase quantity or better-roll chances, but
    level-appropriate bosses remain the best advancement source. Older bosses
    remain useful farms without out-rewarding current-tier challenges.
  - Settled so far: gear may improve combat outcomes and mistake recovery but
    does not widen note-judgment windows. Timing forgiveness comes from
    difficulty and accessibility settings.
  - Players may freely change equipped gear, abilities, instruments, and
    unlocked role specializations outside an active song without Robux or a
    punitive respec grind. Build choices lock when the song begins.
  - A small prepared consumable loadout remains usable through quick-access
    controls during combat. Charges are limited, inventory browsing and
    replenishment are unavailable mid-song, and consumables cannot bypass defeat
    or replace recovery mechanics. Exact slots and effects are downstream design.

- [x] **GV-22: What monetization principles are acceptable?**
  - Settled so far: direct purchases may include permanent instruments,
    defensive equipment, or other durable items that genuinely improve a
    player's chance of winning. This is intentionally distinct from selling
    temporary buffs, revives, or other rescue purchases during an active
    encounter. Paid equipment cannot exceed the normal tier ceiling and must
    have an earnable equivalent or comparably strong non-paid build at the same
    tier.
  - Settled so far: every purchase has a guaranteed, clearly presented outcome.
    Paid loot boxes, gacha, prize wheels, random upgrades, and paid luck
    modifiers are excluded. Rewards earned solely through play may remain
    random.
  - Settled so far: the storefront uses low-pressure, age-appropriate
    presentation. It avoids purchase prompts around battle or defeat, false
    scarcity, resetting countdowns, and repeated nagging. Prices and outcomes
    are clear. Genuine seasonal end dates are allowed when truthful.
  - Initial monetization is limited to direct-purchase cosmetics and permanent
    equipment. Content access, convenience, subscriptions, temporary boosts, and
    other categories are deferred until the core game and economy are proven;
    any future proposal remains bound by rhythm-skill integrity, co-op fairness,
    earnable-equivalent power, and the low-pressure storefront.

- [x] **GV-23: What is the intended shipping platform and device priority?**
  - Settled so far: native Roblox is the only shipping product and supported
    gameplay runtime. The browser prototype is retired from feature investment,
    content production, parity requirements, and release validation. Existing
    browser files may remain temporarily as historical reference pending a
    separate cleanup decision.
  - Device priority is phone and tablet touch first, desktop keyboard and mouse
    second, and gamepad and console third. All supported devices share encounters,
    progression, rewards, matchmaking, rhythm standards, and success criteria;
    controls, layout, calibration, and presentation may adapt.

- [x] **GV-24: What are the tone, accessibility, and social-safety boundaries?**
  - How frightening, grotesque, or intense may demon-like bosses become?
  - How much humor, warmth, and hope should balance the darker material?
  - Which accessibility or assist principles must exist beyond difficulty levels?
  - Must public co-op work without voice chat or unrestricted text chat?
  - Decision: bosses should present serious supernatural danger without
    becoming cartoony, genuinely demonic, or highly frightening. "Demon-like" may
    describe corrupted traits or appearance, but does not require literal demons.
    The intended register is mythic spiritual menace rather than horror; avoid
    grotesque detail, realistic suffering, gore, and imagery designed as nightmare
    fuel. Balance the darker encounters with meaningful warmth, hope, and humor,
    especially through the Order, its members, and moments of recovery. Humor
    should provide relief and affection without making the central threat feel
    silly or turning the world into parody. As a default principle, accessibility
    assists are separate from difficulty and do not reduce progression or rewards.
    Appropriate assists include input and audio calibration, reduced flashing and
    camera shake, high-contrast and non-color-only cues, and control remapping.
    The exact assist set belongs to downstream design and testing. Public co-op
    must be fully playable without voice chat or unrestricted text chat. Core
    coordination uses readable battle cues, pings, and safe preset messages;
    permitted voice or filtered chat can supplement these systems but is never
    required to understand or complete an encounter.

- [x] **GV-25: What is the boundary and success test for the first shippable
  product?**
  - What is the smallest experience that represents the real game rather than a
    technology demo?
  - Which tempting features are explicitly outside the first product, such as
    PvP, user-authored songs, free-roaming worlds, or deep crafting?
  - What observable player reaction would prove that the vision is working?
  - What ongoing content cadence, if any, must the first product be designed to
    support?
  - Decision: the first shippable product is a small but complete native
    Roblox release rather than a technology demo. It includes an Order hub,
    onboarding, three replayable bosses, complete solo play, three-to-six-player
    cooperative play, and a basic equipment, reward, and progression loop. The
    release must express the real rhythm-controlled boss-combat promise on the
    touch-first supported product. PvP, user-authored songs, a free-roaming world,
    deep crafting, and multi-song raids are explicitly outside the first release.
    These exclusions prevent attractive expansion features from displacing the
    core boss-combat experience. The vision is working when target-age players
    understand that their musical performance controls combat, voluntarily replay
    bosses, and want to coordinate or return with friends. Those observable
    reactions matter more than players merely completing onboarding once. The
    game is designed for repeatable additions of new songs and bosses, but does
    not promise a fixed public cadence until actual production time and player
    demand are known.

## Downstream backlog that does not block Vision v1

These are valid questions, but they should not lengthen the vision interview.
Move them into the named design surface when work begins.

- [ ] **DES-01, rhythm charting:** exact phrase length, note reduction, tap/hold
  grammar, difficulty transformations, chart-authoring workflow, automated
  resistance-window suggestions based on beats, time, note density, audio
  analysis, or AI, and a human review/editing surface.
- [ ] **DES-02, timing and feedback:** judgment windows, latency calibration,
  exact preview timing, grading, combo rules, and audio ducking values.
- [ ] **DES-03, combat math:** health or ward values, damage formulas, intent
  multipliers, resource gain, cooldowns, and boss scaling.
- [ ] **DES-04, positioning:** final position count, travel duration, range,
  risk/reward multipliers, cover geometry, and collision rules.
- [ ] **DES-05, group resolution:** exact contribution weights, Cohesion Bonus
  cap and eligibility thresholds, collective tiers, and difficulty curves.
- [ ] **DES-06, progression economy:** item tiers, stat ranges, skill-tree nodes,
  crafting recipes, currencies, respec costs, and drop rates.
- [ ] **DES-07, recovery tuning:** revive timing, last-chance inputs, random
  chances, retry costs, and exploit protections after the vision boundary is set.
- [ ] **DES-08, multiplayer edge cases:** disconnects, AFK players, host loss,
  latency, reconnection, griefing, and matchmaking rating.
- [ ] **DES-09, interface implementation:** precise button placement, keybinds,
  special-ability slot count, cue animation, safe areas, responsive breakpoints,
  and reduced-motion behavior.
- [ ] **DES-10, encounter technology:** runtime schemas, validators, tooling,
  networking authority, anti-cheat, persistence, and analytics events.
- [ ] **CONTENT-01, production scope:** exact launch boss, song, instrument, item,
  region, and story-chapter counts plus the production schedule.
- [ ] **NARRATIVE-01, story bible:** names, cultures, ranks, dialogue, the false
  hero's exact spiritual nature, and detailed campaign scenes. The novice's fate
  remains intentionally unanswered unless the owner reverses that decision.

## Vision interview change log

- **2026-08-15:** The GD-05 owner interview amended GV-03, GV-06, and GV-07 to
  authorize a compact phrase-bounded moving staff and sustained access to
  available instrument notes. This preserves the existing boundary against a
  permanent song-wide highway; no new vision question was required.
- **2026-08-16:** The GD-15 owner interview amended GV-15 so acolytes and humans
  may share formation-enabled tactical locations. Acolytes no longer swap away
  from a claimed location and never consume gameplay capacity; no new vision
  question was required.
- **2026-08-17:** The GD-21 owner interview amended GV-09 by replacing the
  provisional negative cohesion modifier with a capped positive Cohesion Bonus.
  Weak performance can leave potential bonus unearned but never subtracts
  another player's contribution; no new vision question was required.

No questions have been added beyond the 2026-08-14 baseline. Future additions
must include the identifier, date, and why the existing checklist could not
contain the decision.
