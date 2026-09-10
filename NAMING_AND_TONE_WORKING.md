# Bands Battle Naming & Tone Working Record

- **Status:** Interview complete; 9 of 9 questions resolved and reconciled
- **Started:** 2026-09-09
- **Question plan:** [`NAMING_AND_TONE_QUESTIONS.md`](NAMING_AND_TONE_QUESTIONS.md)
- **Required by:** [`GAME_DESIGN.md`](GAME_DESIGN.md#required-product-follow-up)
- **Vision authority:** [`GAME_VISION.md`](GAME_VISION.md)
- **Visual authority:** [`ART_DIRECTION.md`](ART_DIRECTION.md)
- **Canonical result:** [`NAMING_AND_TONE.md`](NAMING_AND_TONE.md)

## 1. Role of this record

This file progressively preserves owner choices, rejected directions, candidate
terms, approved vocabulary, and document handoffs from the Naming & Tone
interview. It is decision evidence for `NAMING_AND_TONE.md`, not the final
language authority.

## 2. Inherited boundary

Naming & Tone owns player-facing vocabulary, voice principles, final label
relationships, glossary rules, future-catalog naming grammar, and the audit that
prevents internal placeholders from shipping.

It does not change gameplay meaning, system ownership, formulas, balance,
content structure, UI hierarchy, accessibility, economy rules, or technical
identities. Existing semantic owners retain those decisions. Internal keys and
localization keys may remain stable while player-facing strings change through
the approved mapping.

The complete inherited decision set is recorded in
[`NAMING_AND_TONE_QUESTIONS.md`](NAMING_AND_TONE_QUESTIONS.md#2-fixed-inherited-decisions).

## 3. Decision record

### Checkpoint A - Voice, audience, and naming system

#### NT-01 - Core voice and emotional register

- **Status:** Resolved 2026-09-09.
- Bands Battle uses five mutually reinforcing voice pillars:
  - **Mythic:** names and world language suggest ancient supernatural stakes,
    monumental forces, corruption, harmony, and restoration without depending on
    generic medieval vocabulary or borrowed real-world occult language.
  - **Electric:** words have the punch, velocity, and spectacle of dark EDM,
    cinematic wall-of-sound production, and K-pop performance energy without
    becoming trendy music-industry slang.
  - **Heroic:** the player is an active musical warrior whose reading, timing,
    courage, positioning, protection, and cooperation produce meaningful results.
    Copy uses decisive active voice rather than treating the avatar as a passive
    stat container.
  - **Hopeful:** darkness is opposed by fellowship, honest performance, recovery,
    and restoration. Victory feels earned and luminous; Defeat leaves a credible
    path forward rather than humiliation or nihilism.
  - **Clear:** the player can understand the action, danger, consequence, cost,
    lock, and next step on a phone without decoding lore or systems jargon.
- The pillars change emphasis by context while remaining one voice:
  - combat warnings are Clear, Electric, and urgent;
  - system/category/option names are Mythic and Electric but still pronounceable;
  - victories, restoration, and rare earned power emphasize Heroic and Hopeful;
  - defeat/recovery/errors are Clear, calm, nonblaming, and forward-looking;
  - lore may carry more Mythic texture after the immediate meaning is clear; and
  - humor is warm relief among allies, not mockery of danger or the player.
- Avoid generic fantasy jargon, heavy-metal parody, edgy horror, direct
  real-world preaching, clinical/engineering system language, forced slang,
  empty superlatives, overly cute quips, and jokes that make the boss or stakes
  feel fake.
- A name is not successful merely because it sounds dramatic. It must also fit
  the actual mechanic, distinguish itself from neighboring terms, work aloud,
  and survive short mobile presentation.

#### NT-02 - Clarity, age range, humor, and thematic boundaries

- **Status:** Resolved 2026-09-09.
- Core buttons, choices, warnings, and recovery actions use familiar direct
  language, normally one to three words. A player should recognize the requested
  action without first learning the world's mythology.
- Flavorful system, power, reward, and world names are encouraged when they are
  distinctive and fit NT-01. Their nearby first sentence explains the function
  and consequence plainly; later optional text may add mythology, imagery, and
  character.
- A label does not stack multiple unfamiliar metaphors. Invented words are rare,
  pronounceable, and introduced beside a familiar explanation. The glossary
  prevents two invented synonyms for one concept.
- Critical combat copy is brief and literal enough to act on immediately. Lore
  never consumes warning time, hides a target/response, or substitutes a poetic
  phrase for an exact cost, lock reason, failure, or recovery action.
- Natural contractions are allowed. Exclamation marks are occasional emphasis,
  not the default voice. Player copy avoids all-caps shouting, forced slang,
  memes, and trendy phrases likely to age quickly.
- Humor comes from allies, personality, affectionate recovery, optional hub
  moments, and the contrast between danger and fellowship. It does not appear in
  critical instructions, boss warnings, Defeat reasons, safety/privacy notices,
  transaction failures, or errors, and it never mocks player performance.
- Dark intensity may evoke supernatural danger, corruption, sacrifice, and
  desperate resistance. It never becomes graphic gore, horror intended to
  frighten, profanity, sexual content, direct real-world preaching, player shame,
  teammate accusation, or coercive purchase language.
- The primary layer is understandable for ages 10–14 on a phone. This is judged
  by read-aloud clarity and target-age playtesting rather than mechanically
  flattening every optional description to one reading-grade formula.

#### NT-03 - Protected anchors, glossary, and internal/player-facing separation

- **Status:** Resolved 2026-09-09.
- `Bands Battle` is definitely not the final player-facing game title. It remains
  a convenient internal project/repository name until the approved replacement
  is chosen. Publication requires the final title; internal paths and historical
  document references do not need a disruptive rename merely to reflect player
  copy.
- The protected world anchors are:
  - **the Order**, the musical-warrior organization;
  - **the Shattered Song**, the broken supernatural whole at the center of the
    world conflict;
  - **the Song**, the living greater musical force when the full name is not
    needed;
  - **shards**, the world destinations/major spaces; and
  - **fragments**, the recovered pieces of the Shattered Song.
- **Resolve** is approved as the final player-facing name for the boss-resistance
  resource and its sequential layers. It fits the Mythic/Heroic voice while
  remaining understandable; no alternate world term is needed.
- Build-system labels, progression tracks, resources, item families, UI
  destinations, and all remaining gameplay terms stay open to their scheduled
  review. An unchanged term will be marked `Keep`, not silently assumed final.
- Internal semantic identities and localization keys remain stable wherever
  practical. Each concept has one official full localized label and may have one
  documented shorter label only when a constrained UI surface needs it.
- Old labels may appear as temporary authoring/search aliases during migration,
  but cannot appear in shipping player strings. An unresolved placeholder or
  missing approved mapping blocks publication.
- Selecting the final game title is deferred to a separate future publication
  task. It does not block the remaining system vocabulary, design documentation,
  or Technical Architecture. It does block presenting `Bands Battle` as the
  approved public title; public branding remains explicitly unfinished.

### Checkpoint B - Builds and core gameplay vocabulary

#### NT-04 - Build system, category, major slot, support slot, and preset names

- **Status:** Resolved 2026-09-09.
- **Battle Style** is the final player-facing name for the editable
  specialization configuration. `Build` may remain a familiar lower-case
  explanatory word, but headings, navigation, and named system references use
  Battle Style.
- **Focus** names one of the four functional option categories. A Focus is a tag
  and browsing aid, not a class, permanent path, required composition, or purity
  rule. Players freely mix Focuses inside one Battle Style.
- **Anthem** names the one major behavior-changing rule. A player chooses one
  Anthem; it carries the defining condition, tradeoff, redistribution, or use-
  pattern change within the approved major budget.
- **Riff** names each smaller supporting rule. A Battle Style contains exactly
  three distinct Riffs. Riffs may reinforce the Anthem, interact with each other
  only through approved synergy, or create a mixed style.
- **Loadout** names a complete saved setup: instrument/role, three gear slots,
  Signature Special, Band Call, consumables, and Battle Style. The three existing
  player-named full spec-preset slots become three player-named Loadouts.
- `Current Loadout` is the complete active hub/staging setup. Applying a saved
  Loadout remains atomic; an incomplete saved Loadout remains visible for repair
  rather than silently dropping or replacing a component.
- The grammar is:
  - one **Battle Style**;
  - one or more **Focuses** represented across its choices;
  - one **Anthem**;
  - three **Riffs**; and
  - a saved or current **Loadout** containing that Battle Style and the other
    existing preparation systems.
- These labels do not rename or merge instruments, gear, abilities, consumables,
  inventory, or the semantic configuration/snapshot records behind the UI.
- Common actions remain clear verbs: Choose/Replace an Anthem or Riff, Preview/
  Apply a Battle Style, and Save/Apply/Rename/Repair a Loadout. Final action-copy
  reconciliation occurs under NT-08.

#### NT-05 - Category, beginner-template, and future option naming grammar

- **Status:** Resolved 2026-09-09.
- The four universal Focuses are:
  - **Drive:** offense, Momentum, and deliberate dangerous-position play. It
    suggests forward force without promising unconditional damage or a locked
    attacker class.
  - **Guard:** Ward, Defend, mitigation, protection, and revival support. It
    describes active protection rather than a passive tank class.
  - **Unity:** teammate support, Band Calls, Crescendos, Cohesion, and shared
    effects. It never implies that solo players or mixed-Focus builds are lesser.
  - **Pulse:** Hype, Signature Specials, movement-triggered utility, resource
    timing, and hybrids. It suggests responsive energy without taking authority
    over the musical clock.
- In navigation/filter contexts, the full forms are `Drive Focus`, `Guard Focus`,
  `Unity Focus`, and `Pulse Focus`. Inside an already labeled Focus list, the
  shorter names stand alone.
- Focus language never labels the player or instrument as a class. An Anthem or
  Riff has one primary Focus for browsing and may carry cross-Focus tags; every
  legal mixture remains available.
- The three beginner Loadouts are:
  - **Break Through:** offense-focused starting configuration;
  - **Guard the Band:** protection/support-focused starting configuration; and
  - **Ready for Anything:** balanced hybrid and safe skipped-selection default.
- These are selectable starter Loadouts, not permanent archetypes, ranked
  recommendations, or additional saved slots. Their plain descriptions show the
  included instrument-independent choices before Apply.
- Future **Anthems** normally use strong two-to-four-word statement or image
  names that feel defining. Future **Riffs** normally use concise one-to-three-
  word action/image names that feel supportive. The distinction is tonal, not a
  hidden mechanical rule; clarity wins when a longer or shorter name is needed.
- A name cannot promise an effect the option does not own, imply unrestricted
  power, conceal a drawback, or reuse a protected system/resource term in a way
  that confuses the UI.
- Every option pairs its flavor name with an immediate plain description of the
  trigger/condition, tradeoff, effect, fallback, and important cap. Future
  catalogs use this grammar but choose final names only after the actual option
  definitions exist.

#### NT-06 - Combat, encounter, cooperation, and role-term audit

- **Status:** Resolved 2026-09-09.
- Keep these player-facing terms:
  - **Attack**, **Defend**, and **Special** for the three stable combat choices/
    controls;
  - **Ward** for personal survival protection;
  - **Resolve** and **Resolve layer** for boss resistance and its sequential
    sections;
  - **Momentum** for banked successful pressure while a later Resolve layer is
    unavailable;
  - **Hype** for the resource that readies the player's defining special;
  - **Band Call** for a player-started cooperative action;
  - **Crescendo** for an authored encounter-started group performance;
  - **Risk** and **Risk Bonus** for deliberately banked positional exposure;
  - **Revive** for bandmates' cooperative rescue of a downed player; and
  - the familiar supporting words **instrument**, **role**, **band**, **boss**,
    **encounter**, **passage**, and **phrase**.
- Use **music staff** the first time the moving rhythm surface is taught or when
  `staff` would be ambiguous. Later compact labels may use **staff**.
- Use **Signature** as the normal name of the player's equipped defining
  ability. The persistent control remains **Special**. UI may say `Signature
  Ready` or name the actual Signature; it does not need the repetitive phrase
  `Signature Special`.
- Use **Band Harmony** as the player-facing name for the nonnegative cooperative
  participation bonus currently identified internally as Cohesion/Cohesion
  Bonus. It reflects broad successful participation without ranking or blaming
  individual bandmates. Internal formulas/events may retain stable Cohesion keys.
- Use **Order Ally** for the NPC support currently identified internally as an
  acolyte. The label connects the NPC to the protected Order, avoids unnecessary
  religious/obscure language, and does not imply a human bandmate. Internal
  acolyte identities may remain stable.
- Use **Last Stand** for the one-use solo emergency recovery opportunity. The
  event pairs the flavor name with a direct instruction such as `Hit the beat to
  get back up.` It does not imply another revive, automatic success, or a second
  recovery use.
- `Near`, `Middle`, and `Rear` remain internal universal distance/risk bands and
  balance references. Player-facing arenas show an authored location name plus a
  direct **High Risk**, **Balanced**, or **Low Risk** indicator. Players choose a
  real place, not a button named after an abstract multiplier.
- `First Clash`, `Escalation`, and `Climax` may remain useful internal/authoring
  labels for the three normal Resolve layers; gameplay does not require those
  names to appear as universal chapter titles.
- `Attack`, `Defend`, `Special`, `Ward`, and other retained functional words are
  intentionally clearer than more elaborate substitutes. Their effects and
  flavor may be dramatic without renaming the controls into lore puzzles.

### Checkpoint C - Progression, UI voice, and publication

#### NT-07 - Progression, rewards, economy, and item-family names

- **Status:** Resolved 2026-09-09.
- The three progression tracks are:
  - **Song Restoration:** first-clear campaign progress that recovers a Song
    Fragment and opens the next destination;
  - **Order Rank:** broad personal progression that opens systems, Loadouts,
    Battle Style choices, and wider equipment access while retaining finite
    direct power; and
  - **Boss Mastery:** familiar per-boss progress for lore, cosmetics, recipes,
    Battle Style options, titles, milestones, and records.
- Order Rank is private progression, not a public competitive rank or skill
  label. Boss Mastery remains shared across instruments for one boss while
  personal records remain instrument/difficulty-specific.
- **Resonance** is the general earned resource used for ordinary upgrades,
  crafting, Battle Supplies, and normal shop purchases. It is always presented
  as earned value, never a premium currency or bought shortcut.
- **Boss Echoes** is the functional family for boss-specific materials. Each boss
  may use a short themed form such as `Blackened Crown Echoes` when it remains
  clearly identified as that boss's material. Boss Echoes never use `Fragment`,
  which is reserved for campaign restoration.
- **Song Fragment** names one recovered campaign piece of the protected Shattered
  Song. `Fragment` may be used after context is established; the full label is
  used where it could be confused with materials or salvage.
- Keep these clear item families: **Instrument**, **Ward Core**, **Resonator**,
  **Trait**, **Cosmetic**, **Recipe**, and **Record**. Resonators may relate
  thematically to Resonance, but their mechanical description remains explicit.
- Use **Battle Supplies** instead of player-facing `consumables` for the prepared
  limited-charge item family. Detailed descriptions still state charges, exact
  effects, limits, and that supplies cannot bypass Defeat.
- Keep the familiar verbs/nouns **Upgrade**, **Craft**, **Salvage**, **Unlock**,
  **Rank**, and **Tier**. Use qualified forms such as `Order Rank`, `Mastery
  Rank`, and `Gear Rank` where an isolated `Rank` could be ambiguous.
- Use **Reforge** instead of technical `uplift` for raising eligible old gear to
  the player's current Tier. The first explanation says exactly what carries
  forward, what it costs, and that the item is not rerolled, destroyed, or
  silently changed.
- Boss-specific resource, item, recipe, trait, cosmetic, and title names follow
  the future catalog grammar and actual boss fiction. This pass does not invent
  catalogs before their definitions exist.

#### NT-08 - Hub navigation, onboarding, Results, failure, and action copy

- **Status:** Resolved 2026-09-09.
- The four primary destinations keep their familiar final labels and order:
  **Play**, **Prepare**, **Progress**, and **Band**.
  - Play covers Continue Story, shard/encounter selection, difficulty, and replay.
  - Prepare covers Current Loadout, inventory, gear, upgrades, Battle Styles,
    abilities, and Battle Supplies.
  - Progress covers Song Restoration, Order Rank, Boss Mastery, Records,
    collection, and restored world state.
  - Band covers party, public matching, safe communication, and compatible social
    state.
- Keep **Settings**, **Controls**, **Accessibility**, and **Help** as global
  utilities. **Store** remains a clearly named secondary destination and never a
  primary task, Results action, recovery, or error solution.
- Use **Quick Play** instead of the working `Fast Play`. Keep **Begin Practice**
  and **Skip Practice**; skipping receives a direct control-reference confirmation
  without shame or reward warning.
- Results uses **Victory**, **Defeat**, or **No Contest**. `Invalid` remains an
  internal state qualifier rather than the headline. No Contest always receives
  the plain explanation: `The encounter couldn't finish fairly. This attempt
  doesn't count.` A more specific safe reason may follow.
- Keep the primary next actions **Continue Story**, **Retry**, **Stay with Band**,
  and **Return to Hub**. The existing outcome/eligibility rules decide which one
  leads; no label implies an unavailable or unconfirmed action.
- Keep the standard verbs **Ready**, **Apply**, **Save**, **Cancel**, **Reset**,
  and **Repair**. Use them consistently: Apply changes the current eligible
  configuration, Save preserves a draft/Loadout, Ready commits to staging state,
  and Repair opens the identified incomplete references.
- Cooperative opt-ins use:
  - **Join Call** / **Pass** for a Band Call invitation;
  - **Join Crescendo** / **Pass** for a Crescendo preview; and
  - **Join In** for optional universal beat activity during an instrument
    dropout. Movement automatically exits Join In as already designed.
- `Pass` is a neutral valid choice. It never becomes `Fail`, `Refuse`, `Let Band
  Down`, or another shaming label. A committed/expired invitation reports the
  actual state instead of accepting a stale press.
- Retain the first-release preset pings as **Move**, **Defend**, **Join Call**,
  **Revive**, **Ready**, and **Thanks**. Their contextual target/location and
  automatic protected-cue boundaries remain Multiplayer-owned.
- Error and unavailable-state copy follows three short facts when applicable:
  1. what happened or is unavailable;
  2. what remained safe/preserved; and
  3. the next valid action.
- Player copy does not show raw technical codes, assign blame, make a joke, hide
  the consequence in lore, or route to Store. A private code may remain available
  to support/diagnostics outside the primary message.

#### NT-09 - Style guide, catalog completion, validation, and change control

- **Status:** Resolved 2026-09-09.
- Named systems, resources, world terms, game states, and UI buttons use Title
  Case in English. Ordinary descriptions use sentence case. A generic use of the
  same word may remain lower-case when it does not name the game concept.
- Buttons and compact navigation labels have no terminal punctuation.
  Instructions, status explanations, failure/recovery copy, transaction
  consequences, and errors use complete sentences.
- Gameplay values, costs, ranks, tiers, quantities, percentages, and countdowns
  use numerals for scanning. Narrative prose may spell out small numbers when it
  is not communicating a mechanic.
- Core action/navigation labels normally use one to three English words.
  Distinctive Anthem names may use up to four. These are drafting targets rather
  than permission to abbreviate away meaning or reject a clear localized form.
- Every concept has one official full label and at most one documented short
  form for constrained context. Aliases may support migration/search but do not
  create multiple interchangeable shipping names.
- Names are read aloud before approval. Avoid hard-to-pronounce invention,
  accidental homophones, visually similar neighboring labels, repeated acronyms,
  and sound-alike cues that would fail without text.
- English layouts reserve roughly 30% expansion and are tested with long strings,
  plural/parameter variants, Unicode, text scaling, and the supported assistive
  presentation. A short English name cannot force truncation in other languages.
- Future boss, item, ability, Anthem, Riff, recipe, trait, cosmetic, title, and
  boss-material names are chosen only after the owned definition/content exists.
  Each uses the relevant approved grammar and immediate plain description.
- The publication check verifies:
  - correct semantic meaning and no promise beyond the mechanic;
  - alignment with the five voice pillars and forbidden-tone boundaries;
  - age-appropriate clarity, pronunciation, spelling, capitalization, plural,
    parameters, and optional short form;
  - uniqueness and no confusing reuse across systems, resources, items, bosses,
    actions, cues, or navigation;
  - localization-key mapping, 30%-expansion layout, text scaling, and non-text
    alternatives;
  - current Roblox moderation/platform requirements and a basic public search for
    obvious naming conflicts before publication; and
  - zero unresolved placeholders or internal-key leakage on the affected surface.
- The game owner approves final player-facing strings. The relevant semantic
  owner verifies that the label and description accurately express its mechanic.
  UI/UX, Audio Presentation, localization, accessibility, Content Authoring, or
  policy review participates when the name crosses that boundary. One person may
  cover several responsibilities during development.
- Canonical `NAMING_AND_TONE.md` owns the glossary and style rules. Each semantic
  catalog owns its actual entries and localization keys while referencing this
  contract. Conflicts resolve through explicit status/version updates, not
  undocumented synonyms.
- The authoritative document pass updates player-facing recommendations and
  records internal-to-player mappings. Stable internal semantic names may remain
  where needed for technical precision, especially in historical working records.
- A change to protected world terms, final build hierarchy, Focus/template names,
  retained/replaced core gameplay vocabulary, progression/resource families,
  navigation/Results patterns, or the five voice pillars requires an explicit
  amendment citing the superseded decision. Ordinary future catalog entries may
  be approved under this grammar without reopening the entire pass.
- Final game-title selection remains a separate future publication task. It must
  receive basic public conflict/moderation review but does not block Technical
  Architecture or the approved system-language reconciliation.

## 4. Candidate and decision glossary

| Internal concept | Candidate player-facing term | Status | Notes |
|---|---|---|---|
| Project/game title | — | Deferred publication task; `Bands Battle` rejected | Working repository title only; not a Technical Architecture blocker |
| Boss resistance resource | Resolve | Approved | Applies to the resource and sequential Resolve layers |
| Specialization configuration | Battle Style | Approved | Overall build-editing system; lower-case `build` may explain it |
| Discipline | Focus | Approved replacement | Four freely mixable category tags; plural `Focuses` |
| Build Core | Anthem | Approved replacement | One major behavior-changing rule |
| Technique | Riff | Approved replacement | Three distinct smaller supporting rules |
| Full spec preset | Loadout | Approved replacement | Saved complete instrument/gear/abilities/consumables/Battle Style configuration |
| Offense/Momentum/danger Focus | Drive | Approved | Freely mixable category; not an attacker class |
| Ward/Defend/revival Focus | Guard | Approved | Active protection category; not a tank class |
| Teammate/Call/Crescendo Focus | Unity | Approved | Cooperative category that remains valid in mixed and solo-capable styles |
| Hype/Signature/movement Focus | Pulse | Approved | Responsive utility/hybrid category; does not own song timing |
| Offense beginner template | Break Through | Approved | Complete offense-focused starter Loadout |
| Protection/support beginner template | Guard the Band | Approved | Complete protection/support starter Loadout |
| Balanced beginner template | Ready for Anything | Approved | Complete hybrid starter Loadout and safe default |
| Signature Special | Signature | Approved short form | Equipped defining ability; the stable control remains `Special` |
| Cohesion Bonus | Band Harmony | Approved player-facing alias | Nonnegative broad-participation bonus; internal Cohesion key may remain |
| Acolyte | Order Ally | Approved player-facing alias | Solo NPC support; never represented as a human bandmate |
| Solo emergency recovery | Last Stand | Approved | One-use solo recovery opportunity with direct beat instruction |
| Near/Middle/Rear position bands | Authored location + High Risk/Balanced/Low Risk | Approved player-facing pattern | Internal bands and numeric ratios remain stable |
| Campaign progression | Song Restoration | Approved replacement | First-clear campaign track and Song Fragment recovery |
| General player progression | Order Rank | Approved replacement | Private broad system/unlock progression, not competitive skill rank |
| Boss mastery | Boss Mastery | Approved keep | Per-boss track shared across instruments |
| General earned resource | Resonance | Approved | Nonpremium general resource |
| Boss-specific material | Boss Echoes | Approved family | Boss-themed form permitted; never called a Song Fragment |
| Campaign fragment | Song Fragment | Approved | Recovered piece of the Shattered Song |
| Consumables | Battle Supplies | Approved replacement | Prepared limited-charge item family |
| Uplift | Reforge | Approved player-facing replacement | Raises eligible old gear to current Tier without reroll/destruction |
| Fast Play | Quick Play | Approved replacement | Opens the same eligible encounter-card flow; not instant deployment |
| Invalid / No Contest | No Contest | Approved player headline | Plain fairness explanation follows; Invalid remains internal qualifier |
| Band Call opt-in | Join Call / Pass | Approved | Neutral optional participation labels |
| Crescendo opt-in | Join Crescendo / Pass | Approved | Neutral optional participation labels |

The table will grow only as terms are actually considered. Rejected candidates
remain visible with a short reason so later drafting does not accidentally
reintroduce them.

## 5. Confirmed downstream handoffs

No new handoffs are confirmed yet. The completed pass will route final strings
and naming patterns to the existing semantic owners, UI/UX, Audio Presentation,
Content Authoring, localization, and Technical Architecture.

## 6. Change log

- **2026-09-09:** Created the working record with all nine questions open and
  checkpoint A ready for owner review.
- **2026-09-09:** Approved NT-01. Established Mythic, Electric, Heroic, Hopeful,
  and Clear as the core voice, with contextual emphasis and rejected-tone
  boundaries. Interview progress is 1 of 9 questions.
- **2026-09-09:** Approved NT-02. Recorded short action-first labels, functional
  first sentences, restrained invented language/punctuation, warm optional humor,
  and firm age/safety/tone prohibitions. Interview progress is 2 of 9 questions.
- **2026-09-09:** Partially resolved NT-03. Recorded `Bands Battle` as a rejected
  final title, protected the Order/Shattered Song/Song/shards/fragments, approved
  Resolve, and established stable internal-to-localized label mapping. Final-title
  direction remains pending; interview progress remains 2 of 9.
- **2026-09-09:** Completed NT-03 by routing final-title selection to a separate
  future publication task. The system-language pass and Technical Architecture
  may proceed while public branding remains explicitly unfinished. Checkpoint A
  and interview progress are 3 of 9.
- **2026-09-09:** Approved NT-04. Replaced the rejected build hierarchy with
  Battle Style, Focus, one Anthem, three Riffs, and three complete player-named
  Loadouts. Interview progress is 4 of 9 questions.
- **2026-09-09:** Approved NT-05. Named Drive, Guard, Unity, and Pulse Focuses;
  Break Through, Guard the Band, and Ready for Anything beginner Loadouts; and the
  future Anthem/Riff naming grammar. Interview progress is 5 of 9 questions.
- **2026-09-09:** Approved NT-06. Kept the clear combat/music terms; adopted
  Signature, Band Harmony, Order Ally, and Last Stand; and separated internal
  position bands from authored location/risk labels. Checkpoint B and interview
  progress are 6 of 9.
- **2026-09-09:** Approved NT-07. Named Song Restoration, Order Rank, Resonance,
  Boss Echoes, Song Fragments, Battle Supplies, and Reforge while retaining the
  clear Boss Mastery/item/action terms. Interview progress is 7 of 9 questions.
- **2026-09-09:** Approved NT-08. Finalized primary navigation/utilities, Quick
  Play, practice and Results language, familiar action verbs, neutral cooperative
  opt-ins, preset pings, and direct nonblaming failure copy. Interview progress
  is 8 of 9 questions.
- **2026-09-09:** Approved NT-09. Finalized the capitalization, punctuation,
  numeral, length, pronunciation, localization, uniqueness, publication-check,
  approval, catalog-extension, document-reconciliation, and amendment rules.
  Published canonical `NAMING_AND_TONE.md`; interview progress is complete at 9
  of 9 questions.
