# Bands Battle Naming & Tone Questions

- **Status:** Interview complete; 9 of 9 questions resolved and reconciled
- **Started:** 2026-09-09
- **Required by:** [`GAME_DESIGN.md`](GAME_DESIGN.md#required-product-follow-up)
- **Vision authority:** [`GAME_VISION.md`](GAME_VISION.md)
- **Visual authority:** [`ART_DIRECTION.md`](ART_DIRECTION.md)
- **Primary system input:** [`BUILDS_AND_SPECIALIZATION.md`](BUILDS_AND_SPECIALIZATION.md)
- **UI language input:** [`UI_UX.md`](UI_UX.md)
- **Working record:** [`NAMING_AND_TONE_WORKING.md`](NAMING_AND_TONE_WORKING.md)
- **Canonical result:** [`NAMING_AND_TONE.md`](NAMING_AND_TONE.md)

## 1. Interview method

This is a compact creative interview with three checkpoints of three questions.
It establishes the voice and naming boundaries first, resolves the required
build and core-game vocabulary second, then finishes progression/economy/UI copy
and publication governance.

Questions will be asked one at a time or in small related groups. Answers are
persisted in `NAMING_AND_TONE_WORKING.md`. When all nine questions are resolved,
they will be reconciled into canonical `NAMING_AND_TONE.md`, followed by a
consistent player-facing terminology pass across the approved design documents.

The pass names approved concepts; it does not redesign their mechanics. It also
does not fabricate final names for bosses, items, abilities, or options that have
not been authored. Instead it establishes a strong naming grammar and a
publication gate for those future catalogs.

## 2. Fixed inherited decisions

The interview does not reopen these approved rules:

- The project currently called Bands Battle is a supernatural rhythm-controlled
  boss-combat game with musical action-RPG progression. **Bands Battle is a
  working project title, not the final game title.**
- Players are their own Roblox avatars within an order of musical warriors. The
  game does not replace them with named hero characters or permanent combat
  classes.
- The primary audience is approximately ages 10–14. Core labels and actions must
  be immediately understandable on phones, while optional descriptions and lore
  may carry more flavor.
- The musical identity is dark, dangerous EDM with cinematic wall-of-sound scale
  and relentless K-pop energy.
- The world uses original supernatural, Christian-inspired themes of darkness,
  corruption, courage, sacrifice, fellowship, hope, and restoration without
  direct real-world preaching.
- Threats should feel mythic and serious, not cartoony, genuinely demonic,
  grotesque, gory, sexually suggestive, profane, or intended as horror.
- Darkness is balanced by warmth, hope, and affectionate humor. Humor cannot turn
  the central threat into parody.
- `Discipline`, `Build Core`, and `Technique` are rejected player-facing names.
  They remain internal working terms until this pass replaces them.
- A build still contains exactly one major behavior-changing rule and three
  smaller supporting rules drawn from four freely mixable functional categories.
  Naming cannot turn those categories into locked classes.
- The four functional categories remain: offense/Momentum/dangerous position;
  Ward/Defend/mitigation/revival; teammate/Band Call/Crescendo/Cohesion; and
  Hype/Signature/movement utility/hybrids.
- Beginner templates remain offense-focused, protection/support-focused, and
  balanced hybrid. They are complete starting configurations, not classes or
  permanent choices.
- Instruments are song-specific musical identities, not fixed combat roles.
  Piano, synthesizer, strings, percussion, vocals, or other authentic roles may
  appear or be absent according to the arrangement.
- Mechanics and semantic ownership behind Attack, Defend, Special, Ward,
  Resolve, Momentum, Hype, Signature Special, Band Call, Crescendo, Cohesion,
  Risk, positions, recovery, and Results remain unchanged unless a separate
  design amendment says otherwise.
- The three progression tracks remain campaign access/restoration, general
  player progression, and per-boss mastery. Their final labels are open.
- The economy still has one general earned resource and boss-specific materials.
  Final resource/item-family names are open, but naming cannot change how they
  are earned, spent, upgraded, crafted, or purchased.
- The UI retains four primary task destinations in the same order: Play,
  Prepare, Progress, and Band are functional placeholders for those destinations.
- Player-facing language remains nonshaming, nonaccusatory, age-appropriate, and
  low pressure. It never blames a teammate, mocks failure, exposes accessibility
  or spending, or manufactures purchase urgency.
- Essential actions, cues, and states use clear text plus their approved
  icon/shape/placement/audio alternatives. Flavor cannot obscure consequences,
  costs, lock reasons, errors, or recovery.
- Player strings use external stable localization keys with parameter/plural/
  number/time support, not concatenated fragments or text baked into images.
- Layouts allow Unicode, long words/names, and roughly 30% localization
  expansion. Internal system/category/option/event keys never leak into shipping
  player text.
- Naming must align with the visual language: ancient monumental ruins, musical
  weapons, cyan player energy, rare white-gold earned power, violet corruption,
  and mythic spiritual menace rather than generic medieval or neon-cyberpunk
  language.

## 3. Question plan

### Checkpoint A - Voice, audience, and naming system

#### NT-01 - Core voice and emotional register

- **Status:** Resolved 2026-09-09.
- **Decision:** The core voice is Mythic, Electric, Heroic, Hopeful, and Clear.
  Ancient supernatural stakes and dark concert energy frame the player's brave,
  skillful action; restoration and fellowship keep the world from nihilism; and
  player instructions remain immediately understandable. Avoid generic fantasy
  jargon, heavy-metal parody, edgy horror, direct preaching, clinical system
  language, and overly cute jokes.
- **Decision needed:** What should Bands Battle sound like when the game names a
  powerful system, addresses the player, describes danger, or celebrates success?
- **Must resolve:** Three to five voice pillars; relative weight of supernatural
  myth, concert/performance energy, combat heroism, danger, hope, and humor;
  desired intensity; sentence rhythm; words/attitudes that feel native; and tones
  that would make the game feel generic, childish, try-hard, preachy, horrific,
  clinical, or parody-heavy.

#### NT-02 - Clarity, age range, humor, and thematic boundaries

- **Status:** Resolved 2026-09-09.
- **Decision:** Core buttons/warnings use familiar action-first language, normally
  one to three words. Flavor belongs in distinctive names and optional supporting
  copy after the first sentence explains function. Use invented words sparingly,
  allow natural contractions, and use exclamation rarely. Avoid all-caps/slang;
  keep humor out of critical instructions, danger, Defeat, and errors; and retain
  the inherited horror, profanity, preaching, shame, and purchase-pressure bans.
- **Decision needed:** How much worldbuilding flavor may appear in core labels and
  instructions before clarity suffers for ages 10–14?
- **Must resolve:** Plain action-first labels versus flavorful descriptions;
  reading level and sentence length; danger without horror; Christian-inspired
  resonance without direct preaching; acceptable warmth/humor; defeat and error
  tone; use of slang, exclamation, all-caps, metaphor, and invented words; and
  language that is always prohibited.

#### NT-03 - Protected anchors, glossary, and internal/player-facing separation

- **Status:** Resolved 2026-09-09.
- **Decision:** Protect `the Order`, `the Shattered Song`, `the Song`,
  `shards`, `fragments`, and `Resolve`. Treat `Bands Battle` as an internal
  working title that must not ship; selecting its replacement is a separate
  future publication task and does not block this system-language pass or
  Technical Architecture. Other gameplay/progression/resource/UI terms remain
  reviewable. Internal identities stay stable behind one official localized
  label and an optional UI-short label; unresolved shipping placeholders block
  their affected publication surface.
- **Decision needed:** Which existing names are protected world anchors, which
  are open to review, and how will internal keys map to final localized labels?
- **Must resolve:** Status of `Bands Battle`, `the Order`, `Shattered Song`,
  shards/fragments, named combat/co-op terms, and functional UI labels; glossary
  fields; one canonical label versus permitted short/long variants; stable
  internal/localization keys; capitalization/plural forms; aliases during
  migration; and how placeholder leakage blocks publication.

### Checkpoint B - Builds and core gameplay vocabulary

#### NT-04 - Build system, category, major slot, support slot, and preset names

- **Status:** Resolved 2026-09-09.
- **Decision:** `Battle Style` names the editable specialization system; `Focus`
  names a freely mixable functional category; `Anthem` names the single major
  behavior-changing rule; `Riff` names each of three smaller supporting rules;
  and `Loadout` names the complete saved instrument/gear/abilities/consumables/
  Battle Style setup. A Battle Style contains one Anthem and three Riffs; three
  player-named Loadouts replace all player-facing `full spec preset` language.
- **Decision needed:** What coherent music-warrior vocabulary replaces
  `Discipline`, `Build Core`, `Technique`, and the functional full-spec-preset
  language?
- **Must resolve:** Final player-facing name for the overall specialization/edit
  system; one option category; the single major behavior-changing slot; each of
  the three smaller supporting slots; a saved complete configuration/preset;
  editor action verbs; singular/plural/short forms; and clear distinction from
  instruments, gear, abilities, and permanent classes.

#### NT-05 - Category, beginner-template, and future option naming grammar

- **Status:** Resolved 2026-09-09.
- **Decision:** The four Focuses are `Drive` for offense/Momentum/danger,
  `Guard` for Ward/Defend/mitigation/revival, `Unity` for teammate/Call/
  Crescendo/Cohesion play, and `Pulse` for Hype/Signature/movement utility/
  hybrids. Beginner Loadouts are `Break Through`, `Guard the Band`, and `Ready
  for Anything`. Anthems use strong two-to-four-word statement names; Riffs use
  concise one-to-three-word names; every option immediately explains its real
  condition, tradeoff, and effect.
- **Decision needed:** How should the four functional categories, three beginner
  templates, and future major/supporting options be named as one family?
- **Must resolve:** Final labels for offense/danger, protection/recovery,
  teamwork/cooperation, and Hype/Signature/utility categories; final names for
  offense-focused, protection/support-focused, and balanced templates; category
  icon/metaphor compatibility; option-name length and grammatical pattern;
  whether major and supporting options use different naming forms; example names
  for already-described behaviors; and a rule that prevents a category from
  sounding like a locked class.

#### NT-06 - Combat, encounter, cooperation, and role-term audit

- **Status:** Resolved 2026-09-09.
- **Decision:** Keep Attack, Defend, Special, Ward, Resolve, Momentum, Hype, Band
  Call, Crescendo, Risk, Revive, instrument, role, band, boss, encounter, passage,
  phrase, and music staff. Shorten `Signature Special` to `Signature` outside the
  Special control; replace player-facing `Cohesion Bonus` with `Band Harmony`,
  `acolyte` with `Order Ally`, and solo emergency recovery with `Last Stand`.
  Keep Near/Middle/Rear internal; player-facing locations use authored names plus
  High Risk/Balanced/Low Risk indicators.
- **Decision needed:** Which established gameplay terms should remain, change, or
  receive a clearer short/long form?
- **Must resolve:** Attack, Defend, Special; Ward; boss Resolve and its layers;
  Momentum; Hype and Signature Special; Band Call and Crescendo; Cohesion; Risk;
  Near/Middle/Rear positions; movement/recovery/revival language; instrument,
  role, band, acolyte, encounter, boss, passage, phrase, and staff; term collisions
  across systems; and whether any strong internal semantic term should differ
  from the player-facing world term.

### Checkpoint C - Progression, UI voice, and publication

#### NT-07 - Progression, rewards, economy, and item-family names

- **Status:** Resolved 2026-09-09.
- **Decision:** Use `Song Restoration`, `Order Rank`, and `Boss Mastery` for the
  three progression tracks; `Resonance` for the general resource; `Boss Echoes`
  with boss-themed forms for specific materials; and `Song Fragment` for a
  recovered campaign piece. Keep Instrument, Ward Core, Resonator, Trait,
  Cosmetic, Recipe, Record, Upgrade, Craft, Salvage, Unlock, Rank, and Tier.
  Replace consumables with `Battle Supplies` and technical `uplift` with
  player-facing `Reforge` for raising old equipment to the current tier.
- **Decision needed:** What vocabulary makes long-term growth and rewards feel
  like restoration through music rather than generic RPG accounting?
- **Must resolve:** Final names for campaign restoration, general player growth,
  and per-boss mastery; general resource; boss-material naming pattern; Shattered
  Song fragment language; item-family review for Instruments, Ward Cores,
  Resonators, traits, cosmetics, consumables, recipes, and records; upgrade,
  uplift, craft, salvage, unlock, rank, and tier verbs/nouns; and how boss-specific
  names vary without hiding function.

#### NT-08 - Hub navigation, onboarding, Results, failure, and action copy

- **Status:** Resolved 2026-09-09.
- **Decision:** Finalize Play/Prepare/Progress/Band; Settings/Controls/
  Accessibility/Help; Store; Quick Play/Begin Practice/Skip Practice; Victory/
  Defeat/No Contest; Continue Story/Retry/Stay with Band/Return to Hub; and the
  familiar Ready/Apply/Save/Cancel/Reset/Repair actions. Use Join Call/Pass, Join
  Crescendo/Pass, and Join In for the distinct opt-ins. Retain the six plain
  preset pings. No Contest and errors state what happened, what was preserved,
  and the next valid action without codes, blame, jokes, or lore.
- **Decision needed:** What are the final concise labels and voice patterns for
  the most common player journeys?
- **Must resolve:** Four primary destinations currently called Play, Prepare,
  Progress, and Band; Settings/Controls/Accessibility/Help/Store relationship;
  Begin/Skip Practice; Continue Story, Retry, Stay with Band, Return to Hub;
  Victory, Defeat, Invalid/No Contest and exact-reason presentation; Ready,
  Apply, Cancel, Reset, lock/unavailable/error/recovery language; group invitation
  accept/decline; safe preset messages; and when clarity requires retaining a
  familiar functional word.

#### NT-09 - Style guide, catalog completion, validation, and change control

- **Status:** Resolved 2026-09-09.
- **Decision:** Use Title Case for named systems/resources/world terms/buttons and
  sentence case for prose; no terminal punctuation on buttons; complete sentences
  for instructions/errors; numerals for scannable gameplay values; short,
  pronounceable, unique labels with at most one documented short form; and 30%
  localization-expansion tolerance. Future names follow actual definitions.
  Publication checks meaning, tone, clarity, language mechanics, localization,
  accessibility, moderation, obvious conflicts, and placeholder leakage. The
  game owner approves player strings; semantic owners verify accuracy. Protected
  language changes require amendment; final-title selection remains separate.
- **Decision needed:** What compact rules make future names and copy consistent,
  localizable, reviewable, and safe to publish?
- **Must resolve:** Capitalization, punctuation, contractions, numerals, term
  length, pronunciation, abbreviations, tooltips, capitalization of world terms,
  glossary ownership, localization-key mapping, duplicate/confusing-name checks,
  Roblox moderation/platform review, basic originality/conflict checking,
  accessibility and 30%-expansion review, names for not-yet-authored catalogs,
  document-update scope, placeholder audit, approver, and material-change rule.

## 4. Completion criteria

The Naming & Tone pass is complete only when:

- NT-01 through NT-09 are explicitly resolved;
- final-title selection is explicitly tracked as a separate publication gate and
  `Bands Battle` is never mistaken for an approved shipping title;
- the rejected `Discipline`, `Build Core`, and `Technique` player labels have
  coherent replacements;
- the build system hierarchy, four categories, beginner templates, and future
  option naming grammar are unambiguous without implying permanent classes;
- every established combat, cooperation, progression, economy, item-family,
  navigation, onboarding, Results, and common-action term is marked `Keep`,
  `Replace`, `Player-facing alias`, or `Internal only`;
- all final terms align with the supernatural music-warrior fantasy, target-age
  clarity, nonshaming tone, visual language, and functional mechanics;
- the canonical glossary distinguishes internal identity, localization key,
  full label, short label, description use, capitalization, and status;
- not-yet-authored boss/item/ability/build-option catalogs have a usable naming
  grammar and publication gate rather than fabricated final inventories;
- UI copy rules preserve exact consequence, price, lock, failure, safety, and
  accessibility meaning;
- localization, length, pronunciation, collision, moderation, and placeholder
  checks are operationally clear;
- approved canonical documents are reconciled so obsolete placeholders do not
  continue to appear as player-facing recommendations; and
- `NAMING_AND_TONE.md` passes local link, terminology, glossary uniqueness,
  placeholder, capitalization, and cross-document consistency audits.

## 5. Change log

- **2026-09-09:** Created the nine-question interview plan and opened checkpoint A.
- **2026-09-09:** Approved NT-01's five voice pillars. Interview progress is 1 of
  9 questions.
- **2026-09-09:** Approved NT-02's action-first clarity, restrained flavor, humor,
  and prohibited-language rules. Interview progress is 2 of 9 questions.
- **2026-09-09:** Partially resolved NT-03: `Bands Battle` is explicitly not the
  final title; the Order, Shattered Song/Song, shards/fragments, and Resolve are
  protected. Final-title direction remains open, so progress remains 2 of 9.
- **2026-09-09:** Completed NT-03 by deferring final-title selection to a separate
  future publication task while preserving the protected anchors and internal/
  player-string boundary. Checkpoint A is complete at 3 of 9 questions.
- **2026-09-09:** Approved NT-04's Battle Style, Focus, Anthem, Riff, and Loadout
  hierarchy. Interview progress is 4 of 9 questions.
- **2026-09-09:** Approved NT-05's four Focuses, three beginner Loadouts, and
  Anthem/Riff naming grammar. Interview progress is 5 of 9 questions.
- **2026-09-09:** Approved NT-06's core gameplay-term audit and four player-facing
  cleanups. Checkpoint B is complete at 6 of 9 questions.
- **2026-09-09:** Approved NT-07's progression, resource, item-family, and Reforge
  vocabulary. Interview progress is 7 of 9 questions.
- **2026-09-09:** Approved NT-08's final navigation, onboarding, Results, opt-in,
  ping, action, and failure-copy patterns. Interview progress is 8 of 9 questions.
- **2026-09-09:** Approved NT-09, completing the interview. Reconciled all nine
  decisions into canonical `NAMING_AND_TONE.md` and began the authoritative
  document terminology pass.
