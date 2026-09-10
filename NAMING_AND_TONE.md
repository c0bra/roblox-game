# Bands Battle Naming & Tone

- **Status:** Approved
- **Approved:** 2026-09-09
- **Working-title notice:** `Bands Battle` is an internal project title, not the
  final public game title
- **Required by:** [`GAME_DESIGN.md`](GAME_DESIGN.md#required-product-follow-up)
- **Vision authority:** [`GAME_VISION.md`](GAME_VISION.md)
- **Visual authority:** [`ART_DIRECTION.md`](ART_DIRECTION.md)
- **Build-system input:** [`BUILDS_AND_SPECIALIZATION.md`](BUILDS_AND_SPECIALIZATION.md)
- **UI language input:** [`UI_UX.md`](UI_UX.md)
- **Decision source:** [`NAMING_AND_TONE_WORKING.md`](NAMING_AND_TONE_WORKING.md)
- **Interview plan:** [`NAMING_AND_TONE_QUESTIONS.md`](NAMING_AND_TONE_QUESTIONS.md)

## 1. Role and authority

This document defines the approved player-facing vocabulary, voice, and naming
grammar for the project currently called Bands Battle. It replaces the rejected
build placeholders, resolves established gameplay/progression/UI terms, and
prevents internal names from leaking into published player text.

It owns:

- the five voice pillars and prohibited tonal directions;
- protected world-language anchors;
- the final player-facing Battle Style hierarchy;
- final Focus, beginner Loadout, gameplay, progression, economy, navigation,
  Results, and common-action vocabulary;
- the canonical internal-to-player glossary;
- future boss/item/ability/Anthem/Riff/catalog naming grammar;
- capitalization, punctuation, length, pronunciation, and localization rules;
  and
- player-language publication validation and amendment rules.

It does not change gameplay behavior, semantic ownership, UI hierarchy,
formulas, balance, content structure, accessibility, economy, or technical
identity. The owning system remains authoritative for what a concept does.
Internal identities and localization keys may remain stable while the approved
player-facing label changes.

The final game title is deliberately not selected here. `Bands Battle` remains a
repository/project label only. Final-title selection is a separate publication
task and does not block Technical Architecture or use of the approved system
language below.

## 2. Five voice pillars

### Mythic

Language suggests ancient supernatural stakes, monumental forces, corruption,
harmony, and restoration. It does not fall back to generic medieval jargon or
borrowed real-world occult names merely to sound mysterious.

### Electric

Words carry the punch, speed, and spectacle of dark EDM, cinematic wall-of-sound
production, and K-pop performance energy. They do not depend on trendy music-
industry slang, forced memes, or guitar-only heavy-metal parody.

### Heroic

The player is an active musical warrior. Timing, reading, courage, movement,
protection, and cooperation cause the result. Copy uses decisive active voice
rather than treating the avatar as a passive collection of stats.

### Hopeful

Darkness is opposed by fellowship, honest performance, recovery, and
restoration. Victory is earned and luminous. Defeat leaves a credible next step
instead of humiliation, hopelessness, or blame.

### Clear

A player can understand the action, danger, consequence, cost, lock reason, and
next step on a phone without first learning the mythology or decoding a system
term.

Context changes emphasis without changing voice:

- combat warnings are Clear, Electric, and urgent;
- system/category/option names are Mythic and Electric but pronounceable;
- victory/restoration emphasize Heroic and Hopeful;
- Defeat, recovery, unavailable states, and errors are Clear, calm, and
  nonblaming;
- lore may become more Mythic after immediate meaning is established; and
- humor is affectionate relief among allies, not mockery of the player or threat.

## 3. Clarity and thematic boundaries

Core buttons, choices, warnings, and recovery actions use familiar direct
language, normally one to three English words. A player should recognize the
requested action without learning the world's lore.

Flavorful names are encouraged for systems, powers, rewards, and world concepts
when they remain distinctive and accurate. Their nearby first sentence explains
the function and consequence plainly. Later optional text may add mythology,
imagery, and character.

Do not stack unfamiliar metaphors in one label. Invented words are rare,
pronounceable, and introduced beside a familiar explanation. Critical copy never
uses poetry to hide a target, response, cost, lock, failure, or recovery action.

Natural contractions are allowed. Exclamation marks are occasional emphasis,
not the default voice. Avoid all-caps shouting, forced slang, memes, and phrases
likely to age quickly.

Humor belongs in allies, character moments, recovery, and optional hub content.
It does not belong in critical instructions, boss warnings, Defeat reasons,
safety/privacy notices, transaction failures, or errors.

Language may evoke supernatural danger, corruption, sacrifice, and desperate
resistance. It never becomes graphic gore, horror designed to frighten, profanity,
sexual content, direct real-world preaching, player shame, teammate accusation,
or coercive purchase pressure.

## 4. Protected anchors and title boundary

These world anchors are protected:

- **the Order:** the musical-warrior organization;
- **the Shattered Song:** the broken supernatural whole at the center of the
  conflict;
- **the Song:** the living greater musical force when the full name is not
  required;
- **shards:** the world destinations and major spaces; and
- **Song Fragments:** recovered campaign pieces of the Shattered Song.

**Resolve** is the approved player-facing boss-resistance resource and applies to
its sequential Resolve layers.

`Bands Battle` is explicitly rejected as the final public title. It may remain
in repository paths, internal documents, and historical references until a
separate title task selects and validates the replacement. Public branding may
not represent it as an approved title.

## 5. Canonical glossary contract

Every player-facing concept records:

- stable internal concept or identity;
- owning semantic system;
- localization key;
- one official full label;
- at most one approved short label for constrained contexts;
- meaning and appropriate surfaces;
- capitalization and plural/parameter behavior;
- `Approved`, `Internal only`, `Rejected`, `Deferred`, or `Retired` status; and
- replacement/alias history when a prior player label existed.

Aliases may support authoring search and migration but do not create multiple
interchangeable shipping names. Missing approved mappings and unresolved
placeholders block publication on the affected surface.

The table below is the canonical player-language baseline. Concrete localization
keys are assigned by each owning catalog/Technical Architecture without changing
the labels.

| Internal concept | Approved player-facing label | Status and use |
|---|---|---|
| Project/game title | — | Deferred separate publication task; `Bands Battle` is internal only |
| Musical-warrior organization | the Order | Protected world anchor |
| Broken supernatural whole | the Shattered Song | Protected world anchor |
| Greater living musical force | the Song | Protected short world anchor |
| World destination | shard | Protected generic world term |
| Campaign collectible | Song Fragment | Protected campaign term; `Fragment` only when context is unambiguous |
| Boss resistance resource | Resolve | Approved; includes Resolve layers |
| Specialization configuration | Battle Style | Approved overall editable system |
| Functional build category | Focus | Approved; plural `Focuses`; never a class/path lock |
| Major build rule | Anthem | Approved; exactly one per Battle Style |
| Supporting build rule | Riff | Approved; exactly three distinct Riffs per Battle Style |
| Complete saved setup/full spec preset | Loadout | Approved; includes instrument, gear, abilities, Battle Supplies, and Battle Style |
| Offense/Momentum/danger Focus | Drive | Approved; not an attacker class |
| Ward/Defend/Revive Focus | Guard | Approved; not a tank class |
| Teammate/Call/Crescendo Focus | Unity | Approved; not a support-class requirement |
| Hype/Signature/movement Focus | Pulse | Approved utility/hybrid Focus; does not own musical time |
| Offense beginner template | Break Through | Approved complete starter Loadout |
| Protection/support beginner template | Guard the Band | Approved complete starter Loadout |
| Balanced beginner template | Ready for Anything | Approved hybrid starter Loadout and safe default |
| Combat choices/controls | Attack / Defend / Special | Approved familiar controls |
| Player survival protection | Ward | Approved keep |
| Banked closed-layer pressure | Momentum | Approved keep |
| Signature readiness resource | Hype | Approved keep |
| Signature Special | Signature | Approved normal term; persistent control remains `Special` |
| Player-started cooperative action | Band Call | Approved keep |
| Encounter-started group performance | Crescendo | Approved keep |
| Cohesion/Cohesion Bonus | Band Harmony | Approved player-facing alias; internal Cohesion identity may remain |
| Banked positional exposure | Risk / Risk Bonus | Approved keep |
| Cooperative downed-player recovery | Revive | Approved keep |
| Solo emergency recovery | Last Stand | Approved; direct beat instruction always accompanies first use |
| Acolyte NPC | Order Ally | Approved player-facing alias; never a human bandmate |
| Near/Middle/Rear position bands | Authored location + High Risk / Balanced / Low Risk | Internal distance bands do not become abstract location buttons |
| Rhythm display | music staff | Use on first teaching/ambiguous mention; `staff` is approved short form |
| Campaign progression | Song Restoration | Approved first-clear/destination track |
| General player progression | Order Rank | Approved private progression; not a competitive skill rank |
| Per-boss progression | Boss Mastery | Approved keep |
| General earned resource | Resonance | Approved nonpremium resource |
| Boss-specific material family | Boss Echoes | Approved family; boss-authored form may vary without using `Fragment` |
| Consumables | Battle Supplies | Approved prepared limited-charge item family |
| Old-item tier elevation/uplift | Reforge | Approved; raises eligible gear to current Tier without reroll/destruction |
| Primary destinations | Play / Prepare / Progress / Band | Approved final labels and order |
| Global utilities | Settings / Controls / Accessibility / Help | Approved final labels |
| Commerce destination | Store | Approved secondary label; never a forced recovery/Results action |
| Fast Play | Quick Play | Approved replacement |
| Practice actions | Begin Practice / Skip Practice | Approved |
| Encounter outcomes | Victory / Defeat / No Contest | Approved; `Invalid` remains internal qualifier |
| Results next actions | Continue Story / Retry / Stay with Band / Return to Hub | Approved |
| Common configuration actions | Ready / Apply / Save / Cancel / Reset / Repair | Approved |
| Band Call invitation | Join Call / Pass | Approved neutral opt-in |
| Crescendo invitation | Join Crescendo / Pass | Approved neutral opt-in |
| Dropout universal-beat opt-in | Join In | Approved; movement exits automatically |
| Preset pings | Move / Defend / Join Call / Revive / Ready / Thanks | Approved first-release set |
| Discipline | Focus | Rejected former player label; internal/historical use only |
| Build Core | Anthem | Rejected former player label; internal/historical use only |
| Technique | Riff | Rejected former player label; internal/historical use only |

Clear item-family names remain **Instrument**, **Ward Core**, **Resonator**,
**Trait**, **Cosmetic**, **Recipe**, and **Record**. Clear economy/action nouns
remain **Upgrade**, **Craft**, **Salvage**, **Unlock**, **Rank**, and **Tier**.
Use qualified forms such as `Order Rank`, `Mastery Rank`, and `Gear Rank` where
an isolated `Rank` could be ambiguous.

## 6. Battle Style language

A player edits a **Battle Style**. It contains exactly one **Anthem** and three
distinct **Riffs**. Each option has one primary **Focus** and may have cross-Focus
tags. Focuses support browsing and explanation; they are not classes, permanent
paths, purity requirements, or party-composition roles.

A complete saved **Loadout** contains the selected instrument/role, Instrument,
Ward Core, Resonator, Signature, Band Call, Battle Supplies, and Battle Style.
Players receive three player-named Loadouts. `Current Loadout` is the active
complete setup in the hub or staging.

Common language is:

- Choose or Replace an Anthem/Riff.
- Preview or Apply a Battle Style.
- Save, Apply, Rename, or Repair a Loadout.
- A Loadout may be `Incomplete` when a referenced choice is unavailable; the
  Current Loadout remains unchanged until Repair and successful Apply.

Lower-case `build` may explain the familiar idea, but named headings and player
system references use **Battle Style**. None of these names changes internal
configuration, option, preset, loadout, validation, or snapshot semantics.

## 7. Focuses and beginner Loadouts

The four Focuses are:

- **Drive:** offense, Momentum, and deliberate dangerous-position play;
- **Guard:** Ward, Defend, mitigation, protection, and Revive;
- **Unity:** teammate support, Band Calls, Crescendos, and Band Harmony; and
- **Pulse:** Hype, Signatures, movement-triggered utility, resource timing, and
  hybrids.

Use `Drive Focus`, `Guard Focus`, `Unity Focus`, or `Pulse Focus` when the
category noun is not already visible. The short name stands alone inside an
explicitly labeled Focus list.

The beginner Loadouts are:

- **Break Through:** offense-focused;
- **Guard the Band:** protection/support-focused; and
- **Ready for Anything:** balanced hybrid and skipped-selection default.

They are complete starting Loadouts, not permanent archetypes, ranked
recommendations, or extra slots. Players may preview their plain included choices
and freely replace them under the existing rules.

Future **Anthems** normally use strong two-to-four-word statement/image names
that feel defining. Future **Riffs** normally use concise one-to-three-word
action/image names that feel supporting. Clarity may justify an exception.

Every option pairs its flavor name with an immediate plain description of its
trigger/condition, tradeoff, effect, fallback, and important cap. Names cannot
promise unavailable power, conceal a drawback, or collide with a protected
resource/control/system term.

## 8. Combat, encounter, and cooperative language

Keep **Attack**, **Defend**, and **Special** as the stable controls. A player's
equipped defining ability is their **Signature**; the Special control activates
or arms it under the existing rules.

Keep **Ward**, **Resolve**, **Momentum**, **Hype**, **Band Call**, **Crescendo**,
**Risk**, and **Revive**. Their approved mechanics and semantic owners do not
change.

Use **Band Harmony** wherever the player sees the internally named Cohesion Bonus.
It communicates broad successful participation without exposing a rank or
blaming a weaker performer.

Use **Order Ally** for solo-support NPCs. The term distinguishes them from human
bandmates while connecting them to the Order. Their fixed-support limitations
remain visible when relevant.

Use **Last Stand** for the one-use solo emergency recovery. First teaching pairs
it with a literal instruction such as `Hit the beat to get back up.`

Near/Middle/Rear remain internal distance and balance bands. Each arena exposes
an authored physical location name plus **High Risk**, **Balanced**, or **Low
Risk**. The label describes the approved tradeoff without making the location
look like an abstract difficulty button.

`First Clash`, `Escalation`, and `Climax` may remain internal/authoring labels for
the three normal Resolve layers; the player needs the visible layer structure,
not universal chapter names.

## 9. Progression, rewards, and economy language

The three progression tracks are:

1. **Song Restoration:** first-clear progress that recovers a Song
   Fragment and opens the next destination.
2. **Order Rank:** broad private progression that opens systems, Loadouts,
   Battle Style options, and wider equipment access while retaining finite
   direct power.
3. **Boss Mastery:** per-boss progression shared across instruments for that boss,
   while personal Records remain instrument/difficulty-specific.

**Resonance** is the general earned resource used for ordinary Upgrades,
Crafting, Battle Supplies, and normal shop purchases. It is not premium currency
or bought progression.

**Boss Echoes** is the boss-specific material family. A boss may use a short
themed form such as `Blackened Crown Echoes` when the relationship remains clear.
Boss Echoes never use `Fragment`, reserved for Song Restoration.

Use **Battle Supplies** for prepared consumable items. Descriptions still state
charges, exact effects, limits, and inability to bypass Defeat.

Use **Reforge** when raising eligible old gear to the player's current Tier. The
first explanation states cost, carried-forward value, and that the item is not
rerolled, destroyed, or silently changed.

Boss-specific resources, items, recipes, traits, cosmetics, titles, and
Signatures are named only after their actual definitions and fiction exist.

## 10. Navigation, onboarding, Results, and actions

The four primary destinations remain in this order:

1. **Play:** Continue Story, shard/encounter selection, difficulty, and replay.
2. **Prepare:** Current Loadout, inventory, gear, upgrades, Battle Styles,
   abilities, and Battle Supplies.
3. **Progress:** Song Restoration, Order Rank, Boss Mastery, Records, collection,
   and restored world state.
4. **Band:** party, public matching, safe communication, and compatible social
   state.

Keep **Settings**, **Controls**, **Accessibility**, and **Help** global. **Store**
is a secondary voluntary destination, never a primary task, Results action,
recovery, or error solution.

Use **Quick Play**, **Begin Practice**, and **Skip Practice**. Results uses
**Victory**, **Defeat**, or **No Contest**, followed by the exact reason and
already-confirmed value.

No Contest receives this baseline explanation:

> The encounter couldn't finish fairly. This attempt doesn't count.

A more specific safe reason may follow. `Invalid` remains an internal state
qualifier rather than the player headline.

Results actions are **Continue Story**, **Retry**, **Stay with Band**, and
**Return to Hub**. Existing eligibility selects the leading action; copy does not
promise an unavailable route.

Standard actions are **Ready**, **Apply**, **Save**, **Cancel**, **Reset**, and
**Repair**. Their meanings remain consistent across surfaces.

Cooperative invitations use **Join Call / Pass** and **Join Crescendo / Pass**.
`Pass` is a neutral valid choice. Optional universal beat activity uses **Join
In**. First-release pings are **Move**, **Defend**, **Join Call**, **Revive**,
**Ready**, and **Thanks**.

## 11. Failure and recovery copy

Error and unavailable-state copy gives these facts when applicable:

1. what happened or is unavailable;
2. what remained safe or preserved; and
3. the next valid action.

Use complete direct sentences. For example:

> This Loadout is incomplete. Your current Loadout is unchanged. Repair the
> missing choices before you apply it.

Player copy does not expose raw technical codes, assign blame, make a joke, hide
the consequence in lore, or route to Store. A private support/diagnostic code may
exist outside the primary message.

Defeat copy identifies the frozen encounter reason without labeling one player
as the cause. Accessibility, device, age, spending, private performance, or
teammate comparison never becomes a public label.

## 12. English style and localization

- Use **Title Case** for named systems, resources, world terms, game states, and
  UI buttons in English.
- Use sentence case for ordinary descriptions, instructions, status text,
  explanations, and errors.
- Button/navigation labels have no terminal punctuation.
- Instructions, errors, recovery explanations, transaction consequences, and
  status messages use complete sentences.
- Use numerals for gameplay values, quantities, costs, Ranks, Tiers,
  percentages, and countdowns. Narrative prose may spell out small numbers when
  it does not communicate a mechanic.
- Core action/navigation labels normally use one to three English words.
  Distinctive Anthem names may use up to four. Clarity may override the drafting
  target.
- Every concept has one full label and at most one approved short label.
- Read names aloud. Avoid difficult invention, accidental homophones, sound-
  alike critical cues, visually similar neighboring labels, and repeated
  acronyms.
- External localization keys support parameters, plurals, numbers, and time.
  Never build sentences from translated fragments or place the only text inside
  an image.
- Reserve roughly 30% expansion from English and test long strings, Unicode,
  plural/parameter variants, text scaling, and supported assistive presentation.
- A short English source cannot require truncating a clear localized label.

Capitalization marks the named game concept, not every ordinary use of a word.
For example, `Choose a Guard Focus` names the category, while `guard the route`
uses an ordinary verb.

## 13. Future catalog naming

Do not name a boss, item, ability, Anthem, Riff, Recipe, Trait, Cosmetic, title,
or Boss Echoes variant before its owned definition/content exists.

When it exists, the naming record includes:

- stable catalog identity and semantic owner;
- actual function, fiction, source, and player context;
- candidate full/short labels and pronunciation;
- direct plain description;
- relevant family/grammar and protected-term interactions;
- uniqueness/conflict/moderation/localization checks; and
- owner approval and revision status.

Boss-specific terms may be more mythic than core controls, but must remain
pronounceable and functionally explained. Instrument-neutral systems avoid names
that imply every song contains guitar, drums, bass, vocals, or another specific
role.

## 14. Publication validation

Before a player string or named catalog revision publishes, verify:

- the label matches the exact mechanic and promises no unavailable effect;
- voice aligns with Mythic, Electric, Heroic, Hopeful, and Clear;
- language avoids the prohibited horror/preaching/shame/blame/coercion bounds;
- spelling, capitalization, plural/parameter behavior, pronunciation, and any
  short form are approved;
- no confusing reuse exists across systems, resources, items, bosses, actions,
  cues, navigation, or the same screen;
- the localization key and roughly 30%-expanded/text-scaled forms fit and retain
  accessible alternatives;
- current Roblox moderation/platform requirements are satisfied;
- a basic public search finds no obvious naming conflict before public branding
  or a major catalog name is locked; and
- no rejected placeholder or internal key appears on the affected player surface.

The game owner approves the player-facing string. The semantic owner verifies
mechanical accuracy. UI/UX, Audio Presentation, localization, accessibility,
Content Authoring, and policy review join only when their surfaces are affected.
One person may cover multiple responsibilities during development.

## 15. System and document handoffs

- **Gameplay/product systems** retain semantic keys, definitions, and catalog
  entry ownership while using this glossary for player strings.
- **Builds & Specialization** maps its internal configuration/category/major/
  supporting/preset vocabulary to Battle Style/Focus/Anthem/Riff/Loadout.
- **Items & Equipment** uses Loadout, Battle Supplies, and Reforge while retaining
  durable internal record meanings.
- **Abilities** uses Signature, Band Harmony, Order Ally, Last Stand, and the
  approved cooperative invitation labels without changing mechanics.
- **Progression and Rewards** use Song Restoration, Order Rank, Boss Mastery,
  Resonance, Boss Echoes, Song Fragments, and the clear retained action terms.
- **UI/UX and Audio Presentation** own layout, cueing, voice performance, and
  localized surface implementation; player wording follows this contract.
- **Content Authoring** applies the future catalog grammar after actual content
  exists and blocks missing/rejected labels at publication.
- **Player Data** preserves stable identities and localization preferences
  without treating displayed text as durable semantic truth.
- **Technical Architecture** defines localization schemas/catalog loading,
  fallback behavior, versioning, validation tooling, and code identities without
  exposing internal placeholders.

Historical question/working records may retain rejected terms as decision
evidence. Canonical documents use the approved player label or explicitly mark a
stable internal term and its mapping.

## 16. Deferred naming work

This system-language and tone contract is complete. Deliberately deferred work
includes:

- selecting and validating the final public game title;
- naming actual bosses, shards, songs, encounters, arenas, NPCs, Signatures,
  Band Calls, Anthems, Riffs, items, Traits, Recipes, Cosmetics, titles, and boss-
  material variants after their content/definitions exist;
- assigning concrete localization keys and producing translations;
- voice casting/performance direction and localized audio recording; and
- current platform moderation and public-conflict review for each final catalog.

None blocks Technical Architecture. Each becomes a publication gate only for the
affected public title, content package, catalog, or localized surface.

## 17. Approval and change control

The owner interview resolved NT-01 through NT-09 on 2026-09-09. This document is
the canonical Naming & Tone product-language specification.

A material change to the five voice pillars, protected anchors, Battle Style
hierarchy, Focus/beginner-Loadout names, retained/replaced gameplay vocabulary,
progression/resource families, navigation/Results/action patterns, thematic
boundaries, or glossary/publication authority requires an explicit amendment
citing the superseded decision.

Ordinary future catalog entries and copy refinements may be approved under these
rules without reopening the whole specification. They still require semantic
accuracy, catalog/version history, localization/accessibility validation, and no
placeholder leakage.
