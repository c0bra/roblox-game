# Bands Battle Game Vision

- **Status:** Vision v1 baseline; design amendments recorded through 2026-08-17
- **Purpose:** Product-level instructions for development decisions

## 1. Role of this document

This document defines the game Bands Battle is trying to become. It is the
product north star for developers, designers, artists, writers, and content
creators. When several technically valid ideas compete, prefer the idea that
best serves this vision.

Related documents have narrower responsibilities:

- [`GAME_VISION_QUESTIONS.md`](GAME_VISION_QUESTIONS.md) records the completed
  bounded owner interview and governs any future vision questions.
- [`GAME_DESIGN.md`](GAME_DESIGN.md) develops specific mechanics and systems.
- [`ART_DIRECTION.md`](ART_DIRECTION.md) owns the visual language and aesthetic
  standards.
- Files under `roblox/web/` document a retired browser prototype. They may serve
  as implementation history but are not product or design authority.
- OpenSpec changes define the scope and acceptance criteria for individual
  development efforts.
- READMEs and technical specifications describe the repository and its tools.

This document should remain more stable than any prototype or implementation.
It should explain why a feature belongs in the game, not prescribe its code or
asset pipeline.

## 2. Vision statement

Bands Battle is a supernatural boss-combat game controlled through rhythm,
blended with the progression and player expression of a musical action RPG.

Players join an order of musical warriors in a fully supernatural world. They
fight enormous bosses, recover fragments of a Shattered Song, grow in power,
master songs, and develop flexible combat roles. Battles are playable solo but
reach their fullest expression when a band coordinates its individual parts
into group attacks and defenses.

## 3. Intended players

The primary audience is approximately 10 to 14 years old.

The game should be immediately understandable and forgiving at its lowest skill
levels while still offering meaningful challenge. It may become moderately hard
and should provide multiple difficulty levels so new, improving, and highly
skilled players can participate without flattening the experience for everyone.

Difficulty should come from learning rhythm, reading boss behavior, choosing
positions, and coordinating abilities. Complexity should not come from unclear
controls or unreadable presentation.

### Product platform

Bands Battle is a native Roblox experience. Native Roblox is the only shipping
product and supported gameplay runtime. The browser prototype is retired: it
does not require feature parity, new gameplay investment, content production, or
release validation. Existing browser files may remain temporarily as historical
reference until a separate cleanup decision archives or removes them.

Device development priority is phone and tablet touch first, desktop keyboard
and mouse second, and gamepad and console third. This order determines design and
validation investment, not separate versions of the game. All supported devices
share the same encounters, progression, rewards, matchmaking, rhythm standards,
and competitive expectations. Controls, layout, safe areas, latency calibration,
and presentation may adapt to the device without changing what success means.

## 4. Core player promise

Players should feel that they are fighting a real supernatural battle through
music. Their performance should visibly and audibly cause attacks, protection,
recovery, movement, and changes in the battle rather than merely increasing an
abstract score.

The desired emotional rhythm is:

1. Perform under pressure.
2. Read the boss and anticipate danger.
3. Choose how much risk to take.
4. Dash to a better defensive or offensive position.
5. Survive or counter a major attack through rhythmic action.
6. Rejoin the performance with renewed momentum.
7. Combine the band's power for an earned spectacle moment.

Battles need phases and breathing room. Constant dense input would leave no room
to observe the boss, make tactical decisions, appreciate character actions, or
coordinate with the band.

## 5. Combat principles

### Rhythm controls combat

Rhythm is the player's means of acting, not a separate minigame placed over the
battle. Timing and musical performance should drive attacks, defenses, healing,
movement, group abilities, and other combat consequences.

### No permanent song-wide note highway

The intended game does not include a permanent, song-wide scrolling note highway
as a boss, rehearsal, or song-mastery mode. Active performance passages may use
a compact moving staff near the bottom of the screen: notes travel right to left
toward a fixed strike line while the input targets remain stationary. The staff
may continue across several chained phrases, but it recedes when there are no
playable notes or a meaningful encounter event calls for full attention.

This bounded staff is a timing instrument inside the battle, not the primary
visual world of the game. Rhythm cues must preserve attention for the boss,
performers, positions, and arena. The retired browser prototypes, including the
Classic Highway, may remain temporarily as implementation history but are not
development or validation targets.

### Rhythm cues bridge interface and world

Rhythm timing should be communicated through a blend of a compact phrase-bounded
moving staff and cues inside the battle. The staff provides precise, familiar
time-to-impact information. Boss poses, attack paths, performer animation,
position states, environment responses, and sound communicate what the
performance means and what will happen. Neither layer should carry required
information alone. The interface supports attention to the battle instead of
becoming the battle.

Phrases group notes for readability, judgments, combat contribution, and clean
intent boundaries; they do not grant arbitrary permission to perform. When a
player is settled at a valid position and the selected instrument has authored
notes available, those notes should generally be playable. Multiple phrases may
chain into a sustained performance passage without forced downtime. Breaks
should follow the actual arrangement, player movement, boss knockback, a phase
transition, recovery, repositioning, or another meaningful song-authored event
rather than occurring automatically at every phrase boundary.

### The song authors the encounter

The song is the encounter's master structure. Boss behavior, player phrases,
movement windows, cooperative moments, and breathing room should be composed
around its sections, energy, transitions, downbeats, and rests rather than laid
over it as unrelated cooldowns. The music should feel like it is causing the
battle to unfold.

Encounter authors should interpret each song rather than force every song into
one fixed template. Quiet passages may support recovery, repositioning,
telegraphs, story beats, or defensive preparation. Musical builds can create
mounting danger. Intense passages and climaxes can support denser instrument
phrases, powerful boss patterns, and band abilities. These are tendencies, not a
rigid formula; the character of the specific song has final authority.

All phrase and boss-event scheduling shares one musical clock. A major boss
attack must not begin unexpectedly during a committed phrase and require a
mid-phrase stance change for survival. Telegraphs may overlap performance when
they are readable without abandoning the rhythm task, but every unavoidable
impact must respect an authored decision and reaction window. Players who read
the song and boss early should feel prepared rather than ambushed by competing
cues.

Boss progress should not be presented as conventional biological health. A
meter such as Dissonance, Guard, or Resolve should communicate that musical
combat is breaking the boss's resistance and preparing it for destruction at a
musically appropriate ending. The final in-world name is not decided yet.

The meter is divided into sequential resistance layers tied to scheduled windows
on the song's beat and time grid. These windows do not require reliable labels
such as verse, chorus, or bridge. The band must break each layer before it can
attack the next one, and the encounter timeline does not pause for a late break.
Falling behind therefore leaves less time to clear the remaining layers.

Breaking a layer early must not create dead or invulnerable-feeling play. Further
successful actions bank visible Momentum or bonus stacks until the next layer
opens, then strengthen the band's opening pressure against that layer. Exact
math is a tuning decision, but players should always see that strong early
performance is still producing value.

At the final cadence, the boss is destroyed only if the required resistance
layers have been broken and the band succeeds at the designated final phrase or
set of phrases already present in the song. This is the finishing performance;
it does not require extra coda audio. If the band has not progressed far enough
or fails the finishing performance, the boss withstands the attempt and the run
ends in defeat. Randomness may add bounded variation or bonus effects, but it
should not secretly override the band's earned victory or defeat.

### Mobile performance is fixed and glanceable

The primary mobile direction is landscape play with three large, fixed rhythm
pads. A compact staff may move notes right to left toward a fixed strike line,
but the three touch locations themselves do not move. Notes and pads must share
shape, label, and color reinforcement so the mapping remains quickly readable.
Difficulty should come primarily from phrase rhythm, density, duration, boss
pressure, and coordination rather than from hunting for controls. The exact
visual treatment can evolve, but touch targets must remain generous and the
battle must retain most of the screen and the player's attention.

### Combat intent is separate from rhythm execution

The working control direction gives Attack, Defend, and Special stable,
persistent buttons that activate or queue a combat choice. These controls are
not additional note lanes. The player's timed musical performance determines
how well the chosen action succeeds.

Players may switch between available intents during a phrase as an advanced
tactic. A switch should take effect on the next musically legible input boundary,
such as the next beat or phrase step, with immediate audiovisual confirmation;
it should not reinterpret inputs the player has already performed. The
remaining successful inputs then contribute to the new intent. This can reward
foresight, adaptation, and high-level optimization, but baseline encounter
success must never depend on switching mid-phrase. A player who commits to one
appropriate intent for the full phrase must remain viable.

Certain earned specials, especially group attacks or defenses, may appear as a
temporary prominent cue. Temporary cues should use a consistent reserved
location rather than appearing unpredictably around the screen, and should be
reinforced through label, shape, sound, and restrained motion instead of rapid
flashing alone.

### Group abilities have two entry paths

**Band Calls** are player-started specials from a small equipped ability loadout.
Any eligible player may activate one, and it queues to the next clean musical
boundary. The initiator can produce its base effect alone. Other players receive
a brief, prominent invitation to join, and each participant adds a stack,
multiplier, or other contribution. Declining, moving, or being downed means that
player contributes nothing; it does not cancel the initiator's ability or punish
the rest of the band.

**Crescendos** are larger ensemble opportunities offered by the song and
encounter. They occur only at spaced, pre-authored candidate windows where the
required instrument activity and cue space exist. The runtime may select among
valid candidates for variation, but opportunities never appear at arbitrary
collision-prone moments. Each boss and difficulty defines a guaranteed
opportunity budget, and participating players use their instrument phrases in
the shared window.

On easier difficulties, the encounter may activate at most one additional unused
candidate as a recovery opportunity when the band falls substantially behind.
This assistance is clearly presented, does not guarantee victory, and becomes
less available or disappears on harder difficulties. Exact opportunity counts,
ability-slot counts, multipliers, and participation windows are downstream
design and tuning decisions.

### The boss must command attention

Bosses should perform larger attacks at meaningful authored intervals. Their
preparation, targets, and consequences must be readable early enough for players
to react. Players may need to reach cover, change position, or perform a
defensive ability before impact.

The player should divide attention between musical timing and the battle itself.
A feature that forces continuous focus on an interface while the boss becomes
background scenery works against the intended experience.

### Position creates risk and reward

Players can dash between distinct performance positions. Positions closer to the
boss are more difficult and dangerous but provide greater rewards or combat
effectiveness. Safer positions offer cover and reduced pressure at a meaningful
cost.

Players may begin repositioning at any time, including during an active phrase;
movement is not restricted to special authored windows. Leaving a position
immediately suspends phrase participation. The player gives up the progress they
could have earned while moving but does not accumulate artificial misses for
choosing to relocate.

Positions should change the player's decisions, rhythmic demands, and exposure
to boss attacks. They must feel like real places in the arena, not difficulty
buttons disguised as scenery.

### Defense is active

Taking cover is only one part of defense. Players may also perform rhythmic
defensive abilities, contribute to a group shield, support a recovery, or respond
to a specific boss attack. Defensive play should feel musical and heroic rather
than passive.

### Failure accumulates

One missed note should not cause an arbitrary defeat. Repeated poor rhythm,
failed defenses, and boss damage should accumulate through health, ward, or an
equivalent survival resource.

In co-op, bandmates may revive a fully downed player by voluntarily diverting
musical effort into a short revive phrase. One active bandmate can complete the
revival alone, while additional participants accelerate or strengthen it. A
revive therefore creates an active cooperative choice without requiring the
whole band or letting one nonparticipant cancel the rescue.

Solo play should provide a limited last-chance recovery opportunity before the
run ends. The player earns recovery through a short emergency rhythm challenge
connected to the current song pulse and performed with the same familiar rhythm
controls used in normal combat, not through a disconnected puzzle or random
result. It should be brief, frenetic, and meaningfully difficult rather than
automatic. Success should release the pressure through strong audiovisual
feedback and give the player a clear feeling of relief before combat resumes.
Exact revive and solo-recovery inputs, duration, and recovery strength are
downstream design and tuning decisions.

Active-encounter recovery must be earned through the solo challenge or a
bandmate's revive performance. It cannot be purchased or bypassed with Robux or
another paid currency. Broader monetization boundaries are addressed separately,
but spending must not replace the recovery play or undo an earned defeat.

A failed encounter should still acknowledge worthwhile practice. Players earn
modest song and boss mastery progress plus ordinary crafting materials based on
their performance, so an unsuccessful attempt is not empty. Story progression,
recovery of a Shattered Song fragment, and signature boss drops require victory.
Failure rewards should support learning and the desire to retry without making
intentional failure a better progression strategy than defeating the boss.

## 6. Solo and cooperative play

Both solo and cooperative boss battles are core experiences. Solo must feel
complete and intentionally designed, not like a multiplayer session with missing
players. Cooperative play should be the more exciting and expressive form of the
game.

Solo keeps the same arena geometry and tactical positions used for a band rather
than rebuilding or shrinking encounters by player count. Order acolyte NPCs use
formation offsets within those tactical locations so the arena and the
musical-warrior fantasy do not feel empty. A player and one or more acolytes may
share the same gameplay location; the acolytes automatically arrange around the
human performer, do not consume position capacity, and never block a player's
movement or risk/reward choice. Everyone sharing a location remains subject to
that location's attack geometry and danger.

Acolytes are lightweight support characters, not simulated rhythm players: they
receive no instrument phrases or timing judgments, while the full song mix
continues normally. Their contribution comes from predictable passive pressure
and authored support abilities. Their passive contribution may help but cannot
break resistance layers without the player's successful performance. Exact
acolyte abilities, damage values, formation offsets, and timing are downstream
tuning decisions.

For the MVP, acolytes cannot be permanently downed and do not require inventory,
individual builds, rhythm simulation, or detailed commands. Boss attacks may
knock them away or temporarily disable their support without turning solo play
into an escort mission. During a solo group ability, the human performs the
actual rhythm phrase; acolytes join the presentation and supply a predictable
fixed contribution rather than generating artificial performance scores. Solo
encounter tuning may adjust their support and boss pressure, but the human
player's rhythm, positioning, and intent choices remain decisive.

The intended co-op band size is approximately three to six players. Encounters
and group abilities should remain understandable with a full six-player ensemble
and should scale down without requiring a fixed instrument composition. Groups
larger than a conventional four-piece band are part of the intended fantasy, not
an edge case.

Players should be able to enter co-op through either a preformed band or public
matchmaking. Playing with friends and quickly finding other performers are both
valid paths into the same boss encounters; the game should not require an
established social group before cooperative play becomes available.

For the initial game, the band roster locks when the song and boss encounter
begin. New players do not join an active run. This keeps musical timing, group
abilities, difficulty, and rewards coherent; phase-break drop-in can be
reconsidered after the core co-op experience is proven.

For the initial multiplayer version, the song chart is the sole source of
playable instrument notes and phrase grouping. The game does not need separate
systems that invent personal and group schedules or arbitrarily suppress
available notes. At each authored part of the song, a player's selected
instrument determines the material that player receives. Two or more players
who choose the same instrument receive the same instrument chart, and there is
no restriction on duplicate instruments. A band made entirely of drummers is
valid.

Availability must respect the actual arrangement. A player does not receive a
drum part during a drum dropout simply because the combat system wants an
action. That space becomes natural breathing room for that performer. When the
drum chart does contain notes and the player is settled at a valid position,
those notes should generally remain playable even across several phrase
boundaries. Different instruments may receive different material during the
same song section, but all players remain aligned to the same musical clock.

During an instrument dropout, the player may still receive isolated universal
beat actions derived directly from the song's BPM and beat grid. "Universal"
describes where the timing comes from, not a different kind of action. A beat
action outside a phrase uses the same input, timing judgment, selected combat
intent, reward, and failure consequence as a beat action inside an instrument
phrase. It is not a replacement instrument note and should not imply that the
missing part is playing.

These isolated beat actions become available when the player is stationary at
an arena position, has no active instrument phrase, and is not moving between
positions. A stable **Join In** button or key lets the player opt into the
location's available beat actions. Leaving that location or beginning movement
automatically exits the joined state, so the player does not have to disengage
manually and is not penalized for later beats at the location they left.
Choosing to move instead of joining is not a miss. Dropouts provide breathing
room because these actions are sparse and do not form a complex phrase, not
because each beat action is less meaningful or judged by a weaker standard.

Authored instrument performance passages do not require the player to press
**Join In**. The game automatically enrolls the player shortly before their
selected instrument's next passage and provides an advance preview so the
transition from movement or observation into performance is expected. The
working warning is about two seconds, expressed at a readable beat or measure
boundary for the specific song rather than as an arbitrary off-beat timer.

If a phrase begins while the player is still moving, movement does not generate
miss judgments, direct damage, or another explicit failure penalty. The song and
phrase continue without pausing, however, and the player forfeits the resistance
damage, support, or other progress they could have earned during the unavailable
part of that phrase. That lost opportunity is the entire movement-related
consequence; no additional punishment applies.

After the player reaches a valid position, a short settling grace period of
approximately one to one-and-a-half seconds passes before the phrase staff or cue
appears and timing judgments activate. The game then automatically joins the
player to the remaining phrase at its next playable beat or step. Earlier beats
and beats inside the grace period remain unscored rather than becoming misses.
From the join point onward, normal timing judgments, rewards, and failure
consequences apply. If the phrase ends before the grace period completes, the
player simply misses that opportunity without an added penalty.

Independent and coordinated play emerge from this shared chart:

- **Instrument performance:** each player performs the available chart material
  authored for their chosen instrument while handling positioning, survival,
  and combat intent. Phrases group that material without forcing silence between
  consecutive groups.
- **Band coordination:** sections in which the song's full authored ensemble is
  playing create natural candidate windows for group attacks, group defenses,
  shields, or recovery actions. Participating players remain on their own
  instrument parts, but their phrases share the same song section and combine
  into one clearly previewed group result.

Group abilities should be earned through play. They should create memorable band
moments without making every second dependent on perfect synchronization from
every player. Global boss telegraphs, major impacts, and repositioning windows
are shared encounter events even while instrument rhythm phrases differ. The
result of a group ability is primarily additive: each player's performance
determines that player's contribution, so weak execution reduces that share
without erasing the successful work of the rest of the band.

Broad successful participation may earn a capped positive **Cohesion Bonus**.
Weak or absent performance supplies less positive contribution and may leave
some of this bonus unearned, but it never subtracts value already earned by
stronger players. Eligibility thresholds may become stricter with difficulty,
while the initial bonus cap remains around 15%. Group-event tiers scale against
the eligible roster, so one expert can create meaningful value without standing
in for an otherwise inactive six-player band. No ordinary mistake, or even one
player's failed phrase, should by itself cause the entire group ability to fail.
Exact contribution weights, bonus thresholds, and collective tier requirements
should be established through playtesting.

## 7. Instruments, roles, and builds

The player's own Roblox avatar is their performer and story character. The game
does not require the player to replace that identity with a named hero or a
separate character creator. Instruments, equipment, abilities, animation, and
Order presentation should build the musical-warrior fantasy around the player's
avatar. Battle presentation should combine the avatar's recognizable normal
appearance with equipped Order clothing or stagewear, the instrument being
played, role-oriented equipment, and visible buff effects. These additions
should express progression and combat identity without erasing the player's
chosen Roblox identity. The warning against default Roblox-avatar styling in
`ART_DIRECTION.md` should therefore mean that an unintegrated default appearance
is insufficient, not that the game replaces the player's avatar.

An instrument is a player's musical identity, not a permanent combat class.

Every instrument category should be capable of serving offensive, defensive,
healing, or utility roles. A player should not have to abandon the instrument
they enjoy because a group needs a different role.

Specialization comes from equipment and character development:

- Individual instruments and items can lean toward particular roles. For
  example, one drum kit might strengthen wards while another favors damage.
- Role-oriented skill trees can deepen or combine combat specialties.
- Different builds should create new strategies and play styles as well as
  meaningful increases in power.

Co-op groups should benefit from complementary builds without requiring one
specific instrument lineup.

## 8. Progression and mastery

Long-term engagement should center on three mutually supporting pursuits:

- upgrading equipment and abilities;
- growing meaningfully more powerful;
- mastering songs and boss encounters.

Progression should blend statistical power with new abilities and play styles.
Stronger builds may make earlier encounters noticeably easier, but rhythm skill,
positioning, cooperation, and boss knowledge must remain decisive against
current-tier challenges. Equipment should help players succeed; it should not
play the song for them.

Boss tier determines the base quality of encounter rewards. Selected difficulty
and stronger performance may increase drop quantity or the chance of improved
rolls, but bosses appropriate to the player's current progression remain the
best source of advancement. Older bosses become easier, useful farms for their
identity-specific rewards and materials without out-rewarding current-tier
challenges.

Gear may strengthen the combat consequences of rhythm performance, including
damage, ward strength, resource generation, support effects, or recovery from a
mistake. It must not widen note-judgment windows. Timing forgiveness belongs to
difficulty and accessibility settings so musical feedback remains consistent and
players can understand the skill they are developing.

Players may freely change equipped gear, abilities, instruments, and unlocked
role-specialization choices outside an active song. Respeccing must not require
Robux or a punitive grind. When the song begins, those build choices lock for the
performance so players cannot replace equipment or rebuild their role while the
battle is underway.

Consumables are the exception for tactical item use, not an exception to the
locked build. Players prepare a small consumable loadout before the song and may
activate those items through quick-access controls during combat. Charges are
limited, the full inventory cannot be browsed or used to replenish them mid-song,
and their effects should resolve at a musically clean opportunity. Consumables
may restore ward or resources, cleanse an effect, or grant a temporary buff, but
they cannot bypass defeat or replace the solo and co-op recovery mechanics. Exact
slot counts, charges, and item effects are downstream design decisions.

Equipment acquisition blends boss drops with lightweight crafting and upgrades.
Bosses should drop memorable complete items, wardrobe pieces, and materials tied
to their identity. The Order workshop provides straightforward item upgrades,
combines earned boss materials, and occasionally crafts a known item from a
short, clear recipe. This gives players a deliberate path toward a build instead
of leaving progression entirely to random drops. Crafting should support boss
mastery and role development rather than become a separate resource-survival
game with a large recipe catalog.

### Monetization principles

Direct purchases may include permanent instruments, defensive equipment, or
other durable items with genuine combat value. This is paid permanent
progression, not permission to sell a temporary rescue at the moment of danger,
downing, or defeat. Purchase offers should occur outside active encounters.

Paid equipment may accelerate access or offer a distinct build and appearance,
but it must not exceed the normal item-tier ceiling. Every paid item must have an
earnable equivalent or a comparably strong non-paid build at the same tier. No
paid item should become mandatory for a role, current-tier boss, difficulty, or
desirable co-op group composition.

Every purchase must have a guaranteed, clearly presented outcome. The game must
not sell paid loot boxes, gacha, prize wheels, random upgrade success, paid luck
modifiers, or any other random result bought directly or indirectly with Robux.
Boss drops and other rewards earned solely through play may remain random.

The storefront must use low-pressure, age-appropriate presentation. It does not
show purchase prompts during battle, downing, defeat, or immediate retry; create
false scarcity; restart countdowns; or repeatedly nag a player who declines.
Prices and outcomes must be clear before confirmation. Genuine seasonal offers
may have real, clearly stated end dates, but urgency must never be fabricated.

Initial monetization is limited to direct-purchase cosmetics and permanent
equipment governed by these rules. Paid song or boss access, convenience
products, subscriptions, temporary boosts, and other categories are deferred
until the core game and economy are proven. Deferral does not promise that those
categories will later be accepted; any future proposal must preserve rhythm-skill
integrity, co-op fairness, earnable-equivalent power, and the low-pressure
storefront.

## 9. Music and content strategy

The initial catalog should consist of original, creator-directed, AI-assisted
songs. Every song receives human creative and suitability review before it enters
the game, and its generation source, usage rights, and relevant provenance should
be documented. AI assistance is a production method rather than the author of the
game's identity; song selection and final creative judgment remain human-led.

Licensed, commissioned, and community-created music may be considered in future
releases, but the first shippable product must not depend on obtaining or
operating those additional content sources.

The game's defining musical identity is dark, dangerous EDM with cinematic
wall-of-sound scale and relentless K-pop energy. Song selection should preserve
that forceful, high-pressure identity rather than treating the catalog as a
general tour through unrelated genres.

Lyrics and vocals should carry the same supernatural, Christian-inspired themes
as the world and story. They may confront darkness, corruption, danger, pride,
courage, sacrifice, fellowship, hope, and restoration through the game's
original mythology. They should remain broadly approachable for ages 10 to 14:
intense without profanity, sexual content, graphic gore, or direct real-world
religious preaching.

### Tone, accessibility, and social safety

Bosses should communicate serious supernatural danger without becoming
cartoony, genuinely demonic, or highly frightening. "Demon-like" describes some
corrupted traits or appearances but does not require literal demons. The target
is mythic spiritual menace rather than horror. Avoid grotesque detail, realistic
suffering, gore, and imagery intended as nightmare fuel.

Dark encounters should be balanced by meaningful warmth, hope, and humor,
especially through the Order, its members, and moments of recovery. Humor should
provide relief and affection without making the central threat feel silly or
turning the world into parody.

Accessibility assists are separate from difficulty and do not reduce progression
or rewards. The experience should support appropriate input and audio
calibration, reduced flashing and camera shake, high-contrast and non-color-only
cues, and control remapping. The exact assist set belongs to downstream design
and testing rather than the vision interview.

Public cooperative play must work without voice chat or unrestricted text chat.
Core coordination must be possible through readable battle cues, pings, and safe
preset messages. Voice or platform-provided filtered chat may supplement those
systems where appropriate, but cannot be required to understand or complete an
encounter.

## 10. World and story foundation

Bands Battle takes place openly in a fully supernatural fantasy world. Music is
a known force within that world, not a hidden magical layer behind an ordinary
modern setting.

The working mythic foundation is the Shattered Song:

- A living harmony once sustained the world's intended order.
- An inner circle within the Order attempted to bind that harmony to a single
  controlling will.
- Most conspirators intended to seize the Song's power but did not understand
  that their act would shatter it. The hidden mastermind intended the Shattering
  and used the other conspirators to cause it. The apparent insider behind the
  coup was secretly an outside spiritual being that had infiltrated the Order
  under the identity of an ancient hero. The Order remembers and celebrates that
  figure as the hero who sacrificed themselves to limit the Shattering, without
  knowing that the hero engineered it and that a novice's resistance prevented
  its completion. The revelation makes the coup both an internal betrayal enabled
  by the Order's own weaknesses and a larger spiritual attack.
- The false hero dazzled, manipulated, and deceived the Order's senior members.
  Their authority and experience did not protect them from spectacle, promised
  power, or a plan presented as necessary for the world's protection.
- During the binding ritual, a novice recognized that something was wrong and
  knowingly defied the false hero by performing one honest counter-note. The
  novice did not understand the full scale of the intervention, but the greater
  Harmony behind the Song answered that freely offered act. The counter-note
  preserved the Song's living core, so it fractured into recoverable pieces
  instead of being destroyed completely. The novice disappeared during or
  immediately after the Shattering, and the subsequent cover-up erased the
  novice's act from official history. The disappearance is an intentional
  mystery; the story does not need to confirm whether the novice died, survived,
  or can return.
- The mastermind's hubris helped defeat its complete plan. It understood power,
  hierarchy, and control but dismissed the possibility that a low-ranking voice,
  freely offered in resistance, could matter.
- The Shattering scattered fragments that could be captured and wielded
  separately in a way the complete Song could not.
- The fragments are now held by powerful bosses whose dissonance corrupts the
  surrounding regions.
- The bosses must be destroyed rather than redeemed.
- Destroying a boss releases a fragment and restores part of the Shattered Song
  and the world.

Most bosses are hostile spiritual beings and demon-like monsters rather than
former members of the Order. Some major bosses, however, are surviving
conspirators or beings directly involved in the ancient coup. Wielding a fragment
without the harmony or character required to bear it can corrupt such a wielder,
amplify its existing discord, and transform it over time into a demon-like form.
This lets the campaign move from monstrous symptoms of the Shattering toward the
personal agents responsible for it.

Players begin as new recruits in a diminished musical order dedicated to this
restoration. The Order was once far more significant and has a rich, mysterious
history that the players gradually uncover. Its fall began with the ancient coup
rather than an ordinary defeat. Its surviving leaders concealed the betrayal and
their own failure, hoping to preserve the Order. The cover-up instead created
mistrust and competing factions. As people lost faith in the Order's protection,
their shared harmony weakened and the world became increasingly discordant. The
centuries since have turned the original betrayal into a prolonged decline.

The Order's lost knowledge, concealed loyalties, forgotten deeds, and unanswered
failures should make rebuilding it part of the player's journey. In the present,
it provides training, mentors, ranks, a shared headquarters, equipment
development, mission structure, and the foundation for solo and cooperative
deployments.

The Order is ancient, but it is a living tradition rather than a historical
reenactment. Its enduring inheritance is the knowledge and discipline used to
channel the Song, not a fixed set of antique instruments. In every era, the Order
has adopted, refined, and sometimes helped develop the instruments,
amplification, and performance technology available in the world. Present-era
guitars, drum kits, synthesizers, amplifiers, and stage equipment therefore
belong naturally within the Order. Older instruments and equipment may appear as
relics, regional traditions, or rediscovered techniques without implying that
newer tools are less authentic. The exact blend of technological and supernatural
engineering belongs to downstream world and art design, but it should feel native
to this world rather than imported from an ordinary modern setting.

The game's themes should be Christian-inspired but broadly approachable. The
story can emphasize intended order, corruption through pride, courage,
sacrifice, fellowship, hope, and restoration without becoming a direct allegory
or requiring players to share a particular faith.

The setting and mythology must remain original. Inspiration should be translated
into Bands Battle's own characters, cultures, symbols, terminology, conflicts,
and plot.

## 11. Campaign and encounter structure

The main progression follows a story path through increasingly consequential
boss encounters. Victories restore fragments, open regions, reveal clues, and
unlock new songs, abilities, equipment, and challenges. Some boss victories also
expose the participants, motives, and consequences of the ancient coup, allowing
the true history of the Order to emerge over the course of the campaign. Early
encounters may emphasize spiritual monsters created or empowered by the
Shattering. Later encounters can reveal transformed conspirators and culminate in
the discovery that the mastermind who deliberately engineered it is one of the
Order's own celebrated ancient heroes.

Defeated bosses remain replayable for mastery and rewards. In addition to the
main path, special encounters may appear on certain days or times or through
semi-random availability. These events should add surprise and reasons to return
without making permanent story progress depend entirely on being online at a
specific moment.

Encounters are finite musical performances rather than unlimited battles. A
normal boss attempt uses one complete song from beginning to end, and the track's
full length determines the encounter duration rather than an unrelated combat
timer or selected excerpt. Normal boss songs typically run from three to seven
minutes. Most bosses therefore reach an outcome on the song's authored ending.
Later releases may introduce exceptional bosses with a multi-song raid
structure, but even those encounters consist of a bounded sequence of authored
musical stages rather than repeating until the players eventually drain a health
bar. An exceptional raid uses two or three songs, targets roughly ten to twenty
minutes in total, and must not exceed about twenty-five minutes including
transitions. A generated final coda could provide a limited near-miss extension
in a future release. Multi-song raids and generated codas are outside the first
release.

A normal play session should comfortably accommodate two to four complete boss
encounters, including the surrounding choice, matchmaking, reward, and retry
flow. Completing only one encounter must still feel worthwhile; rewards and
progress should not depend on finishing a longer forced chain.

### First shippable product

The first shippable product is a small but complete native Roblox release rather
than a technology demo. It includes:

- an Order hub and onboarding;
- three replayable bosses;
- complete solo play and three-to-six-player cooperative play; and
- a basic equipment, reward, and progression loop.

This release must express the real rhythm-controlled boss-combat promise on the
touch-first supported product. PvP, user-authored songs, a free-roaming world,
deep crafting, and multi-song raids are explicitly outside its scope. These
features may be reconsidered later, but cannot displace completion and validation
of the core boss experience.

The vision is working when target-age players understand that their musical
performance controls combat, voluntarily replay bosses, and want to coordinate
or return with friends. Those observable reactions are more meaningful than
players merely completing onboarding once.

The product should support the repeatable addition of new songs and bosses after
release. Do not promise a fixed public content schedule until actual production
time and player demand are known.

## 12. Instructions for development decisions

When evaluating a feature, favor work that strengthens one or more of these
outcomes:

- Players experience boss combat through musical action.
- Players look at and understand the battle, not only the rhythm interface.
- Musical accuracy produces clear combat and audio consequences.
- Positioning creates a legible choice between danger and reward.
- Battles alternate pressure, decisions, coordinated spectacle, and breathing
  room.
- Solo play remains complete while co-op creates superior band moments.
- Any instrument supports any broad role through builds.
- Progress rewards both growing power and growing skill.
- Boss victories advance restoration of the Shattered Song.
- The experience remains readable and approachable for ages 10 to 14.
- The Order feels ancient in continuity and accumulated history while its music,
  instruments, and performance technology continue to evolve.

Avoid decisions that:

- reduce the boss and player characters to background decoration;
- turn the compact phrase staff into a permanent song-wide highway or make it
  the battle's dominant visual surface;
- require uninterrupted dense note entry through an entire encounter;
- lock necessary combat roles to specific instrument categories;
- allow one isolated mistake to erase an otherwise good performance;
- make equipment replace timing skill, positioning, or boss knowledge;
- make co-op mandatory for basic story completion;
- make solo feel like an incomplete version of co-op;
- freeze the Order into one historical period or treat modern amplified
  instruments as anachronisms;
- introduce religious references, borrowed mythology, or recognizable story
  elements in place of original worldbuilding.

## 13. Bounded interview status

[`GAME_VISION_QUESTIONS.md`](GAME_VISION_QUESTIONS.md) is the authoritative,
finite record of the bounded owner interview. At its 2026-08-14 baseline, 13
foundation questions were resolved and 12 vision-level questions remained.
Current progress is 25 resolved and 0 remaining. The bounded owner interview is
complete, and this document is the stable Vision v1 baseline.

Detailed balance, timing values, interface layout, implementation, production
counts, and proper names do not reopen or extend the vision interview; they are
routed to the downstream backlog and their own design documents. New required
questions may be admitted only under the checklist's change-control rule.
