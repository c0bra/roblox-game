# Roblox Bands Battle game design brief

This document synthesizes `trello_notes.md`, `chatgpt_chat.md`, the current audio
pipeline, and Roblox/rhythm-game implementation research into a working product
brief for new developers.

## One-line pitch

A mobile-friendly Roblox rhythm boss-battle game where players perform AI-generated
K-pop/EDM battle songs as a band, survive boss attacks, earn loot, and upgrade
their instruments.

## Product pillars

1. **Instant rhythm fantasy:** players should feel like they are in a supernatural
   K-pop battle performance within the first 10 seconds.
2. **Mobile-simple input:** the core input must work on touch first, then keyboard
   and gamepad.
3. **Combat feedback:** hits should damage, defend, heal, build hype, or protect
   the band, not just add score.
4. **Audio payoff:** when a player performs well, their instrument should feel alive
   in the mix. Misses should visibly and audibly hurt.
5. **Short repeatable rounds:** 75 to 90 seconds is the target loop. Fast setup,
   clear climax, rewards quickly.
6. **Upgrade chase:** boss drops and instrument upgrades give players a reason to
   replay songs.

## Style target

The canonical visual language, asset rules, palette, and approval criteria live in
[`ART_DIRECTION.md`](ART_DIRECTION.md). This section summarizes the product fantasy.

Music prompts from the raw notes converge on:

```text
Dark high-energy K-pop, EDM, trap or fried 808 bass, soaring layered vocals,
distorted guitar, cinematic, supernatural, battle-ready, glamorous, dangerous.
```

Visual world:

- ruined ancient temples
- craggy mountain tops
- swamps
- ice fields
- black/star portals
- purple or blue supernatural beams
- K-pop stage spectacle layered over combat arenas

Boss examples from notes:

- large stone giant, drops from the sky and shakes the ground
- wispy siren, appears through a black/star portal

## Core round loop

Recommended 90-second structure:

1. **Intro, 0-10s:** song starts, easy notes, boss reveal, all players feel cool.
2. **Core performance, 10-60s:** players hit notes or prompts on their instrument
   lanes. Hits build score, damage, crowd hype, and special meter.
3. **Boss pressure, 25-70s:** boss attacks are authored as chart events. Players
   dodge, recover, or hit defensive phrases.
4. **Specials, 30-80s:** players spend hype meter on solos, shields, heals,
   multipliers, burst damage, or crowd-control moments.
5. **Climax, 60-90s:** note density rises, crowd meter swings faster, boss enters
   final phase.
6. **Outro and rewards:** winner pose, boss defeat or escape, drops, upgrades, and
   a dangling clue toward the next boss.

## Minute-to-minute gameplay options

The likely first prototype should blend these, not build all of them at once.

### Rhythm matching

Classic Guitar Hero / Friday Night Funkin style.

- Notes scroll in lanes.
- Players press/tap the matching lane at the right time.
- Hit: instrument stays loud, crowd cheers, combo grows.
- Miss: sound ducks, flub animation plays, combo breaks, boss gains pressure.

This should be the first playable mechanic because the audio pipeline already
exports lane charts.

### Hype meter

Accurate streaks fill a meter. Spending it creates a tactical choice.

Possible specials:

- spotlight solo for bonus damage
- shield break against boss armor
- crowd scream multiplier for 10 seconds
- heal or revive pulse
- fireworks / light show for score or aggro reset

### Boss attacks

Boss attacks should be authored in chart data, not hard-coded to wall-clock time in
scripts. That keeps attacks synced to the song.

Examples:

- knock players off bonus pads
- silence one instrument lane briefly
- spawn hazards on the note highway
- force tap-to-revive after knockdown
- create risky stage positions that boost score or damage

### Crowd control and recovery

For casual players, misses should not mean instant failure.

Recovery actions can include:

- hype emote
- call-and-response prompt
- short dance chain
- bandmate revive action
- defensive instrument phrase

## Input design

Recommended first version:

- 3 lanes for chart data parity with current scripts.
- Consider 4 lanes for final Guitar Hero-like feel if mobile readability holds.
- Landscape mobile first.
- Large tap targets.
- Keyboard maps to `A S D` or `A S D F`.
- Gamepad maps to face buttons or D-pad.

Timing recommendation:

- Start forgiving on mobile.
- Use tiered judgments such as Perfect, Great, Good, Miss.
- Prototype one broad valid window first, then tune per device.
- Keep judgment on the client for responsiveness, but send summarized results to
  the server for scoring and rewards.

## Chart format recommendation

Current audio output is CSV:

```csv
time_s,lane,pitch,dur_s
7.836735,1,62,0.348299
```

For Roblox runtime, promote this into a small battle chart schema:

```json
{
  "songId": "heavens_edge",
  "difficulty": "medium",
  "durationS": 90,
  "lanes": 3,
  "timeChanges": [
    { "t": 0, "bpm": 128 }
  ],
  "notes": [
    { "t": 7.836735, "lane": 1, "len": 0.348299, "instrument": "vocals" }
  ],
  "events": [
    { "t": 30.0, "type": "boss_attack", "id": "shockwave" },
    { "t": 45.0, "type": "hype_phrase_start" },
    { "t": 52.0, "type": "hype_phrase_end" }
  ]
}
```

Keep the first schema small:

- `notes[]`: timing and lane data
- `events[]`: boss attacks, specials, phase changes, camera cues
- `timeChanges[]`: BPM or tempo map if needed
- `difficulty`: easy, medium, hard, two-lane accessibility
- `instrument`: vocals, guitar, bass, drums, keyboard

## Roblox implementation guidance

The browser game layer is implemented under `roblox/web/`: Classic preserves the
lane-based prototype, while opt-in Arena V2 demonstrates the boss-centered rhythm
combat direction. The native Roblox layer is not implemented yet. When it is
added, use these defaults and the Arena V2 OpenSpec as the current interaction
reference rather than assuming the web prototype's DOM or Babylon architecture
should be copied directly.

### Client

- Render the rhythm highway in `ScreenGui` first.
- Use scale-based UI so the lane works on phones and tablets.
- Use `ViewportFrame` only if the lane needs embedded 3D visuals.
- Keep note travel, hit windows, and local feedback client-side for feel.
- Preload only the current song and immediate UI/audio assets.

### Input

- Use Roblox's Input Action System where possible.
- Otherwise use `ContextActionService` to bind actions once across keyboard,
  touch, and gamepad.
- Do not scatter raw key checks through gameplay scripts.

### Server

- The server is the source of truth for inventory, drops, unlocks, and final battle
  results.
- Clients report performance summaries and important discrete events.
- Use RemoteEvents for discrete messages.
- Avoid sending note-by-note per-frame spam over remotes.

### Persistence

- Use DataStoreService only from the server.
- Use safe update patterns for loot and inventory writes.
- Ordered stores are for leaderboards, not inventory.

### Animation and bosses

- Use `Animator:LoadAnimation()` for humanoid performers.
- Use `AnimationController` for non-humanoid boss rigs.
- Move repeated NPC/boss visual animation work client-side when possible for
  performance.

## Progression and loot

Post-battle drops can be instrument-specific:

- guitar pedals
- kick pedals
- drum sticks
- guitar picks
- microphones
- mic stands
- instrument cables
- guitars, basses, drums, keyboards

Upgrade directions:

- larger hit window
- faster hype generation
- stronger shield/heal special
- bonus damage against certain boss types
- cosmetic aura, stage light, or solo animation
- better drop chance on high combo

Boss difficulty target from the notes: players should beat bosses around 75% of the
time after tuning. Treat this as a live balance target, not a hard-coded guarantee.

## Retention hooks

- Timed bosses with countdowns.
- Daily or weekly special songs.
- Boss clues that point to the next encounter.
- Instrument crafting or unlock requirements.
- Rare cosmetic drops tied to performance tiers.

## MVP scope

Do this first:

1. One song.
2. One chart.
3. One instrument lane set.
4. One boss.
5. One hype meter special.
6. One loot drop table.
7. One result screen.

Do not start with multiplayer battle complexity. Prove one player can complete one
song against one boss on mobile and want to play again.

## Open questions

- Is the first shipped mode solo, co-op band, or band-vs-boss with drop-in players?
- Should first mobile input be 3 lanes or 4 lanes?
- Are players controlling avatars in arena space during notes, or is movement a
  between-phrases mechanic?
- Does each instrument get its own chart, or do all players share a simplified lane
  chart with different sounds?
- How much does a miss affect audio, score, health, and boss pressure?
- What is the first loot stat that actually changes gameplay?
