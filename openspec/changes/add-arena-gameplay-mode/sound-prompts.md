# Arena V2 ElevenLabs Sound-Effect Prompts

This is the copy/paste production list for the first Arena vertical slice. Each prompt describes one isolated sound. Set **Duration** and **Loop** in ElevenLabs Sound Effects using the adjacent columns instead of adding a sequence of unrelated events to the prompt.

The [current ElevenLabs Sound Effects guide](https://elevenlabs.io/docs/eleven-creative/playground/sound-effects) says each website generation returns four variations. “Keep” is the number of meaningfully different approved files the game needs, not the number of times to paste the prompt. For sounds shorter than the generator permits, generate at 0.5 seconds and trim the approved master to the target length. Export or retain the highest-quality available source, then edit and master approved sounds as 48 kHz, 24-bit WAV.

## P0: required for the playable graybox

| Filename stem | Keep | Duration | Loop | Copy into ElevenLabs |
|---|---:|---:|:---:|---|
| `run_count_tick` | 1 | 0.5 s, trim to 0.15 s | No | `Dry magical percussion tick, precise, clean, no reverb, one-shot.` |
| `run_count_go` | 1 | 0.5 s | No | `Bright magical start hit with a tiny rising shimmer, decisive, one-shot.` |
| `arena_phrase_reveal` | 2 | 0.6 s | No | `Soft glassy energy constellation unfolding, anticipatory shimmer, clean, one-shot.` |
| `player_input_ack` | 3 | 0.5 s, trim to 0.1 s | No | `Tiny crisp magical energy pluck, instant button feedback, dry, one-shot.` |
| `player_perform_contact_good` | 2 | 0.5 s | No | `Small bright energy bolt hitting a huge monster, restrained impact, one-shot.` |
| `player_perform_contact_great` | 2 | 0.5 s | No | `Bright magical strike hitting a huge monster, punchy layered impact, one-shot.` |
| `player_perform_contact_perfect` | 3 | 0.6 s | No | `Powerful glassy energy strike hitting a giant boss, brilliant clean impact, one-shot.` |
| `player_perform_flub` | 3 | 0.5 s, trim to 0.25 s | No | `Weak sputtering energy pluck, failed magic attack, dry, no triumphant tail.` |
| `player_reposition_select` | 2 | 0.5 s, trim to 0.2 s | No | `Quick tactical selection ping with a subtle magical whoosh, clean, one-shot.` |
| `player_dash_retreat` | 2 | 0.5 s | No | `Fast backward spectral dash whoosh, light airy energy trail, one-shot.` |
| `player_dash_advance` | 2 | 0.5 s | No | `Fast forward spectral dash whoosh, forceful bright energy surge, one-shot.` |
| `player_anchor_arrive` | 2 | 0.5 s, trim to 0.35 s | No | `Light armored landing with a compact magical energy settle, dry, one-shot.` |
| `boss_sweep_warn` | 1 | 0.8 s | No | `Ominous monster weapon scrape widening into a broad dark-energy sweep warning.` |
| `boss_sweep_charge_loop` | 1 | 2 s | Yes | `Seamless rough dark-energy sweep charge, steadily rising tension, no impact.` |
| `boss_sweep_impact` | 2 | 0.8 s | No | `Huge horizontal dark-energy slash impact, sharp transient, short stone debris tail.` |
| `boss_burst_warn` | 1 | 0.8 s | No | `Hollow supernatural inhale with crystalline pulses, ominous magical burst warning.` |
| `boss_burst_charge_loop` | 1 | 2 s | Yes | `Seamless pulsing void-orb charge, inhaling rhythm, tense upper shimmer, no impact.` |
| `boss_burst_impact` | 2 | 0.8 s | No | `Compact radial void explosion, crystalline snap, heavy central impact, short tail.` |
| `player_evade_success` | 2 | 0.5 s | No | `Fast magical near-miss whoosh resolving into a small bright safety chime.` |
| `player_ward_hit_light` | 1 | 0.5 s, trim to 0.25 s | No | `Light blow absorbed by a glassy magical shield, small crystalline tick, one-shot.` |
| `player_ward_hit_medium` | 1 | 0.5 s | No | `Solid blow absorbed by a glassy magical shield, energy thump and short crack.` |
| `player_ward_hit_heavy` | 1 | 0.6 s | No | `Massive blow absorbed by a magical shield, deep thump and violent crystalline fracture.` |
| `player_ward_crack` | 1 | 0.8 s | No | `Single ominous magical shield fracture, crystalline stress and fading energy.` |
| `player_ward_break` | 1 | 1 s | No | `Magical energy shield shattering violently, crystalline burst and low energy collapse.` |
| `boss_hit` | 3 | 0.5 s | No | `Giant supernatural monster struck by bright magic, heavy body and energy impact.` |
| `boss_stagger_open` | 1 | 1 s | No | `Massive monster staggering, armor groan and collapsing dark aura, clear opening cue.` |
| `phrase_complete` | 2 | 0.6 s | No | `Bright magical resolve flourish, compact and rewarding, nonmelodic, one-shot.` |
| `boss_resolve_gain_light` | 1 | 0.5 s, trim to 0.3 s | No | `Small crack forming in a dark magical barrier, bright energy accent, one-shot.` |
| `boss_resolve_gain_medium` | 1 | 0.5 s | No | `Dark magical barrier weakening with a sharp glassy energy fracture.` |
| `boss_resolve_gain_heavy` | 1 | 0.6 s | No | `Dark magical barrier rupturing under powerful bright energy, heavy fracture accent.` |
| `final_resolve_success` | 1 | 1 s | No | `Supernatural boss seal breaking decisively, triumphant bright energy rupture, no melody.` |
| `final_resolve_failure` | 1 | 1 s | No | `Ritual seal sputtering and collapsing unresolved, dark descending energy, no melody.` |

## P1: required for the finished vertical slice

| Filename stem | Keep | Duration | Loop | Copy into ElevenLabs |
|---|---:|---:|:---:|---|
| `arena_intro_rift` | 1 | 2 s | No | `Supernatural arena rift tearing open, deep air pull and violet electrical crackle, no music.` |
| `boss_intro_vocal` | 2 | 1.5 s | No | `Huge stylized fantasy monster awakening roar, intimidating, nonverbal, no speech.` |
| `boss_sweep_vocal` | 2 | 0.7 s | No | `Short giant monster exertion growl for a sweeping attack, aggressive, nonverbal.` |
| `boss_burst_vocal` | 2 | 0.8 s | No | `Short giant monster inhaling roar for a magical burst attack, ominous, nonverbal.` |
| `boss_phase_transition` | 1 | 1.8 s | No | `Giant monster powering up, low roar and expanding corrupted energy, no music.` |
| `boss_defeat` | 1 | 2 s | No | `Massive supernatural monster collapsing and dissolving into dark energy, heavy, no music.` |
| `world_downbeat_accent` | 3 | 0.5 s, trim to 0.25 s | No | `Subtle supernatural arena pulse, low airy thump, short and unobtrusive, one-shot.` |
| `position_shelter_enter` | 1 | 0.6 s | No | `Protected stone sanctuary activating, warm low shield hum, compact, one-shot.` |
| `position_midline_enter` | 1 | 0.6 s | No | `Balanced magical arena anchor activating, neutral glassy energy chime, one-shot.` |
| `position_spotlight_enter` | 1 | 0.6 s | No | `Exposed high-risk arena spotlight igniting, bright dangerous electrical shimmer.` |
| `result_victory_sting` | 1 | 1.8 s | No | `Short triumphant supernatural combat sting, bright energy, no vocals, no long tail.` |
| `result_defeat_sting` | 1 | 1.8 s | No | `Short dark failed-battle sting, unresolved energy collapse, no vocals, no long tail.` |
| `ui_move` | 2 | 0.5 s, trim to 0.1 s | No | `Tiny neutral spectral interface tick, crisp, soft, dry, one-shot.` |
| `ui_confirm` | 2 | 0.5 s, trim to 0.25 s | No | `Clean bright spectral interface confirmation chime, compact, one-shot.` |
| `ui_back` | 1 | 0.5 s, trim to 0.2 s | No | `Soft descending spectral interface whoosh, brief, clean, one-shot.` |
| `ui_error` | 1 | 0.5 s, trim to 0.25 s | No | `Muted distorted spectral interface buzz, clear but gentle, one-shot.` |

## P2: optional flavor and polish

| Filename stem | Keep | Duration | Loop | Copy into ElevenLabs |
|---|---:|---:|:---:|---|
| `arena_ambience_loop` | 1 | 8 s | Yes | `Seamless ruined supernatural arena ambience, distant wind and faint rift drone, no music.` |
| `cover_debris` | 3 | 0.7 s | No | `Small dry stone chips and dust falling after a nearby heavy impact.` |
| `streak_milestone_light` | 1 | 0.5 s, trim to 0.3 s | No | `Tiny rising energy flourish for an early combat streak, nonmelodic, one-shot.` |
| `streak_milestone_medium` | 1 | 0.5 s | No | `Bright rising energy flourish for a strong combat streak, nonmelodic, one-shot.` |
| `streak_milestone_heavy` | 1 | 0.6 s | No | `Powerful rising glassy energy flourish for a major combat streak, nonmelodic.` |
| `spectral_crowd_react` | 3 | 1.2 s | No | `Distant ghostly crowd swell, subtle, nonverbal, no words, no individual voices.` |

## Generation and selection notes

- Start around medium prompt influence so each generation explores useful variations; raise it only when the result ignores the core material or action.
- Generate families in isolation. Combine vocal, charge, impact, debris, and tonal layers later in an editor so their timing remains controllable.
- For `boss_sweep_*`, favor rough lateral motion and scraping noise. For `boss_burst_*`, favor pulsing, hollow, crystalline, and radial energy. Reject candidates that sound interchangeable through a phone speaker.
- Keep player sounds bright, fast, and glassy; boss sounds rough, dark, and weighty; UI sounds neutral and tiny.
- Reject spoken words, accidental music, long reverb, excessive sub-bass, or leading silence unless the row explicitly requests them.
- Charge loops need a separate edited release tail so interruption, pause, impact, and replay never click.
- Use the audio manifest to record the exact prompt, ElevenLabs settings, selected generation, edits, output filename, markers, and license/original-work status.
