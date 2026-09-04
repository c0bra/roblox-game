# Bands Battle Audio Presentation

- **Status:** Approved
- **Approved:** 2026-09-01
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#77-audio-presentation)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Content dependency:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **UI/settings dependency:** [`UI_UX.md`](UI_UX.md)
- **Decision source:** [`AUDIO_PRESENTATION_WORKING.md`](AUDIO_PRESENTATION_WORKING.md)
- **Interview plan:** [`AUDIO_PRESENTATION_QUESTIONS.md`](AUDIO_PRESENTATION_QUESTIONS.md)

## 1. Role and authority

Audio Presentation owns the runtime audible expression of the approved song,
local performance, gameplay state, cooperative action, world response, and
accessible sound metadata. It defines controllable-layer playback, response
envelopes, cue priority and concurrency, mix buses/dynamic range, device/mono/
spatial treatment, caption/source metadata, haptic requests, asset readiness,
degradation, and audio-state output.

It does not own:

- the musical clock, chart, judgment, or normalized contribution;
- Combat, Survival, Boss, Positioning, Ability, Multiplayer, reward, or outcome
  semantics;
- source music or encounter authoring/approval;
- player setting values/profile scope, caption rendering, or durable storage;
- technical audio engine/network authority; or
- final sound assets, mix values, or platform encoding choices.

Audio consumes identified semantic facts from owners and applies versioned
presentation definitions. It never infers gameplay by listening to a stem,
watching a displayed value, or receiving an anonymous button press.

## 2. Governing invariants

1. **One approved song/clock:** difficulty and presentation never change source
   tempo, pitch, duration, structure, or authored event timing.
2. **Complete music remains:** every player hears the complete arrangement at a
   stable neutral mix; a judgment never silences the song.
3. **Flexible authentic roles:** role audio follows the real arrangement and is
   not limited to drums, vocals, guitar, or bass.
4. **Neutral reconstruction:** every offered role can reconstruct the approved
   full mix at neutral and apply bounded local response around it.
5. **Local errors stay local:** one player's judgment/timing/down-state response
   never damages another player's song mix.
6. **Semantic sounds only:** shared cues start from confirmed owner-domain facts,
   not speculative or ad hoc client effects.
7. **Protected clarity:** critical timing, danger, targeting, recovery, group,
   and integrity cues outrank optional sound while preserving the song pulse.
8. **Meaning is not loudness:** rhythm/register/timbre/envelope/source and
   multimodal reinforcement distinguish cues.
9. **Phone/mono core:** critical/local-role meaning cannot depend on sub-bass,
   extreme treble, stereo width, or precise spatial hearing.
10. **Roster-neutral headroom:** one through six humans and duplicate roles do not
    multiply noise or reveal individual performance.
11. **Audio is optional:** every bus may be muted because critical semantics
    retain visual/caption/optional-haptic alternatives.
12. **Haptics are reinforcement:** they never create a required/easier independent
    rhythm track or carry meaning alone.
13. **Exact lifecycle/idempotency:** one causal event produces one ordered audio
    instance/history despite retries, latency, or duplicate delivery.
14. **No surprise failure:** missing protected audio blocks/degrades/cancels under
    explicit rules and never permits a silent committed impact.
15. **Decorative degradation first:** song, local role, timing, committed danger,
    recovery/group cues, captions, and required alternatives are protected.

## 3. Runtime song map and neutral reconstruction

Every song revision supplies one immutable full-mix reference plus one runtime
audio map for each offered playable role. A role map identifies:

- one neutral backing bed;
- one or more authentic controllable local-role layers or approved equivalent;
- stable role/layer/asset/revision identities;
- start offset, exact duration, channel layout, neutral level, and alignment;
- response capability/treatment references; and
- preload, stream, cache, device, and fallback declarations.

At neutral, backing plus controlled role layers reproduce the approved complete
song with the same start, duration, tempo/phase, musical content, and human-
approved perceived balance. Transcodes may prevent sample identity; automated
alignment/loudness/phase checks and human A/B review establish equivalence.

The selected role is audible at a defined neutral baseline. Performance applies
bounded gain/clarity/filter/transient or equivalent deltas rather than gating the
role. The backing preserves song continuity and every other necessary part.

A role may use one stem, multiple grouped sublayers, a role-specific backing plus
controlled layer, or another approved equivalent. Equivalent treatment must:

- respond audibly to authentic role events;
- avoid fabricating absent material/onsets;
- avoid materially changing unrelated parts; and
- pass neutral and maximum-response A/B review.

Roles are extensible/song-specific. Sparse/ambient roles use actual sustained or
textural behavior rather than invented attacks. Dirty isolation is acceptable
only when leakage/masking remains acceptable throughout the response range. A
role lacking adequate controllable/equivalent audio cannot be playable.

Nonplayable source layers remain in the neutral bed. Duplicate-role players use
the same content map independently; their local emphasis is not summed into a
shared stem mix. Runtime never combines identities from different revisions.

## 4. Local judgment and hold response

Each local Rhythm judgment applies its definition once at the authentic audible
event/attempt. Semantic emphasis remains:

- **Perfect:** crispest/clearest strongest bounded lift;
- **Great:** confident normal accent;
- **Good:** softer/less prominent response; and
- **Miss:** brief duck/filter/stumble without silence or punitive unrelated SFX.

Response affects presentation only, never judgment, contribution, combat, or
reward. Roles express the ordering musically: percussion may use transient
definition; sustained synth/strings may use clarity/presence/filter; vocals may
use a reviewed legato treatment. All preserve ordering and headroom.

Early/Late share one first-release audio response. UI/Results carry direction.
Rapid repeats/alternates merge, retrigger, or extend under role-specific caps;
they do not stack unlimited independent sounds. Repetition strengthens source
performance rather than adding unrelated noise.

Hold onset receives its grade envelope. Accepted maintenance may sustain a
restrained state from live hold evidence. Early release stops future sustain and
returns toward neutral without another judgment/Miss sound. Suspension ends
future response at its exact boundary while preserving heard history.

An unclaimed Late Miss may apply the same bounded Miss treatment. Zero input
never creates a positive accent. Movement, authored inactivity, recovery setup,
disconnect/AFK/input-unavailable, and synchronization suspension return toward
neutral without Miss audio.

Downing moves the local role into a muffled/distant state while the complete song
and protected cues continue. Successful recovery/rejoin returns through an on-
beat bounded swell, then ordinary response resumes only at fair return.

Judgment/down response is private/local. Teammates do not hear another player's
grade, miss, down filter, or timing tendency. Confirmed local feedback is not
audibly rolled back by ordinary network delay; shared audio uses separate
confirmed events.

## 5. Playback, start, pause, and rejoin

Before deployment start, each client:

- loads/verifies its selected-role complete song map;
- acknowledges exact content/audio revision;
- phase-readies every layer;
- loads required initial/guaranteed critical cues; and
- guarantees remaining critical assets before their scheduling horizon.

Optional decorative assets do not block deployment. Multiplayer supplies one
shared future boundary; Audio schedules all layers in phase at that boundary.
Audio reports observation/health but does not own chart/encounter time.

Calibration aligns perceived audio/input without retiming source, changing
speed/pitch, moving events, or widening judgment. Local response uses the
synchronized local representation and does not roll back for routine latency.

Layers never independently restart/free-run/seek. Drift convergence must be
inaudible, outside active strike/reaction windows, and free of audible tempo/
pitch manipulation. Unsafe confidence invokes Rhythm suspension/resync;
unrecoverable outcome-critical clock/audio corruption uses No Contest.

Solo pause freezes song, layers, envelopes, clock-driven cues, and encounter
time at one instant. A separate phase-aligned audible/visible count accepts no
judgments and advances no song time. Resume continues the frozen point.

Co-op never pauses for one player. Output-device change rejoins current shared
position with a short phase-safe fade and may suggest future recalibration.
Unsafe usable-audio/input loss uses established local Rhythm suspension.

Network rejoin validates active revision, seeks current authoritative position,
and fades in at safe return. The local role begins in authoritative neutral,
downed, or return-protected state. Normal completion/terminal stop occurs once
at owner boundary; duplicate stop/end is idempotent.

Missing/misaligned song or selected-role audio blocks deployment. Missing
protected audio cancels/defers before commitment or invokes safety/No Contest as
required. Optional crowd/ambience/variant failure degrades explicitly.

## 6. Protected cue taxonomy and attack stages

Protected critical classes are ordered:

1. shared start/resume, unsafe synchronization/control, and terminal integrity;
2. boss Telegraph/Commit/Impact affecting the player, target/source, and urgent
   recovery;
3. required Crescendo and accepted Band Call/group commitment; and
4. indispensable movement/position state where audio is part of the approved
   multimodal response.

Overall GDD mix priority remains:

1. critical boss/timing cues;
2. local selected role and judgment response;
3. complete song; then
4. other combat, crowd, and ambience.

Lower cues may delay, coalesce, simplify, or omit when masking higher. UI has an
independent visual/caption priority contract.

Each attack family carries a boss-specific identity across:

- **Telegraph:** source/family/likely response with validated lead;
- **Commit:** unmistakable locked target/geometry/time state;
- **Impact:** exact ordered boundary resolution;
- **Recovery:** released pressure/earned-advantage opportunity; and
- **Cancellation:** dissipated same identity, never Impact.

Identity combines rhythmic shape, register, timbre/envelope, cadence, and source/
target. Every protected cue has a mono/phone-safe core.

Critical ducking reduces the minimum lower-priority content for the shortest
validated interval. It primarily affects optional combat, crowd, and ambience;
controlled room in the song/local role must preserve pulse/continuity/timing.
Repeated cues share bounded duck envelopes rather than continuous pumping.

Simultaneous threats use impact-ordered protected patterns plus source/target/
shape channels. Same-family/same-boundary duplicates coalesce. Different threats
obey total headroom/voice caps. Distance/spatial reinforces, while personal
target/response retains centered/caption/visual reinforcement.

Ping mute/rate cannot suppress/impersonate automatic cues. A player may mute
audio because other modalities retain semantics; semantic cue state still emits.
Missing required audio blocks/defers/cancels before Commit. Post-Commit loss uses
safety cancellation or No Contest; silent surprise Impact is forbidden.

## 7. Aggregate combat and survival audio

Shared sound begins only from confirmed identified owner events. Local stem gain,
button press, displayed total, or prediction cannot trigger it.

Local grade, direction, miss, down filter, exact contribution, build/gear detail,
and trend never enter other players' mixes. Shared audio communicates resulting
band/boss/world state.

Routine Attack/Defend/effect packets at the same musical boundary aggregate by
family/target into one bounded response. Strength may use semantic ranges, but
roster size, duplicate roles, message count, and packet fragmentation cannot
multiply voices/loudness beyond cap.

Resolve pressure is restrained; Momentum bank/apply is distinct; layer break is
a protected shared musical/world event. Future locked layers make no damage
sound. Finishing/outcome use frozen owner facts and never imply early victory.

Defend/avoidance/risk is mostly personal. Shared Ward reinforcement/restoration,
major group protection/impact, downing, revival progress/completion, and return
use source/target-aware cues only when teammates need state. Routine personal
stats do not create band-wide notifications.

Signature arm/commit/cancel/guaranteed resolution are distinct for owner;
teammates hear only necessary effect/target. Consumables do not sound committed
before durable spend/effect guarantee.

## 8. Band Calls, Crescendos, and acolytes

Band Call invitation includes initiator, identity/effect, and countdown.
Commitment widens/strengthens ensemble for the authentic performance window.
Resolution aggregates guaranteed base and nonnegative human/fixed shares without
individual ranking. Cancel/shared lockout have restrained distinct states.

Crescendo preview uses a protected authored identity. Commit widens the
participating ensemble. Echo, Crescendo, and Full Crescendo use clear increasing
treatments under one headroom cap. A weak share never adds negative sound or
reduces another player's audible success.

Solo acolytes use non-performance motifs:

- Vanguard pressure;
- Warden cadence/reinforcement;
- Herald readiness/group support;
- suppression; and
- recovery.

They never receive responsive role layers, judgments, combo/score/risk audio, or
human-performer sounds. Fixed contribution remains identified fixed support.

Repetition/concurrency caps favor state change over one sound per value packet.
Shared output contains no damage/grade/contribution rank, blame, accessibility,
purchase, or private build information.

## 9. Hub, onboarding, Results, pings, and ambience

The shard ascent and essential anchors use stable spatial motifs. Shards express
boss/arena identity; Practice, Prepare, Progress/archive, Band/social, and
voluntary Store use consistent landmark language. Sound supplements labels and
visuals, never replaces navigation.

Restoration may enrich/reorchestrate hub music, ambience, population, portals,
and motifs while retaining learned identity. Spectacle cannot mask menus, queue,
or anchors.

Menu feedback distinguishes focus/selection, confirm, back/cancel, unavailable,
pending, success, warning, and failure, but does not sound every focus move when
fatiguing. Fast Play/physical shard use the same encounter-card confirmation.

Practice uses clean authored music/layers and real cue language. Calibration
uses an acoustically simple pulse/tap reference distinct from boss motifs.
Counts are aligned/captioned and not chart input. Teaching may reinforce but not
add a competing timing language or mask real cues. Disabling teaching never
removes automatic gameplay information.

Results uses short skippable distinct Victory, nonshaming Defeat, and neutral No
Contest. Reward/unlock/mastery/restoration responses follow confirmed grants and
never delay Retry. There is no paid/store/rescue sting.

Preset pings use localized caption/icon/audio/optional haptic, context/source,
rate/coalescing, and individual mute. Ping receipt never sounds like consent or
gameplay action. Automatic cues remain.

Dialogue is optional, separately routed, fully subtitled with speaker, and never
carries unique timing/response facts. Ducking is bounded and cannot mask active
phrases/threats. Tone remains age-appropriate.

Crowd/ambience responds to aggregate phase/Resolve/group/outcome in coarse capped
ranges. It cannot expose grade/blame or become timing-critical and degrades early.
Scene transitions use bounded fades/stingers. Encounter song never transitions
early from predicted outcome. Background/device change rejoins authority without
replaying scene cues. Store sound remains voluntary and noncoercive.

## 10. Bus, loudness, and dynamic-range model

Player-facing bus controls are:

- Master;
- Song;
- Local Role;
- Timing and Boss Cues;
- Voices;
- Combat Effects;
- Crowd; and
- Ambience.

Names remain subject to UI naming/localization. Internal sub-buses may separate
backing/control, cue classes, UI, pings, group, or world without more sliders.

Master is final parent. Song controls backing; Local Role controls selected
neutral/responsive layer relative to bed. Timing/Boss owns protected automatic
audio. Voices owns dialogue. Combat owns ordinary/shared effects. Crowd/
Ambience own decorative response. UI/pings follow internal routing/source mute.

Each bus has safe default, bounded range including mute, preview, and reset. No
audio floor is forced because alternatives retain semantics. Muting never stops
semantic emission.

Working dynamic-range presets are Full, Balanced, and Quiet/Compressed. Balanced
is default. Full preserves wider dynamics within safe peaks; Quiet reduces range
for low-volume/night/noisy use without flattening cue order. Names/parameters
remain tunable.

Presets alter compressor/limiter/ducking/permitted calibration, not song timing,
event priority, contribution, or identity. Custom values remain visible and
preset reset explains changed fields.

Assets meet class loudness/true-peak/spectral/phase/noise expectations. Neutral
reconstruction keeps headroom for maximum role emphasis, protected cues, group,
and maximum concurrency. A final limiter catches exceptions but cannot be normal
mix strategy.

Ducking declares source/targets, attack/hold/release, max depth/duration, and
priority. Muted triggers do not pump unexplained. Bus/preset edits preview and
apply with click-free ramp without seek/restart/calibration change. UI owns scope;
Player Data persists; Audio reports application/unsupported/failure.

## 11. Device, mono, and spatial behavior

Complete song, local role/judgments, timing/counts, personal targeting/recovery,
and blocking UI/system feedback remain centered/nonspatial.

Bosses, attacks, positions/routes, portals, acolytes, shared effects, dialogue,
and environment may use bounded spatial reinforcement. Spatial never solely
carries target/direction/distance/response.

Critical cues use midrange transients/envelopes that survive phone speakers,
low volume, mono, narrow bandwidth, and no sub-bass. Bass/air/width/reverb/
localization may enrich headphones/desktop without changing identity.

Every map/cue/effect/dialogue/scene is validated for mono phase cancellation,
masking, level, and source/target distinction. Widening/polarity/decorrelation
cannot weaken centered pulse/local role/protected cues.

Critical distance has readable floor/personal reinforcement. Occlusion/reverb
cannot hide cue, smear boundary, or delay it. Doppler/pitch/time-varying spatial
effects never alter song/timing cue pitch or create false offset.

Output profiles may store bus/preset/spatial/caption/haptic/calibration for device
classes. Detection selects prior explicit match or transparently suggests; it
never overwrites choice or assumes quality. Output/Bluetooth change uses phase-
safe rejoin and future recalibration suggestion. No automatic gain/DR/spatial
jump occurs outside applied ramp/profile. Peaks remain bounded; volume never
auto-rises for weak speaker/noisy room.

## 12. Captions and haptic requests

Every captionable cue declares:

- stable identity/revision and category: speech, meaningful sound, music/timing,
  or system;
- localized short/expanded text;
- speaker/source, target/direction/location;
- criticality/priority;
- musical/exact start and duration/end;
- interruption/cancellation and repetition/coalescing; and
- icon/semantic alternative keys.

Speech preserves speaker/meaning. Sound captions explain gameplay/world fact and
source, not only onomatopoeia. Timing captions identify relevant count/boundary/
change without captioning every note. System captions distinguish connection,
input/audio unavailability, and No Contest without blame.

Player-relative direction retains authored geometry. Captions do not mirror
encounter left/right due to RTL localization. Unknown source is omitted, not
guessed.

Critical transitions emit semantic caption/visual facts when audio is muted/
missing. Identical low-priority repetition coalesces; Commit, target, urgent
recovery, and terminal state are not swallowed.

UI renders caption style/focus/localization. Audio emits synchronized metadata
and actual played/canceled/substituted state.

Haptic requests declare identity/revision, class, source/target, criticality,
boundary, intensity band, duration/envelope, repetition/coalescing, priority,
family cooldown, capability, and reduced/off alternative. Device/UI owns final
application.

Brief haptics may reinforce local contact/judgment, Commit/Impact, movement,
invitation/commit, recovery, and major state. Continuous beat vibration, every-
packet pulses, punitive Miss vibration, and high-rate stacking are forbidden.
Danger outranks decorative/contact under total cap. Haptics never replace song,
create easier timing, or carry meaning alone. Off/unsupported changes nothing.

## 13. Audio definition and lifecycle contract

Every behavior uses a versioned definition with:

- cue/family identity, accepted semantic source events, criticality/priority;
- local/shared/participant scope, privacy, bus/internal route;
- approved asset/treatment variants;
- musical/exact boundary, lead/late policy;
- source/target/location/spatial mode;
- start/duration/loop/stop and gain/filter/transient/envelope;
- dynamic-range variants and neutral/full-mix interaction;
- ducking source/targets;
- concurrency/retrigger/coalescing/cooldown/eviction/cap;
- deterministic tie-breaker and variant inputs;
- caption/icon/alternative/haptic keys;
- device/mono/reduced-effects eligibility; and
- fallback plus publication/event/encounter blocking class.

Runtime states are Scheduled, Playing, Completed, Canceled, Failed, or
Substituted. Transitions carry exact definition/asset/content/balance revisions,
causal event/attempt/player/source/target, musical/exact time, selected variant,
bus/profile, prior/next, and idempotency identity.

One semantic event schedules once. Duplicate/late/out-of-order returns established
state without replay/extend/change. Cancellation uses causal identity/owner rule;
stale cancel cannot stop another instance.

Variant choice is deterministic from definition, cause, approved seed/context,
and device/profile. Every critical variant preserves meaning/lead/mono core.
Variation cannot encode different target/value/threat/grade/reward/private fact.

Substitution requires preapproved same-semantic equivalent with compatible
timing, priority, route, mono/caption/accessibility, and revision. Optional
decorative absence may complete silently. Required absence follows safe failure.
Anonymous/ad hoc shipping sound is invalid; debug sound cannot satisfy coverage.

## 14. Concurrency and eviction

Definitions declare family/group, voice caps, merge/retrigger/extend, same-
boundary coalescing, cooldown, priority eviction, cumulative response/ducking,
and stable ties. Local-role response may use one envelope instead of one decoder
voice per note.

Eviction order is:

1. completed/expired;
2. decorative ambience/crowd;
3. optional world/UI variation;
4. lower-priority ordinary combat; then
5. only approved same-semantic coalescing/substitution.

Protected timing/danger/recovery/group, complete song, selected role, and required
alternatives cannot be evicted. Every eviction/coalescing/substitution is reported
with reason.

## 15. Asset, export, and human-review validation

Authoring preserves highest-practical-quality source/control audio, fingerprints,
and processing history. Each asset revision declares identity/hash, lineage,
purpose, encoding, sample rate, channels, start/duration, loop/seek, loudness/
peak target, dependency, preload/stream/cache, and fallback.

Runtime transcodes are derived. Export may change format/compression/chunk/
reference/layout only with equivalence report proving duration/alignment,
reconstruction, cue timing, channel/mono, loudness/headroom, loop/seek, and
semantic/accessibility parity.

Automated validation covers:

- missing/hash/revision references;
- duration/start/sample/phase alignment;
- backing-plus-role reconstruction;
- clipping, DC, noise, silence, true peak, integrated/short loudness;
- spectral extremes, channel/polarity/mono fold;
- loop/seek discontinuity;
- caption/haptic/alternative keys;
- protected cue lead/duration; and
- concurrency/headroom simulations.

Human review covers neutral/maximum A/B, all role response states, leakage,
phone/headphone/mono clarity, critical masking/worst overlaps, group/acolyte/world,
dialogue/captions, bus/DR extremes, and nonshaming age tone.

Publication requires automated pass plus named musical/mix/accessibility approval
on exact platform export. Browser backing/stem regression evidence helps but
cannot replace Roblox listening.

## 16. Streaming, cache, budgets, and degradation

Selected-role song map and initial/guaranteed protected assets preload/phase-
ready. Remaining critical assets are cached/guaranteed before scheduling horizon.
Streaming cannot make an unavailable cue candidate valid.

Cache keys include exact content/audio/platform variant revision. Other role/
revision cannot satisfy readiness. Eviction cannot remove active layers or
committed/upcoming protected assets and preserves pause/rejoin seek.

Runtime profiles declare compressed/decoded memory, network/start latency,
decoder/stream count, voice, spatial/reverb, update, and CPU/frame budgets.
Architecture sets numeric budgets; packages publish measured usage and must fit.

Degradation order is:

1. extra crowd/ambience variants/detail;
2. decorative reverb/spatial width/occlusion complexity;
3. optional world/UI variation;
4. nonessential ordinary-combat variation; then
5. approved lower-cost same-semantic coalescing/substitution.

It cannot remove/retime complete song, local role, timing reference, committed
danger/target, recovery/group cue, caption, or required alternative.

Missing/corrupt song/role blocks selection/deployment. Missing protected assets
block/defer/cancel or cause safety/No Contest. Optional failure degrades and never
borrows semantically different sound.

## 17. Semantic outputs and privacy

Audio emits identified facts for:

- package/map/asset load/cache/preload/readiness;
- start/count/pause/resume/seek/rejoin/stop;
- observed alignment/drift/correction/unsafe confidence;
- output/profile application and degradation/failure;
- judgment/hold/suspension/down/return response, envelope, merge/cap/neutral;
- cue request/schedule/variant/state/concurrency/coalescing/cooldown/eviction;
- ducking/spatial/source/target/caption/haptic/fallback; and
- validation/export/listening evidence.

Facts carry causal/revision/time/idempotency evidence. UI receives actual state,
caption/source, haptic request. Rhythm receives health without yielding clock.
Multiplayer receives readiness/critical failure. Player Data receives profile
application/save-relevant state. Authoring/QA receives evidence. Analytics gets
privacy-reviewed semantic health.

Raw private audio/microphone, another player's grade/timing/local response/exact
contribution, accessibility/profile values, purchases/build/moderation, and
unneeded dialogue identity are excluded from telemetry/other-player output.

## 18. Content Authoring reconciliation requirements

The completed 2026-09-02 `CONTENT_AUTHORING.md` reconciliation incorporates:

- full-mix reference and per-role backing/control/equivalent maps;
- exact asset/layer alignment, neutral level/channel, response capability;
- neutral and maximum reconstruction evidence;
- role response definitions for every judgment/hold/suspension/down/return;
- authored attack/event/landmark/group/recovery/outcome cue family/stage/
  priority/source/target/duck/cancel behavior;
- practice/calibration, hub/restoration, teaching, Results, dialogue/subtitle,
  crowd/ambience mappings;
- buses/routes, DR, loudness/headroom/ducking/device/mono/spatial constraints;
- caption/source/direction/coalescing and bounded haptic metadata;
- full AP-10 definition/lifecycle/concurrency/fallback/idempotency fields;
- AP-11 asset/export/cache/preload/budget/degradation validators and reviews;
- AP-12 semantic outputs and complete test evidence.

Gameplay truth remains in owner systems; audio packages reference semantic keys.
No runtime-private competing schema or orphan package field may remain.

## 19. Verification and observable gates

Verification crosses:

- every playable role/difficulty;
- solo and two/three/six humans including duplicates;
- every grade/hold/suspension/down/return;
- every boss/group/recovery/acolyte/world/Results lifecycle;
- every protected overlap/order/cancel/failure and max concurrency;
- representative low/ordinary phones/tablets, desktop speakers, wired/wireless
  headphones, stereo/mono, low volume;
- all DR presets and meaningful bus mutes/extremes;
- captions and reduced/off/unsupported haptics;
- output change, calibration suggestion, solo pause, co-op loss, rejoin;
- cache miss, stream failure, corruption, substitution, and critical failure.

Evidence combines objective duration/phase/loudness/peak/mono/memory/voice/
latency/CPU measurements, deterministic/idempotent automated tests, structured/
blind A/B listening, phone Roblox playthroughs, and target-age explanation/
response observation. Metrics never replace musical/gameplay approval.

Completion retains the GDD gate that at least 80% recognize local responsive
role and critical boss cues on ordinary phone speakers/headphones, and every
accessibility combination preserves essential cue/outcome.

## 20. Deferred tuning and technical work

Behavior is complete; downstream work includes:

- final sound assets/motifs/voice performances and localized captions;
- exact grade/hold/down/return envelopes and role-specific parameters;
- loudness/peak/headroom, ducking, concurrency, cooldown, and spatial values;
- final bus/preset names, ranges, defaults, and device profiles;
- final formats/transcodes/stream/cache/decoder/voice/memory/CPU budgets;
- Roblox audio authority, scheduling, seeking, spatial, output-change, and
  critical-failure implementation;
- supported platform caption/haptic/device capabilities;
- first three song/package audio completion and human review; and
- representative device/age/accessibility iteration.

Tuning cannot retime/pitch-shift the song, silence the arrangement on Miss,
fabricate roles/onsets, leak local performance, make critical meaning louder-
only/stereo-only/bass-only, make haptics required, multiply roster noise, allow
anonymous sounds, or degrade protected song/cue/alternative content.

## 21. Approval and change control

The owner interview resolved AP-01 through AP-12 on 2026-09-01. This document is
the canonical Audio Presentation design specification.

A material change to neutral role reconstruction, response semantics, clock/
pause/rejoin, protected priority/ducking, aggregate shared sound, bus/device/
mono/spatial behavior, caption/haptic metadata, cue lifecycle/idempotency,
asset/degradation gates, or privacy/verification requires an explicit amendment
citing the superseded rule. Numeric/asset tuning inside these boundaries creates
a new revision and cannot alter active playback history or gameplay truth.
