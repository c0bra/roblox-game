# Bands Battle Audio Presentation Working Record

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-31
- **Question plan:** [`AUDIO_PRESENTATION_QUESTIONS.md`](AUDIO_PRESENTATION_QUESTIONS.md)
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#77-audio-presentation)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Content dependency:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **UI/settings dependency:** [`UI_UX.md`](UI_UX.md)
- **Canonical result:** [`AUDIO_PRESENTATION.md`](AUDIO_PRESENTATION.md)

## 1. Role of this record

This file preserves the approved answers, refinements, inherited constraints,
and cross-system handoffs from the completed Audio Presentation interview. It is
evidence for the canonical specification, not the final authority.

## 2. Inherited boundary

Audio Presentation owns stable runtime song/layer playback, local-role response,
semantic cues and mix priority, buses/dynamic range/device treatment, spatial/
mono behavior, caption/source metadata, haptic requests, and audible degradation.
It does not own the musical clock, judgments, gameplay state, source authoring,
setting values, caption rendering, persistence implementation, or technical
audio architecture.

The complete inherited decision set is recorded in
[`AUDIO_PRESENTATION_QUESTIONS.md`](AUDIO_PRESENTATION_QUESTIONS.md#2-fixed-inherited-decisions).

## 3. Existing evidence, not binding architecture

The Blackened Crown browser assets currently pair one role-specific backing file
with one role-specific stem file, and the prototype exposes prepare/start/time/
pause/resume/duck/accent/stop behavior. This proves a useful minimal interaction
shape. The specification may retain, expand, or replace that representation as
long as it preserves the approved flexible-role and complete-song semantics.

## 4. Decision record

### Checkpoint A - Song playback, controllable layers, and local response

#### AP-01 - Runtime song/layer model and neutral full-mix reference

- **Status:** Resolved 2026-09-01.
- Every approved song revision supplies one immutable full-mix reference used for
  musical/technical comparison plus one runtime audio map for each offered
  playable role. A role map identifies a neutral backing bed and one or more
  controllable local-role layers, all derived from the authentic arrangement.
- At neutral state, backing plus the role's controlled layers must reconstruct
  the approved complete song with the same start, duration, tempo/phase, musical
  content, and human-approved perceived balance. Encoding/transcode tolerances
  may prevent sample-identical output, so automated alignment/loudness/phase
  checks are paired with human A/B review against the reference.
- The controlled role remains audible at a defined neutral baseline. Performance
  applies bounded gain/clarity/filter/transient or equivalent deltas around that
  baseline; it never gates the role completely on/off. The backing always
  preserves song continuity and other musically necessary parts.
- A role may control one clean stem, several grouped sublayers, an alternate
  role-specific backing plus controlled layer, or another human-approved
  equivalent. An equivalent qualifies only if it responds audibly to authentic
  role events without fabricating material, noticeably changing other parts, or
  breaking neutral reconstruction.
- Role definitions are extensible and song-specific. Drums/vocals/guitar/bass
  are examples only; piano, synthesizer, percussion, strings, layered textures,
  or later roles use the same contract. Source layers not offered as playable
  remain inside the neutral bed or other noninteractive mix assets.
- Sparse/ambient roles use their actual musical behavior rather than invented
  attacks. A role without adequate authentic controllable/equivalent audio cannot
  be offered for that revision. Dirty isolation may pass only when neutral and
  maximum-response A/B review finds leakage/masking acceptable.
- Each client selects the approved map for its local role. Duplicate-role humans
  use the same content map independently; their local emphasis does not sum into
  a louder shared song or require a networked stem mix.
- All layer identities, files/assets, start offsets, duration, neutral levels,
  channel layout, and alignment/reconstruction evidence bind to the exact
  content revision. Runtime never combines layers from different revisions.

#### AP-02 - Judgment, hold, suspension, downing, and recovery response

- **Status:** Resolved 2026-09-01.
- Each local judgment consumes Rhythm's identified semantic result once and
  applies the role definition's time-bounded response envelope. Response is
  aligned to the authentic audible event/attempt and cannot create an onset or
  phrase that is absent from the source arrangement.
- Semantic emphasis remains ordered: Perfect has the crispest/clearest strongest
  bounded lift, Great has the confident normal accent, Good is softer/less
  prominent, and Miss briefly ducks/filters/stumbles without complete silence or
  a punitive unrelated failure sound. Response strength never changes judgment,
  normalized contribution, or combat value.
- Roles express that ordering musically. Percussive material may use transient
  definition; sustained/ambient material may use clarity, presence, or filter
  movement; vocals/legato material may use another reviewed envelope. Every role
  keeps the same semantic ordering and headroom even when parameters differ.
- Early and Late do not receive different first-release audio signatures. UI and
  private Results carry timing direction; the audible response communicates the
  grade without adding another coaching language.
- Rapid Repeat/Alternate events do not stack unrestricted independent sounds.
  Compatible envelopes merge, retrigger, or extend under role-specific
  concurrency/headroom caps while preserving each judgment's semantic evidence.
  The result strengthens the performed source rather than becoming percussion-
  like noise unrelated to the role.
- A Hold onset receives its normal grade envelope. Accepted maintenance may
  sustain a restrained clarity/presence state proportional to live hold state.
  Early release stops future sustain and returns toward neutral without a second
  release judgment or Miss sound. Suspension ends future response at its exact
  boundary while preserving already heard response.
- An unclaimed event's Late Miss may apply the same bounded local Miss treatment.
  Ordinary zero input produces no positive accent. Movement, authored inactivity,
  recovery preparation, disconnect/AFK/input-unavailable, and synchronization
  suspension return the role toward neutral without a Miss cue.
- Downing transitions the selected role to a local muffled/distant state while
  the complete song and protected cues continue. Successful recovery/rejoin
  returns through an on-beat bounded swell, then resumes neutral/eligible response
  only at the owning fair-return boundary.
- Judgment response is private/local. Teammates never hear another player's
  individual grade, miss, down-state filter, or timing tendency. Confirmed local
  response is not rolled back audibly by ordinary network delay; shared combat/
  group audio is driven separately by confirmed semantic events.

#### AP-03 - Start, clock alignment, pause, drift, rejoin, and playback failure

- **Status:** Resolved 2026-09-01.
- Before deployment can lock/start, each client loads and verifies the full
  selected-role song map and every required initial/guaranteed critical cue,
  acknowledges the exact content/audio revision, and proves phase-ready playback.
  Remaining protected assets must be guaranteed available before their authored
  scheduling horizon; optional decorative audio does not block.
- Multiplayer supplies one shared future start boundary. Audio schedules every
  backing/control layer to that boundary in phase. Playback follows the approved
  musical clock and exposes observation/health facts; it does not become a
  competing authority for chart or encounter time.
- Audio/output latency calibration aligns perception without retiming source
  assets, changing song speed/pitch, moving chart/events, or widening judgments.
  Local performance response uses the synchronized local representation for
  immediate feedback and is never visibly/audibly rolled back by routine server
  delay.
- All layers maintain their declared relative start/offset. They cannot restart,
  free-run, or seek independently. Normal drift convergence must be inaudible,
  avoid active strike/reaction windows, and never create audible tempo or pitch
  manipulation. Unsafe confidence invokes Rhythm suspension/resynchronization;
  unrecoverable outcome-critical clock/audio corruption uses No Contest.
- Solo pause freezes the song, all controlled layers, response envelopes,
  clock-driven cues, and encounter time at the same instant. The separate
  visible/audible phase-aligned count-in accepts no judgments and does not advance
  the song; resume continues from the exact frozen point.
- Cooperative play never pauses for one player. A recoverable output-device
  change rejoins the current shared position through a short phase-safe fade and
  may privately suggest recalibration. If the player loses a usable audio/input
  presentation with unsafe timing confidence, the established local Rhythm
  suspension applies rather than altering the shared song.
- A returning network player loads/validates the exact active revision, seeks to
  current authoritative song position, and enters through a short phase-safe
  fade at the safe return boundary. Their local role begins in the authoritative
  neutral/downed/return-protected state; playback never restarts for the band.
- Normal completion/terminal resolution stops or transitions all layers once at
  the exact owner-supplied boundary. Repeated stop/end facts are idempotent and
  cannot restart an asset or replay an outcome cue.
- Missing/misaligned full-song or selected-role audio blocks deployment. A
  missing protected cue before commitment cancels/defer its event according to
  the owner contract; outcome-critical missing/corrupt audio can produce No
  Contest. Optional crowd/ambience/noncritical variants may degrade explicitly
  without changing the song, clock, gameplay, or captions.

### Checkpoint B - Critical cues, shared events, and world response

#### AP-04 - Critical cue taxonomy, priority, masking, and ducking

- **Status:** Resolved 2026-09-01.
- Protected critical cue classes are:
  1. shared start/resume count, unsafe synchronization/control, and terminal
     integrity/No Contest;
  2. boss Telegraph/Commit/Impact affecting the player, explicit target/source,
     and urgent solo/cooperative recovery;
  3. required Crescendo preview/commit and already accepted Band Call/group
     commitment; and
  4. indispensable movement/position readiness or state transition when audio is
     part of the approved multimodal response.
- Mix priority remains the GDD order: critical boss/timing cues; local selected
  role and judgment response; complete song; then other combat, crowd, and
  ambience. A lower cue may be delayed, coalesced, simplified, or omitted when
  it would mask a higher cue. UI/visual alternatives remain independently
  prioritized under `UI_UX.md`.
- Each attack family has one boss-specific audible identity carried across its
  four stages. Telegraph introduces source/family/likely response with validated
  lead; Commit adds an unmistakable lock state without changing target/time;
  Impact resolves on its exact identified boundary; Recovery releases pressure
  and may announce an earned-advantage opportunity. Cancellation audibly
  dissipates the same identity and never sounds like Impact.
- Critical identity uses a combination of rhythmic shape, pitch/register,
  timbre/envelope, cadence, and source/target treatment. Loudness alone is never
  the differentiator. Every cue has a mono/phone-safe core that does not depend
  on sub-bass, extreme treble, stereo width, or precise spatial hearing.
- Critical sidechain/ducking reduces only the minimum lower-priority content for
  the shortest validated interval. It primarily affects optional combat effects,
  crowd, and ambience; it may make controlled room around a cue in the local
  role/full mix but must preserve musical pulse, perceived song continuity, and
  the player's timing reference. Repeated cues share one bounded duck envelope
  rather than pumping the song continuously.
- Simultaneous legal threats use one impact-ordered protected pattern plus
  distinct source/target/shape channels. Same-family/same-boundary duplicates
  coalesce under stable order. Different threats keep recognizable identity but
  obey total critical-voice/headroom caps; the mix never plays one full-volume
  warning per target/player.
- Distance/spatial position may reinforce source, but personal targeting and
  required response always have centered/nonspatial and caption/visual
  reinforcement. Off-screen, occluded, mono, or muted-spatial presentation must
  remain equally actionable.
- Preset ping mute/rate state cannot suppress or impersonate a protected cue.
  User audio settings may reduce/mute an audio channel only because UI/caption/
  optional haptic alternatives retain the complete semantic fact; the runtime
  still emits protected cue state to those consumers.
- Missing/invalid critical audio before Commit blocks, defers, or safely cancels
  the owning event. After Commit, loss of its required audible-plus-alternative
  contract invokes visible/audible safety cancellation where legal or the
  established outcome-critical No Contest path. A surprise silent Impact is
  forbidden.

#### AP-05 - Combat, group actions, acolytes, and aggregate band audio

- **Status:** Resolved 2026-09-01.
- Shared combat audio begins only from confirmed identified owner-domain facts,
  never by listening to local stem gain, raw button presses, displayed totals, or
  predicted network state. Each cue retains causal event/attempt/revision and
  permitted private source/target attribution.
- Local judgment grade, timing direction, individual miss, personal down-state
  filter, exact contribution, build/equipment detail, and performance trend are
  never sent into another player's mix. Shared audio communicates the resulting
  band/boss/world event at a useful aggregate level.
- Routine Attack/Defend/effect packets within the same scoring/musical boundary
  aggregate by family/target into one bounded band response. Strength may map to
  approved semantic ranges, but roster size, duplicate roles, message count, and
  packet fragmentation cannot multiply loudness/voices without cap. One through
  six humans preserve comparable headroom and clarity.
- Resolve pressure uses restrained confirmation; Momentum bank/apply has a
  distinct preparatory/release identity; a layer break is a protected shared
  musical/world event. Future locked layers make no damage sound. Finishing and
  frozen outcome use their exact owner facts and never imply early victory.
- Defend/avoidance/risk feedback is primarily personal and brief. Shared Ward
  reinforcement/restoration, large group protection, major impact, downing,
  revival progress/completion, and return use source/target-aware cues only when
  teammates need the state. Routine personal stat changes do not create band-
  wide notifications.
- Signature arming/commit/cancel/guaranteed resolution are distinct for the
  owner. Teammates receive only shared effect/target information necessary for
  play. Consumables follow the same rule and never sound committed before the
  owning durable spend/effect guarantee.
- A Band Call invitation carries initiator, Call identity/effect, and countdown;
  commitment widens/strengthens the ensemble for the authenticated performance
  window; resolution aggregates the guaranteed base and nonnegative human/fixed
  shares without sonifying individual ranking. Cancel and shared lockout have
  restrained distinct states.
- Crescendo preview has its own protected authored musical identity. Commit
  widens the participating ensemble, and Echo/Crescendo/Full Crescendo produce
  three clear increasing result treatments under one headroom cap. A weak share
  never introduces a negative sound or reduces another player's audible success.
- Solo acolytes use stable non-performance support motifs: Vanguard pressure,
  Warden cadence/reinforcement, Herald readiness/group support, suppression, and
  recovery. They never receive responsive instrument layers, judgments, combo/
  score audio, risk audio, or the sound of a human performer. Fixed contribution
  remains labeled fixed support in semantic/caption evidence.
- Repetition/concurrency caps apply by family, target, boundary, and priority.
  The mix favors meaningful state change over one sound per numeric packet.
  Public/shared output contains no individual damage, grade, contribution rank,
  blame, accessibility, purchase, or private build information.

#### AP-06 - Hub, onboarding, Results, dialogue, pings, crowd, and ambience

- **Status:** Resolved 2026-09-01.
- The shard ascent and each essential hub anchor use stable identifiable spatial
  motifs. Shards distinguish boss/arena identity; Practice, workshop/Prepare,
  archive/Progress, social commons/Band, and voluntary store use a consistent
  landmark language. Motifs supplement labels/visuals and never become the only
  way to navigate.
- Campaign restoration may enrich/reorchestrate hub music, ambience, population,
  portal activity, and landmark variations while retaining each learned motif,
  route, and functional identity. Higher spectacle cannot mask menu feedback,
  queue state, or another essential anchor.
- Hub/menu feedback is restrained and consistent: focus/selection, confirm,
  back/cancel, unavailable, pending, success, warning, and failure are distinct
  but not sounded on every focus move when that would fatigue or mask music.
  Fast Play and physical shard entry use the same encounter-card confirmation
  language.
- Practice uses clean authored music/control layers and the real judgment/cue
  language. Calibration uses an acoustically simple visual-aligned pulse and tap
  reference distinct from boss motifs. Count-ins are phase aligned, bounded,
  captioned, and never accepted as chart input.
- Contextual first-boss teaching may reinforce an already valid cue but cannot
  add a competing timing language, play late, or mask the cue it explains.
  Dismissed/disabled teaching removes explanatory audio, not automatic gameplay
  information.
- Results uses short skippable and semantically distinct Victory, Defeat, and
  neutral Invalid / No Contest treatments. Defeat is serious but nonshaming; No
  Contest does not sound like player failure. Reward/unlock/mastery/restoration
  responses follow confirmed grants and remain skippable/nonblocking so Retry is
  never delayed. There is no paid/store sting or rescue prompt.
- Preset pings have localized text/caption, icon/shape, compact audible identity,
  optional haptic request, context/source, and the approved rate/coalescing rules.
  Individual source mute suppresses only that player's ping audio. Automatic
  cues remain. Ping receipt never sounds like consent or a gameplay action.
- Dialogue/voice is optional to core play, routed separately, fully subtitled
  with speaker identity, and never carries a timing/response fact absent from
  UI/caption/gameplay cues. Dialogue ducking is bounded and cannot obscure active
  phrases or critical threats. Language/tone remains age-appropriate.
- Crowd and ambience respond to confirmed aggregate phase, Resolve, group, and
  outcome state in coarse bounded ranges. They cannot expose an individual grade
  or blame, determine gameplay, or become required timing. They are among the
  first layers simplified/removed under device or masking pressure.
- Hub/encounter/Results transitions use authored bounded fades/stingers and
  explicit state identities. Encounter song never fades/changes early because a
  client predicted outcome. Focus/background/device/platform changes rejoin the
  authoritative current state under AP-03 rather than replaying scene cues.
- Store audio is a voluntary location/surface treatment and cannot be louder,
  repeated, personalized, or inserted elsewhere to create purchase pressure.

### Checkpoint C - Buses, devices, accessibility, captions, and haptics

#### AP-07 - Bus catalog, player settings, loudness, and dynamic range

- **Status:** Resolved 2026-09-01.
- The first-release player-facing bus controls are **Master**, **Song**, **Local
  Role**, **Timing and Boss Cues**, **Voices**, **Combat Effects**, **Crowd**, and
  **Ambience**. Names remain subject to UI/UX's naming/localization gate. Internal
  sub-buses may separate backing/control layers, protected cue classes, UI,
  pings, group events, or world sources without adding required sliders.
- Master is the final parent. Song controls the neutral backing bed; Local Role
  controls the selected role's neutral plus responsive layer relative to that
  bed. Timing/Boss owns protected automatic audio. Voices owns dialogue/story.
  Combat owns ordinary/shared effect sound. Crowd/Ambience own decorative world
  response. UI/pings use declared internal routing and source mute/rate rules.
- Every player-facing bus has a useful safe default, bounded continuous or
  stepped range including mute, current/default value, preview material, and
  reset. No audio floor is forced: a player may mute any/all sound because
  visual/caption/optional-haptic alternatives preserve critical semantics.
  Muting never disables generation of the semantic fact for other modalities.
- The working dynamic-range presets are **Full**, **Balanced**, and
  **Quiet/Compressed**. Balanced is the ordinary default. Full preserves wider
  musical dynamics within safe peak/headroom limits. Quiet/Compressed reduces
  peak-to-average range for low-volume/night/noisy use while retaining the same
  cue order and avoiding constant loudness. Names/parameters remain tunable.
- Presets change compressor/limiter/ducking and permitted bus calibration, not
  authored song timing, event priority, contribution, or sound identity. A
  custom bus value remains visible; applying/resetting a preset explains which
  compatible values it changes and does not silently erase unrelated settings.
- Approved source and cue assets meet revisioned integrated loudness, true-peak,
  spectral/phase, and noise expectations by class. Neutral reconstruction leaves
  headroom for maximum local-role emphasis, protected cues, group events, and
  worst-case allowed concurrency. A final safety limiter catches exceptional
  sums but cannot be the normal mix strategy or pump audibly with every note.
- Ducking uses declared source/target buses, attack/hold/release, maximum depth,
  maximum cumulative duration, and priority. A muted/inaudible trigger does not
  create unexplained pumping. Repeated triggers merge under AP-04; local accents
  and critical cues cannot drive unbounded gain reduction.
- Bus and dynamic-range edits preview in safe settings and may apply during play
  through a click-free bounded ramp because they change presentation only.
  They do not re-seek/restart audio or change calibration. Current application,
  pending/failure, and reset state are reported to UI.
- UI/UX defines per-output-device profile scope and Player Data persists it.
  Audio consumes an exact applied profile, reports unsupported/substituted
  values, and never silently changes an explicit choice because it detected a
  different device.

#### AP-08 - Phone, headphones, mono, spatialization, and output profiles

- **Status:** Resolved 2026-09-01.
- The approved complete song, local-role response, judgments/timing references,
  start/resume counts, personal target state, urgent personal recovery, and
  blocking UI/system feedback are centered or nonspatial. Their meaning cannot
  move with camera orientation, avatar location, or stereo image.
- Bosses, attack sources, positions/routes, portals/shards, acolytes, other
  performers' shared effects, dialogue speakers, and environmental events may
  use bounded spatial reinforcement. Spatial treatment supplements a mono-safe
  core/caption/visual indicator and never becomes the only target, direction,
  distance, or response evidence.
- Critical cues use a validated midrange transient/envelope and survive ordinary
  phone speakers, low-volume playback, mono fold-down, narrow bandwidth, and
  absent sub-bass. Bass extension, air, stereo width, reverberation, and precise
  localization may enrich headphones/desktop but cannot change semantic identity.
- All role maps, cue variants, effects, dialogue, and scene mixes are checked for
  mono phase cancellation, masked midrange, unstable level, and loss of source/
  target distinction. Stereo widening, polarity, and decorrelation cannot weaken
  the centered pulse/local role/protected cue when folded down.
- Critical distance rolloff has a readable floor or personal nonspatial
  reinforcement. Occlusion/reverb may communicate world structure but cannot
  fully hide a protected cue, smear its action boundary, or delay it. Doppler,
  pitch shift, and time-varying spatial effects never alter song/timing cue pitch
  or create a false musical offset.
- Output profiles may contain last approved bus/preset/spatial/caption/haptic and
  latency-calibration preferences for phone speaker, headphones, desktop speaker,
  or other identified device classes. Detection may select a previously explicit
  matching profile or offer a transparent suggestion; it never silently
  overwrites a choice or assumes a device's quality.
- Output/Bluetooth changes during an attempt follow AP-03 phase-safe rejoin and
  preserve the shared clock. A material latency change privately recommends
  recalibration for a future attempt. No automatic gain jump, dynamic-range
  change, or stereo-to-spatial change occurs without the applied profile/ramp.
- Sudden peaks are bounded across profiles and device changes. Master/mix values
  do not climb automatically to compensate for a weak speaker/noisy room.
  Players retain control and protected cues retain nonaudio alternatives.
- Listening acceptance covers representative lower-capability phone speakers,
  ordinary phones/tablets, wired/wireless headphones, desktop speakers, stereo,
  mono, low-volume/Quiet preset, and output change. It retains the GDD target
  that at least 80% recognize the local role response and critical boss cues on
  ordinary phone speakers or headphones.

#### AP-09 - Caption/source metadata, accessible alternatives, and haptic requests

- **Status:** Resolved 2026-09-01.
- Every captionable cue definition supplies stable identity/revision, category
  (**speech subtitle**, **meaningful sound**, **music/timing**, or **system**),
  localized short/expanded text keys, speaker/source, target/direction/location,
  criticality/priority, logical musical/exact start, duration/end, interruption/
  cancellation, repetition/coalescing, and matching icon/semantic cue keys.
- Speech subtitles preserve speaker identity and translated dialogue meaning.
  Meaningful-sound captions explain the gameplay/world fact and source, not only
  an onomatopoeia. Timing/music captions remain concise and identify countdown/
  boundary or relevant musical change without captioning every ordinary note.
  System captions distinguish failure, connection, input/audio unavailability,
  and No Contest without blaming the player.
- Personal target/direction metadata uses player-relative words/indicators while
  retaining stable authored source/geometry identity. Captions do not mirror
  encounter left/right as a localization side effect. Unknown/irrelevant source
  is omitted honestly rather than guessed.
- Critical transitions always produce their caption/visual semantic fact even
  when audio is muted or missing. Repeated identical low-priority sounds coalesce
  with an optional count/updated duration; they do not flood reading. A critical
  Commit, target change before Commit, urgent recovery, or terminal transition
  is never swallowed by coalescing.
- UI/UX renders captions/subtitles using the player's enabled state, text size,
  background, contrast, location, focus/announcement, localization, and content-
  scale rules. Audio owns only synchronized metadata/emission and reports actual
  played/canceled/substituted state so captions do not claim a sound occurred
  when the distinction matters.
- Each haptic request declares stable semantic identity/revision, class, source/
  target, criticality, requested start/boundary, bounded intensity band,
  duration/envelope, repetition/coalescing, priority, family cooldown, device
  capability requirement, and reduced/off alternative. UI/device application
  owns final availability and player preference.
- Brief restrained haptics may reinforce local contact/judgment, Commit/Impact,
  movement readiness/arrival, invitation/commitment, recovery, and major state
  change. Continuous beat vibration, one pulse for every shared numeric packet,
  punitive Miss vibration, and high-rate stacking are forbidden. Critical
  danger outranks decorative/contact haptics under one total rate/intensity cap.
- Haptics never substitute for the song, provide an easier independent rhythm
  guide, change timing, or carry required meaning alone. Turning them off/reducing
  them changes no gameplay/reward/public fact. Unsupported devices simply omit
  them while audio/visual/caption alternatives remain complete.
- Captions and haptic facts obey the same privacy rules as audio: no other
  player's grade/timing trend, exact contribution, accessibility setting,
  purchase, private build, moderation state, or blame is exposed.

### Checkpoint D - Cue definitions, authoring, performance, and completeness

#### AP-10 - Audio definition, event lifecycle, concurrency, and idempotency

- **Status:** Resolved 2026-09-01.
- Every audio behavior comes from a versioned definition containing stable cue/
  family identity, accepted semantic source-event kinds, criticality/priority,
  local/shared/participant scope, privacy class, bus/internal route, and one or
  more approved asset/treatment variants.
- A definition also declares musical/exact scheduling boundary, allowed lead/
  late behavior, source/target/location/spatial mode, start offset/duration/
  loop/stop behavior, gain/filter/transient/envelope, dynamic-range variants,
  ducking source/targets, and neutral/full-mix interaction where applicable.
- Concurrency fields include group/family, per-definition and per-group voice
  caps, retrigger/merge/extend behavior, same-boundary coalescing, family
  cooldown, priority eviction, maximum cumulative response/ducking, and
  deterministic stable-order tie-breakers. Local-role response may be one
  controlled envelope rather than a new decoder voice for every judgment.
- Accessibility/presentation fields include short/expanded caption and speaker/
  source/direction metadata keys, icon/semantic alternative, haptic request key,
  device/mono variant eligibility, reduced-effects behavior, missing-asset
  fallback, and whether absence blocks publication/event/encounter.
- Runtime states are **Scheduled**, **Playing**, **Completed**, **Canceled**,
  **Failed**, or **Substituted**. Each transition carries the exact definition/
  asset/content/balance revisions, causal gameplay event and attempt/player/
  source/target, musical/exact time, selected variant, bus/profile, prior/next
  state, and idempotency identity.
- An owning semantic event schedules once. Repeated/late/out-of-order delivery
  returns the established audio state and cannot replay, extend, spend, or change
  a cue. Cancellation uses the same causal identity and only follows an owner-
  allowed pre-Commit/safety/terminal rule; a stale cancel cannot stop a later
  unrelated instance.
- Variant choice is deterministic from definition, causal identity, approved
  seed/context, and device/profile. Every critical variant carries equivalent
  meaning/lead/mono core. Variation may prevent repetition but cannot encode a
  different target, value, threat, grade, reward, or private fact.
- When caps conflict, completed/expired voices clear first, then decorative
  ambience/crowd, optional world/UI variation, and lower-priority ordinary
  combat. Protected timing/danger/recovery/group cues, complete song, selected
  local role, and required alternatives cannot be evicted. Same-priority order
  is stable and reported.
- Substitution is legal only to a preapproved same-semantic variant/equivalent
  with compatible timing, priority, bus, mono/caption/accessibility behavior,
  and revision. Optional decorative absence may become explicit silent
  completion. A required cue without equivalent uses AP-03/AP-04 safe failure.
- No runtime feature may trigger an anonymous/ad hoc sound outside the catalog.
  Temporary debug audio is nonshipping, clearly marked, and cannot satisfy
  validation or accessibility coverage.

#### AP-11 - Asset/package validation, streaming, budgets, and degradation

- **Status:** Resolved 2026-09-01.
- The authoring project preserves highest-practical-quality source/control audio
  plus fingerprints and processing history. Each platform-neutral/runtime asset
  revision declares identity/hash, source lineage, role/cue/variant purpose,
  encoding, sample rate, channel layout, exact start/duration, loop/seek markers,
  loudness/peak target, dependency, preload/stream/cache class, and fallback.
- Roblox/runtime transcodes are derived artifacts. Export may change format,
  compression, chunking, asset reference, or layout only when an equivalence
  report proves duration/alignment, neutral reconstruction, cue timing, channel/
  mono behavior, loudness/headroom, loop/seek, and semantic/accessibility parity.
- Automated validation covers missing/hash/revision references; duration/start/
  sample/phase alignment; backing-plus-role reconstruction; clipping, DC, noise,
  silence, true peak, integrated/short loudness, spectral extremes, channel/
  polarity/mono fold; loop/seek discontinuity; caption/haptic/alternative keys;
  protected cue lead/duration; and concurrency/headroom simulations.
- Human review covers neutral and maximum-response A/B against the approved full
  mix; every playable role's musical Perfect/Great/Good/Miss/hold/suspension/
  down/return behavior; role leakage; phone/headphone/mono intelligibility;
  critical masking and same-boundary worst cases; group/acolyte/world treatment;
  dialogue/captions; dynamic-range/bus extremes; and nonshaming age-appropriate
  tone.
- An encounter preloads/phase-readies its selected role's complete song map and
  initial/guaranteed protected assets. Remaining critical assets must be locally
  cached/guaranteed before their authoring scheduling horizon. Streaming cannot
  make a candidate appear valid when its required cue is not ready.
- Cache keys include exact content/audio/platform variant revision. A prior or
  other-role file cannot satisfy readiness. Cache eviction never removes an
  active layer or committed/upcoming protected asset and must preserve rejoin/
  pause seek requirements.
- Each supported runtime profile declares budgets for compressed/decoded memory,
  network/start latency, decoder/stream count, simultaneous voices, spatial/
  reverb processing, update rate, and CPU/frame impact. Exact numeric budgets
  remain architecture/device-test values, but content packages publish measured
  usage and cannot exceed the selected supported profile.
- Degradation order is deterministic: extra crowd/ambience variants and detail;
  decorative reverb/spatial width/occlusion complexity; optional world/UI
  variation; then nonessential ordinary-combat variation. Degradation may
  coalesce/substitute approved lower-cost equivalents. It cannot remove/retime
  the complete song, local role, timing reference, committed danger/target,
  recovery/group cue, caption, or required alternative.
- Missing/corrupt full song or local-role map blocks that selection/deployment.
  Missing protected assets block/defer/cancel before Commit or invoke safety/No
  Contest after commitment. Optional assets fail explicitly and degrade; runtime
  never borrows a semantically different convenient sound.
- Publication requires automated pass plus named human musical/mix/accessibility
  approval on the exact platform export. Browser-prototype backing/stem behavior
  is useful regression evidence but does not replace package validation or
  representative Roblox listening.

#### AP-12 - Semantic outputs, test matrix, and Content Authoring reconciliation

- **Status:** Resolved 2026-09-01.
- Audio emits identified state for package/map/asset load, cache, preload and
  readiness; start/count/pause/resume/seek/rejoin/stop; observed alignment/drift/
  correction/unsafe confidence; output-device/profile application; and
  optional/protected degradation or failure.
- Local-response facts cover semantic judgment/hold/suspension/down/return input,
  selected definition/envelope, schedule/play/merge/extend/cap/neutral return,
  applied bus/profile, and omission/failure. They remain private to the player
  and authorized validation/analytics consumers.
- Cue facts cover request/schedule, deterministic variant, Playing/Completed/
  Canceled/Failed/Substituted, concurrency/coalescing/cooldown/eviction, ducking,
  spatial/source/target treatment, caption emission, haptic request, and exact
  fallback. Every fact carries causal/revision/time/idempotency evidence.
- UI receives actual state plus caption/source and haptic-request facts; Rhythm
  receives playback/alignment health without surrendering clock authority;
  Multiplayer receives readiness/critical-failure facts; Player Data receives
  profile application/save-relevant state; authoring/QA receives validation and
  equivalence; Analytics receives privacy-reviewed semantic/health evidence.
- Raw private audio, microphone input, another player's grade/timing/local-role
  response, exact contribution, accessibility/profile values, purchase/build/
  moderation details, and unneeded dialogue identity are excluded from ordinary
  telemetry and other-player output.
- Verification crosses every playable role and difficulty; solo and two/three/
  six humans including duplicate roles; every local grade/hold/suspension/state;
  every boss/group/recovery/acolyte/world/Results cue lifecycle; every protected
  overlap/order/cancel/failure; and full minimum/maximum concurrency.
- Device/settings coverage includes representative low-capability and ordinary
  phones/tablets, desktop speakers, wired/wireless headphones, stereo/mono, low
  volume, all dynamic-range presets, meaningful bus mutes/extremes, captions,
  reduced/off haptics, output change, latency recalibration suggestion, solo
  pause, cooperative loss, network rejoin, cache miss, and streaming/asset fault.
- Evidence combines objective duration/phase/loudness/peak/mono/memory/voice/
  latency/CPU measurements; automated deterministic/idempotent failure tests;
  blind/structured human A/B listening; representative phone Roblox playthroughs;
  and target-age explanation/response observations. Passing metrics never replace
  human musical and gameplay-cue approval.
- Completion retains the GDD gate that at least 80% recognize their responsive
  role and critical boss cues on ordinary phone speakers or headphones, while
  all accessibility combinations preserve every essential cue and outcome.
- Content Authoring final reconciliation must incorporate the complete AP-01
  through AP-12 audio map, cue, metadata, validation, export-equivalence,
  budget, review, and evidence requirements. No runtime-private competing audio
  schema or orphaned package field may remain.

## 5. Content/configuration reconciliation register

- No new authoring requirements have been approved yet.
- Final reconciliation must distinguish source/control audio assets and authored
  cue metadata from runtime mix/priority/settings behavior.
- `CONTENT_AUTHORING.md` must gain every approved layer-map, alignment,
  reconstruction, cue/caption/haptic metadata, device/mono, and listening-review
  requirement after AP-01 through AP-12 resolve them.
- Song packages require one full-mix reference, an exact runtime map for every
  playable role, neutral backing plus one-or-more authentic control layers or
  approved equivalent, stable alignment/level/channel fields, and automated plus
  human neutral/maximum-response reconstruction evidence.
- Runtime/export validation must prove exact revision/duration/phase alignment,
  critical preload/readiness, pause/seek/rejoin equivalence, and safe rejection
  of missing or mixed-revision song/role assets.
- Every authored attack/event/landmark/group/recovery/outcome family requires
  stable stage/priority, mono-safe motif, source/target, ducking, caption,
  alternative-modality, cancellation, and missing-cue behavior plus overlap and
  masking evidence.
- Practice/calibration, hub restoration/landmark variants, first-boss teaching,
  Results stingers, dialogue/subtitles, and crowd/ambience states require exact
  semantic mappings without embedding gameplay truth inside audio playback.
- Audio configuration requires the eight player-facing bus identities, internal
  routing, Full/Balanced/Quiet profiles, loudness/headroom/ducking data, device/
  output profiles, mono/spatial constraints, and safe preview/reset semantics.
- Every meaningful cue needs complete caption/source/direction/timing/coalescing
  metadata and any bounded haptic-request definition, including localized keys
  and nonaudio alternatives.
- Every audio definition requires AP-10's complete scheduling, variant, bus,
  spatial, envelope, ducking, concurrency, lifecycle, accessibility, fallback,
  and idempotency fields. Anonymous shipping sounds are invalid.
- Asset/export gates require AP-11 automated measurements, exact revision/cache/
  preload classes, runtime budget evidence, human A/B/phone/mono/masking review,
  and deterministic degradation/failure behavior.
- The reconciled package and catalogs must emit every AP-12 playback, response,
  cue, mix, caption/haptic, profile, degradation, and failure semantic fact and
  pass the full role/difficulty/roster/device/accessibility matrix.

## 6. Confirmed architecture handoffs

- Content Authoring owns source/control assets, audio maps, authored cue metadata,
  validation evidence, and approved runtime package revision.
- Rhythm owns song time and judgment; gameplay domains own semantic state/event
  identity. Audio consumes those facts and never infers them from playback.
- UI/UX owns setting definitions/profile scope, caption rendering, visual/
  optional-haptic reinforcement, and presentation registry priority.
- Player Data stores settings/profile selections; Audio only reports applied/
  failed runtime state.
- Analytics receives privacy-reviewed semantic/mix/failure evidence, never raw
  private performance audio or another player's local judgment response.

## 7. Change log

- **2026-08-31:** Created the working record. Progress is 0 of 12 questions.
- **2026-09-01:** Approved AP-01 through AP-03. Flexible role-layer maps,
  neutral full-mix reconstruction, deterministic musical local response, exact
  start/pause/rejoin/clock behavior, and critical playback failure rules are
  resolved. Progress is 3 of 12 questions.
- **2026-09-01:** Approved AP-04 through AP-06. Protected staged cue identity/
  ducking, capped aggregate combat/group/acolyte audio, and stable hub/practice/
  Results/ping/dialogue/crowd/ambience behavior are resolved. Progress is 6 of
  12 questions.
- **2026-09-01:** Approved AP-07 through AP-09. Eight player buses, three
  dynamic-range presets, headroom/ducking/profile behavior, phone/mono/spatial
  device rules, and complete caption/haptic alternative metadata are resolved.
  Progress is 9 of 12 questions.
- **2026-09-01:** Approved AP-10 through AP-12. Versioned deterministic cue
  definitions/lifecycles, asset/export/stream/budget/degradation gates, semantic
  outputs/privacy, and objective plus human verification are resolved. All
  twelve answers were reconciled into canonical `AUDIO_PRESENTATION.md`.
