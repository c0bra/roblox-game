# Arena V2 Vertical Slice Evidence

Production-build route: `?mode=arena&qa=1`.

## Observable browser verification

- Babylon.js 8.56.2 initialized a WebGL2 scene and imported the 620,696-byte Quaternius Demon GLB with its material, skeleton, and all 14 animation groups.
- The controller rejects the model before countdown if any required semantic clip is absent.
- One same-build, post-edit visual packet covers all six required states at all three responsive widths: Classic setup (`cert12-classic-{375,768,1280}.png`), Arena setup (`cert12-setup-{375,768,1280}.png`), live stationary phrase (`cert12-phrase-{375,768,1280}.png`), live Rift Sweep (`cert12-sweep-{375,768,1280}.png`), live Void Burst (`cert12-burst-{375,768,1280}.png`), and terminal result (`cert12-result-{375,768,1280}.png`). All 18 files live under `roblox/web/output/playwright/`; their PNG signatures and exact 375×812, 768×900, and 1280×720 dimensions were verified. The setup packet includes the noninteractive tactical boss/performer/anchor preview; the combat packet includes the narrow-screen anchor readability band, fully contained outer anchor rings, and 48 px pause target.
- Rift Sweep and Void Burst use different boss clips, text/glyph warnings, affected/safe anchor states, and live Babylon ground geometry. The burst capture shows a radial target while the sweep capture shows a lateral path, so the attacks remain distinct without relying on hue alone.
- The fresh phrase captures place the static constellation in its own band between the elevated boss and performer. At 375 px the side rings stay entirely on-screen; at 768 px and 1280 px the DOM anchor labels share the same centered 360 px axis as the 3D rings. The QA harness paused live runtime states and temporarily hid only the pause modal for unobstructed composition evidence.
- The Classic 1280×720 regression check now places the primary action bottom at 620.19 px and helper copy bottom at 651.19 px inside the 720 px viewport; neither control is clipped.
- Unsupported Blackened Crown / vocals / hard selection remains in the URL, disables start, offers an explicit switch to the Heaven's Edge / drums / easy Arena demo, and retains a same-selection Classic recovery URL: `roblox/web/output/playwright/arena-v2-unsupported.png`.
- Browser console output contained only the Babylon WebGL2 initialization log and no warnings or errors.

## Automated and manual gates

- `bun test`: 112 passed, 0 failed, 217 assertions across 22 files.
- `bun run check`: exit 0; Biome reports 13 non-blocking CSS/configuration warnings and TypeScript succeeds.
- `bun run build`: exit 0; Vite reports only the existing Babylon chunk-size advisory.
- Arena scene construction is now inside the recoverable startup boundary, and
  partially constructed Babylon resources dispose before the error is rethrown.
  User-facing fallback copy is fixed rather than exposing raw runtime messages.
- Encounter semantic validation now requires globally unique base/bonus phrase-step
  identifiers and bounds every bonus step to its phrase execution interval. Three
  dedicated regression cases cover duplicate, early, and late bonus steps.
- Real-browser QA covered Classic default/isolation, Arena model/audio/encounter loading, touch and keyboard actions, attack cues, pause/resume, replay, unsupported-selection recovery, Classic re-entry, and terminal victory, failed-Resolve, and ward-defeat outcomes.
- Pause now sets every boss `AnimationGroup.speedRatio` to zero alongside audio/controller frames and restores it on resume. Two canvas-only screenshots captured more than two seconds apart while paused are byte-identical (SHA-256 `1595c126d5ffe0c7180a05cb749716bcf5a39007c08aa313772ebb987f6f7d0a`): `pause-freeze-a.png` and `pause-freeze-b.png`.
- The exact success driver produced 7,000 score, 100% accuracy, a 7× best streak, 100% Resolve broken, 40 ward damage, and 9.4 exposure. No-input variants deterministically produced failed Resolve after surviving and ward defeat when remaining at Midline.
- Runtime transfer for the two selected audio files plus boss GLB is 3,599,120 bytes before JavaScript/CSS, inside the 12 MB Arena allocation. The boss is 6,712 triangles with one 1024×1024 texture.
- A 375×812 Chromium viewport on the named Mac recorded 180 animation frames with 8.32 ms mean, 9.2 ms p95, and 9.3 ms maximum frame spacing. This is development evidence, not a substitute for the required physical iPhone Safari run.

## Blender exception

Blender 5.2 repeatedly crashed while opening the legacy blend and while importing the official glTF. Per the user-reported repeat crash, Blender is no longer used for this slice. Structural inspection, conversion, animation validation, and phone-scale silhouette review use the official glTF, glTF Transform, Babylon, and production browser captures instead.

## Deliberately external gates

Physical-device 30 FPS measurement, three-person and five-person attention tests, and audition/approval of generated ElevenLabs effects require real devices or participants and remain acceptance gates rather than inferred results.
