# Bands Battle Art Direction

**Status:** Canonical visual direction  
**Applies to:** characters, bosses, instruments, environments, props, VFX, UI, marketing art, concept art, and generated-image prompts

## 1. Purpose and ownership

This document defines how Bands Battle should look and feel across every visual surface. It is the creative source of truth for visual decisions.

Related documents have narrower responsibilities:

- [`GAME_DESIGN.md`](GAME_DESIGN.md) defines the product fantasy, mechanics, and world scenarios.
- [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md) assigns design ownership and dependencies for presentation systems without overriding this document's aesthetic rules.
- [`roblox/web/DESIGN.md`](roblox/web/DESIGN.md) defines implementation-ready UI tokens, components, interaction states, accessibility, and Arena production constraints.
- [`openspec/changes/add-arena-gameplay-mode/design.md`](openspec/changes/add-arena-gameplay-mode/design.md) defines the scoped technical and production contract for the Arena vertical slice.
- [`prompts/nano_banana_character_image.md`](prompts/nano_banana_character_image.md) applies this direction to isolated character-reference generation.
- The Blender prompts in [`prompts/`](prompts/) define asset-production steps and deliverables.

When guidance conflicts, use this document for aesthetic intent and the most specific production document for technical limits. Any deliberate visual exception should be recorded in the relevant design or asset manifest rather than silently becoming a new style.

## 2. The north star

> A supernatural pop concert at the boundary between heaven and the void, where bold stylised heroes turn musical performance into visible combat magic.

The experience should feel:

- **Glamorous:** confident performers, deliberate color, polished presentation, and stage-scale spectacle.
- **Dangerous:** colossal opponents, unstable supernatural power, monumental ruins, and meaningful impact.
- **Readable:** clear silhouettes, restrained effects, strong value grouping, and gameplay cues that survive on a phone.
- **Adventurous:** wondrous dark fantasy rather than hopeless grimdark or realistic horror.
- **Rhythmic:** shape, light, motion, and effects respond to authored musical moments.

The core visual tension is **calm structure versus explosive performance**. Ancient architecture, dark surfaces, and restrained interface chrome create order. Player actions, boss attacks, and climactic beats break that calm with brief, earned energy.

## 3. Style references

The broad reference family is modern stylised 3D adventure and hero-based game art, including Fortnite, Overwatch, and The Legend of Zelda: Breath of the Wild. These are reference points for shared design principles, not templates to copy.

| Reference | Learn from | Do not copy |
|---|---|---|
| Fortnite | Bold silhouettes, simplified production-friendly forms, clean value groups, distance readability | Character likenesses, costumes, proportions, UI, or franchise motifs |
| Overwatch | Heroic shape language, appealing exaggeration, material separation, expressive pose silhouettes | Specific heroes, armor designs, weapons, logos, or interface treatment |
| Zelda: BOTW | Painterly surfaces, controlled toon influence, harmonious color, monumental natural and ancient forms | Characters, cultures, symbols, creatures, architecture, or exact rendering treatment |

The result must feel native to Bands Battle: dark supernatural K-pop spectacle, ancient thresholds, musical weapons, cyan player energy, white-gold earned power, and violet corruption.

## 4. Visual pillars

### 4.1 Stylised, not toy-like

Use simplified anatomy, broad sculpted planes, clean curves, intentional edges, and controlled exaggeration. Forms should feel designed and substantial. Avoid chibi proportions, plastic toy surfaces, black cartoon outlines, flat 2D cel-animation appearance, and default Roblox-avatar styling.

### 4.2 Dark, not muddy

Darkness creates atmosphere, but it must never hide gameplay or erase form. Separate adjacent shapes by value, temperature, rim light, or material response. Keep the focal subject readable in grayscale and at phone size.

### 4.3 Spectacle must be earned

The resting world is restrained. Strong glow, gold light, particles, camera impulse, and high-contrast flashes belong to successful performance, boss commitment, phase changes, and victory. Constant spectacle weakens the rhythm.

### 4.4 Silhouette before surface

A character, instrument, prop, or attack must communicate its role before texture and VFX are applied. Major masses and negative spaces matter more than small decoration. If an asset fails as a solid-color thumbnail, more detail will not fix it.

### 4.5 Music becomes physical

Performance energy travels through instruments, bodies, the stage, and the boss. Beat pulses, the signature light line, stage responses, and attack timing should make sound feel physically connected to the world.

## 5. World identity

Bands Battle exists where a cold, ordered heaven is being eroded by a living violet void. Every arena is also a performance venue. Every battle should contain both ancient-world weight and modern concert intention.

Recurring world ingredients:

- ruined temples, monumental arches, broken circles, standing stones, and ritual thresholds;
- craggy peaks, ice fields, swamps, cloud seas, and other large natural silhouettes;
- black-star portals, fractured halos, void fissures, and corrupted geometry;
- deliberate stage composition, spotlights, performance axes, and audience-facing hero moments;
- instruments and sound transformed into weapons, wards, projectiles, and environmental pulses.

Avoid generic medieval clutter, generic neon cyberpunk, realistic military styling, gore, demonic occult symbols borrowed from real traditions, and decoration that does not support the music-versus-monster fantasy.

## 6. Shape language

| Visual family | Primary shapes | Character |
|---|---|---|
| Player and performance energy | Upward wedges, diamonds, clean arcs, radiating lines | Intentional, agile, aspirational |
| Heaven and earned power | Vertical pillars, circles, halos, symmetrical rays | Ordered, rare, resolving |
| Void and bosses | Broken rings, hooks, split masses, irregular spikes, controlled asymmetry | Unstable, invasive, threatening |
| Neutral world and ruins | Broad blocks, worn curves, heavy arches, stepped platforms | Ancient, grounded, monumental |
| Gameplay cues | Simple circles, diamonds, squares, paths, and target fields | Immediate, semantic, unmistakable |

Use contrast between families to tell the story. Player shapes should cut cleanly through the environment. Boss shapes may be irregular, but their major pose and attack silhouette must remain legible. Small spikes, straps, fragments, and floating pieces are accents, not the foundation of a design.

## 7. Characters

### Player performers

Player characters combine pop-star confidence with supernatural battle readiness.

- Use heroic, appealing exaggeration with a stable center of mass and readable hands, head, instrument, and action line.
- Build costumes from a few large layers: primary body shape, outer silhouette layer, performance accent, and one memorable signature feature.
- Blend stagewear and fantasy protection rather than producing a conventional medieval adventurer or modern street outfit.
- Keep faces expressive and stylised, with clear brow, eye, mouth, and hair masses. Avoid photoreal skin, anime facial shorthand, doll-like smoothness, and miniature facial decoration.
- Give each performer one dominant silhouette idea and one supporting motif. Do not distribute equal visual importance across every garment and accessory.
- Use cyan as the default player-energy accent. White-gold appears only for earned power, full hype, or victory.

### Bosses

Bosses should feel singular, huge, and stage-worthy rather than merely ugly.

- Establish one unmistakable primary mass and one signature feature visible at phone scale.
- Favor powerful proportion changes and controlled asymmetry over dense surface noise.
- Design every major attack to read from pose, direction, and target geometry before particles or color are added.
- Keep at least two attack silhouettes mechanically distinct: for example, a wide lateral preparation versus a compact centered charge.
- Use violet corruption as an identifying energy family, supported by shape and motion so hue is never the only signal.
- Avoid gore, realistic body horror, indistinct smoke anatomy, and a uniformly spiky silhouette.

### Instruments and performance props

Instruments are functional musical objects transformed into battle artifacts.

- Preserve the recognizable instrument category and playable grip before adding fantasy geometry.
- Use one strong outer silhouette, one focal energy channel, and a restrained set of secondary details.
- Repeat the owning character's shape motif and material language.
- Keep strings, sticks, cables, stands, and thin appendages thick and separated enough for mobile rendering and 3D reconstruction.
- Do not make every instrument black with unrestricted neon trim. Accent color communicates ownership or gameplay meaning.

## 8. Environments

Environments should frame performance and combat, not compete with them.

### Composition

- Start with three large value zones: atmosphere, monumental structure, and playable stage.
- Preserve a clean silhouette area for the boss and a darker, quieter lower field for controls or performance cues.
- Use a strong central or near-central performance axis. Break symmetry selectively with damage, corruption, and secondary props.
- Keep foreground detail sparse around gameplay targets and character feet.
- Scale architecture beyond realism when it strengthens awe and clarifies the arena's axis.

### Construction language

- Combine ancient stone, worn metal, carved stage geometry, and supernatural energy.
- Use large modular architectural pieces with broad bevels and readable damage.
- Let corruption interrupt existing structure rather than covering everything with unrelated tentacles or particles.
- Reserve small rubble, cracks, foliage, and decals for low-attention zones.

The environment may be more atmospheric and detailed than a character asset, but it should share the same simplified value grouping and painterly material control. Photoreal source imagery must be restyled before it defines final production assets.

## 9. Materials and surface treatment

Use stylised physically based materials with a softly hand-painted finish.

- Favor broad color gradients, controlled roughness, restrained highlights, gentle ambient shading, and selective edge emphasis.
- Give each major material a distinct response: stone is broad and chalky, metal has compact highlights, cloth has soft value rolloff, and energy is emissive with a readable solid core.
- Use texture to reinforce form, not replace it.
- Keep important transitions visible under neutral lighting, without relying on bloom.
- Use wear and damage selectively at large scale. One meaningful fracture is stronger than uniform scratches.

Avoid photoreal pores, fabric micro-weave, noisy procedural texture, uniform grime, mirror-like armor, excessive normal-map chatter, and emissive trim around every edge.

## 10. Color system

The world is dark-theme first. Color is semantic and controlled rather than decorative.

| Role | Color | Meaning |
|---|---:|---|
| Void surface | `#05070d` | Deepest environmental and UI ground |
| Stage surface | `#0a1020` | Playable dark-blue structure |
| Strong panel | `#10182d` | Raised interface and structural separation |
| Primary text/light | `#f7f9ff` | Maximum clarity and neutral light |
| Secondary neutral | `#b7c2d9` | Fill light, supporting text, cool stone influence |
| Player energy | `#7ce8ff` | Player action, focus, successful contact |
| Heaven energy | `#ffe6a3` | Full hype, resolution, victory, rare sacred light |
| Void energy | `#a15cff` | Boss power, corruption, supernatural danger |
| Danger | `#ff5470` | Damage, failure, urgent error only |

Use roughly 70 to 80 percent dark and neutral foundation, 15 to 25 percent local material color, and no more than 5 to 10 percent emissive accent in a resting scene. This is a composition guide, not a pixel-counting rule.

Rules:

- Cyan is player-owned. Violet is boss- and corruption-owned. Gold is earned.
- Red is a danger signal, not a default player identity color.
- Adjacent forms require value separation even when their hues differ.
- A gameplay state must never rely on color alone; reinforce it with shape, label, position, motion, or pose.
- Song-specific palettes may shift secondary materials and atmosphere while preserving the semantic energy colors.

## 11. Lighting

Lighting should stage the battle like a supernatural concert while preserving sculptural readability.

- Start with a cool neutral fill that keeps silhouettes and material groups visible.
- Use cyan player light, violet boss/corruption light, and rare white-gold climax accents.
- Keep faces, hands, instruments, and attack poses readable without bloom.
- Use rim light to separate important forms, not to outline every object equally.
- Allow deep backgrounds, but avoid crushed blacks on interactive subjects.
- Keep atmospheric color out of semantic cues when it would blur ownership.

Concept and asset-review renders should include at least one neutral-light view. Dramatic lighting can prove mood, but neutral lighting proves the model and materials actually work.

## 12. VFX and motion

Every VFX element needs a gameplay or musical purpose.

### Signature effects

- **Light line:** a narrow white-gold or cyan strike connecting successful player input through the performance space to the boss.
- **Beat pulse:** restrained brightness or material response in instruments, anchors, and selected world elements.
- **Void fracture:** broken-ring or crystalline deformation that introduces boss power without becoming generic purple fog.
- **Ward:** a solid, readable cyan protective form with distinct intact, cracked, and broken states.

### Rules

- Communicate action with pose and geometry first, then light, then particles.
- Use solid cores, restrained trails, short lifetimes, and clear direction.
- Synchronize meaningful motion to authored beats or combat timing.
- Keep idle movement subtle so attack preparation creates a visible change.
- Scale particles and camera motion down before sacrificing target, timing, or silhouette clarity.
- Reduced-motion treatment preserves timing and state information while removing shake, repeated pulse scaling, long trails, and particle storms.

Avoid ambient particle noise everywhere, transparent effects that hide anatomy, continuous maximum bloom, energy with no visible origin or destination, and effects that continue after their semantic event has ended.

## 13. UI relationship

The interface is the calm frame around the expressive 3D world.

- Use restrained, nearly flat dark chrome so characters, notes, and attacks own the depth.
- Prefer clean geometric glyphs and severe contrast over ornamental fantasy frames.
- Use `Oxanium` for display and compact HUD roles and `Atkinson Hyperlegible` for reading and controls.
- Maintain a 48px minimum touch target, 8px control separation, and WCAG 2.2 AA text contrast.
- Reserve glow for focus, active input, successful hits, and urgent semantic state.
- Align UI geometry with the same diamonds, circles, squares, arcs, and broken rings used in the world.

Exact UI tokens, states, spacing, and motion values live in [`roblox/web/DESIGN.md`](roblox/web/DESIGN.md).

## 14. Camera and presentation

### Gameplay

- Portrait mobile is the primary composition.
- Keep the boss, performer, target geometry, and active prompt within one readable visual axis.
- Use a fixed camera for the Arena vertical slice with a 36 to 42 degree vertical field of view.
- Limit camera movement to concise intro, impact, phase, and victory moments.
- Never require the player to find an off-screen threat.

### Character and asset references

- Use frontal, neutral, full-body views for image-to-3D reconstruction and proportion approval.
- Use three-quarter views to review final form and materials after the design is established.
- Use action poses to prove gameplay silhouette only after a neutral pose passes review.
- Keep concept sheets and turnarounds separate from single-view reconstruction prompts.

### Promotional art

- Lead with one hero action, one boss threat, and one strong musical-energy connection.
- Preserve large value groups and avoid a collage of equally loud effects.
- Show the game fantasy truthfully; do not imply unsupported player counts, movement systems, or environments.

## 15. Platform and production constraints

The style must survive both browser and Roblox-oriented production workflows.

- Prioritize phone-readable silhouettes and major material groups over high-frequency detail.
- Use simplified, riggable geometry and keep appendages thick enough to animate and reconstruct reliably.
- Keep limbs, props, and major appendages visually separated in neutral poses.
- Deliver runtime 3D assets as optimized GLB/glTF 2.0 unless a scoped production contract says otherwise.
- Default to textures no larger than 1024px without a measured visual need.
- Use `BLENDER_EEVEE` for Blender 5.2 Eevee automation.
- Treat generated concept art as direction, not production truth. Runtime meshes, textures, rigs, and effects require authored cleanup and recorded provenance.
- Do not imitate Roblox avatar proportions merely because Roblox is a target platform.

Current Arena transfer budgets and export requirements live in [`roblox/web/DESIGN.md`](roblox/web/DESIGN.md#9-arena-v2-production-contract).

## 16. Existing project references

These assets are useful references with deliberately limited authority. None overrides the principles above.

| Asset | Keep | Change or avoid |
|---|---|---|
| [`roblox/web/public/assets/heavens-edge-stage.png`](roblox/web/public/assets/heavens-edge-stage.png) | Monumental threshold, boss silhouette zone, strong central axis, dark lower play space, heaven-versus-void atmosphere | Photoreal surface density, crushed foreground values, and detail levels unsuitable for the final stylised 3D world |
| [`cool-dangerous-electric-guitar-roblox-game-asset.png`](cool-dangerous-electric-guitar-roblox-game-asset.png) | Instantly readable instrument silhouette, strong focal channel, bold large forms | Red as a default player accent, excessive small edge treatments, and realism that is not integrated with a performer |
| [`roblox/web/public/assets/three-head-ghost-preview.png`](roblox/web/public/assets/three-head-ghost-preview.png) | Supernatural multi-head premise and broad symmetry | Legacy/reference only; surface, silhouette separation, value grouping, and material definition do not set the visual quality bar |
| [`roblox/web/public/assets/arena/models/quaternius-demon.glb`](roblox/web/public/assets/arena/models/quaternius-demon.glb) | Current readable, animated, license-safe Arena boss and production baseline | Not the visual ceiling; lighting, material treatment, staging, and semantic VFX must carry it toward the Bands Battle identity |

## 17. Approval checklist

An asset or visual feature is ready for integration only when the applicable checks pass.

### Identity

- Does it support supernatural musical combat rather than generic fantasy?
- Does it express player, heaven, void, or neutral-world ownership clearly?
- Is it original rather than a recognizable copy of a reference franchise?

### Readability

- Does the silhouette read as a small solid-color thumbnail?
- Are the primary, secondary, and tertiary forms clearly ranked?
- Does it remain understandable in grayscale and without VFX?
- Are gameplay states distinguished by more than hue?

### Craft

- Do sculpted forms carry the design before texture?
- Are materials distinct without micro-detail or bloom?
- Are thin parts, overlaps, and rigging risks controlled?
- Does a neutral-light render still look intentional?

### Integration

- Does it preserve the camera's boss, performer, prompt, and control hierarchy?
- Does it meet the relevant triangle, texture, transfer, and animation budgets?
- Does it remain readable at the target phone viewport?
- Are source, license, generation, and export provenance recorded?

### Motion and effects

- Does the pose communicate the action before particles play?
- Is the timing authored and the effect lifetime bounded?
- Does reduced motion retain all necessary gameplay information?
- Does the scene return to visual calm after the event?

## 18. Quick do and do not

| Do | Do not |
|---|---|
| Use bold stylised 3D silhouettes | Chase photorealism or default Roblox-avatar styling |
| Keep darkness readable | Hide geometry in black, fog, or bloom |
| Build with large cohesive forms | Use detail density as a substitute for design |
| Give energy a semantic owner | Apply cyan, violet, gold, and red as arbitrary decoration |
| Make pose and geometry explain gameplay | Depend on particles or color to explain an attack |
| Blend ancient ruins with concert staging | Build generic medieval, cyberpunk, or sci-fi environments |
| Use painterly PBR materials | Use flat 2D shading or noisy realistic micro-texture |
| Earn spectacle on beats and outcomes | Keep every surface glowing and every moment explosive |
| Design for phone readability first | Approve assets only from close-up desktop renders |
