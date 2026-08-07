You are controlling Blender through the official Blender MCP server.

Your task is to complete Stage 3 of a mobile-friendly Roblox NPC asset pipeline:

1. Inspect the finalized textured model.
2. Create a simple custom skeletal armature.
3. Skin all deforming meshes to the armature.
4. Clean and optimize all vertex weights.
5. Create diagnostic test poses.
6. Verify that the character deforms correctly.
7. Save the completed rigged Blender file and validation report.

This is a custom Roblox NPC rig, not an R15 avatar rig, unless the project inputs explicitly state otherwise.

Do not create final gameplay animations or export the character during this stage.

# Project inputs

Character name:
[CHARACTER_NAME]

Creature type:
[BIPED / QUADRUPED / FLYING / SERPENTINE / MULTI-LIMBED / OTHER]

Stage 2 Blender file:
[ABSOLUTE_PATH_TO_STAGE2_BLEND_FILE]

Output directory:
[ABSOLUTE_OUTPUT_DIRECTORY]

Output Blender file:
[OUTPUT_DIRECTORY]/[CHARACTER_NAME]_stage3_rigged.blend

Rig type:
[DEFAULT: CUSTOM_R1]

Locomotion type:
[WALKING / RUNNING / CRAWLING / FLYING / HOVERING / SLITHERING / STATIONARY]

Important animated features:
[HEAD / JAW / ARMS / LEGS / WINGS / TAIL / TENTACLES / CLAWS / OTHER]

Features that do not need independent animation:
[LIST OR NONE]

Optional weapon or prop:
[OBJECT NAME OR NONE]

Optional rigid armor or shell objects:
[OBJECT NAMES OR NONE]

Expected gameplay scale:
[SMALL / MEDIUM / LARGE / BOSS]

Target bone count:
[DEFAULT: 20–28 INCLUDING ROOT]

Hard bone-count ceiling:
[DEFAULT: 40 INCLUDING ROOT]

# Primary objective

Produce the smallest practical rig that supports the character’s required gameplay movements.

Prioritize:

1. Reliable deformation
2. Simple animation
3. Roblox import compatibility
4. Low bone count
5. Clear bone hierarchy
6. Mobile performance

Do not add bones merely to imitate anatomical realism.

Do not add finger, toe, facial, cloth, fur, or accessory bones unless they visibly contribute to gameplay animation.

# Safety and file-handling rules

1. Open the specified Stage 2 Blender file.
2. Save immediately as the specified Stage 3 output file.
3. Never overwrite the Stage 2 file.
4. Do not delete or alter unrelated objects or collections.
5. Do not access the network.
6. Do not download assets.
7. Do not install add-ons.
8. Do not execute unrelated operating-system commands.
9. Do not use Rigify or another external auto-rigging system.
10. Use one custom armature created with standard Blender bones.
11. Before modifying parenting, vertex groups, or weights, create a backup collection named:
    `[CHARACTER_NAME]_STAGE3_BACKUPS`
12. Duplicate all affected meshes into the backup collection and hide it.
13. Save incremental checkpoints after:

    * Initial audit
    * Armature creation
    * Initial skinning
    * Weight cleanup
    * Deformation testing
    * Final validation

# Scope

Complete:

* Stage 2 asset audit
* Armature creation
* Bone placement
* Bone naming
* Bone hierarchy
* Armature modifier setup
* Initial skinning
* Manual weight cleanup
* Rigid-part weighting
* Influence-count cleanup
* Diagnostic pose creation
* Deformation inspection
* Final rig validation

Do not:

* Change the character’s design
* Change the texture artwork
* Repaint textures
* Rebuild UV maps
* Perform broad remeshing
* Add subdivision
* Add final gameplay animations
* Export FBX or glTF
* Create Roblox Studio scripts
* Create hitboxes or gameplay collision parts
* Create particle effects
* Create sound effects

# Geometry restrictions

Treat the Stage 2 geometry as finalized.

Do not add or remove vertices, edges, or faces during normal rigging work.

You may move existing vertices slightly only when necessary to:

* Separate intersecting limbs
* Improve a neutral joint position
* Prevent immediate deformation collapse
* Correct an obviously misplaced joint center

Do not make changes that significantly alter:

* Silhouette
* Proportions
* UV placement
* Texture alignment
* Triangle count

If topology prevents acceptable deformation, do not hide the problem with excessive bones.

Document the affected area and recommend returning it to Stage 2 for a topology correction.

# Phase 1: Audit the Stage 2 asset

Inspect the entire asset before creating the armature.

Report:

* Final mesh-object names
* Triangle count
* Vertex count
* Material count
* UV-map count
* Texture status
* Current object hierarchy
* Existing armatures, bones, constraints, modifiers, or vertex groups
* Applied transform status
* Character dimensions
* Character facing direction
* Neutral pose quality
* Separate rigid objects
* Separate objects requiring deformation
* Parts likely to require independent bones
* Areas likely to produce difficult deformation

Confirm that:

* Mesh scale is 1, 1, 1
* Mesh rotation is 0, 0, 0
* Character faces negative Y
* Lowest contact point remains at Z = 0
* No existing topology or UV damage is present

Remove obsolete armature data only if it is clearly leftover test data and only after creating the backup.

Provide a concise rigging plan, then continue automatically.

The plan must include:

* Proposed bone hierarchy
* Estimated total bone count
* Bones that deform geometry
* Any non-deforming root bone
* Separate rigid objects and how they will be attached
* Features intentionally left without independent bones
* Expected high-risk deformation areas

# Phase 2: Choose the minimum practical skeleton

Use a direct FK skeleton.

Do not create:

* Rigify rigs
* Complex IK systems
* Drivers
* Custom scripted rig systems
* Bendy Bones
* Stretch-to systems
* Spline IK
* Complex constraint chains
* Hidden mechanism rigs
* Duplicate control and deformation skeletons

The deforming bones must remain directly poseable.

The rig must still work if all Blender-only constraints are removed.

Use one Armature object named:

`[CHARACTER_NAME]_Rig`

Name the armature data:

`[CHARACTER_NAME]_RigData`

Create a top-level bone named:

`Root`

The Root bone must:

* Be located at the world origin
* Be aligned with the character
* Be the parent of the main body or pelvis bone
* Have no vertex weights
* Have Deform disabled
* Exist for hierarchy and whole-character posing
* Be identified in the final report as a zero-influence bone that must be preserved during the later Roblox import stage

Do not create additional root bones unless there is a clear functional requirement.

# Phase 3: Bone naming rules

Use:

* Unique names
* ASCII characters
* No spaces
* Clear anatomical or functional terms
* `_L` and `_R` suffixes for paired bones

Use names such as:

* `Root`
* `Pelvis`
* `Spine_01`
* `Spine_02`
* `Chest`
* `Neck`
* `Head`
* `Jaw`
* `Clavicle_L`
* `UpperArm_L`
* `Forearm_L`
* `Hand_L`
* `Clavicle_R`
* `UpperArm_R`
* `Forearm_R`
* `Hand_R`
* `Thigh_L`
* `Shin_L`
* `Foot_L`
* `Thigh_R`
* `Shin_R`
* `Foot_R`
* `Tail_01`
* `Tail_02`
* `WingUpper_L`
* `WingLower_L`

Do not include Blender-generated names such as:

* `Bone`
* `Bone.001`
* `Bone.002`

Rename all bones intentionally.

# Phase 4: Skeleton templates

Choose the template closest to the creature and adapt it conservatively.

## Standard biped

Preferred hierarchy:

`Root`

* `Pelvis`

  * `Spine_01`

    * `Spine_02`

      * `Chest`

        * `Neck`

          * `Head`

            * Optional `Jaw`
        * `Clavicle_L`

          * `UpperArm_L`

            * `Forearm_L`

              * `Hand_L`
        * `Clavicle_R`

          * `UpperArm_R`

            * `Forearm_R`

              * `Hand_R`
  * `Thigh_L`

    * `Shin_L`

      * `Foot_L`
  * `Thigh_R`

    * `Shin_R`

      * `Foot_R`

Simplify this hierarchy when the character does not need every joint.

For example:

* Omit clavicles for extremely simple creatures.
* Use one or two spine bones rather than three when sufficient.
* Omit hand bones when arms end in rigid weapons or simple stumps.
* Omit feet when the legs end in points or fused shapes.
* Use one jaw bone rather than a facial rig.

## Standard quadruped

Preferred hierarchy:

`Root`

* `Pelvis`

  * `Spine_Rear`

    * `Spine_Mid`

      * `Chest`

        * `Neck_01`

          * Optional `Neck_02`

            * `Head`

              * Optional `Jaw`
        * `FrontUpper_L`

          * `FrontLower_L`

            * `FrontFoot_L`
        * `FrontUpper_R`

          * `FrontLower_R`

            * `FrontFoot_R`
  * `RearUpper_L`

    * `RearLower_L`

      * `RearFoot_L`
  * `RearUpper_R`

    * `RearLower_R`

      * `RearFoot_R`
  * Optional tail chain

Do not force a quadruped into a human bone orientation.

Place joints according to its visible anatomy.

## Flying creature

Use:

* Root
* Central body or pelvis
* Spine or chest
* Neck and head
* One to three deform bones per wing
* Leg bones only when the legs visibly articulate
* Tail bones only when required for silhouette or animation

Do not create one bone for every feather.

Use broad wing sections.

## Serpentine creature

Use:

* Root
* A continuous spine chain
* Head
* Optional jaw
* Optional limbs or fins

Use the fewest spine bones that can create a readable curve.

Typical mobile-friendly starting range:

* Small creature: 5–7 spine bones
* Medium creature: 7–10 spine bones
* Large boss: 10–14 spine bones

Do not create one bone for each mesh loop.

## Tail, tentacle, or appendage chains

Use approximately:

* Short rigid tail: 1–2 bones
* Normal flexible tail: 3–5 bones
* Long focal tail: 5–7 bones
* Small tentacle: 2–4 bones
* Large gameplay-critical tentacle: 4–6 bones

Use fewer bones for minor background motion.

Do not exceed the total bone ceiling.

# Phase 5: Bone placement

Place bones inside the geometry and align them with actual intended joint centers.

Requirements:

* Hips rotate from the visible hip sockets.
* Knees rotate from the narrowest or anatomically appropriate bend point.
* Ankles rotate near the transition into the foot.
* Shoulders rotate from the shoulder socket, not the middle of the upper arm.
* Elbows rotate at the intended elbow bend.
* Wrists rotate at the hand transition.
* Neck begins at the torso-to-neck transition.
* Jaw rotates from a believable hinge point.
* Tail and tentacle bones follow the centerline of the appendage.
* Wing bones follow broad structural sections rather than surface details.

Keep left and right bone placement symmetrical unless the model is intentionally asymmetrical.

Set consistent bone roll.

Mirrored limbs must have matching local-axis orientation.

Use ordinary connected or offset parenting appropriately:

* Use connected bones for continuous chains.
* Use offset parenting where anatomy requires separation.
* Do not connect a clavicle directly through the torso merely to make the hierarchy visually tidy.

Keep the entire armature inside or very close to the character’s body.

# Phase 6: Armature setup

Ensure the armature object has:

* Location appropriate to the world origin
* Rotation 0, 0, 0
* Scale 1, 1, 1

Set the character’s neutral modeling pose as the armature rest pose.

Do not apply the current pose as the rest pose after diagnostic posing has begun.

For each bone:

* Enable Deform only when the bone should influence geometry.
* Disable Deform for Root.
* Disable Deform for any optional organizational or socket bone.
* Do not create empty deform bones.
* Do not create unused bones.

Set a useful viewport display mode such as In Front so bone placement can be inspected.

# Phase 7: Initial skinning

Skin every deforming mesh object to the same armature.

Each deforming mesh must have:

* Exactly one Armature modifier
* The correct armature assigned
* No duplicate Armature modifiers
* Vertex groups matching deform-bone names

Use automatic weights only as an initial approximation.

Do not treat automatic weights as the finished result.

After automatic weighting:

1. Inspect every major joint.
2. Remove influence leaking into nearby unrelated anatomy.
3. Repair collapsed or excessively stretched areas.
4. Normalize all weights.
5. Limit every vertex to no more than four bone influences.
6. Remove negligible or accidental influences.
7. Verify that every deforming vertex has a valid total influence.

Use Auto Normalize during weight painting.

# Phase 8: Weighting rules

The final skinning must follow these rules:

* Maximum four deform-bone influences per vertex
* Total normalized influence of 1.0 per deforming vertex
* No unweighted deforming vertices
* No weights assigned to Root
* No weights assigned to non-deforming bones
* No unintended left-side weights on right-side geometry
* No unintended right-side weights on left-side geometry
* No distant bone influence
* No tiny residual weights causing unexpected movement

Use a cleanup threshold of approximately 0.01 where appropriate, but do not remove a subtle influence when it visibly improves deformation.

After cleanup:

1. Limit total influences to four.
2. Normalize all weights.
3. Check for unweighted vertices.
4. Check for vertices with more than four influences.
5. Check for weight leakage.
6. Check each vertex group for isolated accidental assignments.

# Phase 9: Joint-specific weighting

## Shoulders

Preserve shoulder volume during arm lifting.

Blend influence between:

* Chest or clavicle
* Upper arm
* Nearby torso only where necessary

Avoid pulling chest or neck vertices when the arm rotates.

## Elbows and knees

Create a controlled bend with a clear inside compression area.

Blend primarily between the two bones meeting at the joint.

Avoid:

* Candy-wrapper twisting
* Severe volume collapse
* Sharp spikes
* Vertices crossing through the opposite side
* Excess influence from torso or distant limb bones

## Hips

Blend pelvis and thigh influence carefully.

Test:

* Forward leg lift
* Backward extension
* Side lift
* Squat position

Avoid pulling the abdomen excessively into the thigh.

## Wrists and ankles

Keep the transition simple.

Do not spread wrist or ankle influence far into the limb unless the geometry requires it.

## Neck and head

Head motion should not drag the shoulders substantially.

Neck motion may influence the upper chest slightly when needed.

## Jaw

If a jaw bone exists:

* Weight the lower jaw and lower teeth to the jaw bone.
* Keep the upper skull and upper teeth weighted to the head.
* Prevent cheek, eye, or neck vertices from moving unexpectedly.
* Test mouth opening without visible tearing.

## Tail, wings, and tentacles

Distribute weights progressively along the chain.

Each area should generally be influenced most strongly by the nearest one or two bones.

Avoid distant chain influences.

Avoid sudden hard transitions unless the appendage is intentionally segmented or armored.

# Phase 10: Rigid objects

For rigid armor, eyes, teeth, claws, shell sections, or props, choose the simplest correct method.

Use full 1.0 weighting to one bone when the object should move rigidly with that bone.

Examples:

* Upper teeth to `Head`
* Lower teeth to `Jaw`
* Rigid shoulder armor to `UpperArm_L` or `UpperArm_R`
* Non-rotating eyes to `Head`
* A rigid claw to the nearest hand or foot bone

If an eye must rotate independently, create one eye bone per eye only when the movement will be visible during gameplay.

Do not add eye bones merely because the eye objects are separate.

Do not smooth-weight rigid mechanical or armor objects across multiple bones unless deformation is intentional.

For a weapon or removable prop:

* Keep it as a separate mesh object.
* Do not merge it into the body.
* Do not deform it.
* Align it with the intended hand or attachment location.
* Document the intended attachment bone.
* Do not permanently skin it into the body unless explicitly requested.

# Phase 11: Symmetry

For symmetrical characters:

* Complete and verify one side.
* Mirror weights to the opposite side.
* Use the `_L` and `_R` naming convention.
* Verify that the mirrored weights correspond to the correct bones.
* Inspect the mirrored side manually.

Do not assume mirrored weights are correct without posing both sides.

For asymmetrical characters:

* Do not force symmetrical weights where geometry differs.
* Weight each side independently as needed.

# Phase 12: Diagnostic rig test action

Create one diagnostic Blender action named:

`[CHARACTER_NAME]_RIG_TEST`

This action exists only to test the rig. It is not a final gameplay animation.

Use linear or stepped interpolation so individual test poses are easy to inspect.

Suggested frames:

* Frame 1: Neutral rest pose
* Frame 10: Left limb bend test
* Frame 20: Right limb bend test
* Frame 30: Both arms, wings, or front limbs raised
* Frame 40: Deep elbow, knee, or primary joint bends
* Frame 50: Hip and leg range test or quadruped crouch
* Frame 60: Torso bend and twist
* Frame 70: Neck and head rotation
* Frame 80: Jaw, tail, wing, or tentacle test
* Frame 90: Full-body combat silhouette pose
* Frame 100: Return to neutral pose

Adapt the poses to the creature.

Do not force humanoid poses onto non-humanoid creatures.

Use realistic gameplay ranges rather than impossible anatomical extremes.

Test at least:

* Each primary limb separately
* Paired limb movement
* Main locomotion bend
* Torso or body-chain movement
* Head movement
* Every important appendage
* One combined combat pose

The final saved frame should be frame 1 in the neutral pose.

Keep the diagnostic action in the file, but label it clearly as a test action.

# Phase 13: Deformation inspection

Inspect each diagnostic pose from:

* Front
* Rear
* Left
* Right
* Top
* Bottom
* Three-quarter front
* Three-quarter rear

Look for:

* Collapsing joints
* Severe loss of volume
* Texture stretching
* Mesh intersections
* Detached-looking limbs
* Sharp weight spikes
* Torso movement caused by unrelated limb bones
* Left/right weight leakage
* Moving eyes or teeth that should remain rigid
* Armor deformation that should remain rigid
* Unsupported floating geometry
* Twisting around wrists, ankles, tails, or tentacles
* Visible holes caused by deformation

Make one complete corrective weight-painting pass.

Then repeat all diagnostic poses.

Continue correcting clear weight problems until:

* Normal gameplay poses look acceptable
* No major weight leakage remains
* No vertices are left unweighted
* No vertex has more than four influences
* Rigid objects remain rigid
* The character returns cleanly to the neutral pose

Do not pursue perfect deformation at anatomically extreme poses that will never be used in gameplay.

# Phase 14: Weight validation

Programmatically inspect every skinned vertex.

Report:

* Total weighted vertices
* Unweighted vertices
* Vertices with more than four deform influences
* Vertices with weight totals significantly below 1.0
* Vertices with weight totals significantly above 1.0
* Vertices influenced by Root
* Vertices influenced by non-deforming bones
* Vertex groups with no assigned vertices
* Bones with Deform enabled but no meaningful weights
* Unexpected cross-body influences

Automatically correct straightforward normalization and influence-count issues.

Do not silently delete a bone required by the planned hierarchy.

If a deform bone has no meaningful influence:

1. Decide whether it is actually needed.
2. Remove it if unnecessary.
3. Otherwise assign the correct intended region.
4. Document the decision.

# Phase 15: Rig simplicity audit

Before completion, ask whether every bone is necessary.

Remove bones that:

* Do not noticeably improve silhouette or movement
* Duplicate the function of a nearby bone
* Control details too small to notice on mobile
* Exist only because an automatic system created them
* Have no meaningful skin influence
* Will never be animated

Do not exceed the hard bone-count ceiling.

When close to the ceiling, prioritize bones for:

1. Main locomotion
2. Combat readability
3. Head and attack direction
4. Large silhouette-changing appendages
5. Secondary motion
6. Minor decorative movement

# Phase 16: Final collection organization

Create or preserve these collections:

* `[CHARACTER_NAME]_STAGE2_SOURCE`
* `[CHARACTER_NAME]_STAGE3_BACKUPS`
* `[CHARACTER_NAME]_STAGE3_FINAL`

Place the following in `[CHARACTER_NAME]_STAGE3_FINAL`:

* Final armature
* Final skinned body mesh
* Final skinned secondary meshes
* Final rigid character components
* Any separate weapon or prop intended to remain with the asset

Hide the Stage 2 source and Stage 3 backup collections.

Do not delete them.

# Phase 17: Preview renders

Create neutral, clearly lit previews.

Save:

* `[CHARACTER_NAME]_Rig_Rest_Front.png`
* `[CHARACTER_NAME]_Rig_Rest_Side.png`
* `[CHARACTER_NAME]_Rig_Rest_Rear.png`
* `[CHARACTER_NAME]_Rig_Bones_ThreeQuarter.png`
* `[CHARACTER_NAME]_Rig_LimbTest.png`
* `[CHARACTER_NAME]_Rig_CrouchTest.png`
* `[CHARACTER_NAME]_Rig_AppendageTest.png`
* `[CHARACTER_NAME]_Rig_CombatPose.png`
* `[CHARACTER_NAME]_Rig_WeightIssueCheck.png`

For the bones preview:

* Display the armature through the mesh.
* Show bone names where practical.
* Use a view that makes the hierarchy understandable.

For posed previews:

* Use even lighting.
* Use a neutral background.
* Do not use dramatic camera angles that hide deformation defects.
* Frame the complete character.

# Phase 18: Final validation

Verify:

## Armature

* Exactly one final armature exists
* Armature object is named correctly
* Armature transforms are applied
* Root is the top-level bone
* Root has Deform disabled
* Root has no vertex weights
* Bone names are unique
* Bone names use no spaces
* `_L` and `_R` pairs are consistent
* Bone roll is consistent
* Mirrored limbs have matching axis orientation
* No unnecessary bones remain
* Bone count is below the hard ceiling

## Mesh and modifiers

* Every deforming mesh has exactly one Armature modifier
* Every Armature modifier references the final armature
* No unintended topology changes occurred
* UV maps remain intact
* Materials remain intact
* Texture files remain valid
* Mesh transforms remain applied
* Mesh returns correctly to the rest pose

## Skinning

* Every deforming vertex is weighted
* Every deforming vertex has four or fewer influences
* Every deforming vertex has normalized total weight
* Root influences no geometry
* Non-deforming bones influence no geometry
* No distant weight leakage remains
* Rigid objects remain rigid
* Left/right assignments are correct
* Major gameplay poses deform acceptably

## Actions

* Only the intended diagnostic action was created
* Diagnostic action is named correctly
* Neutral pose exists at frame 1
* Diagnostic poses cover all primary joints
* File is saved at frame 1
* The character is in its neutral pose when the file opens

# Required deliverables

Save:

1. `[CHARACTER_NAME]_stage3_rigged.blend`
2. All requested rig and deformation preview images
3. `[CHARACTER_NAME]_Rig_Hierarchy.txt`
4. `[CHARACTER_NAME]_Stage3_Validation.txt`

The hierarchy file must list every bone using indentation, for example:

Root
Pelvis
Spine_01
Chest
Neck
Head

For every bone, include:

* Parent
* Deform enabled or disabled
* Approximate purpose
* Whether it has weighted vertices
* Number of vertices meaningfully influenced

# Final report

Report:

* Creature type
* Rig type
* Total bone count
* Deforming bone count
* Non-deforming bone count
* Full bone hierarchy
* Mesh-object count
* Armature-modifier count
* Total weighted vertices
* Unweighted vertex count
* Vertices exceeding four influences
* Maximum influences found on any vertex
* Empty vertex groups
* Unused bones
* Root configuration
* Rigid components and their assigned bones
* Weapon or prop attachment recommendation
* Diagnostic poses performed
* Deformation issues corrected
* Remaining deformation limitations
* Areas that may require topology revision
* Whether UVs and textures remained intact
* Whether the asset is ready for Stage 4 animation
* Settings that must be remembered during Roblox import, especially preservation of the zero-influence Root bone

Stop after the rigged and skinned Blender file, diagnostic poses, previews, hierarchy file, and validation report have been created.

Do not begin final animation or export.
