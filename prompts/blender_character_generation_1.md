You are controlling Blender through the official Blender MCP server.

Your task is to create the first-stage, low-poly, rig-ready 3D base model of a Roblox game enemy from the supplied reference image.

## Project inputs

Character name: [CHARACTER_NAME]

Source image: [ATTACHED IMAGE OR ABSOLUTE IMAGE FILE PATH]

Creature type: [BIPED / QUADRUPED / SERPENTINE / FLYING / OTHER]

Intended role: Mobile-friendly Roblox NPC enemy or monster

Planned rig type: [CUSTOM RIG / R15 / UNDECIDED]

Target height: [HEIGHT] Blender units

Target triangle count: [DEFAULT: 5,000]

Hard triangle ceiling: [DEFAULT: 7,000]

Important features that must be preserved:
[LIST THE CHARACTER’S MOST IMPORTANT SILHOUETTE FEATURES]

Features that may be simplified:
[SMALL DETAILS, ORNAMENTS, FUR, CHAINS, SPIKES, ETC.]

Symmetry: [SYMMETRICAL / ASYMMETRICAL]

## Scope

Complete only the base-model stage.

Do not rig, weight paint, animate, perform final UV unwrapping, create detailed textures, or export the model during this task.

The result must be a clean, low-poly, properly proportioned mesh that can be passed into a later rigging and texturing workflow.

## Safety and file-handling rules

1. Do not delete or modify unrelated Blender objects, collections, files, or directories.
2. Create a new collection named `CHARACTER_NAME_WORKING`.
3. Keep all generated objects inside that collection.
4. Do not access the network, download external assets, install add-ons, or execute unrelated system commands.
5. Save only to the explicitly provided output path.
6. Save incremental checkpoints before destructive operations.
7. Do not overwrite the source image.
8. Before executing a destructive operation, duplicate the current primary mesh into a hidden backup collection named `CHARACTER_NAME_BACKUPS`.

Output Blender file:
[ABSOLUTE_OUTPUT_PATH]/[CHARACTER_NAME]_stage1_base.blend

## Modeling requirements

### Reference and proportions

Import the source image as a non-rendering reference image.

Analyze the reference before modeling. Identify:

* Overall silhouette
* Head-to-body ratio
* Limb proportions
* Major masses
* Distinctive appendages
* Large color or material regions
* Areas hidden or ambiguous in the source image

Preserve recognizable silhouette and proportions ahead of small surface details.

Because the image does not show every angle, infer hidden geometry conservatively. Use simple functional forms for unseen surfaces. Do not invent elaborate details that are unsupported by the reference.

Unless the reference clearly requires asymmetry, use mirrored construction to keep the model symmetrical and easier to rig.

### Pose

Create the character in a neutral, rig-ready pose.

For bipeds:

* Use a relaxed A-pose.
* Keep the arms away from the torso.
* Keep the legs slightly separated.
* Keep fingers, claws, wings, horns, and other appendages from intersecting nearby geometry.
* Keep elbows and knees slightly bent rather than perfectly straight.
* Keep the head facing directly forward.

For quadrupeds:

* Use a neutral standing pose.
* Separate all four legs clearly.
* Keep knees, elbows, shoulders, and hips readable.
* Keep the spine close to neutral.
* Keep the head facing forward.

For unusual creatures, choose a neutral pose that exposes every joint intended for later animation.

Do not copy a dramatic action pose from the reference image if it would interfere with rigging.

### Scene orientation

* Use Z as the up axis.
* Center the character on the world origin.
* Place the lowest foot, paw, or body contact point at Z = 0.
* Face the character toward negative Y.
* Apply object rotation and scale before completion.
* Final mesh scale must be 1, 1, 1.
* Final mesh rotation must be 0, 0, 0.

### Geometry

Prefer one primary deforming body mesh.

Separate an object only when it has a clear functional reason, such as:

* Eyeballs that may rotate
* A rigid jaw intended to hinge
* A weapon or removable prop
* A rigid shell or armor component
* An appendage intended to animate independently without deformation

Use no more separate objects than necessary.

The model must:

* Remain below the requested hard triangle ceiling
* Be watertight and manifold wherever appropriate
* Have outward-facing normals
* Contain no N-gons in the completed result
* Contain no zero-area faces
* Contain no loose vertices or edges
* Contain no accidental duplicate geometry
* Avoid self-intersections
* Avoid hidden internal geometry unless structurally necessary
* Avoid extremely thin triangles
* Use deformation-friendly edge placement around shoulders, elbows, wrists, hips, knees, ankles, neck, jaw, tail, wings, and other future joints
* Use additional geometry only where it improves silhouette or future deformation
* Keep flat or nearly flat surfaces simple
* Represent tiny details through later textures rather than geometry

Do not use subdivision surfaces to hide inadequate low-poly geometry.

Modifiers may be used during construction, but apply or resolve them before completion unless keeping a non-destructive Mirror modifier is clearly beneficial for the next stage.

### Mobile optimization

Prioritize:

1. Silhouette
2. Large readable shapes
3. Animation-ready joints
4. Texture-friendly surfaces
5. Small decorative details

Remove geometry that will not noticeably change the character’s appearance at normal Roblox gameplay distance.

Simplify:

* Individual strands of hair or fur
* Tiny teeth
* Surface scratches
* Small chains
* Small cloth folds
* Minor spikes
* Invisible rear-facing details
* Layered ornaments that can be represented in a texture

Prefer one material slot on the primary body mesh. Use temporary flat colors only to distinguish major regions. Do not create a complex shader network.

### Future rigging considerations

Even though this stage does not include rigging, construct the mesh so it can later support:

* A root or center-of-mass bone
* Pelvis and torso movement
* Head and neck movement
* Limb bending
* Major appendage movement
* A maximum of four bone influences per vertex during the later skinning stage

Do not merge limbs into the torso in ways that would prevent clean shoulder or hip deformation.

Do not create dense geometry around joints merely to make them smooth. Use a small number of well-positioned deformation loops.

## Working process

Complete the work in these phases:

### Phase 1: Analysis

Inspect the reference image and provide a concise modeling plan containing:

* Intended primary shapes
* Expected triangle allocation
* Symmetry decision
* Planned separate objects
* Ambiguous areas and the assumptions you will use

Then proceed without asking for clarification unless the source image is completely unusable.

### Phase 2: Blockout

Create the character using simple primitives or low-resolution constructed meshes.

Match the silhouette and proportions before adding secondary details.

Create front, side, rear, and three-quarter viewport previews.

### Phase 3: Low-poly refinement

Refine the blockout while remaining within the triangle budget.

Add only geometry that materially improves silhouette, readability, or deformation.

### Phase 4: Cleanup

Apply transforms.

Remove duplicate, hidden, invalid, and unnecessary geometry.

Correct normals.

Resolve non-manifold geometry.

Triangulate only for final triangle-count inspection. Preserve editable quad-oriented topology where practical.

### Phase 5: Verification

Inspect the finished model from:

* Front
* Left side
* Rear
* Three-quarter front
* Three-quarter rear
* Top
* Bottom

Check for intersections, holes, floating pieces, extreme thickness changes, poor proportions, and unsupported invented details.

Compare the model’s silhouette against the source reference.

Make one automatic correction pass for any clearly visible problems.

## Required deliverables

At completion:

1. Save the Blender file to the specified output path.
2. Name the primary body mesh `CHARACTER_NAME_Body`.
3. Name optional objects consistently, such as:

   * `CHARACTER_NAME_Eye_L`
   * `CHARACTER_NAME_Eye_R`
   * `CHARACTER_NAME_Jaw`
   * `CHARACTER_NAME_Weapon`
4. Create a collection named `CHARACTER_NAME_STAGE1_FINAL`.
5. Place the completed model inside it.
6. Create and save six preview renders:

   * Front
   * Side
   * Rear
   * Three-quarter front
   * Three-quarter rear
   * Wireframe three-quarter
7. Provide a final validation report containing:

   * Total triangles
   * Total vertices
   * Number of mesh objects
   * Number of material slots
   * Dimensions
   * Whether transforms are applied
   * Whether non-manifold geometry remains
   * Whether N-gons remain
   * Whether duplicate vertices remain
   * Any assumptions made for unseen geometry
   * Any issues that should be addressed before rigging

Stop after producing the base mesh, previews, saved Blender file, and validation report.
