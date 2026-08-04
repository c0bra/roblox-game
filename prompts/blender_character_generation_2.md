You are controlling Blender through the official Blender MCP server.

Your task is to complete Stage 2 of a Roblox NPC asset pipeline:

1. Audit and technically finalize the existing low-poly model.
2. Create a clean UV unwrap.
3. Create a simple, stylized, mobile-friendly texture.
4. Validate and save the finished textured model.

Do not rig, skin, animate, or export the model during this stage.

# Project inputs

Character name: [CHARACTER_NAME]

Stage 1 Blender file:
[ABSOLUTE_PATH_TO_STAGE1_BLEND_FILE]

Reference images:

* Front: [PATH OR ATTACHED IMAGE]
* Rear: [OPTIONAL PATH OR NONE]
* Left side: [OPTIONAL PATH OR NONE]
* Right side: [OPTIONAL PATH OR NONE]
* Additional concept art: [OPTIONAL PATH OR NONE]

Output directory:
[ABSOLUTE_OUTPUT_DIRECTORY]

Output Blender file:
[OUTPUT_DIRECTORY]/[CHARACTER_NAME]_stage2_textured.blend

Texture style:
[STYLIZED / HAND-PAINTED / DARK FANTASY / CARTOON / OTHER]

Texture resolution:
[DEFAULT: 1024]

Optional reduced texture resolution:
[DEFAULT: 512]

Important colors and markings:
[DESCRIBE IMPORTANT COLORS, PATTERNS, SCARS, EYES, ARMOR, CLOTH, ETC.]

Important material types:
[SKIN / BONE / CLOTH / METAL / STONE / WOOD / SLIME / OTHER]

Parts intended to glow in Roblox:
[LIST PARTS OR NONE]

Target triangle count:
[STAGE 1 TARGET]

Hard triangle ceiling:
[STAGE 1 HARD CEILING]

# Safety and file-handling rules

1. Open the specified Stage 1 Blender file.
2. Do not modify or delete unrelated objects, collections, files, or directories.
3. Save immediately as the specified Stage 2 output file before making changes.
4. Never overwrite the Stage 1 file.
5. Do not access the network.
6. Do not download assets.
7. Do not install Blender add-ons.
8. Do not run unrelated operating-system commands.
9. Use Blender-native functionality and Blender Python only where necessary.
10. Before any destructive geometry operation, duplicate the affected mesh into a hidden collection named `[CHARACTER_NAME]_STAGE2_BACKUPS`.
11. Save an incremental checkpoint after:

    * Geometry audit
    * UV completion
    * Initial texture
    * Final validation

# Scope

Complete only:

* Mesh audit and minor repair
* Final smoothing and normals
* UV seam creation
* UV unwrapping
* UV packing
* Checker-map testing
* Base-color texture creation
* Optional simple roughness and metalness maps
* Material setup
* Preview renders
* Validation report

Do not:

* Create an armature
* Add bones
* Parent the model to a rig
* Weight paint
* Create animations
* Export FBX or glTF
* Add collision geometry
* Create Roblox Studio scripts
* Make significant changes to the character’s silhouette

# Phase 1: Inspect the Stage 1 model

Locate the final Stage 1 collection and primary body mesh.

Before changing anything, report:

* Mesh object names
* Triangle count
* Vertex count
* Existing UV maps
* Existing material slots
* Existing image textures
* Non-manifold edge count
* N-gon count
* Loose geometry count
* Duplicate vertex problems
* Normal-direction problems
* Unapplied transforms
* Major topology problems that would interfere with UV mapping or rigging

Identify any separate objects that could reasonably be combined.

Do not combine an object when it needs to:

* Move independently
* Rotate independently
* Use a separate Roblox material
* Become a weapon or detachable prop
* Use Roblox Neon or transparency independently

Provide a concise plan, then continue automatically.

# Phase 2: Final geometry audit

The Stage 1 geometry should already be close to final.

Make only repairs necessary for:

* Clean UV unwrapping
* Good shading
* Future deformation
* Removal of invalid geometry
* Reduction of obvious wasted geometry

Preserve:

* Overall silhouette
* Character proportions
* Major facial structure
* Major appendages
* Triangle budget
* Rig-ready neutral pose

Correct, when necessary:

* Duplicate vertices
* Internal faces
* Loose edges or vertices
* Zero-area faces
* Reversed normals
* Non-manifold holes
* Accidental self-intersections
* Extreme long, thin triangles
* Excess geometry invisible during gameplay
* Poorly placed joint loops that clearly prevent later deformation

Do not perform broad automatic remeshing unless the current topology is unusable.

Do not use voxel remesh as a routine cleanup tool.

Do not subdivide the model.

Do not significantly increase the triangle count.

If topology changes after UV creation, re-unwrap all affected geometry.

Apply object transforms before UV unwrapping.

Ensure:

* Scale is 1, 1, 1
* Rotation is 0, 0, 0
* Character remains centered properly
* Lowest contact point remains at Z = 0
* Character continues facing negative Y

# Phase 3: Decide the texture and object structure

The preferred setup is:

* One primary body mesh
* One UV set named `UVMap`
* One material assigned to each mesh object
* One base-color image per mesh object

Whenever practical, use one primary body texture for the whole character.

A separate mesh may have a separate material only when necessary, such as:

* Glowing eyes
* Transparent wings
* A removable weapon
* A rigid metal shell
* A separately animated jaw
* A visually distinct effect surface

Do not use multiple material slots on one mesh as a substitute for a proper texture atlas.

If multiple material slots already exist on the primary body mesh, consolidate them into one material and one texture wherever practical.

# Phase 4: Create UV seams

Create deliberate seams based on the character’s anatomy and areas that will be less visible during normal gameplay.

For a biped, generally place seams along areas such as:

* Inner arms
* Inner legs
* Back of the torso
* Back or underside of the head
* Underside of hands and feet
* Rear or underside of tails
* Underside of wings
* Boundaries between major armor or clothing regions

For quadrupeds, generally place seams along:

* Inner legs
* Belly
* Underside of neck
* Rear of the head
* Underside of tail
* Boundaries between major body regions

Avoid seams:

* Across the center of the face
* Across important markings
* Across highly visible chest areas
* Through large, smooth focal surfaces
* Directly over likely bending areas when another location is available

Use Smart UV Project only for small rigid or geometrically simple objects.

Do not use Smart UV Project as the default unwrap for the primary organic character mesh.

# Phase 5: UV unwrap and packing

Create exactly one UV set named `UVMap` for each final mesh object.

All UV coordinates must remain inside the 0–1 UV space.

Unwrap the model and organize islands logically.

Requirements:

* No unintended overlapping UV islands
* Intentional mirrored overlap is allowed for symmetrical areas
* Do not mirror UVs for areas requiring unique left/right markings
* Maintain reasonably consistent texel density
* Give additional UV space to:

  * Face
  * Head
  * Hands or claws
  * Important chest markings
  * Other major visual focal areas
* Give less UV space to:

  * Soles
  * Hidden undersides
  * Small rear-facing areas
* Straighten simple strip-like islands where practical
* Minimize unnecessary island fragmentation
* Orient similar body regions consistently
* Use sufficient island padding for mipmapping

For a 1024×1024 map, target at least 12–16 pixels of effective padding between major islands.

For a 512×512 map, target at least 6–8 pixels.

Export a UV layout image named:

`[CHARACTER_NAME]_UV_Layout.png`

# Phase 6: Checker-map validation

Create and assign a numbered or colored checker texture.

Inspect the model from:

* Front
* Rear
* Left
* Right
* Top
* Bottom
* Three-quarter front
* Three-quarter rear

Look specifically for:

* Texture stretching
* Abrupt texel-density changes
* Mirrored text or asymmetrical details
* Seams crossing important facial features
* Islands packed too closely
* Distortion around shoulders
* Distortion around hips
* Distortion around knees and elbows
* Distortion around jaw, tail, wings, or other appendages

Correct visible UV problems before creating the final texture.

Save a checker-map preview render named:

`[CHARACTER_NAME]_UV_Checker_Preview.png`

# Phase 7: Create the base-color texture

Create an image named:

`[CHARACTER_NAME]_BaseColor_[RESOLUTION].png`

Use the supplied reference images as visual guidance.

Do not simply project the original concept image across the entire model as the final texture.

Direct projection commonly creates:

* Stretched sides
* Incorrect rear surfaces
* Baked lighting
* Baked shadows
* Misaligned details
* Distorted facial features

Projection may be used selectively as a starting point for visible details, but clean and repaint the result.

Create a simplified, stylized game texture using:

* Large readable color regions
* Clear separation between skin, armor, cloth, bone, fur, or other materials
* Moderate hand-painted shading
* Important markings visible from normal gameplay distance
* Controlled edge highlights where appropriate
* Limited surface noise
* Stronger contrast around the face and important combat features

Do not bake scene lighting into the base-color texture.

Avoid:

* Directional cast shadows
* Strong highlights tied to one light direction
* Photographic noise
* Tiny unreadable details
* Excessive dirt
* Random scratches
* High-frequency skin pores
* Text
* Logos
* Watermarks
* Artifacts copied from the background of the reference image

For surfaces unseen in the references:

* Infer simple, coherent colors
* Continue visible patterns conservatively
* Avoid inventing elaborate artwork
* Prefer broad color zones over detailed speculation

The texture should still look acceptable under flat lighting.

# Phase 8: Mobile texture optimization

Default to a base-color-only texture unless an additional map creates a clear visual benefit.

Do not generate a normal map merely because Roblox supports one.

A normal map is justified only when it noticeably represents important medium-scale detail that would otherwise require substantial geometry, such as:

* Major scales
* Deep carved armor
* Large stone cracks
* Major muscle or tendon forms
* Broad cloth seams

Do not use normal maps for tiny skin pores, scratches, or noise.

If a normal map is justified:

* Create an OpenGL tangent-space normal map
* Name it `[CHARACTER_NAME]_Normal_[RESOLUTION].png`
* Verify that it does not introduce visible seams or inverted shading

Create a roughness map only when the character combines substantially different surface types.

Examples:

* Wet skin and dry cloth
* Polished metal and rough leather
* Slime and bone
* Stone and glass

Name it:

`[CHARACTER_NAME]_Roughness_[RESOLUTION].png`

Create a metalness map only if the model includes genuinely metallic regions.

Name it:

`[CHARACTER_NAME]_Metalness_[RESOLUTION].png`

Do not treat shiny stone, wet skin, or glossy chitin as metal.

# Phase 9: Glow and special surfaces

If parts are intended to glow in Roblox, do not rely on a Blender emission shader being transferred automatically as the final game effect.

Instead:

1. Keep the glowing region as a separate mesh object where practical.
2. Name it descriptively, such as:

   * `[CHARACTER_NAME]_EyesGlow`
   * `[CHARACTER_NAME]_RuneGlow`
3. Give that mesh one simple material.
4. Apply the intended glow color for preview purposes.
5. Include it in the validation report so it can later receive a Neon material or other treatment in Roblox Studio.

Avoid creating many tiny glow meshes.

Combine nearby glow regions when it does not interfere with animation.

# Phase 10: Material setup

Create a simple Blender material named:

`[CHARACTER_NAME]_Body_MAT`

Use a Principled BSDF shader.

Connect the base-color image correctly.

If roughness, metalness, or normal maps exist:

* Connect them correctly
* Set non-color data appropriately
* Use an OpenGL-compatible normal-map setup
* Keep the node graph simple

Do not create procedural shader effects that cannot be exported meaningfully.

Do not depend on Blender-only displacement.

Do not create a complex shader network.

# Phase 11: Reduced-resolution texture

If the optional reduced texture size is specified:

1. Create a high-quality reduced copy of the final base-color texture.
2. Save it as:
   `[CHARACTER_NAME]_BaseColor_[REDUCED_RESOLUTION].png`
3. Inspect it on the model.
4. Confirm that important eyes, facial features, markings, and material boundaries remain readable.
5. Do not replace the primary texture with the reduced version in the Stage 2 file unless explicitly instructed.

# Phase 12: Preview renders

Create neutral preview renders using simple, even lighting.

Do not use dramatic lighting that hides texture defects.

Save:

* `[CHARACTER_NAME]_Textured_Front.png`
* `[CHARACTER_NAME]_Textured_Rear.png`
* `[CHARACTER_NAME]_Textured_Left.png`
* `[CHARACTER_NAME]_Textured_Right.png`
* `[CHARACTER_NAME]_Textured_ThreeQuarter.png`
* `[CHARACTER_NAME]_Textured_Wireframe.png`
* `[CHARACTER_NAME]_Texture_FlatLighting.png`

Use a plain neutral background.

Frame the entire character consistently.

# Phase 13: Final validation

Before completion, verify:

## Geometry

* Triangle count remains below the hard ceiling
* No new non-manifold geometry
* No loose geometry
* No accidental duplicate vertices
* No reversed normals
* No unintended internal faces
* Transforms remain applied

## UVs

* Exactly one UV set per mesh object
* UV set is named `UVMap`
* UVs remain in 0–1 space
* No unintended overlap
* Adequate island padding
* Reasonably consistent texel density
* No major visible stretching
* Face and focal areas receive sufficient texture space

## Textures

* Texture images are saved externally
* Texture images use supported formats
* Base color contains no baked cast shadows
* Important details are readable at gameplay distance
* Reduced texture remains readable, if created
* No unnecessary texture maps were created
* Material nodes reference valid image files

## Materials

* No mesh object uses more than one final material
* Material names are consistent
* Shader network is simple
* Special glow objects are clearly identified

# Required deliverables

Save:

1. `[CHARACTER_NAME]_stage2_textured.blend`
2. `[CHARACTER_NAME]_UV_Layout.png`
3. `[CHARACTER_NAME]_BaseColor_[RESOLUTION].png`
4. Optional reduced base-color texture
5. Optional justified PBR maps
6. Checker-map preview
7. All requested preview renders
8. A validation report

Pack the image textures into the Blender file after confirming that external PNG copies have been saved successfully.

Place the final Stage 2 objects in a collection named:

`[CHARACTER_NAME]_STAGE2_FINAL`

Keep the Stage 1 collection hidden but intact.

# Final report

Report:

* Total triangles
* Total vertices
* Final mesh-object count
* Material count per mesh object
* UV-set count per mesh object
* Texture filenames
* Texture dimensions
* Whether any UV overlap is intentional
* Estimated texture-memory concerns
* Whether normal, roughness, or metalness maps were created
* Why each optional map was or was not created
* Glow objects that require Roblox Studio configuration
* Assumptions made for unseen surfaces
* Problems that remain before rigging
* Whether the asset is ready for Stage 3 rigging

Stop after the textured Blender file, texture files, previews, and validation report have been created.
