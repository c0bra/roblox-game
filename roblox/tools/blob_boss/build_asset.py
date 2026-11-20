#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run through Blender:
#      /Applications/Blender.app/Contents/MacOS/Blender --background --python roblox/tools/blob_boss/build_asset.py
# 3. Open the resulting file:
#      open roblox/assets/blob_boss/output/resonant_blob_warden.blend
# ──────────────────

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import bpy
from mathutils import Vector

ROOT: Final = Path(__file__).resolve().parents[2]
OUTPUT: Final = ROOT / "assets" / "blob_boss" / "output"


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    name: str
    color: tuple[float, float, float, float]
    metallic: float
    roughness: float
    emission: tuple[float, float, float, float] | None = None
    emission_strength: float = 0.0


@dataclass(frozen=True, slots=True)
class PartSpec:
    name: str
    location: tuple[float, float, float]
    scale: tuple[float, float, float]
    material: bpy.types.Material
    subdivisions: int = 2


@dataclass(frozen=True, slots=True)
class TendrilSpec:
    name: str
    points: tuple[tuple[float, float, float], ...]
    radii: tuple[float, ...]
    bevel_depth: float
    material: bpy.types.Material


def make_material(spec: MaterialSpec) -> bpy.types.Material:
    material = bpy.data.materials.new(spec.name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = spec.color
    shader.inputs["Metallic"].default_value = spec.metallic
    shader.inputs["Roughness"].default_value = spec.roughness
    if spec.emission is not None:
        shader.inputs["Emission Color"].default_value = spec.emission
        shader.inputs["Emission Strength"].default_value = spec.emission_strength
    return material


def make_blob_part(spec: PartSpec, collection: bpy.types.Collection) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=spec.subdivisions,
        radius=1.0,
        location=spec.location,
    )
    part = bpy.context.object
    part.name = spec.name
    part.scale = spec.scale
    part.data.materials.append(spec.material)
    for polygon in part.data.polygons:
        polygon.use_smooth = True
    bpy.context.collection.objects.unlink(part)
    collection.objects.link(part)
    return part


def make_tendril(spec: TendrilSpec, collection: bpy.types.Collection) -> bpy.types.Object:
    curve = bpy.data.curves.new(spec.name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = spec.bevel_depth
    curve.bevel_resolution = 2
    curve.resolution_u = 8
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(spec.points) - 1)
    for point, coordinate, radius in zip(
        spline.bezier_points,
        spec.points,
        spec.radii,
        strict=True,
    ):
        point.co = coordinate
        point.radius = radius
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    tendril = bpy.data.objects.new(spec.name, curve)
    collection.objects.link(tendril)
    curve.materials.append(spec.material)
    return tendril


def make_arc(
    name: str,
    angles: tuple[int, int],
    material: bpy.types.Material,
) -> bpy.types.Object:
    points = tuple(
        (
            3.65 * math.cos(math.radians(angle)),
            0.72,
            3.85 + 3.65 * math.sin(math.radians(angle)),
        )
        for angle in range(angles[0], angles[1] + 1, 8)
    )
    radii = tuple(1.0 for _ in points)
    return make_tendril(TendrilSpec(name, points, radii, 0.13, material), boss_collection)


def point_camera(camera: bpy.types.Object, target: tuple[float, float, float]) -> None:
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def add_light(
    name: str,
    location: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = 850.0
    data.color = color
    data.shape = "DISK"
    data.size = 5.0
    light = bpy.data.objects.new(name, data)
    preview_collection.objects.link(light)
    light.location = location
    point_camera(light, (0.0, 0.0, 3.0))


def keyframe_loop(objects: tuple[bpy.types.Object, ...]) -> None:
    scene = bpy.context.scene
    scene.timeline_markers.new("IDLE", frame=1)
    scene.timeline_markers.new("TELEGRAPH", frame=25)
    scene.timeline_markers.new("IMPACT", frame=37)
    scene.timeline_markers.new("LOOP", frame=49)
    for part in objects:
        original_scale = part.scale.copy()
        original_rotation = part.rotation_euler.copy()
        part.keyframe_insert("scale", frame=1)
        part.keyframe_insert("rotation_euler", frame=1)
        part.scale = original_scale * Vector((1.04, 1.02, 1.10))
        part.rotation_euler.z = original_rotation.z + math.radians(4.0)
        part.keyframe_insert("scale", frame=25)
        part.keyframe_insert("rotation_euler", frame=25)
        part.scale = original_scale
        part.rotation_euler = original_rotation
        part.keyframe_insert("scale", frame=49)
        part.keyframe_insert("rotation_euler", frame=49)
    scene.frame_start = 1
    scene.frame_end = 49


def configure_scene() -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world = bpy.data.worlds.new("ResonantVoid")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.004, 0.006, 0.018, 1.0)
    background.inputs["Strength"].default_value = 0.12

    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    preview_collection.objects.link(camera)
    camera_data.lens = 56.0
    scene.camera = camera

    scene.use_nodes = True
    compositor = scene.node_tree
    compositor.nodes.clear()
    render_layers = compositor.nodes.new("CompositorNodeRLayers")
    glare = compositor.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    glare.threshold = 0.7
    glare.size = 7
    composite = compositor.nodes.new("CompositorNodeComposite")
    compositor.links.new(render_layers.outputs["Image"], glare.inputs["Image"])
    compositor.links.new(glare.outputs["Image"], composite.inputs["Image"])
    return camera


def render_view(
    camera: bpy.types.Object,
    filepath: Path,
    location: tuple[float, float, float],
) -> None:
    scene = bpy.context.scene
    camera.location = location
    point_camera(camera, (0.0, 0.0, 3.25))
    scene.render.filepath = str(filepath)
    bpy.ops.render.render(write_still=True)


OUTPUT.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)

boss_collection = bpy.data.collections.new("BOSS__ResonantBlobWarden")
preview_collection = bpy.data.collections.new("PREVIEW__DoNotExport")
bpy.context.scene.collection.children.link(boss_collection)
bpy.context.scene.collection.children.link(preview_collection)

body_material = make_material(MaterialSpec("Body_InkViolet", (0.055, 0.018, 0.16, 1.0), 0.1, 0.3))
accent_material = make_material(MaterialSpec("Body_Amethyst", (0.19, 0.045, 0.42, 1.0), 0.12, 0.26))
void_material = make_material(MaterialSpec("Eye_Void", (0.002, 0.002, 0.008, 1.0), 0.1, 0.12))
cyan_material = make_material(MaterialSpec("Ward_Cyan", (0.01, 0.19, 0.27, 1.0), 0.25, 0.2, (0.01, 0.75, 1.0, 1.0), 7.0))
magenta_material = make_material(MaterialSpec("Core_Magenta", (0.28, 0.01, 0.18, 1.0), 0.2, 0.18, (1.0, 0.01, 0.35, 1.0), 9.0))
stage_material = make_material(MaterialSpec("Stage", (0.008, 0.012, 0.035, 1.0), 0.3, 0.35))

root = bpy.data.objects.new("ROOT__ResonantBlobWarden", None)
boss_collection.objects.link(root)
root["design"] = "anchored modular blob boss"
root["gameplay_read"] = "single eye, split ward halo, six reusable tendrils"

body_parts = (
    make_blob_part(PartSpec("Body_Core", (0.0, 0.0, 3.3), (2.55, 1.65, 2.45), body_material, 3), boss_collection),
    make_blob_part(PartSpec("Body_Base", (0.0, 0.08, 1.55), (3.0, 1.7, 1.15), accent_material, 2), boss_collection),
    make_blob_part(PartSpec("Lobe_Left", (-2.15, 0.1, 2.55), (1.25, 1.18, 1.5), body_material), boss_collection),
    make_blob_part(PartSpec("Lobe_Right", (2.05, 0.05, 2.7), (1.35, 1.15, 1.7), body_material), boss_collection),
)
for part in body_parts:
    part.parent = root

tentacle_specs = (
    TendrilSpec("Tendril_Front_L", ((-0.9, -0.2, 1.45), (-2.0, -0.75, 0.75), (-3.6, -1.0, 0.28), (-4.45, -0.62, 0.75)), (1.3, 1.05, 0.72, 0.22), 0.43, accent_material),
    TendrilSpec("Tendril_Front_R", ((0.85, -0.25, 1.45), (2.1, -0.85, 0.72), (3.7, -1.05, 0.3), (4.55, -0.82, 0.95)), (1.3, 1.0, 0.66, 0.2), 0.43, body_material),
    TendrilSpec("Tendril_Sweep_L", ((-2.0, 0.1, 2.1), (-3.65, 0.0, 2.4), (-4.65, -0.25, 3.7), (-4.0, -0.6, 4.65)), (1.2, 0.95, 0.58, 0.18), 0.38, body_material),
    TendrilSpec("Tendril_Sweep_R", ((2.05, 0.05, 2.15), (3.45, 0.2, 2.55), (4.45, -0.15, 3.55), (4.05, -0.55, 4.45)), (1.2, 0.92, 0.55, 0.18), 0.38, accent_material),
    TendrilSpec("Tendril_Back_L", ((-1.25, 0.5, 1.7), (-2.4, 1.0, 0.9), (-3.5, 1.2, 0.45), (-3.9, 0.8, 0.2)), (1.1, 0.9, 0.5, 0.16), 0.34, body_material),
    TendrilSpec("Tendril_Back_R", ((1.35, 0.45, 1.7), (2.55, 0.9, 0.95), (3.5, 1.1, 0.5), (3.95, 0.7, 0.28)), (1.1, 0.88, 0.48, 0.16), 0.34, body_material),
)
tentacles = tuple(make_tendril(spec, boss_collection) for spec in tentacle_specs)
for tentacle in tentacles:
    tentacle.parent = root

eye_void = make_blob_part(PartSpec("Eye_Void", (0.0, -1.66, 3.85), (1.15, 0.28, 1.38), void_material, 3), boss_collection)
eye_iris = make_blob_part(PartSpec("Eye_Iris", (0.0, -1.95, 3.85), (0.72, 0.16, 0.98), cyan_material, 3), boss_collection)
eye_pupil = make_blob_part(PartSpec("Eye_Pupil", (0.0, -2.11, 3.85), (0.15, 0.09, 0.7), magenta_material, 2), boss_collection)
for eye_part in (eye_void, eye_iris, eye_pupil):
    eye_part.parent = root

bpy.ops.mesh.primitive_torus_add(major_radius=1.35, minor_radius=0.12, major_segments=48, minor_segments=8, location=(0.0, -1.78, 3.85), rotation=(math.pi / 2.0, 0.0, 0.0))
eye_ring = bpy.context.object
eye_ring.name = "Eye_WardRing"
eye_ring.data.materials.append(magenta_material)
bpy.context.collection.objects.unlink(eye_ring)
boss_collection.objects.link(eye_ring)
eye_ring.parent = root

halo_left = make_arc("Ward_Halo_Left", (38, 158), cyan_material)
halo_right = make_arc("Ward_Halo_Right", (202, 322), magenta_material)
halo_left.parent = root
halo_right.parent = root

for index, x_position in enumerate((-0.82, 0.0, 0.82), start=1):
    bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=0.34, radius2=0.04, depth=1.55, location=(x_position, -0.05, 6.05))
    spike = bpy.context.object
    spike.name = f"Ward_Crown_{index}"
    spike.data.materials.append(cyan_material if index != 2 else magenta_material)
    bpy.context.collection.objects.unlink(spike)
    boss_collection.objects.link(spike)
    spike.parent = root

bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=5.8, depth=0.45, location=(0.0, 0.0, -0.25))
stage = bpy.context.object
stage.name = "Preview_Pedestal"
stage.data.materials.append(stage_material)
bpy.context.collection.objects.unlink(stage)
preview_collection.objects.link(stage)

camera = configure_scene()
add_light("Key_Cyan", (-5.5, -7.0, 8.5), (0.1, 0.65, 1.0))
add_light("Fill_Magenta", (5.5, -4.0, 5.5), (1.0, 0.08, 0.35))
add_light("Rim_Violet", (0.0, 4.5, 8.0), (0.48, 0.12, 1.0))

keyframe_loop(body_parts + (eye_ring, halo_left, halo_right) + tentacles)
bpy.context.scene.frame_set(25)
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
render_view(camera, OUTPUT / "resonant_blob_warden_hero.png", (0.0, -15.5, 4.7))
render_view(camera, OUTPUT / "resonant_blob_warden_three_quarter.png", (8.8, -14.0, 6.2))
bpy.context.scene.render.resolution_x = 1280
bpy.context.scene.render.resolution_y = 720
render_view(camera, OUTPUT / "resonant_blob_warden_gameplay.png", (0.0, -20.0, 7.6))

bpy.context.scene.render.filepath = str(OUTPUT / "resonant_blob_warden_hero.png")
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "resonant_blob_warden.blend"))
