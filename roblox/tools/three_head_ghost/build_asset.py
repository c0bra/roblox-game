#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "bpy==5.0.1",
#     "numpy>=1.26,<2",
#     "pillow>=11,<13",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run tools/three_head_ghost/build_asset.py
# 3. Or make executable and run:
#      chmod +x tools/three_head_ghost/build_asset.py && ./tools/three_head_ghost/build_asset.py
# ──────────────────

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageFilter, ImageOps

TEXTURE_SIZE: Final = 1024
TARGET_TRIANGLES: Final = 18_000


@dataclass(frozen=True, slots=True)
class AssetPaths:
    root: Path
    raw_mesh: Path
    material_source: Path
    output: Path


@dataclass(frozen=True, slots=True)
class BoneSegment:
    name: str
    head: tuple[float, float, float]
    tail: tuple[float, float, float]


def _create_maps(paths: AssetPaths) -> None:
    paths.output.mkdir(parents=True, exist_ok=True)
    with Image.open(paths.material_source) as source:
        albedo = ImageOps.fit(
            source.convert("RGB"),
            (TEXTURE_SIZE, TEXTURE_SIZE),
            method=Image.Resampling.LANCZOS,
        )
    albedo.save(paths.output / "three_head_ghost_albedo.png")

    gray = np.asarray(albedo.convert("L"), dtype=np.float32) / 255.0
    roughness = np.clip(205.0 + (1.0 - gray) * 42.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(roughness, mode="L").save(
        paths.output / "three_head_ghost_roughness.png"
    )

    blurred = albedo.convert("L").filter(ImageFilter.GaussianBlur(radius=1.1))
    height = np.asarray(blurred, dtype=np.float32) / 255.0
    dy, dx = np.gradient(height)
    normal = np.dstack((-dx * 4.0, -dy * 4.0, np.ones_like(height)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    encoded = np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(
        paths.output / "three_head_ghost_normal.png"
    )
    Image.new("L", (TEXTURE_SIZE, TEXTURE_SIZE), color=0).save(
        paths.output / "three_head_ghost_metalness.png"
    )


def _load_and_prepare_mesh(paths: AssetPaths) -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(
        filepath=str(paths.raw_mesh),
        merge_vertices=True,
        import_shading="NORMALS",
    )
    mesh_object = next(item for item in bpy.context.scene.objects if item.type == "MESH")
    bpy.context.view_layer.objects.active = mesh_object
    mesh_object.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    mesh_object.data.remesh_voxel_size = 0.025
    bpy.ops.object.voxel_remesh()
    decimate = mesh_object.modifiers.new(name="RobloxTriangleBudget", type="DECIMATE")
    decimate.ratio = min(1.0, TARGET_TRIANGLES / (2 * len(mesh_object.data.polygons)))
    decimate.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=decimate.name)
    mesh_object.data.validate(verbose=True, clean_customdata=False)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66.0),
        island_margin=0.02,
        scale_to_bounds=True,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    for polygon in mesh_object.data.polygons:
        polygon.use_smooth = True
    mesh_object.name = "ThreeHeadGhostMesh"
    mesh_object.data.name = "ThreeHeadGhostMesh"
    return mesh_object


def _apply_material(mesh_object: bpy.types.Object, paths: AssetPaths) -> None:
    material = bpy.data.materials.new(name="ThreeHeadGhostPBR")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    for name, filename, colorspace, socket in (
        ("Albedo", "three_head_ghost_albedo.png", "sRGB", "Base Color"),
        ("Roughness", "three_head_ghost_roughness.png", "Non-Color", "Roughness"),
        ("Metalness", "three_head_ghost_metalness.png", "Non-Color", "Metallic"),
    ):
        texture = nodes.new("ShaderNodeTexImage")
        texture.name = name
        texture.image = bpy.data.images.load(str(paths.output / filename))
        texture.image.colorspace_settings.name = colorspace
        material.node_tree.links.new(texture.outputs["Color"], shader.inputs[socket])

    normal_texture = nodes.new("ShaderNodeTexImage")
    normal_texture.name = "Normal"
    normal_texture.image = bpy.data.images.load(
        str(paths.output / "three_head_ghost_normal.png")
    )
    normal_texture.image.colorspace_settings.name = "Non-Color"
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = 0.55
    material.node_tree.links.new(normal_texture.outputs["Color"], normal_map.inputs["Color"])
    material.node_tree.links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    mesh_object.data.materials.append(material)


def _rig_segments(mesh_object: bpy.types.Object) -> tuple[BoneSegment, ...]:
    coordinates = np.asarray([vertex.co[:] for vertex in mesh_object.data.vertices])
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    center_y = float((minimum[1] + maximum[1]) * 0.5)
    width = float(maximum[0] - minimum[0])
    height = float(maximum[2] - minimum[2])
    bottom = float(minimum[2])
    return (
        BoneSegment("Root", (0.0, center_y, bottom), (0.0, center_y, bottom + 0.12 * height)),
        BoneSegment("Spine", (0.0, center_y, bottom + 0.12 * height), (0.0, center_y, bottom + 0.62 * height)),
        BoneSegment("HeadCenter", (0.0, center_y, bottom + 0.60 * height), (0.0, center_y, bottom + 0.96 * height)),
        BoneSegment("HeadLeft", (-0.12 * width, center_y, bottom + 0.60 * height), (-0.25 * width, center_y, bottom + 0.91 * height)),
        BoneSegment("HeadRight", (0.12 * width, center_y, bottom + 0.60 * height), (0.25 * width, center_y, bottom + 0.91 * height)),
        BoneSegment("UpperArm.L", (-0.12 * width, center_y, bottom + 0.57 * height), (-0.34 * width, center_y, bottom + 0.39 * height)),
        BoneSegment("Forearm.L", (-0.34 * width, center_y, bottom + 0.39 * height), (-0.45 * width, center_y, bottom + 0.18 * height)),
        BoneSegment("Hand.L", (-0.45 * width, center_y, bottom + 0.18 * height), (-0.48 * width, center_y, bottom + 0.06 * height)),
        BoneSegment("UpperArm.R", (0.12 * width, center_y, bottom + 0.57 * height), (0.34 * width, center_y, bottom + 0.39 * height)),
        BoneSegment("Forearm.R", (0.34 * width, center_y, bottom + 0.39 * height), (0.45 * width, center_y, bottom + 0.18 * height)),
        BoneSegment("Hand.R", (0.45 * width, center_y, bottom + 0.18 * height), (0.48 * width, center_y, bottom + 0.06 * height)),
    )


def _create_and_bind_rig(
    mesh_object: bpy.types.Object, segments: tuple[BoneSegment, ...]
) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new("ThreeHeadGhostRig")
    armature = bpy.data.objects.new("ThreeHeadGhostRig", armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = {}
    for segment in segments:
        bone = armature_data.edit_bones.new(segment.name)
        bone.head = segment.head
        bone.tail = segment.tail
        edit_bones[segment.name] = bone
    for name in ("Spine",):
        edit_bones[name].parent = edit_bones["Root"]
    for name in ("HeadCenter", "HeadLeft", "HeadRight", "UpperArm.L", "UpperArm.R"):
        edit_bones[name].parent = edit_bones["Spine"]
    edit_bones["Forearm.L"].parent = edit_bones["UpperArm.L"]
    edit_bones["Hand.L"].parent = edit_bones["Forearm.L"]
    edit_bones["Forearm.R"].parent = edit_bones["UpperArm.R"]
    edit_bones["Hand.R"].parent = edit_bones["Forearm.R"]
    bpy.ops.object.mode_set(mode="OBJECT")

    deform_segments = segments[1:]
    vertex_groups = {
        segment.name: mesh_object.vertex_groups.new(name=segment.name)
        for segment in deform_segments
    }
    for vertex in mesh_object.data.vertices:
        point = np.asarray(vertex.co[:], dtype=np.float64)
        distances: list[tuple[float, str]] = []
        for segment in deform_segments:
            head = np.asarray(segment.head)
            tail = np.asarray(segment.tail)
            axis = tail - head
            projection = np.clip(np.dot(point - head, axis) / np.dot(axis, axis), 0.0, 1.0)
            distances.append((float(np.linalg.norm(point - (head + projection * axis))), segment.name))
        nearest = sorted(distances)[:4]
        inverse = np.asarray([1.0 / max(distance, 0.02) ** 2 for distance, _ in nearest])
        weights = inverse / inverse.sum()
        for weight, (_, name) in zip(weights, nearest, strict=True):
            vertex_groups[name].add([vertex.index], float(weight), "REPLACE")

    modifier = mesh_object.modifiers.new(name="ThreeHeadGhostRig", type="ARMATURE")
    modifier.object = armature
    mesh_object.parent = armature
    return armature


def _render_preview(
    paths: AssetPaths, mesh_object: bpy.types.Object, armature: bpy.types.Object
) -> None:
    coordinates = np.asarray([mesh_object.matrix_world @ vertex.co for vertex in mesh_object.data.vertices])
    target = Vector(coordinates.mean(axis=0))
    extent = float(np.ptp(coordinates, axis=0).max())
    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = target + Vector((0.0, -3.2 * extent, 0.15 * extent))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = 58
    bpy.context.scene.camera = camera
    for location, energy, size in (
        ((-2.0, -3.0, 3.0), 95.0, 3.0),
        ((2.5, -1.0, 1.5), 70.0, 2.5),
        ((0.0, 2.0, 3.0), 80.0, 2.0),
    ):
        light_data = bpy.data.lights.new(name="PreviewLight", type="AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new("PreviewLight", light_data)
        bpy.context.collection.objects.link(light)
        light.location = target + Vector(location)
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new(name="PreviewWorld")
    scene.world.color = (0.035, 0.035, 0.045)
    scene.render.filepath = str(paths.output / "three_head_ghost_preview.png")
    bpy.ops.render.render(write_still=True)
    armature.hide_render = False


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = AssetPaths(
        root=root,
        raw_mesh=root / "assets/three_head_ghost/source/three_head_ghost_raw.glb",
        material_source=root
        / "assets/three_head_ghost/source/three_head_ghost_material_generated.png",
        output=root / "assets/three_head_ghost/output",
    )
    _create_maps(paths)
    mesh_object = _load_and_prepare_mesh(paths)
    _apply_material(mesh_object, paths)
    armature = _create_and_bind_rig(mesh_object, _rig_segments(mesh_object))
    bpy.context.view_layer.objects.active = armature
    mesh_object.select_set(True)
    armature.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(paths.output / "three_head_ghost_rigged.glb"),
        export_format="GLB",
        use_selection=True,
        export_skins=True,
        export_animations=False,
        export_apply=True,
        export_yup=True,
        export_influence_nb=4,
        export_all_influences=False,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(paths.output / "three_head_ghost_rigged.blend"))
    _render_preview(paths, mesh_object, armature)
    print(
        f"Built {len(mesh_object.data.vertices)} vertices, "
        f"{len(mesh_object.data.polygons)} triangles, "
        f"{len(armature.data.bones)} bones."
    )


if __name__ == "__main__":
    main()
