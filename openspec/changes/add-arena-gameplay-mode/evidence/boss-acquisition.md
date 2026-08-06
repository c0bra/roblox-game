# Arena Boss Acquisition Decision

## Decision

Use the Quaternius Ultimate Monsters `Demon` under CC0 1.0 for the first Arena vertical slice.

The earlier N-Hance Stylized Demon Boss preference was not acquired because no paid purchase or repository/browser redistribution approval exists. The user authorized selecting the best workable boss, so the license-safe CC0 fallback is the correct acquisition path. Existing repository models and GLBs remain explicitly rejected.

## Evidence

- Official creator page: <https://quaternius.com/packs/ultimatemonsters.html>
- Official download folder: <https://drive.google.com/drive/folders/18m4KpzpEzhC9wl7jzr6dUc0N8Jozr79C>
- Public-domain license: CC0 1.0.
- Preserved source: `roblox/assets/arena_v2/source/quaternius-ultimate-monsters/Demon.gltf`.
- Preserved license text: `roblox/assets/arena_v2/source/quaternius-ultimate-monsters/LICENSE.txt`.
- Public runtime derivative: `roblox/web/public/assets/arena/models/quaternius-demon.glb`.
- Full measurements and checksums: `roblox/assets/arena_v2/manifests/quaternius-demon.json`.

## Tool-path exception

Blender 5.2.0 crashes before loading both the pack's legacy `.blend` and its official glTF on the named Apple M5 Pro machine. Repeated Blender retries are therefore stopped. The official self-contained glTF is inspected structurally and converted reproducibly with `@gltf-transform/cli copy`; Babylon import, material, clip, silhouette, and animation behavior are validated in the real browser showcase.
