# Roblox Studio MCP patterns

Read this reference before changing a live place through Roblox Studio MCP.

## Connection and edit discipline

1. Call `list_roblox_studios` before mutations.
2. Call `set_active_studio` when multiple places are connected or the intended place is not active.
3. Call `get_studio_state` before selecting an Edit, Client, or Server DataModel.
4. Inspect the target hierarchy and relevant properties before running Luau.
5. Prefer one idempotent Luau operation over many manual property mutations.
6. Move replaced scenery into a named `ServerStorage` archive rather than deleting it.

Studio MCP edits modify the open DataModel but may not save the `.rbxl` file. Return to Edit mode and tell the user to save when no explicit save tool is available.

## Generated mesh templates

Mesh generation inserts a source Model into `Workspace`. Preserve that Model because cloning retains protected mesh and texture asset properties.

```lua
local source = assert(workspace:FindFirstChild("Generated Ridge"))
local library = game:GetService("ServerStorage"):FindFirstChild("BackdropMeshLibrary")
if not library then
    library = Instance.new("Folder")
    library.Name = "BackdropMeshLibrary"
    library.Parent = game:GetService("ServerStorage")
end

local clone = source:Clone()
clone.Name = "RearRidge01"
clone.Parent = workspace.Environment.RearTransition

local part = assert(clone:FindFirstChildWhichIsA("MeshPart", true))
part.Size = Vector3.new(90, 24, 30)
part.CFrame = CFrame.new(0, part.Size.Y / 2, 155)
part.Anchored = true
part.CanCollide = false
part.CanTouch = false

source.Parent = library
```

Do not depend on assigning `MeshPart.MeshId` from Luau. Studio can reject that write with a capability error even when other properties remain writable.

## Structural assertion

Adapt this check to the environment's names and expected counts:

```lua
local environment = assert(workspace:FindFirstChild("Environment"))
local folders = {
    assert(environment:FindFirstChild("Foreground")),
    assert(environment:FindFirstChild("Midground")),
    assert(environment:FindFirstChild("RearTransition")),
}

local meshCount = 0
local meshIds = {}
for _, folder in folders do
    for _, item in folder:GetDescendants() do
        if item:IsA("BasePart") then
            assert(item.Anchored, item:GetFullName() .. " is not anchored")
            assert(not item.CanCollide, item:GetFullName() .. " blocks movement")
            assert(not item.CanTouch, item:GetFullName() .. " receives touches")
        end
        if item:IsA("MeshPart") then
            meshCount += 1
            meshIds[item.MeshId] = true
        end
    end
end

local variantCount = 0
for _ in meshIds do
    variantCount += 1
end

assert(meshCount > 0, "No layered scenery meshes found")
assert(variantCount >= 3, "Use at least three distinct mesh families")
return { meshes = meshCount, variants = variantCount }
```

Treat mesh-ID diversity as a guardrail, not proof of good composition. Different meshes can still form an obvious cadence, and one mesh can sometimes support secondary variation through carefully authored attachments. Always inspect the rendered result.

## Play validation

- Start Play mode and wait for `LocalPlayer.Character` and `HumanoidRootPart`.
- Use `character_navigation` for at least three meaningful routes, including a movement extreme and a return to spawn.
- Re-run collision assertions against the Client DataModel.
- Capture one frame at spawn and one after translating the player; rotate as needed to expose the transition and cubemap seam.
- Read the Studio console after movement and capture.
- Stop Play even when a capture or navigation tool fails.

If capture repeatedly hangs, terminate the pending call, restart the Play session once, and retry. Do not keep polling a stalled capture. Report the evidence limitation and let the user inspect the live viewport.
