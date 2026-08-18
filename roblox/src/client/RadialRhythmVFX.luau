--!strict

local Lighting = game:GetService("Lighting")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local TweenService = game:GetService("TweenService")

local OrangeImpact = require(ReplicatedStorage.BandsBattle.OrangeImpact)
local RadialRhythmUI = require(script.Parent:WaitForChild("RadialRhythmUI"))

type Lane = "Z" | "X" | "C"
type Grade = "Perfect" | "Great" | "Good" | "Miss"
type View = RadialRhythmUI.View

local RadialRhythmVFX = {}

local bloom = Instance.new("BloomEffect")
bloom.Name = "RhythmHitBloom"
bloom.Intensity = 0.35
bloom.Size = 32
bloom.Threshold = 0.82
bloom.Parent = Lighting

local colorFlash = Instance.new("ColorCorrectionEffect")
colorFlash.Name = "RhythmHitColor"
colorFlash.Brightness = 0
colorFlash.Contrast = 0.04
colorFlash.Saturation = -0.05
colorFlash.Parent = Lighting

local function ringBurst(view: View, lane: Lane, color: Color3)
	local target = view.laneTargets[lane]
	local burst = Instance.new("Frame")
	burst.Name = "HitRing"
	burst.AnchorPoint = Vector2.new(0.5, 0.5)
	burst.Position = target.Position
	burst.Size = UDim2.fromOffset(32, 32)
	burst.BackgroundTransparency = 1
	burst.ZIndex = 18
	burst.Parent = view.staff
	local corner = Instance.new("UICorner")
	corner.CornerRadius = UDim.new(1, 0)
	corner.Parent = burst
	local stroke = Instance.new("UIStroke")
	stroke.Color = color
	stroke.Thickness = 3
	stroke.Parent = burst

	TweenService:Create(
		burst,
		TweenInfo.new(0.32, Enum.EasingStyle.Quint, Enum.EasingDirection.Out),
		{ Size = UDim2.fromOffset(78, 78), Rotation = 45 }
	):Play()
	local fade = TweenService:Create(
		stroke,
		TweenInfo.new(0.32, Enum.EasingStyle.Quad, Enum.EasingDirection.Out),
		{ Transparency = 1, Thickness = 1 }
	)
	fade:Play()
	fade.Completed:Once(function()
		burst:Destroy()
	end)

	for index = 1, 8 do
		local angle = ((index - 1) / 8) * math.pi * 2
		local shard = Instance.new("Frame")
		shard.Name = "EnergyShard"
		shard.AnchorPoint = Vector2.new(0.5, 0.5)
		shard.Position = target.Position
		shard.Size = UDim2.fromOffset(3, 9)
		shard.BackgroundColor3 = color:Lerp(Color3.new(1, 1, 1), 0.45)
		shard.Rotation = math.deg(angle) + 90
		shard.ZIndex = 17
		shard.Parent = view.staff
		local destination = UDim2.new(
			target.Position.X.Scale,
			math.cos(angle) * 34,
			target.Position.Y.Scale,
			math.sin(angle) * 34
		)
		local shardTween = TweenService:Create(
			shard,
			TweenInfo.new(0.36, Enum.EasingStyle.Quart, Enum.EasingDirection.Out),
			{ Position = destination, Size = UDim2.fromOffset(2, 5), BackgroundTransparency = 1 }
		)
		shardTween:Play()
		shardTween.Completed:Once(function()
			shard:Destroy()
		end)
	end
end

local function pulseTarget(target: Frame, scaleAmount: number)
	local scale = target:FindFirstChild("TargetScale") :: UIScale
	TweenService:Create(
		scale,
		TweenInfo.new(0.09, Enum.EasingStyle.Back, Enum.EasingDirection.Out),
		{ Scale = scaleAmount }
	):Play()
	task.delay(0.09, function()
		TweenService:Create(
			scale,
			TweenInfo.new(0.18, Enum.EasingStyle.Quint, Enum.EasingDirection.Out),
			{ Scale = 1 }
		):Play()
	end)
end

function RadialRhythmVFX.hit(view: View, lane: Lane, grade: Grade)
	local color = RadialRhythmUI.color(lane)
	ringBurst(view, lane, color)
	pulseTarget(view.laneTargets[lane], if grade == "Perfect" then 1.24 else 1.16)

	TweenService:Create(
		view.staffScale,
		TweenInfo.new(0.08, Enum.EasingStyle.Back, Enum.EasingDirection.Out),
		{ Scale = if grade == "Perfect" then 1.045 else 1.025 }
	):Play()
	task.delay(0.08, function()
		TweenService:Create(
			view.staffScale,
			TweenInfo.new(0.2, Enum.EasingStyle.Quint, Enum.EasingDirection.Out),
			{ Scale = 1 }
		):Play()
	end)

	bloom.Intensity = if grade == "Perfect" then 2.2 else 1.35
	colorFlash.TintColor = color:Lerp(Color3.new(1, 1, 1), 0.7)
	colorFlash.Brightness = if grade == "Perfect" then 0.12 else 0.06
	TweenService:Create(bloom, TweenInfo.new(0.3), { Intensity = 0.35 }):Play()
	TweenService:Create(colorFlash, TweenInfo.new(0.26), { Brightness = 0, TintColor = Color3.new(1, 1, 1) }):Play()

	local camera = workspace.CurrentCamera
	if camera then
		OrangeImpact.emit(camera.CFrame * CFrame.new(0, -2.4, -16), {
			color = color,
			scale = if grade == "Perfect" then 0.42 else 0.3,
		})
	end
end

function RadialRhythmVFX.miss(view: View, lane: Lane)
	local target = view.laneTargets[lane]
	pulseTarget(target, 0.86)
	local stroke = target:FindFirstChildOfClass("UIStroke")
	if stroke then
		local originalColor = stroke.Color
		stroke.Color = Color3.fromRGB(255, 72, 104)
		TweenService:Create(stroke, TweenInfo.new(0.35), { Color = originalColor }):Play()
	end
	colorFlash.TintColor = Color3.fromRGB(255, 160, 178)
	colorFlash.Brightness = -0.08
	TweenService:Create(colorFlash, TweenInfo.new(0.3), { Brightness = 0, TintColor = Color3.new(1, 1, 1) }):Play()
end

return table.freeze(RadialRhythmVFX)
