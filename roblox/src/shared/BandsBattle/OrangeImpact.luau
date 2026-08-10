--!strict

local Debris = game:GetService("Debris")
local TweenService = game:GetService("TweenService")

local OrangeImpact = {}

export type Config = {
	color: Color3?,
	scale: number?,
}

local DEFAULT_COLOR = Color3.fromRGB(255, 106, 18)
local RING_TEXTURE = "rbxassetid://1266170131"
local FIRE_TEXTURE = "rbxasset://textures/particles/fire_main.dds"
local SPARK_TEXTURE = "rbxasset://textures/particles/sparkles_main.dds"

local function makePart(
	parent: Instance,
	name: string,
	shape: Enum.PartType,
	color: Color3,
	size: Vector3,
	cframe: CFrame,
	transparency: number
): Part
	local part = Instance.new("Part")
	part.Name = name
	part.Anchored = true
	part.CanCollide = false
	part.CanQuery = false
	part.CanTouch = false
	part.CastShadow = false
	part.Material = Enum.Material.Neon
	part.Shape = shape
	part.Color = color
	part.Size = size
	part.CFrame = cframe
	part.Transparency = transparency
	part.Parent = parent
	return part
end

local function addParticles(anchor: BasePart, color: Color3, scale: number)
	local attachment = Instance.new("Attachment")
	attachment.Name = "ImpactParticles"
	attachment.Parent = anchor

	local hotColor = color:Lerp(Color3.new(1, 1, 1), 0.4)
	local emberColor = Color3.fromRGB(255, 174, 46)

	local shockRing = Instance.new("ParticleEmitter")
	shockRing.Name = "ShockRing"
	shockRing.Texture = RING_TEXTURE
	shockRing.Color = ColorSequence.new(hotColor, color)
	shockRing.LightEmission = 1
	shockRing.Lifetime = NumberRange.new(0.48)
	shockRing.Rate = 0
	shockRing.Rotation = NumberRange.new(0, 360)
	shockRing.Size = NumberSequence.new({
		NumberSequenceKeypoint.new(0, 1.1 * scale),
		NumberSequenceKeypoint.new(0.14, 4.8 * scale),
		NumberSequenceKeypoint.new(1, 10.5 * scale),
	})
	shockRing.Transparency = NumberSequence.new({
		NumberSequenceKeypoint.new(0, 0.05),
		NumberSequenceKeypoint.new(0.18, 0.12),
		NumberSequenceKeypoint.new(1, 1),
	})
	shockRing.Parent = attachment

	local flames = Instance.new("ParticleEmitter")
	flames.Name = "Flames"
	flames.Texture = FIRE_TEXTURE
	flames.Color = ColorSequence.new({
		ColorSequenceKeypoint.new(0, hotColor),
		ColorSequenceKeypoint.new(0.35, emberColor),
		ColorSequenceKeypoint.new(1, color),
	})
	flames.LightEmission = 1
	flames.Lifetime = NumberRange.new(0.28, 0.62)
	flames.Rate = 0
	flames.Speed = NumberRange.new(2.5 * scale, 7 * scale)
	flames.Drag = 3
	flames.Acceleration = Vector3.new(0, 8 * scale, 0)
	flames.SpreadAngle = Vector2.new(360, 360)
	flames.Rotation = NumberRange.new(0, 360)
	flames.RotSpeed = NumberRange.new(-150, 150)
	flames.Size = NumberSequence.new({
		NumberSequenceKeypoint.new(0, 1.35 * scale),
		NumberSequenceKeypoint.new(0.35, 2.15 * scale),
		NumberSequenceKeypoint.new(1, 0),
	})
	flames.Transparency = NumberSequence.new({
		NumberSequenceKeypoint.new(0, 0.02),
		NumberSequenceKeypoint.new(0.55, 0.18),
		NumberSequenceKeypoint.new(1, 1),
	})
	flames.Parent = attachment

	local sparks = Instance.new("ParticleEmitter")
	sparks.Name = "RadialSparks"
	sparks.Texture = SPARK_TEXTURE
	sparks.Color = ColorSequence.new(hotColor, emberColor)
	sparks.LightEmission = 1
	sparks.Lifetime = NumberRange.new(0.32, 0.7)
	sparks.Rate = 0
	sparks.Speed = NumberRange.new(18 * scale, 36 * scale)
	sparks.Drag = 4
	sparks.Acceleration = Vector3.new(0, -18 * scale, 0)
	sparks.SpreadAngle = Vector2.new(360, 360)
	sparks.Orientation = Enum.ParticleOrientation.VelocityParallel
	sparks.Rotation = NumberRange.new(0, 360)
	sparks.Size = NumberSequence.new({
		NumberSequenceKeypoint.new(0, 0.34 * scale),
		NumberSequenceKeypoint.new(0.22, 0.16 * scale),
		NumberSequenceKeypoint.new(1, 0),
	})
	sparks.Transparency = NumberSequence.new(0, 1)
	sparks.Parent = attachment

	local embers = Instance.new("ParticleEmitter")
	embers.Name = "Embers"
	embers.Texture = SPARK_TEXTURE
	embers.Color = ColorSequence.new(emberColor, color)
	embers.LightEmission = 1
	embers.Lifetime = NumberRange.new(0.8, 1.6)
	embers.Rate = 0
	embers.Speed = NumberRange.new(4 * scale, 11 * scale)
	embers.Drag = 3
	embers.Acceleration = Vector3.new(0, 6 * scale, 0)
	embers.SpreadAngle = Vector2.new(360, 360)
	embers.Rotation = NumberRange.new(0, 360)
	embers.RotSpeed = NumberRange.new(-90, 90)
	embers.Size = NumberSequence.new({
		NumberSequenceKeypoint.new(0, 0.22 * scale),
		NumberSequenceKeypoint.new(0.7, 0.12 * scale),
		NumberSequenceKeypoint.new(1, 0),
	})
	embers.Transparency = NumberSequence.new({
		NumberSequenceKeypoint.new(0, 0.05),
		NumberSequenceKeypoint.new(0.75, 0.2),
		NumberSequenceKeypoint.new(1, 1),
	})
	embers.Parent = attachment

	shockRing:Emit(2)
	flames:Emit(24)
	sparks:Emit(34)
	embers:Emit(42)
end

local function addRadialFlare(parent: Instance, origin: CFrame, color: Color3, scale: number)
	local random = Random.new()
	local hotColor = color:Lerp(Color3.new(1, 1, 1), 0.32)

	for index = 1, 18 do
		local angle = ((index - 1) / 18) * math.pi * 2 + random:NextNumber(-0.09, 0.09)
		local length = random:NextNumber(5.5, 11.5) * scale
		local width = random:NextNumber(0.075, 0.18) * scale
		local rayCFrame = origin * CFrame.Angles(0, angle, 0) * CFrame.new(0, 0, -length * 0.38)
		local ray = makePart(
			parent,
			`Ray{index}`,
			Enum.PartType.Block,
			hotColor,
			Vector3.new(width, width, 0.08 * scale),
			rayCFrame,
			0.04
		)

		local grow = TweenService:Create(
			ray,
			TweenInfo.new(0.12, Enum.EasingStyle.Quart, Enum.EasingDirection.Out),
			{ Size = Vector3.new(width, width, length), Transparency = 0.12 }
		)
		local fade = TweenService:Create(
			ray,
			TweenInfo.new(0.5, Enum.EasingStyle.Quad, Enum.EasingDirection.In, 0, false, 0.12),
			{ Size = Vector3.new(0.02, 0.02, length * 1.18), Transparency = 1 }
		)
		grow:Play()
		fade:Play()
	end
end

function OrangeImpact.emit(origin: CFrame, config: Config?)
	local color = if config and config.color then config.color else DEFAULT_COLOR
	local scale = if config and config.scale then config.scale else 1
	local hotColor = color:Lerp(Color3.new(1, 1, 1), 0.42)
	local flashColor = color:Lerp(Color3.new(1, 1, 1), 0.88)

	local effect = Instance.new("Model")
	effect.Name = "OrangeImpactVFX"
	effect.Parent = workspace

	local anchor = makePart(
		effect,
		"Anchor",
		Enum.PartType.Block,
		color,
		Vector3.new(0.1, 0.1, 0.1),
		origin,
		1
	)

	local core = makePart(
		effect,
		"EnergyCore",
		Enum.PartType.Ball,
		hotColor,
		Vector3.new(0.25, 0.25, 0.25) * scale,
		origin,
		0
	)

	local aura = makePart(
		effect,
		"EnergyAura",
		Enum.PartType.Ball,
		color,
		Vector3.new(0.4, 0.4, 0.4) * scale,
		origin,
		0.42
	)

	local flash = makePart(
		effect,
		"ImpactFlash",
		Enum.PartType.Ball,
		flashColor,
		Vector3.new(0.3, 0.3, 0.3) * scale,
		origin,
		0.08
	)

	local light = Instance.new("PointLight")
	light.Name = "ImpactLight"
	light.Color = color
	light.Brightness = 6
	light.Range = 24 * scale
	light.Shadows = true
	light.Parent = anchor

	TweenService:Create(
		core,
		TweenInfo.new(0.18, Enum.EasingStyle.Back, Enum.EasingDirection.Out),
		{ Size = Vector3.new(2.8, 2.8, 2.8) * scale }
	):Play()
	TweenService:Create(
		core,
		TweenInfo.new(0.62, Enum.EasingStyle.Quad, Enum.EasingDirection.In, 0, false, 0.28),
		{ Size = Vector3.new(0.08, 0.08, 0.08), Transparency = 1 }
	):Play()
	TweenService:Create(
		aura,
		TweenInfo.new(0.62, Enum.EasingStyle.Quint, Enum.EasingDirection.Out),
		{ Size = Vector3.new(6.2, 6.2, 6.2) * scale, Transparency = 1 }
	):Play()
	TweenService:Create(
		flash,
		TweenInfo.new(0.2, Enum.EasingStyle.Exponential, Enum.EasingDirection.Out),
		{ Size = Vector3.new(7.5, 7.5, 7.5) * scale, Transparency = 1 }
	):Play()
	TweenService:Create(
		light,
		TweenInfo.new(0.55, Enum.EasingStyle.Quad, Enum.EasingDirection.Out),
		{ Brightness = 0, Range = 5 * scale }
	):Play()

	addParticles(anchor, color, scale)
	addRadialFlare(effect, origin, color, scale)
	Debris:AddItem(effect, 2.5)
end

return OrangeImpact
