--!strict

local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RhythmChart = require(ReplicatedStorage.BandsBattle.RhythmChart)

type Lane = RhythmChart.Lane

local RhythmStaffUI = {}

local STRIKE_X = 0.2
local SPAWN_X = 1.04
local LANE_Y: { [Lane]: number } = {
	Z = 0.22,
	X = 0.5,
	C = 0.78,
}
local COLORS: { [Lane]: Color3 } = {
	Z = Color3.fromRGB(61, 221, 255),
	X = Color3.fromRGB(255, 210, 90),
	C = Color3.fromRGB(211, 104, 255),
}

export type View = {
	screenGui: ScreenGui,
	input: BindableEvent,
	staff: Frame,
	staffScale: UIScale,
	laneTargets: { [Lane]: Frame },
	feedback: TextLabel,
	combo: TextLabel,
	score: TextLabel,
	accuracy: TextLabel,
	status: TextLabel,
	progress: Frame,
	replay: TextButton,
}

local function addCorner(parent: GuiObject, radius: UDim)
	parent.BorderSizePixel = 0
	local corner = Instance.new("UICorner")
	corner.CornerRadius = radius
	corner.Parent = parent
end

local function addStroke(parent: GuiObject, color: Color3, thickness: number, transparency: number): UIStroke
	local stroke = Instance.new("UIStroke")
	stroke.Color = color
	stroke.Thickness = thickness
	stroke.Transparency = transparency
	stroke.Parent = parent
	return stroke
end

local function textLabel(parent: Instance, name: string, text: string, size: UDim2, position: UDim2): TextLabel
	local label = Instance.new("TextLabel")
	label.Name = name
	label.BackgroundTransparency = 1
	label.BorderSizePixel = 0
	label.Size = size
	label.Position = position
	label.Font = Enum.Font.GothamMedium
	label.Text = text
	label.TextColor3 = Color3.fromRGB(232, 246, 255)
	label.TextScaled = true
	label.Parent = parent
	return label
end

local function makeTarget(staff: Frame, lane: Lane): Frame
	local target = Instance.new("Frame")
	target.Name = `{lane}Target`
	target.AnchorPoint = Vector2.new(0.5, 0.5)
	target.Position = UDim2.fromScale(STRIKE_X, LANE_Y[lane])
	target.Size = UDim2.fromOffset(58, 42)
	target.BackgroundColor3 = Color3.fromRGB(7, 18, 32)
	target.BackgroundTransparency = 0.02
	target.ZIndex = 12
	target.Parent = staff
	addCorner(target, UDim.new(0, 10))
	addStroke(target, COLORS[lane], 3, 0.04)

	local scale = Instance.new("UIScale")
	scale.Name = "TargetScale"
	scale.Parent = target

	local key = textLabel(target, "Key", lane, UDim2.fromScale(1, 1), UDim2.fromScale(0, 0))
	key.Font = Enum.Font.GothamBlack
	key.TextColor3 = COLORS[lane]
	return target
end

function RhythmStaffUI.create(playerGui: PlayerGui): View
	local oldGui = playerGui:FindFirstChild("RhythmStaff")
	if oldGui then
		oldGui:Destroy()
	end

	local screenGui = Instance.new("ScreenGui")
	screenGui.Name = "RhythmStaff"
	screenGui.IgnoreGuiInset = true
	screenGui.ResetOnSpawn = false
	screenGui.DisplayOrder = 80
	screenGui.ZIndexBehavior = Enum.ZIndexBehavior.Sibling
	screenGui.Parent = playerGui

	local input = Instance.new("BindableEvent")
	input.Name = "RhythmInput"
	input.Parent = screenGui

	local header = Instance.new("Frame")
	header.Name = "Header"
	header.AnchorPoint = Vector2.new(0.5, 0)
	header.Position = UDim2.fromScale(0.5, 0.025)
	header.Size = UDim2.new(0.88, 0, 0, 62)
	header.BackgroundColor3 = Color3.fromRGB(4, 13, 27)
	header.BackgroundTransparency = 0.14
	header.Parent = screenGui
	addCorner(header, UDim.new(0, 16))
	addStroke(header, Color3.fromRGB(78, 198, 240), 1.5, 0.62)

	local title = textLabel(header, "Track", string.upper(RhythmChart.trackTitle), UDim2.fromScale(0.37, 0.44), UDim2.fromScale(0.035, 0.12))
	title.TextXAlignment = Enum.TextXAlignment.Left
	title.Font = Enum.Font.GothamBlack
	local artist = textLabel(header, "Artist", `{RhythmChart.trackArtist}  ·  CHART SYNC`, UDim2.fromScale(0.4, 0.23), UDim2.fromScale(0.035, 0.61))
	artist.TextXAlignment = Enum.TextXAlignment.Left
	artist.TextColor3 = Color3.fromRGB(129, 175, 205)

	local combo = textLabel(header, "Combo", "0×", UDim2.fromScale(0.12, 0.58), UDim2.fromScale(0.48, 0.1))
	combo.Font = Enum.Font.GothamBlack
	combo.TextColor3 = Color3.fromRGB(255, 225, 139)
	local score = textLabel(header, "Score", "000000", UDim2.fromScale(0.18, 0.44), UDim2.fromScale(0.61, 0.1))
	score.Font = Enum.Font.RobotoMono
	local accuracy = textLabel(header, "Accuracy", "100%", UDim2.fromScale(0.12, 0.3), UDim2.fromScale(0.83, 0.49))
	accuracy.TextColor3 = Color3.fromRGB(124, 232, 255)

	local progressTrack = Instance.new("Frame")
	progressTrack.Name = "ProgressTrack"
	progressTrack.Position = UDim2.fromScale(0.48, 0.79)
	progressTrack.Size = UDim2.fromScale(0.47, 0.055)
	progressTrack.BackgroundColor3 = Color3.fromRGB(48, 70, 94)
	progressTrack.Parent = header
	addCorner(progressTrack, UDim.new(1, 0))
	local progress = Instance.new("Frame")
	progress.Name = "Progress"
	progress.Size = UDim2.fromScale(0, 1)
	progress.BackgroundColor3 = Color3.fromRGB(94, 229, 255)
	progress.Parent = progressTrack
	addCorner(progress, UDim.new(1, 0))

	local staff = Instance.new("Frame")
	staff.Name = "NoteHighway"
	staff.AnchorPoint = Vector2.new(0.5, 1)
	staff.Position = UDim2.fromScale(0.5, 0.91)
	staff.Size = UDim2.new(0.88, 0, 0, 224)
	staff.BackgroundColor3 = Color3.fromRGB(3, 11, 23)
	staff.BackgroundTransparency = 0.08
	staff.ClipsDescendants = true
	staff.Parent = screenGui
	addCorner(staff, UDim.new(0, 20))
	addStroke(staff, Color3.fromRGB(78, 198, 240), 2, 0.36)

	local staffGradient = Instance.new("UIGradient")
	staffGradient.Color = ColorSequence.new({
		ColorSequenceKeypoint.new(0, Color3.fromRGB(10, 26, 46)),
		ColorSequenceKeypoint.new(0.2, Color3.fromRGB(4, 17, 31)),
		ColorSequenceKeypoint.new(1, Color3.fromRGB(12, 8, 31)),
	})
	staffGradient.Rotation = 0
	staffGradient.Parent = staff

	local staffScale = Instance.new("UIScale")
	staffScale.Parent = staff

	local clef = textLabel(staff, "Clef", "♫", UDim2.fromOffset(62, 142), UDim2.fromScale(0.015, 0.18))
	clef.Font = Enum.Font.GothamBlack
	clef.TextColor3 = Color3.fromRGB(88, 185, 224)
	clef.TextTransparency = 0.28
	clef.ZIndex = 3

	for _, lane in RhythmChart.lanes do
		local line = Instance.new("Frame")
		line.Name = `{lane}LaneLine`
		line.AnchorPoint = Vector2.new(0, 0.5)
		line.Position = UDim2.fromScale(0.07, LANE_Y[lane])
		line.Size = UDim2.fromScale(0.93, 0.012)
		line.BackgroundColor3 = COLORS[lane]
		line.BackgroundTransparency = 0.75
		line.ZIndex = 2
		line.Parent = staff
		addCorner(line, UDim.new(1, 0))
	end

	local finishGlow = Instance.new("Frame")
	finishGlow.Name = "FinishGlow"
	finishGlow.AnchorPoint = Vector2.new(0.5, 0.5)
	finishGlow.Position = UDim2.fromScale(STRIKE_X, 0.5)
	finishGlow.Size = UDim2.new(0, 18, 0.88, 0)
	finishGlow.BackgroundColor3 = Color3.fromRGB(78, 226, 255)
	finishGlow.BackgroundTransparency = 0.82
	finishGlow.ZIndex = 5
	finishGlow.Parent = staff
	addCorner(finishGlow, UDim.new(1, 0))

	local finishLine = Instance.new("Frame")
	finishLine.Name = "FinishLine"
	finishLine.AnchorPoint = Vector2.new(0.5, 0.5)
	finishLine.Position = UDim2.fromScale(STRIKE_X, 0.5)
	finishLine.Size = UDim2.new(0, 4, 0.88, 0)
	finishLine.BackgroundColor3 = Color3.fromRGB(221, 252, 255)
	finishLine.ZIndex = 6
	finishLine.Parent = staff
	addCorner(finishLine, UDim.new(1, 0))

	local finishCaption = textLabel(staff, "FinishCaption", "PLAY", UDim2.fromOffset(74, 18), UDim2.new(STRIKE_X, -37, 0, 3))
	finishCaption.Font = Enum.Font.GothamBlack
	finishCaption.TextColor3 = Color3.fromRGB(178, 238, 255)
	finishCaption.ZIndex = 8

	local targets: { [Lane]: Frame } = {} :: { [Lane]: Frame }
	for _, lane in RhythmChart.lanes do
		targets[lane] = makeTarget(staff, lane)
	end

	local feedback = textLabel(screenGui, "Feedback", "READY", UDim2.new(0.4, 0, 0, 52), UDim2.fromScale(0.3, 0.575))
	feedback.Font = Enum.Font.GothamBlack
	feedback.TextColor3 = Color3.fromRGB(226, 250, 255)
	feedback.TextStrokeTransparency = 0.45
	feedback.ZIndex = 20

	local status = textLabel(screenGui, "Status", "Loading chart audio…", UDim2.new(0.74, 0, 0, 28), UDim2.fromScale(0.13, 0.93))
	status.TextColor3 = Color3.fromRGB(216, 239, 255)
	status.Font = Enum.Font.GothamMedium
	status.TextStrokeColor3 = Color3.fromRGB(3, 11, 23)
	status.TextStrokeTransparency = 0.18

	local replay = Instance.new("TextButton")
	replay.Name = "Replay"
	replay.AnchorPoint = Vector2.new(0.5, 0.5)
	replay.Position = UDim2.fromScale(0.5, 0.55)
	replay.Size = UDim2.fromOffset(190, 52)
	replay.BackgroundColor3 = Color3.fromRGB(83, 219, 255)
	replay.Text = "PLAY AGAIN"
	replay.TextColor3 = Color3.fromRGB(2, 14, 24)
	replay.TextSize = 18
	replay.Font = Enum.Font.GothamBlack
	replay.Visible = false
	replay.Parent = screenGui
	addCorner(replay, UDim.new(0, 14))
	addStroke(replay, Color3.fromRGB(220, 252, 255), 2, 0.15)

	return {
		screenGui = screenGui,
		input = input,
		staff = staff,
		staffScale = staffScale,
		laneTargets = targets,
		feedback = feedback,
		combo = combo,
		score = score,
		accuracy = accuracy,
		status = status,
		progress = progress,
		replay = replay,
	}
end

function RhythmStaffUI.createNote(view: View, lane: Lane): Frame
	local note = Instance.new("Frame")
	note.Name = `{lane}Note`
	note.AnchorPoint = Vector2.new(0.5, 0.5)
	note.Position = UDim2.fromScale(SPAWN_X, LANE_Y[lane])
	note.Size = UDim2.fromOffset(38, 30)
	note.BackgroundColor3 = COLORS[lane]
	note.ZIndex = 10
	note.Parent = view.staff
	addCorner(note, UDim.new(0, 9))
	addStroke(note, Color3.fromRGB(235, 253, 255), 2, 0.08)
	local gradient = Instance.new("UIGradient")
	gradient.Color = ColorSequence.new(COLORS[lane]:Lerp(Color3.new(1, 1, 1), 0.58), COLORS[lane])
	gradient.Rotation = 90
	gradient.Parent = note
	return note
end

function RhythmStaffUI.positionNote(note: Frame, lane: Lane, progress: number)
	local travel = math.clamp(progress, 0, 1.22)
	local x = SPAWN_X + (STRIKE_X - SPAWN_X) * travel
	note.Position = UDim2.fromScale(x, LANE_Y[lane])
	local focus = 1 - math.min(math.abs(x - STRIKE_X) / 0.18, 1)
	note.Size = UDim2.fromOffset(38 + 8 * focus, 30 + 6 * focus)
end

function RhythmStaffUI.color(lane: Lane): Color3
	return COLORS[lane]
end

return table.freeze(RhythmStaffUI)
