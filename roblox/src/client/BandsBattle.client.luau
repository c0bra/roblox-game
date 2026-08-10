--!strict

local ContentProvider = game:GetService("ContentProvider")
local Players = game:GetService("Players")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local RunService = game:GetService("RunService")
local SoundService = game:GetService("SoundService")
local UserInputService = game:GetService("UserInputService")

local RhythmChart = require(ReplicatedStorage.BandsBattle.RhythmChart)
local RhythmJudge = require(ReplicatedStorage.BandsBattle.RhythmJudge)
local RhythmInput = require(script.Parent.RadialRhythmInput)
local RhythmStaffUI = require(script.Parent.RadialRhythmUI)
local RhythmVFX = require(script.Parent.RadialRhythmVFX)

type Lane = RhythmChart.Lane
type Grade = RhythmJudge.Grade

local player = Players.LocalPlayer
local playerGui = player:WaitForChild("PlayerGui") :: PlayerGui
local view = RhythmStaffUI.create(playerGui)

local function makeMusic(name: string, assetId: string, volume: number): Sound?
	if assetId == "" then
		return nil
	end
	local sound = Instance.new("Sound")
	sound.Name = name
	sound.SoundId = assetId
	sound.Volume = volume
	sound.Looped = false
	sound.Parent = SoundService
	return sound
end

local music = makeMusic("HeavensEdgeMusic", RhythmChart.audioId, 0.82)

local activeNotes: { [number]: Frame } = {}
local spawned: { [number]: boolean } = {}
local judged: { [number]: boolean } = {}
local running = false
local finished = false
local audioStarted = false
local prerollStartedAt = 0
local score = 0
local combo = 0
local bestCombo = 0
local judgedCount = 0
local accuracyWeight = 0

local function hideBossModels()
	for _, modelName in { "MonsterOctopus", "ThreeHeadGhost" } do
		local model = workspace:FindFirstChild(modelName)
		if model then
			for _, descendant in model:GetDescendants() do
				if descendant:IsA("BasePart") then
					descendant.LocalTransparencyModifier = 1
				end
			end
		end
	end
end

local function songTime(): number
	if audioStarted and music then
		return music.TimePosition - RhythmChart.audioClipStart + RhythmChart.inputOffset
	end
	return os.clock() - prerollStartedAt - RhythmChart.approachTime
end

local function updateHud(currentTime: number)
	view.combo.Text = `{combo}×`
	view.score.Text = string.format("%06d", score)
	local accuracy = if judgedCount == 0 then 1 else accuracyWeight / judgedCount
	view.accuracy.Text = `{math.floor(accuracy * 100 + 0.5)}%`
	view.progress.Size = UDim2.fromScale(math.clamp(currentTime / RhythmChart.duration, 0, 1), 1)
end

local function clearNotes()
	for _, noteFrame in activeNotes do
		noteFrame:Destroy()
	end
	table.clear(activeNotes)
	table.clear(spawned)
	table.clear(judged)
end

local function recordJudgement(index: number, grade: Grade)
	if judged[index] then
		return
	end
	judged[index] = true
	judgedCount += 1
	accuracyWeight += RhythmJudge.weight(grade)

	local note = RhythmChart.notes[index]
	local frame = activeNotes[index]
	if frame then
		frame:Destroy()
		activeNotes[index] = nil
	end

	if grade == "Miss" then
		combo = 0
		view.feedback.Text = "MISS"
		view.feedback.TextColor3 = Color3.fromRGB(255, 91, 121)
		RhythmVFX.miss(view, note.lane)
	else
		combo += 1
		bestCombo = math.max(bestCombo, combo)
		score += RhythmJudge.score(grade) + math.min(combo, 30) * 12
		view.feedback.Text = string.upper(grade)
		view.feedback.TextColor3 = RhythmStaffUI.color(note.lane)
		RhythmVFX.hit(view, note.lane, grade)
	end
	updateHud(songTime())
end

local function closestNote(lane: Lane, currentTime: number): (number?, number)
	local closestIndex: number? = nil
	local closestDistance = math.huge
	for index, note in RhythmChart.notes do
		if not judged[index] and note.lane == lane then
			local distance = math.abs(note.time - currentTime)
			if distance < closestDistance then
				closestIndex = index
				closestDistance = distance
			end
		end
	end
	return closestIndex, closestDistance
end

local function strike(lane: Lane)
	if not running or not audioStarted then
		return
	end
	local currentTime = songTime()
	local index, distance = closestNote(lane, currentTime)
	if index and distance <= RhythmJudge.goodWindow then
		recordJudgement(index, RhythmJudge.grade(distance))
	else
		combo = 0
		view.feedback.Text = "OFF BEAT"
		view.feedback.TextColor3 = Color3.fromRGB(255, 91, 121)
		RhythmVFX.miss(view, lane)
		updateHud(currentTime)
	end
end

local function finishRun()
	if finished then
		return
	end
	finished = true
	running = false
	if music then
		music:Stop()
	end
	for index, _note in RhythmChart.notes do
		if not judged[index] then
			recordJudgement(index, "Miss")
		end
	end
	local accuracy = if judgedCount == 0 then 0 else accuracyWeight / judgedCount
	view.feedback.Text = if accuracy >= 0.9 then "STAFF CLEAR" elseif accuracy >= 0.7 then "SONG COMPLETE" else "KEEP THE BEAT"
	view.feedback.TextColor3 = if accuracy >= 0.9 then Color3.fromRGB(255, 229, 151) else Color3.fromRGB(183, 232, 255)
	view.status.Text = `Best combo {bestCombo}×  ·  {math.floor(accuracy * 100 + 0.5)}% accuracy`
	view.replay.Visible = true
	for _, target in view.laneTargets do
		target.Visible = false
	end
	updateHud(RhythmChart.duration)
end

local function beginRun()
	if music then
		music:Stop()
		music.TimePosition = RhythmChart.audioClipStart
	end
	clearNotes()
	running = true
	finished = false
	audioStarted = false
	prerollStartedAt = os.clock()
	score = 0
	combo = 0
	bestCombo = 0
	judgedCount = 0
	accuracyWeight = 0
	view.replay.Visible = false
	for _, target in view.laneTargets do
		target.Visible = true
	end
	view.feedback.Text = "GET READY"
	view.feedback.TextColor3 = Color3.fromRGB(226, 250, 255)
	view.status.Text = "Z · X · C  ·  Arrow keys also work"
	updateHud(0)
end

local function startAudio()
	audioStarted = true
	if music then
		music.TimePosition = RhythmChart.audioClipStart
		music:Play()
	end
	view.feedback.Text = "PLAY"
	view.status.Text = "Hit a note as it crosses the vertical line"
end

local function update(currentTime: number)
	for index, note in RhythmChart.notes do
		if not spawned[index] and currentTime >= note.time - RhythmChart.approachTime then
			spawned[index] = true
			activeNotes[index] = RhythmStaffUI.createNote(view, note.lane)
		end
		local frame = activeNotes[index]
		if frame and not judged[index] then
			local progress = 1 - ((note.time - currentTime) / RhythmChart.approachTime)
			RhythmStaffUI.positionNote(frame, note.lane, progress)
			if currentTime > note.time + RhythmJudge.goodWindow then
				recordJudgement(index, "Miss")
			end
		end
	end
	updateHud(currentTime)
	if audioStarted and currentTime >= RhythmChart.duration then
		finishRun()
	end
end

UserInputService.InputBegan:Connect(function(input: InputObject, gameProcessed: boolean)
	if gameProcessed then
		return
	end
	local lane = RhythmInput.laneFor(input.KeyCode)
	if lane then
		strike(lane)
	end
end)

view.input.Event:Connect(strike)
RunService.Heartbeat:Connect(function()
	if not running then
		return
	end
	local currentTime = songTime()
	if not audioStarted and currentTime >= 0 then
		startAudio()
		currentTime = songTime()
	end
	update(currentTime)
end)

view.replay.Activated:Connect(beginRun)

hideBossModels()
if not music then
	view.feedback.Text = "AUDIO ASSET NEEDED"
	view.feedback.TextColor3 = Color3.fromRGB(255, 193, 92)
	view.status.Text = "Import audio/Heavens_Edge/heavens_edge.mp3 into Roblox"
else
	view.status.Text = "Loading chart audio…"
	local loaded, loadError = pcall(function()
		ContentProvider:PreloadAsync({ music })
	end)
	if loaded then
		task.delay(0.5, beginRun)
	else
		view.feedback.Text = "AUDIO UNAVAILABLE"
		view.feedback.TextColor3 = Color3.fromRGB(255, 91, 121)
		view.status.Text = `Could not load the chart audio: {tostring(loadError)}`
	end
end
