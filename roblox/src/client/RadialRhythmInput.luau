--!strict

local RhythmInput = {}

export type Lane = "Z" | "X" | "C"

local KEY_LANES: { [Enum.KeyCode]: Lane } = {
	[Enum.KeyCode.Z] = "Z",
	[Enum.KeyCode.X] = "X",
	[Enum.KeyCode.C] = "C",
	[Enum.KeyCode.Left] = "Z",
	[Enum.KeyCode.Down] = "X",
	[Enum.KeyCode.Right] = "C",
}

function RhythmInput.laneFor(keyCode: Enum.KeyCode): Lane?
	return KEY_LANES[keyCode]
end

return table.freeze(RhythmInput)
