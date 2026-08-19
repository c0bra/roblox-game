--!strict

local RhythmJudge = {}

export type Grade = "Perfect" | "Great" | "Good" | "Miss"

RhythmJudge.perfectWindow = 0.06
RhythmJudge.greatWindow = 0.11
RhythmJudge.goodWindow = 0.17

function RhythmJudge.grade(errorSeconds: number): Grade
	local distance = math.abs(errorSeconds)
	if distance <= RhythmJudge.perfectWindow then
		return "Perfect"
	elseif distance <= RhythmJudge.greatWindow then
		return "Great"
	elseif distance <= RhythmJudge.goodWindow then
		return "Good"
	end
	return "Miss"
end

function RhythmJudge.score(grade: Grade): number
	if grade == "Perfect" then
		return 1000
	elseif grade == "Great" then
		return 650
	elseif grade == "Good" then
		return 300
	end
	return 0
end

function RhythmJudge.weight(grade: Grade): number
	if grade == "Perfect" then
		return 1
	elseif grade == "Great" then
		return 0.75
	elseif grade == "Good" then
		return 0.45
	end
	return 0
end

return table.freeze(RhythmJudge)
