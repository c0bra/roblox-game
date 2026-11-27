-- Run in a synced disposable place with the fixture file text as the argument.
return function(fixtureJson: string)
	local fixture = game:GetService("HttpService"):JSONDecode(fixtureJson)
	local read = require(game.ReplicatedStorage.BandsBattle.Contracts.Timeline).read
	local parsed = read(fixture.timeline)
	assert(parsed.contentRevision == "instrumental-r3" and parsed.balanceRevision == "balance-r2")
	assert(#parsed.roles == 1 and parsed.roles[1] == "piano")
	local times = {-250001, 1234567, 1234567}
	assert(#parsed.events == #times)
	for i, event in ipairs(parsed.events) do
		assert(event.eventId == fixture.expectedIds[i] and event.timeUs == times[i] and event.roleId == "piano")
	end
	for _, patch in ipairs({
		{schemaVersion = 2},
		{balanceRevision = ""},
		{roles = {"synth"}},
		{roles = {"piano", "piano"}},
		{events = {fixture.timeline.events[1], fixture.timeline.events[1]}},
		{events = {{eventId = "bad-time", roleId = "piano", timeUs = 1.5, globalTieOrder = 0}}},
	}) do
		local input = table.clone(fixture.timeline)
		for key, value in patch do input[key] = value end
		assert(not pcall(read, input), "Accepted invalid timeline")
	end
	return "Timeline fixture and six rejection cases passed"
end
