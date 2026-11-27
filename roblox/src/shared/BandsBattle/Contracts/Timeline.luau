--!strict

type Event = { eventId: string, roleId: string, timeUs: number, globalTieOrder: number }
local function identifier(value: unknown): string
	assert(type(value) == "string" and #value > 0, "Expected a nonempty identifier")
	return value
end

local function integer(value: unknown): number
	assert(type(value) == "number" and math.abs(value) <= 9007199254740991 and value % 1 == 0, "Expected a safe integer")
	return value
end

local function arraySize(value): number
	assert(type(value) == "table", "Expected an array")
	local count = 0
	for index in value do
		assert(type(index) == "number" and index > 0 and index % 1 == 0, "Expected an array index")
		count += 1
	end
	for index = 1, count do
		assert(value[index] ~= nil, "Expected a dense array")
	end
	return count
end

-- Read the single-domain-phase projection used by the shared contract fixture.
local function read(value)
	assert(type(value) == "table" and value.schemaVersion == 1, "Unsupported timeline version")
	local contentRevision = identifier(value.contentRevision)
	local balanceRevision = identifier(value.balanceRevision)
	local roles: {string} = {}
	local knownRoles: {[string]: boolean} = {}
	assert(arraySize(value.roles) > 0, "Expected a role")
	for _, role in ipairs(value.roles) do
		local id = identifier(role)
		assert(not knownRoles[id], "Duplicate role ID")
		knownRoles[id] = true
		table.insert(roles, id)
	end

	arraySize(value.events)
	local events: {Event} = {}
	local seen: {[string]: boolean} = {}
	for _, item in ipairs(value.events) do
		local id = identifier(item.eventId)
		local roleId = identifier(item.roleId)
		local order = integer(item.globalTieOrder)
		assert(knownRoles[roleId] and not seen[id] and order >= 0, "Invalid timeline event")
		seen[id] = true
		table.insert(events, {eventId = id, roleId = roleId, timeUs = integer(item.timeUs), globalTieOrder = order})
	end
	table.sort(events, function(a, b)
		if a.timeUs ~= b.timeUs then return a.timeUs < b.timeUs end
		if a.globalTieOrder ~= b.globalTieOrder then return a.globalTieOrder < b.globalTieOrder end
		return a.eventId < b.eventId
	end)
	return { schemaVersion = 1, contentRevision = contentRevision, balanceRevision = balanceRevision, roles = roles, events = events }
end

return { read = read }
