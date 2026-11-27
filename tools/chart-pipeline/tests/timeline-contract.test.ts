import { expect, test } from "bun:test";
import fixture from "../../../roblox/tests/fixtures/instrumental-timeline.json";
import { timelineSchema } from "../src/timeline-contract";

test("reads a piano timeline in master time and authored order", () => {
  const timeline = timelineSchema.parse(fixture.timeline);
  expect<string[]>(timeline.events.map((event) => event.eventId)).toEqual(
    fixture.expectedIds,
  );
  expect<number[]>(timeline.events.map((event) => event.timeUs)).toEqual([
    -250001, 1234567, 1234567,
  ]);
  expect<string[]>(timeline.roles).toEqual(["piano"]);
  expect<string[]>([
    timeline.contentRevision,
    timeline.balanceRevision,
  ]).toEqual(["instrumental-r3", "balance-r2"]);
});

for (const [name, patch] of Object.entries({
  version: { schemaVersion: 2 },
  revision: { balanceRevision: "" },
  roleReference: { roles: ["synth"] },
  duplicateRole: { roles: ["piano", "piano"] },
  duplicateEvent: {
    events: [...fixture.timeline.events, ...fixture.timeline.events],
  },
  fractionalTime: {
    events: fixture.timeline.events.map((e) => ({ ...e, timeUs: 1.5 })),
  },
})) {
  test(`rejects invalid ${name}`, () => {
    expect(
      timelineSchema.safeParse({ ...fixture.timeline, ...patch }).success,
    ).toBe(false);
  });
}
