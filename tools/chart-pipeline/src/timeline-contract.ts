import { z } from "zod";

const id = z.string().min(1);
const roleId = id.brand<"RoleId">();
const event = z.object({
  eventId: id.brand<"EventId">(),
  roleId,
  timeUs: z.int().brand<"MasterTimeUs">(),
  globalTieOrder: z.int().nonnegative(),
});

// This projection contains events from one domain phase (architecture §9.1).
export const timelineSchema = z
  .object({
    schemaVersion: z.literal(1),
    contentRevision: id.brand<"ContentRevision">(),
    balanceRevision: id.brand<"BalanceRevision">(),
    roles: z.array(roleId).min(1),
    events: z.array(event),
  })
  .superRefine((timeline, ctx) => {
    const roles = new Set(timeline.roles);
    if (roles.size !== timeline.roles.length) {
      ctx.addIssue({
        code: "custom",
        path: ["roles"],
        message: "Duplicate role ID",
      });
    }
    const seen = new Set<string>();
    for (const [index, item] of timeline.events.entries()) {
      if (!roles.has(item.roleId) || seen.has(item.eventId)) {
        ctx.addIssue({
          code: "custom",
          path: ["events", index],
          message: "Unknown role or duplicate event ID",
        });
      }
      seen.add(item.eventId);
    }
  })
  .transform((timeline) => ({
    ...timeline,
    events: timeline.events.toSorted(
      (a, b) =>
        a.timeUs - b.timeUs ||
        a.globalTieOrder - b.globalTieOrder ||
        Buffer.compare(Buffer.from(a.eventId), Buffer.from(b.eventId)),
    ),
  }));
