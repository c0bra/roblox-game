import type { ArenaBossEvent, ArenaEncounter } from "./encounter-schema";

export type ArenaEncounterIssue = {
  readonly path: string;
  readonly message: string;
};

const issue = (path: string, message: string): ArenaEncounterIssue => ({
  path,
  message,
});

const sorted = (values: readonly number[]): boolean =>
  values.every(
    (value, index) => index === 0 || value >= (values[index - 1] ?? 0),
  );

const duplicateIds = (
  path: string,
  values: readonly { readonly id: string }[],
): readonly ArenaEncounterIssue[] => {
  const seen = new Set<string>();
  return values.flatMap(({ id }) => {
    if (seen.has(id)) return [issue(path, `Duplicate id ${id}`)];
    seen.add(id);
    return [];
  });
};

const validateBossEvent = (
  event: ArenaBossEvent,
  index: number,
  duration: number,
  positionIds: ReadonlySet<string>,
): readonly ArenaEncounterIssue[] => {
  const path = `bossEvents.${index}`;
  const issues: ArenaEncounterIssue[] = [];
  if (
    !(
      event.telegraphStart <= event.criticalStart &&
      event.criticalStart < event.impactTime &&
      event.impactTime < event.recoveryEnd &&
      event.recoveryEnd <= event.openingEnd
    )
  ) {
    issues.push(issue(path, `Invalid timing order for ${event.id}`));
  }
  if (event.openingEnd > duration) {
    issues.push(issue(path, `${event.id} exceeds encounter duration`));
  }
  for (const positionId of [
    ...event.affectedPositionIds,
    ...event.safePositionIds,
  ]) {
    if (!positionIds.has(positionId)) {
      issues.push(issue(path, `${event.id} references ${positionId}`));
    }
  }
  for (const positionId of event.affectedPositionIds) {
    if (event.safePositionIds.includes(positionId)) {
      issues.push(
        issue(path, `${positionId} cannot be both safe and affected`),
      );
    }
  }
  return issues;
};

export const validateArenaEncounter = (
  encounter: ArenaEncounter,
): readonly ArenaEncounterIssue[] => {
  const issues: ArenaEncounterIssue[] = [];
  const positionIds = new Set<string>(encounter.positions.map(({ id }) => id));
  const positionOrders = new Map<string, number>(
    encounter.positions.map(({ id, order }) => [id, order]),
  );
  const bossEvents = new Map(
    encounter.bossEvents.map((event) => [event.id, event]),
  );

  if (!sorted(encounter.beats))
    issues.push(issue("beats", "Times must be sorted"));
  if (!sorted(encounter.downbeats)) {
    issues.push(issue("downbeats", "Times must be sorted"));
  }
  if (encounter.beats.some((time) => time > encounter.duration)) {
    issues.push(issue("beats", "Beat exceeds encounter duration"));
  }
  if (encounter.finalCadence > encounter.duration) {
    issues.push(issue("finalCadence", "Final cadence exceeds duration"));
  }
  if (encounter.resolveVictoryThreshold > encounter.initialResolve) {
    issues.push(
      issue("resolveVictoryThreshold", "Threshold exceeds initial Resolve"),
    );
  }

  issues.push(...duplicateIds("positions", encounter.positions));
  issues.push(...duplicateIds("phrases", encounter.phrases));
  issues.push(
    ...duplicateIds("repositionWindows", encounter.repositionWindows),
  );
  issues.push(...duplicateIds("bossEvents", encounter.bossEvents));
  issues.push(...duplicateIds("phases", encounter.phases));
  issues.push(
    ...duplicateIds(
      "phraseSteps",
      encounter.phrases.flatMap((phrase) => [
        ...phrase.steps,
        ...phrase.positionBonusSteps.flatMap((bonus) => bonus.steps),
      ]),
    ),
  );

  const requiredPositions: readonly string[] = [
    "shelter",
    "midline",
    "spotlight",
  ];
  for (const [order, positionId] of requiredPositions.entries()) {
    if (positionOrders.get(positionId) !== order) {
      issues.push(issue("positions", `${positionId} must have order ${order}`));
    }
  }

  encounter.phrases.forEach((phrase, index) => {
    const path = `phrases.${index}`;
    const previewBeats = encounter.beats.filter(
      (time) => time >= phrase.previewStart && time < phrase.executionStart,
    ).length;
    if (
      !(
        phrase.previewStart < phrase.executionStart &&
        phrase.executionStart < phrase.end
      )
    ) {
      issues.push(
        issue(path, `${phrase.id} has invalid preview/execution order`),
      );
    }
    if (previewBeats < 2) {
      issues.push(
        issue(path, `${phrase.id} requires at least two preview beats`),
      );
    }
    if (!sorted(phrase.steps.map(({ time }) => time))) {
      issues.push(issue(path, `${phrase.id} steps must be sorted`));
    }
    for (const step of phrase.steps) {
      if (step.time < phrase.executionStart || step.time > phrase.end) {
        issues.push(issue(path, `${step.id} is outside phrase execution`));
      }
    }
    for (const bonus of phrase.positionBonusSteps) {
      if (!positionIds.has(bonus.positionId)) {
        issues.push(
          issue(path, `${phrase.id} bonus references ${bonus.positionId}`),
        );
      }
      if (!sorted(bonus.steps.map(({ time }) => time))) {
        issues.push(issue(path, `${phrase.id} bonus steps must be sorted`));
      }
      for (const step of bonus.steps) {
        if (
          step.time < phrase.executionStart ||
          step.time > phrase.end ||
          step.time > encounter.duration
        ) {
          issues.push(
            issue(path, `${step.id} bonus is outside phrase execution`),
          );
        }
      }
    }
    if (phrase.end > encounter.duration) {
      issues.push(issue(path, `${phrase.id} exceeds encounter duration`));
    }
    if (
      encounter.difficulty === "easy" &&
      encounter.bossEvents.some(
        (event) =>
          phrase.previewStart >= event.criticalStart &&
          phrase.previewStart < event.impactTime,
      )
    ) {
      issues.push(
        issue(path, `${phrase.id} preview collides with a critical telegraph`),
      );
    }
  });

  encounter.bossEvents.forEach((event, index) => {
    issues.push(
      ...validateBossEvent(event, index, encounter.duration, positionIds),
    );
  });

  encounter.repositionWindows.forEach((window, index) => {
    const path = `repositionWindows.${index}`;
    const bossEvent = bossEvents.get(window.bossEventId);
    if (!bossEvent) {
      issues.push(issue(path, `${window.id} references unknown boss event`));
      return;
    }
    if (
      !(
        window.start <= window.decisionTime &&
        window.decisionTime <= window.deadline
      )
    ) {
      issues.push(issue(path, `${window.id} has invalid decision order`));
    }
    if (window.deadline + window.travelDuration > bossEvent.impactTime) {
      issues.push(
        issue(path, `${window.id} travel cannot finish before impact`),
      );
    }
    for (const choice of window.choices) {
      const from = positionOrders.get(choice.from);
      const to = positionOrders.get(choice.to);
      if (from === undefined || to === undefined || Math.abs(from - to) > 1) {
        issues.push(
          issue(path, `${window.id} contains an invalid adjacent choice`),
        );
      }
    }
  });

  if (!sorted(encounter.phases.map(({ start }) => start))) {
    issues.push(issue("phases", "Phase times must be sorted"));
  }
  if (encounter.phases.some(({ start }) => start > encounter.duration)) {
    issues.push(issue("phases", "Phase exceeds encounter duration"));
  }
  return issues;
};
