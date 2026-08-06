const arenaCssTokens = {
  void: "--void",
  stage: "--stage",
  panel: "--panel-strong",
  cyan: "--cyan",
  gold: "--gold",
  violet: "--violet",
  danger: "--danger",
  fill: "--arena-fill",
} as const;

export type ArenaColorToken = keyof typeof arenaCssTokens;

export const cssArenaColor = (token: ArenaColorToken): string => {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(arenaCssTokens[token])
    .trim();
  if (!value)
    throw new Error(`Missing Arena color token ${arenaCssTokens[token]}`);
  return value;
};
