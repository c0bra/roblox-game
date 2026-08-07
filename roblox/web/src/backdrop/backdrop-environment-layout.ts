export type IcePerimeterCluster = {
  readonly angle: number;
  readonly radius: number;
  readonly rockScale: readonly [number, number, number];
  readonly crystalHeight: number;
  readonly spread: number;
};

export const icePerimeterClusters: readonly IcePerimeterCluster[] = [
  {
    angle: 0.1,
    radius: 31,
    rockScale: [2.8, 3.1, 2.2],
    crystalHeight: 4.6,
    spread: 2.2,
  },
  {
    angle: 0.48,
    radius: 23,
    rockScale: [2.2, 2.6, 1.9],
    crystalHeight: 0,
    spread: 1.8,
  },
  {
    angle: 0.92,
    radius: 34,
    rockScale: [3.2, 4.1, 2.4],
    crystalHeight: 5.4,
    spread: 2.5,
  },
  {
    angle: 1.42,
    radius: 27,
    rockScale: [2.5, 3.4, 2.1],
    crystalHeight: 3.6,
    spread: 2,
  },
  {
    angle: 1.94,
    radius: 36,
    rockScale: [3.4, 4.4, 2.6],
    crystalHeight: 0,
    spread: 2.7,
  },
  {
    angle: 2.38,
    radius: 25,
    rockScale: [2.7, 3.2, 2],
    crystalHeight: 4.2,
    spread: 2.1,
  },
  {
    angle: 2.82,
    radius: 32,
    rockScale: [3.1, 3.8, 2.3],
    crystalHeight: 0,
    spread: 2.4,
  },
  {
    angle: 3.27,
    radius: 22,
    rockScale: [2.4, 2.8, 1.8],
    crystalHeight: 3.8,
    spread: 1.9,
  },
  {
    angle: 3.72,
    radius: 35,
    rockScale: [3.6, 4.6, 2.7],
    crystalHeight: 5.8,
    spread: 2.8,
  },
  {
    angle: 4.08,
    radius: 24,
    rockScale: [2.8, 3.6, 2.1],
    crystalHeight: 0,
    spread: 2.2,
  },
  {
    angle: 4.5,
    radius: 29,
    rockScale: [3.2, 4.2, 2.4],
    crystalHeight: 6.2,
    spread: 2.5,
  },
  {
    angle: 4.88,
    radius: 21,
    rockScale: [2.6, 3.3, 2],
    crystalHeight: 4.4,
    spread: 2,
  },
  {
    angle: 5.24,
    radius: 34,
    rockScale: [3.5, 4.8, 2.8],
    crystalHeight: 0,
    spread: 2.7,
  },
  {
    angle: 5.68,
    radius: 25,
    rockScale: [2.9, 3.7, 2.2],
    crystalHeight: 5.2,
    spread: 2.3,
  },
  {
    angle: 6.02,
    radius: 37,
    rockScale: [3.8, 4.5, 2.9],
    crystalHeight: 0,
    spread: 2.9,
  },
] as const;
