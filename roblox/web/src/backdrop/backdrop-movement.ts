export type BackdropPosition = {
  readonly x: number;
  readonly z: number;
};

export function clampBackdropPosition(
  position: BackdropPosition,
  maximumRadius: number,
): BackdropPosition {
  const distance = Math.hypot(position.x, position.z);
  if (distance <= maximumRadius) return position;

  const scale = maximumRadius / distance;
  return {
    x: position.x * scale,
    z: position.z * scale,
  };
}
