// A fixed categorical palette so a given segment/channel keeps its colour across
// charts and re-renders.

export const PALETTE = [
  "#5b8cff",
  "#4dd6b0",
  "#ffb454",
  "#ff6b6b",
  "#b78cff",
  "#4db8ff",
  "#f77fbe",
  "#8bd450",
  "#ffd166",
  "#06d6a0",
  "#ef476f",
  "#a0aec0",
];

export function colorFor(index: number): string {
  return PALETTE[index % PALETTE.length];
}
