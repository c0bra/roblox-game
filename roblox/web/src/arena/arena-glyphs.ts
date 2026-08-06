const svg = (path: string, className = "arena-glyph"): string =>
  `<svg class="${className}" viewBox="0 0 32 32" aria-hidden="true">${path}</svg>`;

export const arenaGlyphs = {
  retreat: svg('<path d="M15 6 5 16l10 10M6 16h17"/><path d="M23 9v14"/>'),
  perform: svg(
    '<path d="m16 3 10 13-10 13L6 16 16 3Z"/><circle cx="16" cy="16" r="4"/><path d="M16 8v4M16 20v4"/>',
  ),
  advance: svg('<path d="m17 6 10 10-10 10M26 16H9"/><path d="M9 9v14"/>'),
  shelter: svg(
    '<path d="M5 27V15a11 11 0 0 1 22 0v12M10 27V15a6 6 0 0 1 12 0v12"/><path d="M3 27h26"/>',
  ),
  midline: svg(
    '<path d="M7 6h7v21H7zM18 6h7v21h-7z"/><path d="m16 11 4 5-4 5-4-5 4-5Z"/>',
  ),
  spotlight: svg(
    '<path d="m16 3 3 9 9 4-9 4-3 9-3-9-9-4 9-4 3-9Z"/><circle cx="16" cy="16" r="3"/>',
  ),
  sweep: svg(
    '<path d="M3 11h20l6 5-6 5H3"/><path d="m8 7-5 4 5 4M8 17l-5 4 5 4"/>',
  ),
  burst: svg(
    '<circle cx="16" cy="16" r="4"/><circle cx="16" cy="16" r="10"/><path d="M16 1v5M16 26v5M1 16h5M26 16h5"/>',
  ),
} as const;
