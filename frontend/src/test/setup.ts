import "@testing-library/jest-dom/vitest";

// Recharts' ResponsiveContainer measures its parent with ResizeObserver, which
// jsdom does not implement. Provide a no-op so chart components mount without
// throwing during tests (we assert on the text/table content they render, not on
// pixel geometry).
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver ?? (ResizeObserverStub as never);
