// Pure, unit-tested transforms that turn the raw typed rows into the exact shapes
// the charts consume. Kept out of the components so the numbers can be tested
// without rendering anything.

import type {
  AttributionCreditRow,
  ClvPerCustomerRow,
  KSelectionRow,
  SegmentRow,
} from "./types";

export interface ConcentrationPoint {
  /** Cumulative share of customers, 0..1, richest segment first. */
  cumCustomers: number;
  /** Cumulative share of value, 0..1. */
  cumValue: number;
  segment: string;
}

/**
 * Value-concentration (Lorenz-style) curve: order segments by value per
 * customer, then accumulate. A diagonal would mean value is spread evenly; the
 * further the curve bows above the diagonal, the more concentrated the value.
 * Starts at the origin so the area/gap reads correctly.
 */
export function valueConcentrationCurve(
  segments: SegmentRow[],
): ConcentrationPoint[] {
  const totalCustomers = sum(segments.map((s) => s.customers));
  const totalValue = sum(segments.map((s) => s.clv_total));
  if (totalCustomers === 0 || totalValue === 0) return [];

  const ordered = [...segments].sort(
    (a, b) => b.clv_total / b.customers - a.clv_total / a.customers,
  );
  const points: ConcentrationPoint[] = [
    { cumCustomers: 0, cumValue: 0, segment: "" },
  ];
  let cc = 0;
  let cv = 0;
  for (const s of ordered) {
    cc += s.customers / totalCustomers;
    cv += s.clv_total / totalValue;
    points.push({ cumCustomers: cc, cumValue: cv, segment: s.segment });
  }
  return points;
}

export interface HistogramBin {
  /** Left edge of the bin. */
  x0: number;
  /** Right edge of the bin. */
  x1: number;
  /** Midpoint label for the axis. */
  mid: number;
  count: number;
}

/**
 * Fixed-width histogram of predicted CLV, clipped at `clip` so a long tail does
 * not flatten the mass near zero. The final bin is an inclusive ">= clip" bucket.
 */
export function clvHistogram(
  rows: ClvPerCustomerRow[],
  binCount = 40,
  clip = 2000,
): HistogramBin[] {
  const width = clip / binCount;
  const bins: HistogramBin[] = Array.from({ length: binCount }, (_, i) => ({
    x0: i * width,
    x1: (i + 1) * width,
    mid: i * width + width / 2,
    count: 0,
  }));
  for (const r of rows) {
    const v = r.predicted_clv;
    if (!Number.isFinite(v)) continue;
    let idx = Math.floor(v / width);
    if (idx >= binCount) idx = binCount - 1; // clip tail into last bin
    if (idx < 0) idx = 0;
    bins[idx].count++;
  }
  return bins;
}

export interface MethodError {
  method: string;
  channel: string;
  credit: number;
  truth: number;
  error: number;
}

/** All per-channel rows for one attribution method, channels ordered by truth. */
export function attributionForMethod(
  rows: AttributionCreditRow[],
  method: string,
): MethodError[] {
  return rows
    .filter((r) => r.method === method)
    .sort((a, b) => b.truth - a.truth)
    .map((r) => ({
      method: r.method,
      channel: r.channel,
      credit: r.credit,
      truth: r.truth,
      error: r.error,
    }));
}

/** Distinct method names present in the attribution table, "truth" excluded. */
export function attributionMethods(rows: AttributionCreditRow[]): string[] {
  const seen = new Set<string>();
  for (const r of rows) if (r.method !== "truth") seen.add(r.method);
  return [...seen];
}

/** The single channel whose true effect is zero (the planted incrementality
 * control). Returns undefined if the truth table has no such channel. */
export function zeroEffectChannel(
  rows: AttributionCreditRow[],
): string | undefined {
  const truth = rows.find((r) => r.method === "truth" && r.truth === 0);
  return truth?.channel;
}

export function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

export interface SegmentHighlights {
  /** Segment with the largest share of total predicted value. */
  top: SegmentRow;
  /** Segment with the highest predicted value per customer. */
  richest: SegmentRow;
  /** Segment with the lowest predicted value per customer. */
  poorest: SegmentRow;
}

/** The segments worth naming in the narrative, picked from the data. */
export function segmentHighlights(
  segments: SegmentRow[],
): SegmentHighlights | undefined {
  if (segments.length === 0) return undefined;
  const maxBy = (f: (s: SegmentRow) => number) =>
    segments.reduce((a, b) => (f(b) > f(a) ? b : a));
  return {
    top: maxBy((s) => s.share_of_value),
    richest: maxBy((s) => s.clv_mean),
    poorest: maxBy((s) => -s.clv_mean),
  };
}

export interface KSelectionSummary {
  chosen: KSelectionRow;
  /** The row for k = chosen - 1, if it was tested. */
  previous?: KSelectionRow;
  /** The k with the highest adjusted eta^2 (forward separation). */
  bestForward: KSelectionRow;
  /** chosen adjusted eta^2 as a fraction of the best one. */
  shareOfBestForward: number;
  silhouetteK: number;
  daviesBouldinK: number;
  stabilityK: number;
  /** Every other k whose bootstrap stability (mean ARI) beats the chosen k. */
  moreStableKs: number[];
}

/** What the k-selection table actually says about a chosen k. */
export function kSelectionSummary(
  rows: KSelectionRow[],
  chosenK: number,
): KSelectionSummary | undefined {
  const chosen = rows.find((r) => r.k === chosenK);
  if (!chosen) return undefined;
  const maxBy = (f: (r: KSelectionRow) => number) =>
    rows.reduce((a, b) => (f(b) > f(a) ? b : a));
  const bestForward = maxBy((r) => r.adjusted_eta_squared);
  return {
    chosen,
    previous: rows.find((r) => r.k === chosenK - 1),
    bestForward,
    shareOfBestForward:
      bestForward.adjusted_eta_squared > 0
        ? chosen.adjusted_eta_squared / bestForward.adjusted_eta_squared
        : 0,
    silhouetteK: maxBy((r) => r.silhouette).k,
    daviesBouldinK: maxBy((r) => -r.davies_bouldin).k,
    stabilityK: maxBy((r) => r.mean_ari).k,
    moreStableKs: rows
      .filter((r) => r.k !== chosenK && r.mean_ari > chosen.mean_ari)
      .map((r) => r.k),
  };
}

/** Pounds sterling, no decimals. The source data is a UK retailer in GBP. */
export function formatGbp(n: number): string {
  return n.toLocaleString("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0,
  });
}

export function formatPct(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}
