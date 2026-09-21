import { describe, expect, it } from "vitest";
import {
  attributionForMethod,
  attributionMethods,
  clvHistogram,
  formatGbp,
  formatPct,
  kSelectionSummary,
  segmentHighlights,
  sum,
  valueConcentrationCurve,
  zeroEffectChannel,
} from "./transforms";
import type {
  AttributionCreditRow,
  ClvPerCustomerRow,
  KSelectionRow,
  SegmentRow,
} from "./types";

function seg(partial: Partial<SegmentRow>): SegmentRow {
  return {
    segment: "Segment X",
    customers: 0,
    share_of_customers: 0,
    cal_frequency: 0,
    cal_monetary: 0,
    recency_days: 0,
    churn_rate: 0,
    holdout_orders: 0,
    holdout_spend: 0,
    clv_total: 0,
    clv_mean: 0,
    share_of_value: 0,
    ...partial,
  };
}

describe("valueConcentrationCurve", () => {
  const segments = [
    seg({ segment: "A", customers: 100, clv_total: 1000 }), // £10/cust
    seg({ segment: "B", customers: 100, clv_total: 9000 }), // £90/cust
  ];

  it("starts at the origin and ends at (1,1)", () => {
    const curve = valueConcentrationCurve(segments);
    expect(curve[0]).toMatchObject({ cumCustomers: 0, cumValue: 0 });
    const last = curve[curve.length - 1];
    expect(last.cumCustomers).toBeCloseTo(1);
    expect(last.cumValue).toBeCloseTo(1);
  });

  it("orders richest-per-customer first, so value leads customers", () => {
    const curve = valueConcentrationCurve(segments);
    // After the first (richest) segment: 50% of customers, 90% of value.
    expect(curve[1].segment).toBe("B");
    expect(curve[1].cumCustomers).toBeCloseTo(0.5);
    expect(curve[1].cumValue).toBeCloseTo(0.9);
  });

  it("is empty when there are no customers or no value", () => {
    expect(valueConcentrationCurve([])).toEqual([]);
    expect(
      valueConcentrationCurve([seg({ customers: 10, clv_total: 0 })]),
    ).toEqual([]);
  });
});

describe("clvHistogram", () => {
  const rows: ClvPerCustomerRow[] = [
    { predicted_clv: 5 },
    { predicted_clv: 15 },
    { predicted_clv: 15 },
    { predicted_clv: 9999 }, // tail -> clipped into last bin
  ].map((r, i) => ({
    customer_id: i,
    segment: "S",
    holdout_spend: 0,
    cal_frequency: 0,
    recency_days: 0,
    ...r,
  }));

  it("counts every customer exactly once", () => {
    const bins = clvHistogram(rows, 30, 600);
    expect(sum(bins.map((b) => b.count))).toBe(rows.length);
  });

  it("folds the tail into the final bin", () => {
    const bins = clvHistogram(rows, 30, 600);
    expect(bins[bins.length - 1].count).toBe(1);
  });

  it("places values in the correct fixed-width bin", () => {
    const bins = clvHistogram(rows, 30, 600); // width = 20
    // 5, 15, 15 all fall in bin 0 (floor(v/20) === 0); the tail is in bin 29.
    expect(bins[0].count).toBe(3);
    expect(bins[0]).toMatchObject({ x0: 0, x1: 20 });
    expect(bins[1].count).toBe(0);
  });
});

describe("attribution helpers", () => {
  const rows: AttributionCreditRow[] = [
    { method: "shapley", channel: "a", credit: 0.2, truth: 0.1, error: 0.1 },
    { method: "shapley", channel: "b", credit: 0.3, truth: 0.5, error: -0.2 },
    { method: "truth", channel: "a", credit: 0.1, truth: 0.1, error: 0 },
    { method: "truth", channel: "z", credit: 0, truth: 0, error: 0 },
  ];

  it("lists methods excluding truth", () => {
    expect(attributionMethods(rows)).toEqual(["shapley"]);
  });

  it("returns one method's channels ordered by truth desc", () => {
    const out = attributionForMethod(rows, "shapley");
    expect(out.map((r) => r.channel)).toEqual(["b", "a"]);
  });

  it("finds the planted zero-effect channel", () => {
    expect(zeroEffectChannel(rows)).toBe("z");
  });
});

describe("formatters", () => {
  it("formats GBP with no decimals", () => {
    expect(formatGbp(1234.56)).toBe("£1,235");
    expect(formatGbp(36055.43)).toBe("£36,055");
    expect(formatGbp(0)).toBe("£0");
  });
  it("formats fractions as percentages", () => {
    expect(formatPct(0.766)).toBe("76.6%");
    expect(formatPct(0.5, 0)).toBe("50%");
  });
});

describe("segmentHighlights", () => {
  // Real rows from segments.csv (UCI Online Retail II), trimmed to the fields used.
  const segments = [
    seg({ segment: "Segment 0", customers: 1133, share_of_customers: 0.2313, clv_mean: 153.08, share_of_value: 0.0465, churn_rate: 0.755 }),
    seg({ segment: "Segment 3", customers: 34, share_of_customers: 0.0069, clv_mean: 23297.92, share_of_value: 0.2125, cal_frequency: 62.235 }),
    seg({ segment: "Segment 4", customers: 1615, share_of_customers: 0.3297, clv_mean: 1135.24, share_of_value: 0.4918 }),
  ];

  it("picks the top-value, richest-per-customer and poorest segments", () => {
    const hl = segmentHighlights(segments);
    expect(hl?.top.segment).toBe("Segment 4");
    expect(hl?.richest.segment).toBe("Segment 3");
    expect(hl?.poorest.segment).toBe("Segment 0");
  });

  it("is undefined with no segments", () => {
    expect(segmentHighlights([])).toBeUndefined();
  });
});

describe("kSelectionSummary", () => {
  // Real k_selection.csv rows.
  const k = (
    kk: number,
    silhouette: number,
    davies_bouldin: number,
    mean_ari: number,
    adjusted_eta_squared: number,
  ): KSelectionRow => ({
    k: kk,
    silhouette,
    calinski_harabasz: 0,
    davies_bouldin,
    inertia: 0,
    mean_ari,
    adjusted_eta_squared,
    smallest_cluster_share: 0,
  });
  const rows = [
    k(2, 0.2551, 1.5903, 0.9341, 0.025),
    k(3, 0.2169, 1.4782, 0.7658, 0.036),
    k(4, 0.2552, 1.2548, 0.9538, 0.0387),
    k(5, 0.2329, 1.4303, 0.7356, 0.0659),
    k(6, 0.2053, 1.3912, 0.8717, 0.0699),
    k(7, 0.2083, 1.3346, 0.8952, 0.0708),
    k(8, 0.204, 1.3596, 0.7858, 0.0703),
  ];

  it("reports forward separation at k-1, k and its peak", () => {
    const s = kSelectionSummary(rows, 5)!;
    expect(s.previous?.adjusted_eta_squared).toBe(0.0387);
    expect(s.chosen.adjusted_eta_squared).toBe(0.0659);
    expect(s.bestForward.k).toBe(7);
    expect(s.shareOfBestForward).toBeCloseTo(0.0659 / 0.0708);
  });

  it("reports which k the geometric criteria pick", () => {
    const s = kSelectionSummary(rows, 5)!;
    expect(s.silhouetteK).toBe(4);
    expect(s.daviesBouldinK).toBe(4);
    expect(s.stabilityK).toBe(4);
  });

  it("does not claim k=5 is the most stable: every other k beats it", () => {
    const s = kSelectionSummary(rows, 5)!;
    expect(s.moreStableKs).toEqual([2, 3, 4, 6, 7, 8]);
  });

  it("is undefined when the chosen k was not tested", () => {
    expect(kSelectionSummary(rows, 12)).toBeUndefined();
  });
});
