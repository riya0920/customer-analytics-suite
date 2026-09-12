import { describe, expect, it } from "vitest";
import {
  attributionForMethod,
  attributionMethods,
  clvHistogram,
  formatPct,
  formatUsd,
  sum,
  valueConcentrationCurve,
  zeroEffectChannel,
} from "./transforms";
import type {
  AttributionCreditRow,
  ClvPerCustomerRow,
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
    seg({ segment: "A", customers: 100, clv_total: 1000 }), // $10/cust
    seg({ segment: "B", customers: 100, clv_total: 9000 }), // $90/cust
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
  it("formats USD with no decimals", () => {
    expect(formatUsd(1234.56)).toBe("$1,235");
  });
  it("formats fractions as percentages", () => {
    expect(formatPct(0.766)).toBe("76.6%");
    expect(formatPct(0.5, 0)).toBe("50%");
  });
});
