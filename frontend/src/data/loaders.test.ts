import { describe, expect, it } from "vitest";
import { parsers } from "./loaders";

// Sample strings copied verbatim from the exported CSVs so the parsers are
// pinned to the real column names and formats.

describe("parsers", () => {
  it("parses segments rows into typed objects", () => {
    const csv =
      "segment,customers,share_of_customers,cal_frequency,cal_monetary,recency_days,churn_rate,holdout_orders,holdout_spend,clv_total,clv_mean,share_of_value\n" +
      "Segment 0,605,0.0756,39.625,71.583,55.3,0.309,10.251,724.54,432907.49,715.55,0.3733";
    const rows = parsers.segments(csv);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      segment: "Segment 0",
      customers: 605,
      share_of_value: 0.3733,
      clv_mean: 715.55,
    });
  });

  it("parses the quoted metric names in clv_summary", () => {
    const csv =
      'metric,value\n"Spearman (rank, holdout)",0.7366\nTop-20% share of value,0.766';
    const rows = parsers.clvSummary(csv);
    expect(rows.find((r) => r.metric === "Spearman (rank, holdout)")?.value).toBe(
      0.7366,
    );
  });

  it("parses the single kpis row into one object", () => {
    const csv =
      "total_customers,n_segments,top20pct_value_share,clv_rank_spearman,best_attribution_mae,n_channels\n" +
      "8000,5,0.766,0.7366,0.0292,12";
    const kpis = parsers.kpis(csv);
    expect(kpis.total_customers).toBe(8000);
    expect(kpis.n_channels).toBe(12);
  });

  it("parses attribution long rows including the truth method", () => {
    const csv =
      "method,channel,credit,truth,error\n" +
      "shapley,retargeting,0.1065,0.0,0.1065\n" +
      "truth,retargeting,0.0,0.0,0.0";
    const rows = parsers.attribution(csv);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      method: "shapley",
      channel: "retargeting",
      truth: 0,
    });
  });
});
