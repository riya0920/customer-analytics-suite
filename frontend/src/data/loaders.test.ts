import { describe, expect, it } from "vitest";
import { parsers } from "./loaders";

// Sample strings copied verbatim from the exported CSVs so the parsers are
// pinned to the real column names and formats.

describe("parsers", () => {
  it("parses segments rows into typed objects", () => {
    const csv =
      "segment,customers,share_of_customers,cal_frequency,cal_monetary,recency_days,churn_rate,holdout_orders,holdout_spend,clv_total,clv_mean,share_of_value\n" +
      "Segment 3,34,0.0069,62.235,2440.982,26.5,0.088,23.471,36055.43,792129.12,23297.92,0.2125";
    const rows = parsers.segments(csv);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      segment: "Segment 3",
      customers: 34,
      share_of_value: 0.2125,
      clv_mean: 23297.92,
    });
  });

  it("parses the quoted metric names in clv_summary", () => {
    const csv =
      'metric,value\n"Spearman (rank, holdout)",0.6284\nTop-20% share of value,0.6885';
    const rows = parsers.clvSummary(csv);
    expect(rows.find((r) => r.metric === "Spearman (rank, holdout)")?.value).toBe(
      0.6284,
    );
  });

  it("parses the single kpis row into one object", () => {
    const csv =
      "total_customers,n_segments,top20pct_value_share,clv_rank_spearman,best_attribution_mae,n_channels\n" +
      "4899,5,0.6885,0.6284,0.035,12";
    const kpis = parsers.kpis(csv);
    expect(kpis.total_customers).toBe(4899);
    expect(kpis.n_channels).toBe(12);
  });

  it("parses attribution long rows including the truth method", () => {
    const csv =
      "method,channel,credit,truth,error\n" +
      "shapley,retargeting,0.1397,0.0,0.1397\n" +
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
