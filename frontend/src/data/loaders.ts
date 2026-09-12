// Fetch each CSV from the app's static /data folder and map it to its typed row
// shape. Paths are resolved against import.meta.env.BASE_URL so the same code
// works at "/" in dev and under "/customer-analytics-suite/" on GitHub Pages.

import { parseRows } from "./csv";
import type {
  AttributionCreditRow,
  BudgetOutcomeRow,
  ClvByChannelRow,
  ClvModelRow,
  ClvPerCustomerRow,
  ClvSummaryRow,
  Dataset,
  KpiRow,
  KSelectionRow,
  MethodScoreRow,
  SegmentRow,
} from "./types";

async function fetchText(file: string): Promise<string> {
  const url = `${import.meta.env.BASE_URL}data/${file}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${file}: HTTP ${res.status}`);
  }
  return res.text();
}

export const parsers = {
  segments: (t: string): SegmentRow[] =>
    parseRows(t, (g, n) => ({
      segment: g("segment"),
      customers: n("customers"),
      share_of_customers: n("share_of_customers"),
      cal_frequency: n("cal_frequency"),
      cal_monetary: n("cal_monetary"),
      recency_days: n("recency_days"),
      churn_rate: n("churn_rate"),
      holdout_orders: n("holdout_orders"),
      holdout_spend: n("holdout_spend"),
      clv_total: n("clv_total"),
      clv_mean: n("clv_mean"),
      share_of_value: n("share_of_value"),
    })),
  kSelection: (t: string): KSelectionRow[] =>
    parseRows(t, (_g, n) => ({
      k: n("k"),
      silhouette: n("silhouette"),
      calinski_harabasz: n("calinski_harabasz"),
      davies_bouldin: n("davies_bouldin"),
      inertia: n("inertia"),
      mean_ari: n("mean_ari"),
      adjusted_eta_squared: n("adjusted_eta_squared"),
      smallest_cluster_share: n("smallest_cluster_share"),
    })),
  clvByChannel: (t: string): ClvByChannelRow[] =>
    parseRows(t, (g, n) => ({
      channel: g("channel"),
      converters_touched: n("converters_touched"),
      mean_clv: n("mean_clv"),
      median_clv: n("median_clv"),
      total_clv: n("total_clv"),
      clv_index: n("clv_index"),
    })),
  clvSummary: (t: string): ClvSummaryRow[] =>
    parseRows(t, (g, n) => ({ metric: g("metric"), value: n("value") })),
  clvModels: (t: string): ClvModelRow[] =>
    parseRows(t, (g, n) => ({
      model: g("model"),
      spearman: n("spearman"),
      mae: n("mae"),
    })),
  clvPerCustomer: (t: string): ClvPerCustomerRow[] =>
    parseRows(t, (g, n) => ({
      customer_id: n("customer_id"),
      segment: g("segment"),
      predicted_clv: n("predicted_clv"),
      holdout_spend: n("holdout_spend"),
      cal_frequency: n("cal_frequency"),
      recency_days: n("recency_days"),
    })),
  attribution: (t: string): AttributionCreditRow[] =>
    parseRows(t, (g, n) => ({
      method: g("method"),
      channel: g("channel"),
      credit: n("credit"),
      truth: n("truth"),
      error: n("error"),
    })),
  methodScores: (t: string): MethodScoreRow[] =>
    parseRows(t, (g, n) => ({
      method: g("method"),
      mae: n("mae"),
      credit_to_zero_effect: n("credit_to_zero_effect"),
    })),
  budgetOutcomes: (t: string): BudgetOutcomeRow[] =>
    parseRows(t, (g, n) => ({
      allocation: g("allocation"),
      conversions: n("conversions"),
      conversions_lost_vs_truth: n("conversions_lost_vs_truth"),
      pct_conversions_lost: n("pct_conversions_lost"),
      spend_on_zero_effect: n("spend_on_zero_effect"),
      customer_value: n("customer_value"),
      pct_value_lost: n("pct_value_lost"),
    })),
  kpis: (t: string): KpiRow => {
    const rows = parseRows(t, (_g, n) => ({
      total_customers: n("total_customers"),
      n_segments: n("n_segments"),
      top20pct_value_share: n("top20pct_value_share"),
      clv_rank_spearman: n("clv_rank_spearman"),
      best_attribution_mae: n("best_attribution_mae"),
      n_channels: n("n_channels"),
    }));
    if (rows.length === 0) throw new Error("kpis.csv is empty");
    return rows[0];
  },
};

/** Load every CSV in parallel and assemble the typed Dataset. */
export async function loadDataset(): Promise<Dataset> {
  const [
    segments,
    kSelection,
    clvByChannel,
    clvSummary,
    clvModels,
    clvPerCustomer,
    attribution,
    methodScores,
    budgetOutcomes,
    kpis,
  ] = await Promise.all([
    fetchText("segments.csv").then(parsers.segments),
    fetchText("k_selection.csv").then(parsers.kSelection),
    fetchText("clv_by_channel.csv").then(parsers.clvByChannel),
    fetchText("clv_summary.csv").then(parsers.clvSummary),
    fetchText("clv_model_comparison.csv").then(parsers.clvModels),
    fetchText("clv_per_customer.csv").then(parsers.clvPerCustomer),
    fetchText("attribution_credit_long.csv").then(parsers.attribution),
    fetchText("method_scores.csv").then(parsers.methodScores),
    fetchText("budget_outcomes.csv").then(parsers.budgetOutcomes),
    fetchText("kpis.csv").then(parsers.kpis),
  ]);
  return {
    segments,
    kSelection,
    clvByChannel,
    clvSummary,
    clvModels,
    clvPerCustomer,
    attribution,
    methodScores,
    budgetOutcomes,
    kpis,
  };
}
