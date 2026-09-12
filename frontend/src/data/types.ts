// Row shapes for each exported CSV the dashboard reads. Column names match the
// files in bi_export/ exactly, so a change to the pipeline's output surfaces here
// as a type error rather than a silent undefined.

export interface SegmentRow {
  segment: string;
  customers: number;
  share_of_customers: number;
  cal_frequency: number;
  cal_monetary: number;
  recency_days: number;
  churn_rate: number;
  holdout_orders: number;
  holdout_spend: number;
  clv_total: number;
  clv_mean: number;
  share_of_value: number;
}

export interface KSelectionRow {
  k: number;
  silhouette: number;
  calinski_harabasz: number;
  davies_bouldin: number;
  inertia: number;
  mean_ari: number;
  adjusted_eta_squared: number;
  smallest_cluster_share: number;
}

export interface ClvByChannelRow {
  channel: string;
  converters_touched: number;
  mean_clv: number;
  median_clv: number;
  total_clv: number;
  clv_index: number;
}

export interface ClvSummaryRow {
  metric: string;
  value: number;
}

export interface ClvModelRow {
  model: string;
  spearman: number;
  mae: number;
}

export interface ClvPerCustomerRow {
  customer_id: number;
  segment: string;
  predicted_clv: number;
  holdout_spend: number;
  cal_frequency: number;
  recency_days: number;
}

export interface AttributionCreditRow {
  method: string;
  channel: string;
  credit: number;
  truth: number;
  error: number;
}

export interface MethodScoreRow {
  method: string;
  mae: number;
  credit_to_zero_effect: number;
}

export interface BudgetOutcomeRow {
  allocation: string;
  conversions: number;
  conversions_lost_vs_truth: number;
  pct_conversions_lost: number;
  spend_on_zero_effect: number;
  customer_value: number;
  pct_value_lost: number;
}

export interface KpiRow {
  total_customers: number;
  n_segments: number;
  top20pct_value_share: number;
  clv_rank_spearman: number;
  best_attribution_mae: number;
  n_channels: number;
}

// Everything the app needs, loaded once.
export interface Dataset {
  segments: SegmentRow[];
  kSelection: KSelectionRow[];
  clvByChannel: ClvByChannelRow[];
  clvSummary: ClvSummaryRow[];
  clvModels: ClvModelRow[];
  clvPerCustomer: ClvPerCustomerRow[];
  attribution: AttributionCreditRow[];
  methodScores: MethodScoreRow[];
  budgetOutcomes: BudgetOutcomeRow[];
  kpis: KpiRow;
}
