import type { Dataset } from "../data/types";

// A small but structurally complete Dataset for tests. Numbers are chosen to be
// easy to assert on, not to match the real export.
export function makeDataset(overrides: Partial<Dataset> = {}): Dataset {
  return {
    kpis: {
      total_customers: 8000,
      n_segments: 5,
      top20pct_value_share: 0.766,
      clv_rank_spearman: 0.7366,
      best_attribution_mae: 0.0292,
      n_channels: 12,
    },
    segments: [
      {
        segment: "Segment 4",
        customers: 3142,
        share_of_customers: 0.3927,
        cal_frequency: 9.7,
        cal_monetary: 79.44,
        recency_days: 108.7,
        churn_rate: 0.411,
        holdout_orders: 2.57,
        holdout_spend: 191.44,
        clv_total: 593309,
        clv_mean: 188.83,
        share_of_value: 0.5116,
      },
      {
        segment: "Segment 0",
        customers: 605,
        share_of_customers: 0.0756,
        cal_frequency: 39.6,
        cal_monetary: 71.58,
        recency_days: 55.3,
        churn_rate: 0.309,
        holdout_orders: 10.25,
        holdout_spend: 724.54,
        clv_total: 432907,
        clv_mean: 715.55,
        share_of_value: 0.3733,
      },
    ],
    kSelection: [
      {
        k: 5,
        silhouette: 0.1857,
        calinski_harabasz: 1782,
        davies_bouldin: 1.55,
        inertia: 20731,
        mean_ari: 0.6707,
        adjusted_eta_squared: 0.2987,
        smallest_cluster_share: 0.1779,
      },
    ],
    clvByChannel: [
      {
        channel: "sms",
        converters_touched: 1379,
        mean_clv: 158.99,
        median_clv: 28.33,
        total_clv: 219249,
        clv_index: 1.081,
      },
    ],
    clvSummary: [
      { metric: "Spearman (rank, holdout)", value: 0.7366 },
      { metric: "Pearson (individual)", value: 0.8227 },
      { metric: "Top-20% share of value", value: 0.766 },
    ],
    clvModels: [
      { model: "BG/NBD + Gamma-Gamma", spearman: 0.7464, mae: 91.94 },
      { model: "GBM challenger", spearman: 0.7287, mae: 97.48 },
    ],
    clvPerCustomer: [
      {
        customer_id: 0,
        segment: "Segment 4",
        predicted_clv: 170.58,
        holdout_spend: 0,
        cal_frequency: 10,
        recency_days: 21.6,
      },
    ],
    attribution: [
      { method: "shapley", channel: "paid_search", credit: 0.15, truth: 0.2055, error: -0.0554 },
      { method: "shapley", channel: "retargeting", credit: 0.1065, truth: 0, error: 0.1065 },
      { method: "truth", channel: "paid_search", credit: 0.2055, truth: 0.2055, error: 0 },
      { method: "truth", channel: "retargeting", credit: 0, truth: 0, error: 0 },
    ],
    methodScores: [
      { method: "shapley", mae: 0.0292, credit_to_zero_effect: 0.1065 },
      { method: "markov_removal", mae: 0.1082, credit_to_zero_effect: 0.1166 },
    ],
    budgetOutcomes: [
      {
        allocation: "shapley",
        conversions: 2722.6,
        conversions_lost_vs_truth: 51.7,
        pct_conversions_lost: 1.864,
        spend_on_zero_effect: 159.69,
        customer_value: 402614,
        pct_value_lost: 1.869,
      },
      {
        allocation: "TRUTH",
        conversions: 2774.3,
        conversions_lost_vs_truth: 0,
        pct_conversions_lost: 0,
        spend_on_zero_effect: 0,
        customer_value: 410281,
        pct_value_lost: 0,
      },
    ],
    ...overrides,
  };
}
