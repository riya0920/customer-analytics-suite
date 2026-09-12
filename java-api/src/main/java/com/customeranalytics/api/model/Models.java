package com.customeranalytics.api.model;

import java.util.List;
import java.util.Map;

/**
 * Response DTOs as Java records, mirroring the Pydantic models in
 * ../api/models.py and the TypeScript row types in
 * ../frontend/src/data/types.ts. Field names match the exported CSV columns
 * exactly, so the JSON produced here is byte-for-byte the same contract the
 * FastAPI service serves. Records give immutable, typed shapes that Jackson
 * serialises directly.
 */
public final class Models {
    private Models() {}

    public record Segment(
            String segment,
            int customers,
            double share_of_customers,
            double cal_frequency,
            double cal_monetary,
            double recency_days,
            double churn_rate,
            double holdout_orders,
            double holdout_spend,
            double clv_total,
            double clv_mean,
            double share_of_value) {}

    public record KSelection(
            int k,
            double silhouette,
            double calinski_harabasz,
            double davies_bouldin,
            double inertia,
            double mean_ari,
            double adjusted_eta_squared,
            double smallest_cluster_share) {}

    public record ClvByChannel(
            String channel,
            int converters_touched,
            double mean_clv,
            double median_clv,
            double total_clv,
            double clv_index) {}

    public record ClvSummary(String metric, double value) {}

    public record ClvModel(String model, double spearman, double mae) {}

    public record ClvPerCustomer(
            int customer_id,
            String segment,
            double predicted_clv,
            double holdout_spend,
            double cal_frequency,
            double recency_days) {}

    public record AttributionCredit(
            String method,
            String channel,
            double credit,
            double truth,
            double error) {}

    public record MethodScore(String method, double mae, double credit_to_zero_effect) {}

    public record BudgetOutcome(
            String allocation,
            double conversions,
            double conversions_lost_vs_truth,
            double pct_conversions_lost,
            double spend_on_zero_effect,
            double customer_value,
            double pct_value_lost) {}

    public record Kpi(
            int total_customers,
            int n_segments,
            double top20pct_value_share,
            double clv_rank_spearman,
            double best_attribution_mae,
            int n_channels) {}

    /** A slice of a larger collection, cursor echoed back (matches Page in the
     * FastAPI service). */
    public record Page<T>(List<T> items, int total, int limit, int offset) {}

    public record Health(String status, String data_dir, Map<String, Integer> row_counts) {}

    /** Uniform error body returned for 404/422 (matches ErrorResponse). */
    public record ErrorResponse(String error, String detail) {}
}
