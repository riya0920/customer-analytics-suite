package com.customeranalytics.api.data;

import com.customeranalytics.api.model.Models.*;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.springframework.stereotype.Component;

/**
 * Loads every exported CSV into typed records once at startup and answers queries
 * over them in memory — the same read-only-over-a-static-export design as the
 * FastAPI service.
 *
 * <p>The data directory is resolved in this order so the app runs both from a
 * fresh clone and from a tree that has just run the pipeline:
 * <ol>
 *   <li>$CAS_DATA_DIR, if set</li>
 *   <li>&lt;repo&gt;/bi_export (checked as ./ and ../ relative to the cwd)</li>
 *   <li>&lt;repo&gt;/frontend/public/data (likewise)</li>
 * </ol>
 */
@Component
public class DataRepository {

    private final Path dataDir;
    private final List<Segment> segments;
    private final List<KSelection> kSelection;
    private final List<ClvByChannel> clvByChannel;
    private final List<ClvSummary> clvSummary;
    private final List<ClvModel> clvModels;
    private final List<ClvPerCustomer> clvPerCustomer;
    private final List<AttributionCredit> attribution;
    private final List<MethodScore> methodScores;
    private final List<BudgetOutcome> budgetOutcomes;
    private final Kpi kpis;

    public DataRepository() {
        this(resolveDataDir(System.getenv("CAS_DATA_DIR")));
    }

    public DataRepository(Path dataDir) {
        this.dataDir = dataDir;
        this.segments = read(dataDir.resolve("segments.csv"), r -> new Segment(
                r.get("segment"), i(r, "customers"), d(r, "share_of_customers"),
                d(r, "cal_frequency"), d(r, "cal_monetary"), d(r, "recency_days"),
                d(r, "churn_rate"), d(r, "holdout_orders"), d(r, "holdout_spend"),
                d(r, "clv_total"), d(r, "clv_mean"), d(r, "share_of_value")));
        this.kSelection = read(dataDir.resolve("k_selection.csv"), r -> new KSelection(
                i(r, "k"), d(r, "silhouette"), d(r, "calinski_harabasz"),
                d(r, "davies_bouldin"), d(r, "inertia"), d(r, "mean_ari"),
                d(r, "adjusted_eta_squared"), d(r, "smallest_cluster_share")));
        this.clvByChannel = read(dataDir.resolve("clv_by_channel.csv"), r -> new ClvByChannel(
                r.get("channel"), i(r, "converters_touched"), d(r, "mean_clv"),
                d(r, "median_clv"), d(r, "total_clv"), d(r, "clv_index")));
        this.clvSummary = read(dataDir.resolve("clv_summary.csv"), r -> new ClvSummary(
                r.get("metric"), d(r, "value")));
        this.clvModels = read(dataDir.resolve("clv_model_comparison.csv"), r -> new ClvModel(
                r.get("model"), d(r, "spearman"), d(r, "mae")));
        this.clvPerCustomer = read(dataDir.resolve("clv_per_customer.csv"), r -> new ClvPerCustomer(
                i(r, "customer_id"), r.get("segment"), d(r, "predicted_clv"),
                d(r, "holdout_spend"), d(r, "cal_frequency"), d(r, "recency_days")));
        this.attribution = read(dataDir.resolve("attribution_credit_long.csv"), r -> new AttributionCredit(
                r.get("method"), r.get("channel"), d(r, "credit"), d(r, "truth"), d(r, "error")));
        this.methodScores = read(dataDir.resolve("method_scores.csv"), r -> new MethodScore(
                r.get("method"), d(r, "mae"), d(r, "credit_to_zero_effect")));
        this.budgetOutcomes = read(dataDir.resolve("budget_outcomes.csv"), r -> new BudgetOutcome(
                r.get("allocation"), d(r, "conversions"), d(r, "conversions_lost_vs_truth"),
                d(r, "pct_conversions_lost"), d(r, "spend_on_zero_effect"),
                d(r, "customer_value"), d(r, "pct_value_lost")));
        List<Kpi> k = read(dataDir.resolve("kpis.csv"), r -> new Kpi(
                i(r, "total_customers"), i(r, "n_segments"), d(r, "top20pct_value_share"),
                d(r, "clv_rank_spearman"), d(r, "best_attribution_mae"), i(r, "n_channels")));
        if (k.isEmpty()) {
            throw new IllegalStateException("kpis.csv is empty");
        }
        this.kpis = k.get(0);
    }

    // --- resolution & parsing ---------------------------------------------

    static Path resolveDataDir(String explicit) {
        List<Path> candidates = new ArrayList<>();
        if (explicit != null && !explicit.isBlank()) {
            candidates.add(Path.of(explicit));
        }
        for (String base : new String[] {".", ".."}) {
            candidates.add(Path.of(base, "bi_export"));
            candidates.add(Path.of(base, "frontend", "public", "data"));
        }
        for (Path c : candidates) {
            if (Files.isDirectory(c) && Files.exists(c.resolve("kpis.csv"))) {
                return c.toAbsolutePath().normalize();
            }
        }
        throw new IllegalStateException(
                "No data directory with kpis.csv found. Run `python export_bi.py` "
                        + "or set CAS_DATA_DIR. Looked in: " + candidates);
    }

    private static <T> List<T> read(Path path, Function<Map<String, String>, T> make) {
        try {
            String text = Files.readString(path, StandardCharsets.UTF_8);
            List<T> out = new ArrayList<>();
            for (Map<String, String> rec : Csv.parseRecords(text)) {
                out.add(make.apply(rec));
            }
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read " + path, e);
        }
    }

    private static double d(Map<String, String> r, String key) {
        String raw = r.getOrDefault(key, "").trim();
        return raw.isEmpty() ? Double.NaN : Double.parseDouble(raw);
    }

    private static int i(Map<String, String> r, String key) {
        return (int) Double.parseDouble(r.get(key).trim());
    }

    // --- queries -----------------------------------------------------------

    public Path getDataDir() {
        return dataDir;
    }

    public Kpi getKpis() {
        return kpis;
    }

    public List<Segment> getSegments() {
        return segments;
    }

    public List<KSelection> getKSelection() {
        return kSelection;
    }

    public List<ClvByChannel> getClvByChannel() {
        return clvByChannel;
    }

    public List<ClvSummary> getClvSummary() {
        return clvSummary;
    }

    public List<ClvModel> getClvModels() {
        return clvModels;
    }

    public List<MethodScore> getMethodScores() {
        return methodScores;
    }

    public List<BudgetOutcome> getBudgetOutcomes() {
        return budgetOutcomes;
    }

    public boolean hasSegment(String segment) {
        return segments.stream().anyMatch(s -> s.segment().equals(segment));
    }

    public boolean hasMethod(String method) {
        return attribution.stream().anyMatch(a -> a.method().equals(method));
    }

    public List<String> attributionMethods() {
        List<String> seen = new ArrayList<>();
        for (AttributionCredit a : attribution) {
            if (!a.method().equals("truth") && !seen.contains(a.method())) {
                seen.add(a.method());
            }
        }
        return seen;
    }

    public List<AttributionCredit> attributionFor(String method) {
        if (method == null) {
            return attribution;
        }
        List<AttributionCredit> out = new ArrayList<>();
        for (AttributionCredit a : attribution) {
            if (a.method().equals(method)) {
                out.add(a);
            }
        }
        return out;
    }

    /** A page of per-customer CLV, optionally filtered to one segment. Returns
     * the slice and the total matching count. */
    public Page<ClvPerCustomer> customers(int limit, int offset, String segment) {
        List<ClvPerCustomer> rows = clvPerCustomer;
        if (segment != null) {
            rows = new ArrayList<>();
            for (ClvPerCustomer c : clvPerCustomer) {
                if (c.segment().equals(segment)) {
                    rows.add(c);
                }
            }
        }
        int total = rows.size();
        int from = Math.min(offset, total);
        int to = Math.min(offset + limit, total);
        return new Page<>(new ArrayList<>(rows.subList(from, to)), total, limit, offset);
    }

    public Map<String, Integer> rowCounts() {
        Map<String, Integer> m = new LinkedHashMap<>();
        m.put("segments", segments.size());
        m.put("k_selection", kSelection.size());
        m.put("clv_by_channel", clvByChannel.size());
        m.put("clv_summary", clvSummary.size());
        m.put("clv_models", clvModels.size());
        m.put("clv_per_customer", clvPerCustomer.size());
        m.put("attribution", attribution.size());
        m.put("method_scores", methodScores.size());
        m.put("budget_outcomes", budgetOutcomes.size());
        return m;
    }
}
