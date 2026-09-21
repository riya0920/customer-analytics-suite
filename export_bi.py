"""Export the segmentation / CLV / attribution results into tidy CSVs a BI tool
can chart directly (Looker Studio, Tableau Public, Power BI).

Every number here traces to something the pipeline produced:

  - aggregate tables are read straight out of out/analytics_metrics.json and
    out/complete_metrics.json, which are written by run_analytics.py /
    run_complete.py;
  - the two customer-level frames (segment membership and predicted CLV) are
    recomputed here by calling the SAME src.clv functions with the SAME
    deterministic steps run_analytics.py uses (StandardScaler -> KMeans
    random_state=0; BG/NBD x Gamma-Gamma), and the script ASSERTS that the
    per-segment counts it reproduces match the ones already in the JSON, so the
    CSVs cannot silently drift from the report.

Run after the pipeline:
    python run_analytics.py            # writes out/analytics_metrics.json
    python run_complete.py             # writes out/complete_metrics.json
    python export_bi.py                # writes bi_export/*.csv
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import clv as CLV  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
BI = os.path.join(HERE, "bi_export")
K_SEGMENTS = 5


def _load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def _write(name, fieldnames, rows):
    path = os.path.join(BI, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("  %-34s %4d rows" % (name, len(rows)))


def _num(x):
    """JSON carries Infinity; CSV/BI tools want a finite cell or blank."""
    if x is None:
        return ""
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return ""
    return x


def recompute_customer_level():
    """Reproduce, deterministically, the per-customer segment label and predicted
    CLV that run_analytics.py computes -- so the customer-level CSVs match the
    report rather than being an independent second opinion."""
    txn = np.load(os.path.join(DATA, "transactions.npy"))
    with open(os.path.join(DATA, "TRUTH.json")) as f:
        truth = json.load(f)
    cal_end = float(truth["calibration_days"])
    obs_end = float(truth["observation_days"])
    n_cust = int(txn[:, 0].max()) + 1

    s = CLV.summarise(txn, cal_end)
    hold_n, hold_spend = CLV.holdout_counts(txn, cal_end, obs_end, n_cust)
    recency = s["T"] - s["t_x"]

    feats = pd.DataFrame(dict(recency=recency, frequency=s["x"],
                             monetary=s["monetary"], tenure=s["T"]))
    extra = []
    for c in range(n_cust):
        rows = txn[(txn[:, 0] == c) & (txn[:, 1] <= cal_end)]
        extra.append((rows[:, 3].mean() if len(rows) else 0.0,
                      rows[:, 4].mean() if len(rows) else 0.0))
    extra = np.array(extra)
    feats["product_breadth"] = extra[:, 0]
    feats["return_rate"] = extra[:, 1]

    X = StandardScaler().fit_transform(feats.to_numpy())
    labels = KMeans(n_clusters=K_SEGMENTS, n_init=10,
                    random_state=0).fit(X).labels_

    bg = CLV.BGNBD().fit(s["x"], s["t_x"], s["T"])
    horizon = obs_end - cal_end
    pred_n = bg.expected_purchases(horizon, s["x"], s["t_x"], s["T"])
    gg = CLV.GammaGamma().fit(s["x"], s["monetary"])
    pred_clv = pred_n * gg.expected_value(s["x"], s["monetary"])

    return dict(labels=labels, pred_clv=pred_clv, hold_spend=hold_spend,
                hold_n=hold_n, recency=recency, freq=s["x"],
                monetary=s["monetary"], n_cust=n_cust)


def main():
    os.makedirs(BI, exist_ok=True)
    am = _load("analytics_metrics.json")
    cm = _load("complete_metrics.json")

    print("Recomputing customer-level frames from src.clv ...")
    cl = recompute_customer_level()
    labels, pred_clv = cl["labels"], cl["pred_clv"]

    # --- faithfulness check: recomputed segment sizes must equal the JSON ---
    fwd = am["segmentation"]["forward"]
    json_n = sorted(int(v["n"]) for v in fwd.values())
    recomp_n = sorted(int((labels == k).sum()) for k in range(K_SEGMENTS))
    assert json_n == recomp_n, (
        "recomputed segment sizes %s != report %s" % (recomp_n, json_n))
    print("  faithfulness check ok: segment sizes match report %s" % json_n)

    print("Writing CSVs to bi_export/ ...")

    # === 1. SEGMENTATION ==================================================
    # per-segment aggregates from the report, plus CLV share recomputed here
    seg_clv_total = {k: float(pred_clv[labels == k].sum())
                     for k in range(K_SEGMENTS)}
    grand = sum(seg_clv_total.values())
    n_all = cl["n_cust"]
    rows = []
    for k in sorted(fwd, key=lambda x: int(x)):
        v = fwd[k]
        ki = int(k)
        rows.append(dict(
            segment="Segment %d" % ki,
            customers=int(v["n"]),
            share_of_customers=round(v["n"] / n_all, 4),
            cal_frequency=round(v["cal_frequency"], 3),
            cal_monetary=round(v["cal_monetary"], 3),
            recency_days=round(v["recency"], 1),
            churn_rate=round(v["churn_rate_T1"], 4),
            holdout_orders=round(v["holdout_orders"], 3),
            holdout_spend=round(v["holdout_spend"], 2),
            clv_total=round(seg_clv_total[ki], 2),
            clv_mean=round(seg_clv_total[ki] / v["n"], 2),
            share_of_value=round(seg_clv_total[ki] / grand, 4)))
    _write("segments.csv",
           ["segment", "customers", "share_of_customers", "cal_frequency",
            "cal_monetary", "recency_days", "churn_rate", "holdout_orders",
            "holdout_spend", "clv_total", "clv_mean", "share_of_value"], rows)

    # k-selection sweep
    ks = cm["k_selection"]["table"]
    _write("k_selection.csv",
           ["k", "silhouette", "calinski_harabasz", "davies_bouldin", "inertia",
            "mean_ari", "adjusted_eta_squared", "smallest_cluster_share"],
           [dict(k=r["k"], silhouette=r["silhouette"],
                 calinski_harabasz=r["calinski_harabasz"],
                 davies_bouldin=r["davies_bouldin"], inertia=round(r["inertia"], 1),
                 mean_ari=r["mean_ari"],
                 adjusted_eta_squared=r["adjusted_eta_squared"],
                 smallest_cluster_share=r["smallest_cluster_share"]) for r in ks])

    picks = cm["k_selection"]["picks"]
    _write("k_selection_picks.csv", ["criterion", "k_chosen"],
           [dict(criterion=c, k_chosen=k) for c, k in picks.items()])

    _write("hdbscan_sweep.csv",
           ["min_cluster_size", "n_clusters", "noise_share", "largest_cluster"],
           [dict(min_cluster_size=r["min_cluster_size"], n_clusters=r["n_clusters"],
                 noise_share=round(r["noise_share"], 4), largest_cluster=r["largest"])
            for r in cm["k_selection"]["hdbscan"]])

    # === 2. CLV ==========================================================
    clv = am["clv"]
    kv = [("BG/NBD r", clv["params"]["r"]), ("BG/NBD alpha", clv["params"]["alpha"]),
          ("BG/NBD a", clv["params"]["a"]), ("BG/NBD b", clv["params"]["b"]),
          ("Gamma-Gamma p", clv["params"]["p"]), ("Gamma-Gamma q", clv["params"]["q"]),
          ("Gamma-Gamma v", clv["params"]["v"]),
          ("Spearman (rank, holdout)", clv["spearman"]),
          ("Pearson (individual)", clv["individual_pearson"]),
          ("Corr(frequency, order value)", clv["freq_monetary_corr"]),
          ("Top-20% share of value", clv["top20_value_share"])]
    _write("clv_summary.csv", ["metric", "value"],
           [dict(metric=m, value=round(v, 4)) for m, v in kv])

    _write("clv_model_comparison.csv", ["model", "spearman", "mae"],
           [dict(model=m, spearman=v["spearman"], mae=v["mae"])
            for m, v in clv["challenger"].items()])

    cv = am["clv_handoff"]["channel_value"]
    _write("clv_by_channel.csv",
           ["channel", "converters_touched", "mean_clv", "median_clv",
            "total_clv", "clv_index"],
           [dict(channel=ch, converters_touched=v["converters_touched"],
                 mean_clv=round(v["mean_clv"], 2), median_clv=round(v["median_clv"], 2),
                 total_clv=round(v["total_clv"], 2), clv_index=v["clv_index"])
            for ch, v in cv.items()])

    # per-customer CLV for a distribution / concentration chart
    rows = [dict(customer_id=i, segment="Segment %d" % int(labels[i]),
                 predicted_clv=round(float(pred_clv[i]), 2),
                 holdout_spend=round(float(cl["hold_spend"][i]), 2),
                 cal_frequency=int(cl["freq"][i]),
                 recency_days=round(float(cl["recency"][i]), 1))
            for i in range(n_all)]
    _write("clv_per_customer.csv",
           ["customer_id", "segment", "predicted_clv", "holdout_spend",
            "cal_frequency", "recency_days"], rows)

    # === 3. ATTRIBUTION ==================================================
    attr = am["attribution"]
    truth = attr["** TRUTH **"]
    channels = [c for c in truth if not c.startswith("_")]
    long_rows = []
    for method, credits in attr.items():
        if method == "** TRUTH **":
            continue
        for ch in channels:
            long_rows.append(dict(
                method=method, channel=ch, credit=round(credits[ch], 4),
                truth=round(truth[ch], 4),
                error=round(credits[ch] - truth[ch], 4)))
    # include truth as its own "method" row so a chart can show it as a series
    for ch in channels:
        long_rows.append(dict(method="truth", channel=ch,
                              credit=round(truth[ch], 4), truth=round(truth[ch], 4),
                              error=0.0))
    _write("attribution_credit_long.csv",
           ["method", "channel", "credit", "truth", "error"], long_rows)

    _write("attribution_mae.csv", ["method", "mae"],
           sorted([dict(method=m, mae=round(v["_mae"], 4))
                   for m, v in attr.items() if m != "** TRUTH **"],
                  key=lambda r: r["mae"]))

    pc = am["planted_channel"]
    _write("attribution_zero_effect_credit.csv",
           ["method", "credit_to_zero_effect_channel", "truth"],
           sorted([dict(method=m, credit_to_zero_effect_channel=round(v, 4), truth=0.0)
                   for m, v in pc["credited"].items()],
                  key=lambda r: r["credit_to_zero_effect_channel"]))

    # budget outcomes: merge conversions view with the value-weighted view
    budget = am["budget"]
    alloc = am["clv_handoff"]["allocations"]
    rows = []
    for m in budget:
        b = budget[m]
        a = alloc.get(m, {})
        rows.append(dict(
            allocation=m,
            conversions=round(b["conversions"], 1),
            conversions_lost_vs_truth=round(b["conversions_lost_vs_truth"], 1),
            pct_conversions_lost=round(b["pct_lost"], 3),
            spend_on_zero_effect=round(b["spend_on_zero_effect"], 2),
            customer_value=round(a.get("customer_value", float("nan")), 0)
            if a else "",
            pct_value_lost=round(a.get("value_lost_pct", float("nan")), 3)
            if a else ""))
    _write("budget_outcomes.csv",
           ["allocation", "conversions", "conversions_lost_vs_truth",
            "pct_conversions_lost", "spend_on_zero_effect", "customer_value",
            "pct_value_lost"], rows)

    # economics: observational + incremental CAC/ROAS per channel
    obs = {r["channel"]: r for r in cm["economics"]["observational"]}
    inc = {r["channel"]: r for r in cm["economics"]["incremental"]}
    rows = []
    for ch in obs:
        o, i = obs[ch], inc.get(ch, {})
        rows.append(dict(
            channel=ch, spend=round(o["spend"], 2),
            conversions_touched=o["conversions_touched"],
            observational_cac=round(o["cac"], 4),
            incremental_cac=_num(round(i["incremental_cac"], 4)
                                 if "incremental_cac" in i and
                                 math.isfinite(i["incremental_cac"]) else
                                 i.get("incremental_cac")),
            observational_roas=round(o["roas"], 2),
            incremental_roas=round(i.get("incremental_roas", float("nan")), 2)
            if i else ""))
    _write("economics.csv",
           ["channel", "spend", "conversions_touched", "observational_cac",
            "incremental_cac", "observational_roas", "incremental_roas"], rows)

    # shapley convergence: the diagnosis (exact-set plateau) vs the fix (closure)
    fix = cm["shapley_fix"]
    closure = {r["n_perms"]: r["mean_abs_error"] for r in fix["curve"]}
    exact = {r["n_perms"]: r["mean_abs_error"] for r in fix["curve_exact_set"]}
    _write("shapley_convergence.csv",
           ["n_perms", "exact_set_error", "closure_error"],
           [dict(n_perms=n, exact_set_error=round(exact[n], 5),
                 closure_error=round(closure[n], 5))
            for n in sorted(closure)])

    _write("shapley_method_mae.csv", ["method", "mae"],
           sorted([dict(method=m, mae=round(v, 4)) for m, v in fix["mae"].items()],
                  key=lambda r: r["mae"]))

    # === CONSOLIDATED FILES FOR THE DASHBOARD =============================
    # method_scores: attribution error and confound-credit in one table so a
    # single BI data source powers both method charts.
    credited = pc["credited"]
    _write("method_scores.csv",
           ["method", "mae", "credit_to_zero_effect"],
           sorted([dict(method=m, mae=round(v["_mae"], 4),
                        credit_to_zero_effect=round(credited.get(m, float("nan")), 4))
                   for m, v in attr.items() if m != "** TRUTH **"],
                  key=lambda r: r["mae"]))

    # kpis: single wide row for scorecards spanning all three result areas.
    best_mae = min(v["_mae"] for m, v in attr.items() if m != "** TRUTH **")
    _write("kpis.csv",
           ["total_customers", "n_segments", "top20pct_value_share",
            "clv_rank_spearman", "best_attribution_mae", "n_channels"],
           [dict(total_customers=n_all, n_segments=K_SEGMENTS,
                 top20pct_value_share=round(clv["top20_value_share"], 4),
                 clv_rank_spearman=round(clv["spearman"], 4),
                 best_attribution_mae=round(best_mae, 4),
                 n_channels=len(channels))])

    print("\nDone. %d CSVs in bi_export/." %
          len([f for f in os.listdir(BI) if f.endswith(".csv")]))


if __name__ == "__main__":
    main()
