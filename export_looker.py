"""Tidy CSVs for the Looker Studio / Tableau dashboards -> bi_export/dashboard/.

Built from the same data as the web dashboard (build_dashboard.build_data), so
the three dashboards show identical numbers. Run after export_bi.py.
"""
import csv
import os

from build_dashboard import build_data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bi_export", "dashboard")
METHOD = {"last_touch": "Last touch", "first_touch": "First touch", "linear": "Linear",
          "time_decay": "Time decay", "markov_removal": "Markov", "shapley": "Shapley",
          "shapley_sampled": "Sampled Shapley", "TRUTH": "True effects",
          "CLV-WEIGHTED TRUTH": "True effects, value-weighted"}


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {name:28s} {len(rows)} rows")


def main():
    d = build_data()
    write("segments.csv", [dict(
        segment=s["name"], segment_id=int(s["id"]), customers=int(s["customers"]),
        share_of_customers=round(s["share_of_customers"], 4),
        share_of_value=round(s["share_of_value"], 4),
        orders_so_far=round(s["cal_frequency"], 2), days_since_last=round(s["recency_days"]),
        no_purchase_next_6m=round(s["churn_rate"], 4),
        avg_spend_next_6m_gbp=round(s["holdout_spend"], 2),
        predicted_value_gbp=round(s["clv_total"], 2)) for s in d["segments"]])
    write("clv_deciles.csv", [dict(
        decile=f"D{r['d']:02d}", predicted_gbp=round(r["predicted"], 2),
        actual_gbp=round(r["actual"], 2),
        ratio=round(r["predicted"] / r["actual"], 3),
        direction="Over-predicts" if r["predicted"] >= r["actual"] else "Under-predicts")
        for r in d["deciles"]])
    write("clv_models.csv", [dict(model=k, rank_correlation=v["spearman"], mae_gbp=v["mae"])
                             for k, v in d["clv"]["models"].items()])
    write("zero_effect_credit.csv", [dict(
        method=METHOD.get(c["method"], c["method"]), credit=c["credit"], correct_credit=0.0)
        for c in d["zero"]["credit"]])
    write("channel_value.csv", [dict(
        channel=c["channel"].replace("_", " "), value_index=c["clv_index"],
        is_zero_effect="Zero-effect channel" if c["channel"] == d["zero"]["channel"]
        else "Other channels",
        converters_reached=int(c["converters_touched"])) for c in d["channels"]])
    write("budget.csv", [dict(
        budget_split_by=METHOD.get(b["allocation"], b["allocation"]),
        error_vs_true_credit=b["mae"] if b["mae"] is not None else "",
        sales_lost_pct=round(b["pct_conversions_lost"] / 100, 4),
        spent_on_zero_effect_gbp=round(b["spend_on_zero_effect"], 2))
        for b in d["budget"]])
    c = d["clean"]
    write("kpis.csv", [dict(
        customers=c["cohort_customers"], purchase_days=c["cohort_occasions"],
        raw_invoice_lines=c["raw_lines"], top20_value_share=round(d["clv"]["top20"], 4),
        clv_rank_correlation=round(d["clv"]["spearman"], 4),
        last_touch_credit_to_zero_effect=next(
            x["credit"] for x in d["zero"]["credit"] if x["method"] == "last_touch"))])


if __name__ == "__main__":
    main()
