"""Build the public dashboard (docs/index.html) from the pipeline's own output.

Run after run_analytics.py, run_complete.py and export_bi.py. Every number on the
page is read from out/*.json, bi_export/*.csv and data/cleaning_report.json.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _json(*p):
    with open(os.path.join(HERE, *p), encoding="utf-8") as f:
        return json.load(f)


def _csv(name):
    with open(os.path.join(HERE, "bi_export", name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(rows):
    out = []
    for r in rows:
        o = {}
        for k, v in r.items():
            try:
                o[k] = float(v)
            except (TypeError, ValueError):
                o[k] = v
        out.append(o)
    return out


def name_segments(segs):
    """Plain names from each segment's own numbers, so a re-run that renumbers
    the k-means labels still names them correctly."""
    left = {s["segment"]: s for s in segs}
    names = {}

    def take(key, label, pick=max, pool=None):
        pool = pool or list(left.values())
        s = pick(pool, key=lambda r: r[key])
        names[s["segment"]] = label
        left.pop(s["segment"])

    take("cal_monetary", "Wholesale accounts")
    take("cal_frequency", "Regulars")
    ones = [s for s in left.values() if s["cal_frequency"] < 1.5]
    if len(ones) >= 2:
        take("recency_days", "Lapsed one-timers", pool=ones)
        ones = [s for s in left.values() if s["cal_frequency"] < 1.5]
        take("recency_days", "Recent one-timers", pick=min, pool=ones)
    for s in list(left.values()):
        names[s["segment"]] = "Occasional"
        left.pop(s["segment"])
    return names


def build_data():
    am = _json("out", "analytics_metrics.json")
    cm = _json("out", "complete_metrics.json")
    clean = _json("data", "cleaning_report.json")
    stats = _json("data", "stats.json")
    segs = _num(_csv("segments.csv"))
    names = name_segments(segs)
    for s in segs:
        s["name"] = names[s["segment"]]
        s["id"] = s["segment"].replace("Segment ", "")

    per = _num(_csv("clv_per_customer.csv"))
    pred = np.array([r["predicted_clv"] for r in per])
    act = np.array([r["holdout_spend"] for r in per])
    dec = np.floor(10 * np.argsort(np.argsort(pred)) / len(pred)).astype(int)
    deciles = [dict(d=int(d + 1), predicted=float(pred[dec == d].mean()),
                    actual=float(act[dec == d].mean())) for d in range(10)]

    attr = am["attribution"]
    truth = attr["** TRUTH **"]
    zc = am["planted_channel"]["channel"]
    zero_credit = [dict(method=m, credit=v[zc]) for m, v in attr.items()
                   if m != "** TRUTH **"]
    zero_credit.sort(key=lambda r: -r["credit"])

    budget = _num(_csv("budget_outcomes.csv"))
    methods = {r["method"]: r for r in _num(_csv("method_scores.csv"))}
    for b in budget:
        m = methods.get(b["allocation"])
        b["mae"] = m["mae"] if m else None

    chan = _num(_csv("clv_by_channel.csv"))
    chan.sort(key=lambda r: -r["clv_index"])

    fwd = cm.get("k_selection", {}).get("table", [])
    return dict(
        clean=clean, stats=stats, segments=segs, deciles=deciles,
        clv=dict(spearman=am["clv"]["spearman"], pearson=am["clv"]["individual_pearson"],
                 top20=am["clv"]["top20_value_share"],
                 freq_value_corr=am["clv"]["freq_monetary_corr"],
                 models=am["clv"]["challenger"]),
        seg_meta=dict(ari=am["segmentation"]["ari_mean"],
                      ari_sweep=next((r["mean_ari"] for r in fwd if r["k"] == len(segs)), None)),
        k_table=[dict(k=r["k"], adj=r["adjusted_eta_squared"]) for r in fwd],
        zero=dict(channel=zc, credit=zero_credit, n_channels=len(truth) - 1,
                  lift=am["planted_channel"]["apparent_lift"]),
        budget=budget, channels=chan,
    )


def main():
    data = build_data()
    with open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8") as f:
        page = f.read()
    page = page.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    with open(os.path.join(HERE, "docs", "index.html"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(page)
    print(f"wrote docs/index.html ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
