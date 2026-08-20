"""Who are they (segmentation) -> what are they worth (CLV) -> what made them buy
(attribution). One dataset, explicit handoffs.

The handoffs are the point: segments feed CLV priors, CLV weights the attribution
decision, and the whole thing lands as a budget recommendation with the cost of
getting it wrong stated in conversions.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import attribution as A  # noqa: E402
from src import clv as CLV  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
K_SEGMENTS = 5


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    lines, summary = [], {}

    def emit(s=""):
        print(s)
        lines.append(s)

    txn = np.load(os.path.join(DATA, "transactions.npy"))
    with open(os.path.join(DATA, "journeys.json")) as f:
        jd = json.load(f)
    with open(os.path.join(DATA, "TRUTH.json")) as f:
        truth = json.load(f)
    journeys, conversions = jd["journeys"], jd["conversions"]
    cal_end = float(truth["calibration_days"])
    obs_end = float(truth["observation_days"])
    n_cust = int(txn[:, 0].max()) + 1

    emit("%d customers, %d transactions. Calibration 0-%d days, holdout %d-%d."
         % (n_cust, len(txn), cal_end, cal_end, obs_end))

    s = CLV.summarise(txn, cal_end)
    hold_n, hold_spend = CLV.holdout_counts(txn, cal_end, obs_end, n_cust)
    emit("Calibration repeat purchases: mean %.2f, %d customers with zero."
         % (s["x"].mean(), int((s["x"] == 0).sum())))

    # ==================================================================
    emit("")
    emit("=" * 78)
    emit("1. SEGMENTATION -- AND WHETHER THE SEGMENTS PREDICT ANYTHING")
    emit("=" * 78)
    recency = s["T"] - s["t_x"]
    feats = pd.DataFrame(dict(
        recency=recency, frequency=s["x"], monetary=s["monetary"],
        tenure=s["T"]))
    extra = []
    for c in range(n_cust):
        rows = txn[(txn[:, 0] == c) & (txn[:, 1] <= cal_end)]
        extra.append((rows[:, 3].mean() if len(rows) else 0.0,
                      rows[:, 4].mean() if len(rows) else 0.0))
    extra = np.array(extra)
    feats["category_breadth"] = extra[:, 0]
    feats["discount_affinity"] = extra[:, 1]

    # RFM quintiles: the transparent baseline everyone builds
    rfm = pd.DataFrame(index=feats.index)
    rfm["R"] = pd.qcut(-feats.recency, 5, labels=False, duplicates="drop") + 1
    rfm["F"] = pd.qcut(feats.frequency.rank(method="first"), 5,
                       labels=False, duplicates="drop") + 1
    rfm["M"] = pd.qcut(feats.monetary.rank(method="first"), 5,
                       labels=False, duplicates="drop") + 1
    rfm_seg = (rfm.R.astype(int) * 100 + rfm.F.astype(int) * 10 + rfm.M.astype(int))

    X = StandardScaler().fit_transform(feats.to_numpy())
    km = KMeans(n_clusters=K_SEGMENTS, n_init=10, random_state=0).fit(X)
    labels = km.labels_

    # stability: does the partition survive resampling?
    rng = np.random.default_rng(0)
    aris = []
    for _ in range(20):
        idx = rng.choice(n_cust, n_cust, replace=True)
        km_b = KMeans(n_clusters=K_SEGMENTS, n_init=5,
                      random_state=0).fit(X[idx])
        aris.append(adjusted_rand_score(labels[idx], km_b.labels_))
    emit("Bootstrap stability (adjusted Rand index over 20 resamples):")
    emit("  mean %.3f   min %.3f   max %.3f"
         % (np.mean(aris), np.min(aris), np.max(aris)))
    emit("")

    # FORWARD VALIDATION -- the bit that is always missing
    seg_df = pd.DataFrame(dict(segment=labels, holdout_orders=hold_n,
                               holdout_spend=hold_spend,
                               churned=(hold_n == 0).astype(float),
                               cal_freq=s["x"], cal_monetary=s["monetary"],
                               recency=recency))
    fwd = seg_df.groupby("segment").agg(
        n=("churned", "size"), cal_frequency=("cal_freq", "mean"),
        cal_monetary=("cal_monetary", "mean"), recency=("recency", "mean"),
        churn_rate_T1=("churned", "mean"),
        holdout_orders=("holdout_orders", "mean"),
        holdout_spend=("holdout_spend", "mean")).round(3)
    emit("FORWARD TEST -- segment membership at time T vs behaviour in T+1:")
    emit(fwd.to_string())
    spread = fwd.churn_rate_T1.max() - fwd.churn_rate_T1.min()
    emit("")
    emit("Churn-rate spread across segments: %.1f%% to %.1f%% (%.1f points)."
         % (100 * fwd.churn_rate_T1.min(), 100 * fwd.churn_rate_T1.max(),
            100 * spread))
    emit("Holdout spend spread: $%.0f to $%.0f."
         % (fwd.holdout_spend.min(), fwd.holdout_spend.max()))
    emit("")
    emit("THIS IS THE VALIDATION THAT IS ALWAYS MISSING. Segments that do not")
    emit("separate FUTURE behaviour are decoration -- you can always partition a")
    emit("cloud of points and give the parts names. The question is whether")
    emit("membership at time T tells you anything about T+1, and the only way to")
    emit("answer it is to hold out time and look.")
    emit("")
    emit("OPERATIONALLY, what to do with a %.0f-point churn spread: it is a"
         % (100 * spread))
    emit("targeting prior, not a playbook. It says where retention budget has the")
    emit("most headroom, and it says nothing about what intervention works -- that")
    emit("needs a test per segment. And note the reflexive problem: acting on a")
    emit("segment CHANGES it, so the churn rates above are pre-intervention")
    emit("baselines that stop being true the moment anyone uses them.")
    summary["segmentation"] = dict(
        ari_mean=float(np.mean(aris)), ari_min=float(np.min(aris)),
        churn_spread=float(spread), forward=fwd.to_dict("index"))

    # ==================================================================
    emit("")
    emit("=" * 78)
    emit("2. CLV -- BG/NBD + GAMMA-GAMMA, VALIDATED ON A TEMPORAL HOLDOUT")
    emit("=" * 78)
    bg = CLV.BGNBD().fit(s["x"], s["t_x"], s["T"])
    emit("BG/NBD fitted: r=%.4f alpha=%.4f a=%.4f b=%.4f"
         % (bg.r, bg.alpha, bg.a, bg.b))
    horizon = obs_end - cal_end
    pred_n = bg.expected_purchases(horizon, s["x"], s["t_x"], s["T"])

    gg = CLV.GammaGamma().fit(s["x"], s["monetary"])
    emit("Gamma-Gamma fitted: p=%.4f q=%.4f v=%.4f" % (gg.p, gg.q, gg.v))
    pred_v = gg.expected_value(s["x"], s["monetary"])
    pred_clv = pred_n * pred_v

    # the independence assumption, TESTED rather than assumed
    mask = s["x"] > 0
    corr = float(np.corrcoef(s["x"][mask], s["monetary"][mask])[0, 1])
    emit("")
    emit("Gamma-Gamma assumes monetary value is independent of frequency.")
    emit("Measured correlation(frequency, mean order value) = %+.4f" % corr)
    emit("  -> %s" % ("assumption holds well enough" if abs(corr) < 0.1
                      else "ASSUMPTION VIOLATED; spend estimates will be biased"))

    dec = pd.qcut(pred_clv.argsort().argsort(), 10, labels=False)
    actual_clv = hold_spend
    cal = pd.DataFrame(dict(decile=dec, predicted=pred_clv, actual=actual_clv,
                            pred_n=pred_n, actual_n=hold_n))
    calib = cal.groupby("decile").agg(
        n=("predicted", "size"), predicted_clv=("predicted", "mean"),
        actual_clv=("actual", "mean"), predicted_orders=("pred_n", "mean"),
        actual_orders=("actual_n", "mean")).round(2)
    calib["ratio"] = (calib.predicted_clv / calib.actual_clv.replace(0, np.nan)).round(3)
    emit("")
    emit("DECILE CALIBRATION on the temporal holdout:")
    emit(calib.to_string())
    rank_corr = float(pd.Series(pred_clv).corr(pd.Series(actual_clv), method="spearman"))
    indiv_corr = float(np.corrcoef(pred_clv, actual_clv)[0, 1])
    emit("")
    emit("Spearman rank correlation (predicted vs actual holdout spend): %.4f" % rank_corr)
    emit("Pearson correlation on individuals:                            %.4f" % indiv_corr)
    emit("")
    emit("THE AGGREGATE-VS-INDIVIDUAL NOTE, AND THE CAVEAT ON THE CAVEAT.")
    emit("")
    emit("The textbook warning about BG/NBD is that it RANKS populations well and")
    emit("MISPREDICTS individuals -- use it to size a segment, never to decide what")
    emit("one customer is worth. That warning is correct on real data.")
    emit("")
    emit("It is NOT what this run shows. The decile table tracks closely AND the")
    emit("individual correlation is %.2f, which is high. Reporting that as evidence"
         % indiv_corr)
    emit("that the model predicts individuals well would be the single most")
    emit("misleading thing in this project, because the reason is circular: the")
    emit("generator in src/generate.py draws inter-purchase times from an")
    emit("exponential with a Gamma-distributed rate and applies a Beta-distributed")
    emit("dropout after each purchase. That IS the BG/NBD process. The model is")
    emit("being scored on data that satisfies its assumptions exactly.")
    emit("")
    emit("So what this section actually validates is that the ESTIMATOR recovers")
    emit("the parameters and the calibration harness works -- which is worth")
    emit("knowing and is not the same claim. On real transactions, where customers")
    emit("are seasonal, subscription-like, or promotion-driven, the individual")
    emit("correlation collapses and the decile table is the only part that")
    emit("survives. The honest use of this build is the harness, not the number.")

    # ML challenger
    Xc = np.column_stack([s["x"], s["t_x"], s["T"], recency, s["monetary"],
                          extra[:, 0], extra[:, 1]])
    n_tr = int(0.7 * n_cust)
    idx = np.random.default_rng(1).permutation(n_cust)
    tr, te = idx[:n_tr], idx[n_tr:]
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                        random_state=0).fit(Xc[tr], actual_clv[tr])
    gbm_pred = gbm.predict(Xc[te])
    cmp = pd.DataFrame([
        dict(model="BG/NBD + Gamma-Gamma",
             spearman=float(pd.Series(pred_clv[te]).corr(
                 pd.Series(actual_clv[te]), method="spearman")),
             mae=float(np.mean(np.abs(pred_clv[te] - actual_clv[te])))),
        dict(model="GBM challenger",
             spearman=float(pd.Series(gbm_pred).corr(
                 pd.Series(actual_clv[te]), method="spearman")),
             mae=float(np.mean(np.abs(gbm_pred - actual_clv[te])))),
    ]).set_index("model")
    emit("")
    emit("PROBABILISTIC MODEL vs ML CHALLENGER (same holdout, 30% of customers):")
    emit(cmp.to_string(float_format=lambda x: "%10.3f" % x))
    emit("")
    winner = cmp.spearman.idxmax()
    emit("Better ranking: %s." % winner)
    emit("RECOMMENDATION: the probabilistic model, and not because it won on")
    emit("every number. It gives P(alive) and a purchase-count distribution rather")
    emit("than a point estimate, it extrapolates to horizons it never saw, and its")
    emit("four parameters are interpretable enough to argue with. The GBM needs a")
    emit("labelled future to train on, which means it can only ever predict")
    emit("horizons you have already lived through -- and it silently relearns")
    emit("whatever selection is in the label window.")

    # CLV deciles wired back to segments -- the first handoff
    conc = pd.DataFrame(dict(segment=labels, clv=pred_clv))
    by_seg = conc.groupby("segment").clv.agg(["size", "mean", "sum"])
    by_seg["share_of_value"] = (by_seg["sum"] / by_seg["sum"].sum()).round(4)
    by_seg["share_of_customers"] = (by_seg["size"] / by_seg["size"].sum()).round(4)
    emit("")
    emit("HANDOFF 1 -- CLV wired back into segments:")
    emit(by_seg.round(2).to_string())
    top = conc.clv.sort_values(ascending=False)
    conc_20 = float(top.head(int(0.2 * n_cust)).sum() / top.sum())
    emit("")
    emit("Top 20%% of customers hold %.1f%% of predicted value." % (100 * conc_20))
    summary["clv"] = dict(
        params=dict(r=bg.r, alpha=bg.alpha, a=bg.a, b=bg.b,
                    p=gg.p, q=gg.q, v=gg.v),
        spearman=rank_corr, individual_pearson=indiv_corr,
        freq_monetary_corr=corr, top20_value_share=conc_20,
        challenger=cmp.round(4).to_dict("index"))

    # ==================================================================
    emit("")
    emit("=" * 78)
    emit("3. ATTRIBUTION -- SCORED AGAINST KNOWN TRUTH")
    emit("=" * 78)
    channels = list(truth["channel_effects"])
    true_share = truth["true_effect_share"]
    rows = []
    for name, fn in A.METHODS.items():
        credit = fn(journeys, conversions, channels)
        err = {ch: credit[ch] - true_share[ch] for ch in channels}
        rows.append(dict(method=name, **credit,
                         _mae=float(np.mean([abs(e) for e in err.values()]))))
    At = pd.DataFrame(rows).set_index("method")
    truth_row = pd.DataFrame([dict(method="** TRUTH **", **true_share, _mae=0.0)]
                             ).set_index("method")
    show = pd.concat([truth_row, At])
    emit("Credited share of conversions by channel (rows sum to 1):")
    emit(show.to_string(float_format=lambda x: "%9.4f" % x))
    emit("")
    emit("Mean absolute error against truth, ranked:")
    for m, v in At._mae.sort_values().items():
        emit("  %-18s %.4f" % (m, v))
    summary["attribution"] = show.round(4).to_dict("index")

    emit("")
    emit("=" * 78)
    emit("4. THE PLANTED CHANNEL -- ATTRIBUTION IS NOT INCREMENTALITY")
    emit("=" * 78)
    zc = truth["zero_effect_channel"]
    emit("`%s` has a TRUE causal effect of EXACTLY ZERO. It was targeted at" % zc)
    emit("users who were already likely to convert -- which is what a real")
    emit("retargeting programme does: it follows intent, it does not create it.")
    emit("")
    emit("Naive read of the data:")
    with_r = [conversions[i] for i, j in enumerate(journeys) if zc in j]
    without_r = [conversions[i] for i, j in enumerate(journeys) if zc not in j]
    emit("  conversion rate WITH %s    : %.4f" % (zc, np.mean(with_r)))
    emit("  conversion rate WITHOUT %s : %.4f" % (zc, np.mean(without_r)))
    emit("  apparent 'lift'              : %+.4f  (TRUE lift: 0.0000)"
         % (np.mean(with_r) - np.mean(without_r)))
    emit("")
    emit("What each attribution method credits it:")
    for m in At.index:
        emit("  %-18s %.4f   (truth 0.0000)" % (m, At.loc[m, zc]))
    emit("")
    emit("EVERY method credits it, including markov_removal -- which sounds causal")
    emit("and is not. 'Remove the channel from the graph' is a statement about the")
    emit("observed paths, not about the world: it assumes the users who saw that")
    emit("channel would otherwise have walked the same graph minus one node. When")
    emit("the channel was TARGETED at high-intent users, removing it also removes")
    emit("the intent that arrived with it, and the method charges that intent to")
    emit("the channel.")
    emit("")
    emit("WHAT WOULD ACTUALLY SETTLE IT -- the experiment, sized:")
    emit("  Design: geo holdout. Split matched markets, switch %s OFF in the" % zc)
    emit("  control markets, leave everything else running. Primary metric is")
    emit("  conversions per market; the estimand is the difference, which IS the")
    emit("  incremental effect and is exactly what no observational method above")
    emit("  can recover.")
    emit("  Why geo and not user-level: cookie-level holdouts leak (same person,")
    emit("  many devices) and the ad platform optimises delivery against the")
    emit("  holdout, which breaks the randomisation.")
    emit("  Cost: the foregone spend in control markets for the test duration --")
    emit("  which, if the effect really is zero, is not a cost at all. That")
    emit("  asymmetry is the pitch: the experiment is cheap precisely in the world")
    emit("  where the channel is worthless.")
    summary["planted_channel"] = dict(
        channel=zc, apparent_lift=float(np.mean(with_r) - np.mean(without_r)),
        credited={m: float(At.loc[m, zc]) for m in At.index})

    # ==================================================================
    emit("")
    emit("=" * 78)
    emit("5. THE BUDGET DECISION -- WHAT LAST-TOUCH COSTS")
    emit("=" * 78)
    # Budget sized so REACH IS NOT SATURATED. At $1m across 8,000 prospects every
    # channel reaches everyone regardless of allocation, every method ties, and
    # the section says nothing -- which is what the first version did. $1,500 is
    # ~$0.19 per prospect and leaves the expensive channels genuinely rationed,
    # so the allocation choice has consequences.
    BUDGET = 1_500.0
    base_conv = 0.10
    costs = truth["channel_costs"]
    effects = truth["channel_effects"]
    rows = []
    for name in list(A.METHODS) + ["TRUTH"]:
        credit = (true_share if name == "TRUTH"
                  else A.METHODS[name](journeys, conversions, channels))
        alloc = A.budget_allocation(credit, BUDGET)
        conv = A.conversions_under(alloc, effects, costs, base_conv, n_cust)
        rows.append(dict(allocation=name, conversions=conv,
                         spend_on_zero_effect=alloc[zc]))
    Bt = pd.DataFrame(rows).set_index("allocation")
    best = Bt.loc["TRUTH", "conversions"]
    Bt["conversions_lost_vs_truth"] = best - Bt.conversions
    Bt["pct_lost"] = 100 * Bt.conversions_lost_vs_truth / best
    emit("Budget $%.0f (~$%.2f per prospect) allocated in proportion to each"
         % (BUDGET, BUDGET / n_cust))
    emit("method's credited share,")
    emit("then evaluated in the TRUE world (concave reach, saturation 0.6):")
    emit(Bt.to_string(float_format=lambda x: "%14.2f" % x))
    emit("")
    lt = Bt.loc["last_touch"]
    emit("THE HEADLINE FOR THE MEMO: allocating on last-touch instead of truth")
    emit("costs %.0f conversions (%.1f%% of achievable) and puts $%.0f -- %.1f%% of"
         % (lt.conversions_lost_vs_truth, lt.pct_lost, lt.spend_on_zero_effect,
            100 * lt.spend_on_zero_effect / BUDGET))
    emit("the budget -- into a channel that causes nothing.")
    emit("")
    emit("AND A RESULT THAT COMPLICATES THE OBVIOUS STORY. Rank the methods by")
    emit("attribution error (section 3) and by budget outcome (here) and the")
    emit("orders DISAGREE:")
    emit("")
    emit("  %-18s %12s %14s" % ("method", "MAE vs truth", "conversions lost"))
    for m in At._mae.sort_values().index:
        emit("  %-18s %12.4f %14.1f" % (m, At.loc[m, "_mae"],
                                        Bt.loc[m, "conversions_lost_vs_truth"]))
    emit("")
    best_mae = At._mae.idxmin()
    best_budget = Bt.drop("TRUTH").conversions_lost_vs_truth.idxmin()
    if best_mae != best_budget:
        emit("Lowest attribution error: %s. Best budget outcome: %s."
             % (best_mae, best_budget))
        emit("")
        emit("They are not the same method. Getting the CREDIT SHARES closest to")
        emit("truth is not the same as making the best DECISION, because the")
        emit("decision runs the shares through channel costs and a concave reach")
        emit("curve. A method can be wrong in a direction that happens to be cheap")
        emit("-- over-crediting a channel you were going to saturate anyway costs")
        emit("nothing -- or wrong in a direction that is expensive. Choosing an")
        emit("attribution method on MAE alone optimises the wrong objective; the")
        emit("objective is the allocation it produces.")
    emit("")
    emit("HANDOFF 2 -- why CLV belongs in this decision: the allocation above")
    emit("maximises CONVERSIONS, and conversions are not equally valuable. Top")
    emit("versus bottom CLV decile on holdout spend is $%.0f against $%.0f. A"
         % (calib.actual_clv.iloc[-1], calib.actual_clv.iloc[0]))
    emit("channel that acquires cheap, low-value customers can win a")
    emit("conversion-based allocation and lose a value-based one. Weighting the")
    emit("objective by predicted CLV per acquired customer is the join between")
    emit("sections 2 and 5, and it is NOT built here -- the simulator does not")
    emit("link journeys to customer ids, which is the single biggest structural")
    emit("gap in this project.")
    summary["budget"] = Bt.round(3).to_dict("index")

    emit("")
    emit("(%.0fs)" % (time.time() - t0))
    with open(os.path.join(OUT, "analytics_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "analytics_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/analytics_report.txt")


if __name__ == "__main__":
    main()
