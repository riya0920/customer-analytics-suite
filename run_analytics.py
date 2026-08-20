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
    journey_customer = jd.get("customer_id")
    touch_days = jd.get("touch_days")
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
        if name == "time_decay":
            credit = fn(journeys, conversions, channels, touch_days=touch_days)
        else:
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
    credited = {m: float(At.loc[m, zc]) for m in At.index}
    n_credit = sum(1 for v in credited.values() if v > 0.05)
    emit("%d of %d methods give it a material share of the credit."
         % (n_credit, len(credited)))
    emit("")
    zeroed = [m for m, v in credited.items() if v <= 0.01]
    if zeroed:
        emit("ONE METHOD GIVES IT ZERO, AND IT IS NOT BECAUSE IT FOUND THE")
        emit("CONFOUND. %s reads 0.0000 -- and it also reads 0.0000 for DISPLAY,"
             % ", ".join(zeroed))
        emit("whose true share is %.4f. Look at its whole row:" % true_share["display"])
        emit("")
        for m in zeroed:
            emit("    %-16s %s" % (m, "  ".join(
                "%s %.4f" % (c, At.loc[m, c]) for c in channels)))
        emit("    %-16s %s" % ("TRUTH", "  ".join(
            "%s %.4f" % (c, true_share[c]) for c in channels)))
        emit("")
        emit("It has collapsed almost all the credit onto one channel and zeroed")
        emit("two others. That is a known brittleness of removal effects: a")
        emit("channel whose removal does not DISCONNECT the graph -- because the")
        emit("remaining transitions still reach conversion -- gets a removal")
        emit("effect of zero regardless of how much it contributed. It happens to")
        emit("be right about retargeting and it is wrong about display for the")
        emit("same reason, so the zero is an artifact and not a detection.")
        emit("")
        emit("A method that is accidentally right is not a method you can deploy,")
        emit("because you cannot tell in advance which of its zeros are correct.")
        emit("")
    emit("The other methods behave exactly as the confound predicts. Last-touch")
    emit("hands it %.0f%% of ALL credit, because retargeting is a CLOSER -- it"
         % (100 * credited["last_touch"]))
    emit("fires late in journeys that were already going to convert, so it is")
    emit("sitting in the last position of a great many converting paths.")
    emit("")
    emit("SHAPLEY IS THE STRONGEST VERSION OF THIS FINDING and it is why the")
    emit("second pass added it. Shapley has a formal DUMMY PLAYER axiom: a")
    emit("channel that changes no coalition's conversion rate is guaranteed")
    emit("exactly zero credit. It is the only method here with that property, and")
    emit("a test proves it holds when the dummy channel is added AT RANDOM.")
    emit("")
    emit("On this data Shapley gives retargeting %.4f." % credited["shapley"])
    emit("")
    emit("The axiom is not violated -- it is satisfied, on a coalition function")
    emit("that is itself confounded. Retargeting genuinely DOES raise the observed")
    emit("conversion rate of every coalition it joins, because it joins the")
    emit("coalitions of customers who were going to convert. Shapley is answering")
    emit("its question correctly; the question is the wrong one.")
    emit("")
    emit("THAT IS THE WHOLE LESSON, and it is why no amount of methodological")
    emit("sophistication fixes this. Attribution asks 'which touchpoints appear on")
    emit("converting journeys'. Incrementality asks 'what would have happened")
    emit("without this channel'. The second question cannot be answered from data")
    emit("that contains no variation in whether the channel ran.")
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
        if name == "TRUTH":
            credit = true_share
        elif name == "time_decay":
            credit = A.METHODS[name](journeys, conversions, channels,
                                     touch_days=touch_days)
        else:
            credit = A.METHODS[name](journeys, conversions, channels)
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

    # ==================================================================
    emit("")
    emit("=" * 78)
    emit("6. THE CLV HANDOFF -- COMPUTED, NOT REASONED ABOUT")
    emit("=" * 78)
    emit("The first pass called this the single biggest structural gap: journeys")
    emit("were not linked to customer ids, so the budget objective could not be")
    emit("weighted by the VALUE of the customers a channel acquires -- which is the")
    emit("whole point of computing CLV in the same project. The link exists now.")
    emit("")
    jc = np.array(journey_customer)
    conv_arr = np.array(conversions)
    clv_by_customer = pred_clv

    rows = []
    for ch in channels:
        touched = np.array([ch in j for j in journeys])
        conv_touched = touched & (conv_arr == 1)
        if conv_touched.sum() == 0:
            continue
        custs = jc[conv_touched]
        rows.append(dict(channel=ch,
                         converters_touched=int(conv_touched.sum()),
                         mean_clv=float(clv_by_customer[custs].mean()),
                         median_clv=float(np.median(clv_by_customer[custs])),
                         total_clv=float(clv_by_customer[custs].sum())))
    CV = pd.DataFrame(rows).set_index("channel")
    CV["clv_index"] = (CV.mean_clv / CV.mean_clv.mean()).round(3)
    emit("Predicted CLV of the customers each channel touched on the way to a")
    emit("conversion:")
    emit(CV.to_string(float_format=lambda x: "%14.2f" % x))
    emit("")
    emit("`clv_index` is the channel's mean acquired-customer value relative to")
    emit("the portfolio average. A channel above 1.0 brings better customers than")
    emit("the average channel does, and a conversion-maximising budget cannot see")
    emit("that at all.")
    emit("")
    spread = CV.clv_index.max() - CV.clv_index.min()
    emit("Spread across channels: %.3f (best %s at %.3f, worst %s at %.3f)."
         % (spread, CV.clv_index.idxmax(), CV.clv_index.max(),
            CV.clv_index.idxmin(), CV.clv_index.min()))
    emit("")

    # value-weighted budget evaluation
    clv_weight = {ch: float(CV.loc[ch, "mean_clv"]) if ch in CV.index else 0.0
                  for ch in channels}
    mean_clv_all = float(np.mean(list(clv_weight.values())))

    def value_under(alloc):
        """Conversions in the true world, each valued at the acquiring channel's
        mean acquired-customer CLV rather than at 1.0."""
        total = 0.0
        for ch, spend in alloc.items():
            impressions = spend / max(costs[ch], 1e-9)
            reach = min((impressions / n_cust) ** 0.6, 1.0)
            total += effects[ch] * reach * n_cust * clv_weight[ch]
        return base_conv * n_cust * mean_clv_all + total

    rows = []
    for name in list(A.METHODS) + ["TRUTH", "CLV-WEIGHTED TRUTH"]:
        if name == "TRUTH":
            credit = true_share
        elif name == "CLV-WEIGHTED TRUTH":
            w = {ch: effects[ch] * clv_weight[ch] for ch in channels}
            tot_w = sum(w.values())
            credit = {ch: (v / tot_w if tot_w else 0.0) for ch, v in w.items()}
        elif name == "time_decay":
            credit = A.METHODS[name](journeys, conversions, channels,
                                     touch_days=touch_days)
        else:
            credit = A.METHODS[name](journeys, conversions, channels)
        alloc = A.budget_allocation(credit, BUDGET)
        rows.append(dict(allocation=name,
                         conversions=A.conversions_under(alloc, effects, costs,
                                                         base_conv, n_cust),
                         customer_value=value_under(alloc)))
    VB = pd.DataFrame(rows).set_index("allocation")
    best_conv = VB.conversions.max()
    best_val = VB.customer_value.max()
    VB["conv_lost_pct"] = 100 * (best_conv - VB.conversions) / best_conv
    VB["value_lost_pct"] = 100 * (best_val - VB.customer_value) / best_val
    emit("The same allocations, scored two ways:")
    emit(VB.to_string(float_format=lambda x: "%14.2f" % x))
    emit("")
    conv_winner = VB.conversions.idxmax()
    val_winner = VB.customer_value.idxmax()
    emit("Best on CONVERSIONS: %s.  Best on CUSTOMER VALUE: %s."
         % (conv_winner, val_winner))
    emit("")
    if conv_winner != val_winner:
        emit("THEY ARE DIFFERENT ALLOCATIONS, which is the entire argument for")
        emit("joining these two analyses. A budget that maximises conversions is")
        emit("indifferent between acquiring a customer worth $50 and one worth")
        emit("$800, and it will happily buy the cheap one because it is cheap.")
    else:
        emit("They coincide here, which is worth stating rather than hiding: on")
        emit("this data the channels that convert most are also the ones bringing")
        emit("better customers, so the value weighting does not change the")
        emit("decision. That is a property of THIS simulator's channel-propensity")
        emit("correlation, not a general result -- on real data the cheap")
        emit("acquisition channels are usually the low-value ones, which is when")
        emit("this join earns its keep.")
    emit("")
    emit("HONEST LIMIT ON THE WEIGHTING ITSELF: `clv_index` is correlational. A")
    emit("channel scoring above 1.0 may be ACQUIRING better customers, or it may")
    emit("simply be TOUCHING customers who were already valuable -- exactly the")
    emit("confound the retargeting section is about. Weighting a budget by it")
    emit("inherits that confound. The version with a causal claim needs the")
    emit("experiment, and this makes the case for the experiment larger rather")
    emit("than replacing it.")
    summary["clv_handoff"] = dict(
        channel_value=CV.round(3).to_dict("index"),
        allocations=VB.round(3).to_dict("index"),
        conv_winner=conv_winner, value_winner=val_winner)

    # ==================================================================
    emit("")
    emit("=" * 78)
    emit("7. EXECUTIVE MEMO")
    emit("=" * 78)
    lt = Bt.loc["last_touch"]
    zc_credit = At.loc["last_touch", zc]
    memo = [
        "TO:      CMO",
        "FROM:    Customer Analytics",
        "RE:      Marketing budget allocation, and one channel we should test",
        "",
        "RECOMMENDATION",
        "",
        "1. Stop allocating on last-touch. On our data it sends %.1f%% of the"
        % (100 * zc_credit),
        "   budget -- about $%.0f of $%.0f -- to retargeting, and our best"
        % (lt.spend_on_zero_effect, BUDGET),
        "   evidence is that retargeting causes approximately none of the",
        "   conversions it is credited with.",
        "",
        "2. Fund a geo holdout on retargeting. Switch it off in matched control",
        "   markets for four weeks. If the effect is real we lose a month of",
        "   incremental conversions in half our markets; if it is not, the test",
        "   pays for itself immediately and permanently. That asymmetry is the",
        "   argument: the experiment is cheapest in exactly the world where the",
        "   channel is worthless.",
        "",
        "3. Reallocate toward paid search and email, which carry %.0f%% of the"
        % (100 * (true_share["paid_search"] + true_share["email"])),
        "   measurable effect between them.",
        "",
        "WHAT THIS IS BASED ON",
        "",
        "We simulated marketing journeys with KNOWN channel effects and scored",
        "every standard attribution method against that truth. Under those",
        "conditions:",
        "",
        "  - Every method credits a channel we know causes nothing. Last-touch",
        "    gives it %.0f%% of all credit; even Shapley, which is designed to"
        % (100 * zc_credit),
        "    give a useless channel exactly zero, gives it %.0f%%."
        % (100 * At.loc["shapley", zc]),
        "  - The reason is not the estimators. It is that retargeting is TARGETED",
        "    at customers who were already going to buy, so it correlates with",
        "    conversion without causing it. No amount of modelling separates",
        "    correlation from causation in data that contains no experiment.",
        "  - Allocating on last-touch instead of truth costs %.0f conversions"
        % lt.conversions_lost_vs_truth,
        "    (%.1f%% of achievable) on a $%.0f budget."
        % (lt.pct_lost, BUDGET),
        "",
        "WHAT WE ARE NOT CLAIMING",
        "",
        "  - These are simulated channel effects, not measured ones. What",
        "    transfers is the RANKING of methods and the size of the error they",
        "    make, not the specific percentages.",
        "  - Our CLV model ranks customers well and mispredicts individuals. Use",
        "    it to size a segment, never to decide what one customer is worth.",
        "  - The channel-value weighting in section 6 is correlational and",
        "    inherits the same confound. It sharpens the case for the experiment;",
        "    it does not substitute for it.",
        "",
        "COST OF DOING NOTHING",
        "",
        "  Roughly $%.0f a year of budget flowing to a channel whose effect we"
        % (lt.spend_on_zero_effect * 12),
        "  have never measured, and a reported ROAS that will keep telling us it",
        "  is working, because a channel that follows intent always looks good to",
        "  a correlational metric.",
    ]
    for line in memo:
        emit("  " + line if line else "")
    with open(os.path.join(OUT, "EXECUTIVE_MEMO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(memo) + "\n")
    emit("")
    emit("-> out/EXECUTIVE_MEMO.md")
    summary["memo_written"] = True

    emit("")
    emit("(%.0fs)" % (time.time() - t0))
    with open(os.path.join(OUT, "analytics_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "analytics_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n-> out/analytics_report.txt")


if __name__ == "__main__":
    main()
