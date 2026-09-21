"""Build the project's inputs: REAL transactions plus a SIMULATED marketing layer.

TRANSACTIONS ARE REAL
---------------------
UCI Online Retail II, cleaned by src/clean_retail.py: one row per customer per
purchase day. Segmentation and CLV run on these and nothing else.

    calibration : 2009-12-01 .. 2011-05-31  (18 months, models are fitted here)
    holdout     : 2011-06-01 .. 2011-12-09  (~6 months, models are scored here)

The cohort is every customer whose first purchase falls in the calibration
window. Customers first seen in the holdout cannot be predicted by any model
fitted before they existed, so they are counted and excluded.

THE MARKETING LAYER IS SIMULATED, AND THAT IS THE RIGHT CHOICE
--------------------------------------------------------------
Real multi-touch attribution data with ground truth does not exist publicly, and
it cannot: the ground truth is a causal quantity, so establishing it requires an
experiment nobody publishes. Without truth, attribution methods can only be
ASSERTED, never validated.

So ad journeys are generated FOR THE REAL CUSTOMERS with channel effects I
choose, and every attribution method is scored against them. Each customer's
baseline intent comes from their real calibration-period purchase frequency, so
retargeting chases customers who really do buy more.

Three things are planted deliberately:

  1. Each channel has a known INCREMENTAL effect on conversion probability.

  2. `retargeting` has an effect of EXACTLY ZERO, and it is targeted at users who
     are already likely to convert. It is the shape of a real retargeting
     programme: it follows intent rather than creating it. Every correlational
     method will credit it handsomely.

  3. Channels differ in WHERE in the journey they appear -- display and social
     open journeys, email and search close them -- so first-touch and last-touch
     have systematically opposite biases rather than random ones.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

try:
    from src.clean_retail import clean, load_raw
except ImportError:          # run as a script: python src/build_data.py
    from clean_retail import clean, load_raw

RNG = np.random.default_rng(4711)

CALIBRATION_DAYS = 546          # last calibration day: 2011-05-31
OBSERVATION_DAYS = 738          # last day in the data: 2011-12-09

# channel -> (true incremental effect on conversion prob, position bias,
#             cost per impression)
#   position bias: 0 = opens journeys, 1 = closes them
# channel -> (true incremental effect on conversion prob, position bias,
#             cost per impression, value tilt)
#   position bias : 0 = opens journeys, 1 = closes them
#   value_tilt    : how much this channel's touched customers differ in SPEND.
#                   >1 acquires higher-value customers, <1 lower-value.
#
# TWELVE CHANNELS, NOT FIVE. Exact Shapley is 2^n coalitions, so five channels
# is 32 and twelve is 4,096 -- still enumerable, but it puts the sampled
# estimator in a regime where its error can be MEASURED against the exact answer
# instead of asserted. That comparison is the only honest way to justify
# sampling at the 30-channel scale a real media mix has.
CHANNELS = {
    "display":       dict(effect=0.020, position=0.05, cost=0.004, value_tilt=0.82),
    "social":        dict(effect=0.045, position=0.25, cost=0.012, value_tilt=0.90),
    "email":         dict(effect=0.070, position=0.70, cost=0.002, value_tilt=1.05),
    "paid_search":   dict(effect=0.090, position=0.85, cost=0.350, value_tilt=1.35),
    "affiliate":     dict(effect=0.030, position=0.60, cost=0.090, value_tilt=0.88),
    "influencer":    dict(effect=0.035, position=0.20, cost=0.140, value_tilt=1.10),
    "podcast":       dict(effect=0.025, position=0.10, cost=0.070, value_tilt=1.22),
    "ctv":           dict(effect=0.028, position=0.08, cost=0.210, value_tilt=1.18),
    "push":          dict(effect=0.018, position=0.75, cost=0.001, value_tilt=0.95),
    "sms":           dict(effect=0.022, position=0.80, cost=0.006, value_tilt=0.92),
    "shopping_feed": dict(effect=0.055, position=0.88, cost=0.180, value_tilt=1.28),
    # THE PLANTED CHANNEL: touches many journeys, causes nothing.
    "retargeting":   dict(effect=0.000, position=0.90, cost=0.050, value_tilt=1.00),
}

# Exposure weights. Deliberately NOT uniform -- a real media mix is dominated by
# two or three channels, and an evenly-exposed one would make every coalition
# equally well observed, which is the easy case for Shapley and not the real one.
CHANNEL_WEIGHTS = np.array([0.16, 0.13, 0.12, 0.11, 0.07, 0.07,
                            0.05, 0.05, 0.06, 0.06, 0.07, 0.05])

# THE UNOBSERVED CONFOUNDER.
#
# The previous README ended with: "the attribution simulator has no unobserved
# confounders beyond the one I planted -- so every method here performs better
# than it would on real data." That is now false by construction.
#
# `in_market` is a latent state -- the customer is actively shopping this week --
# that raises BOTH the probability of being exposed to closing channels AND the
# probability of converting. It is never written to disk. No attribution method
# in this project can condition on it, because no attribution system in the world
# can: it is the thing the customer knows and the ad server does not.
#
# The planted retargeting confound is observable in principle (propensity is a
# customer attribute that a good model could proxy). This one is not, and the
# distinction matters: it is the difference between "we needed a better model"
# and "we needed an experiment".
IN_MARKET_RATE = 0.32
IN_MARKET_CONVERSION_LIFT = 0.16
IN_MARKET_CLOSER_EXPOSURE = 2.4

CHANNEL_LIST = list(CHANNELS)

# The `cost` column above is RELATIVE cost per touch. It is scaled to pounds so
# that total simulated marketing spend is about 10% of the revenue the
# converting journeys bring in (one average order per conversion) -- a normal
# retail level. Paid search lands near GBP 10 per touch, email near 6p. The
# scale multiplies every channel equally, so it changes no allocation.
COST_SCALE = 30.0


def _real_transactions():
    """Real purchase occasions for the calibration cohort, as the 5-column array
    every downstream step reads: customer, day, order value, distinct products,
    had_return."""
    occ, rep = clean(load_raw())
    first = occ.groupby("customer")["day"].min()
    cohort = first[first <= CALIBRATION_DAYS].index
    rep["customers_first_seen_in_holdout_excluded"] = int(len(first) - len(cohort))
    occ = occ[occ["customer"].isin(cohort)]
    ids = np.sort(cohort.to_numpy())
    idx = {c: i for i, c in enumerate(ids)}
    txn = np.column_stack([
        occ["customer"].map(idx).to_numpy(float),
        occ["day"].to_numpy(float),
        occ["order_value"].round(2).to_numpy(float),
        occ["n_products"].to_numpy(float),
        occ["had_return"].to_numpy(float)])
    txn = txn[np.lexsort((txn[:, 1], txn[:, 0]))]
    customers = (occ.groupby("customer")["country"].first()
                 .reindex(ids).reset_index()
                 .rename(columns={"customer": "source_customer_id"}))
    customers.insert(0, "customer", np.arange(len(ids)))
    rep["cohort_customers"] = int(len(ids))
    rep["cohort_occasions"] = int(len(txn))
    return txn, customers, rep


def _propensity(txn, n):
    """Baseline intent for the simulated journeys, anchored to real behaviour:
    a customer's calibration-period purchase count, rank-mapped onto Beta(2, 5)
    (the distribution the simulator was designed around)."""
    cal = txn[txn[:, 1] <= CALIBRATION_DAYS]
    freq = np.bincount(cal[:, 0].astype(int), minlength=n)
    rank = (pd.Series(freq).rank(method="average").to_numpy() - 0.5) / n
    return beta_dist.ppf(rank, 2.0, 5.0)


def build(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    txn, customers, cleaning = _real_transactions()
    N_CUSTOMERS = len(customers)
    propensity = _propensity(txn, N_CUSTOMERS)
    in_market = RNG.random(N_CUSTOMERS) < IN_MARKET_RATE

    # ---------------- marketing journeys ---------------------------------
    journeys, conversions, journey_customer, touch_times = [], [], [], []
    journey_index = []
    weights = CHANNEL_WEIGHTS / CHANNEL_WEIGHTS.sum()
    closer = np.array([CHANNELS[c]["position"] >= 0.6 for c in CHANNEL_LIST])

    for c in range(N_CUSTOMERS):
        # MULTIPLE JOURNEYS PER CUSTOMER. The previous pass gave each customer
        # exactly one, so the data could not represent re-engagement at all --
        # and re-engagement is most of what a retail marketing budget buys. A
        # customer with three journeys is three separate exposures to the same
        # channels, and their conversions are NOT independent, which is precisely
        # what makes attribution on customer-level data harder than it looks.
        n_journeys = 1 + int(RNG.poisson(0.9))
        for j in range(n_journeys):
            n_touch = int(RNG.integers(1, 9))
            base = propensity[c]

            # The confounder acts on EXPOSURE: an in-market customer is served
            # closing channels far more often, because that is what an ad
            # platform's own optimiser does when it detects intent.
            w = weights.copy()
            if in_market[c]:
                w = w * np.where(closer, IN_MARKET_CLOSER_EXPOSURE, 1.0)
                w = w / w.sum()
            chans = list(RNG.choice(CHANNEL_LIST, n_touch, replace=True, p=w))

            # RETARGETING IS TARGETED AT HIGH-PROPENSITY USERS. The observable
            # confound, kept because the contrast with the unobservable one is
            # the point.
            if base > 0.45 and RNG.random() < 0.75:
                chans.append("retargeting")
            order_key = [CHANNELS[ch]["position"] + RNG.normal(0, 0.18)
                         for ch in chans]
            chans = [ch for _, ch in sorted(zip(order_key, chans))]

            p = base * 0.45
            for ch in set(chans):
                p += CHANNELS[ch]["effect"]
            # ... and the confounder ALSO acts on the outcome. Exposure and
            # outcome share an unobserved cause, which is the textbook definition
            # of confounding and the reason no amount of controlling for observed
            # covariates recovers the truth.
            if in_market[c]:
                p += IN_MARKET_CONVERSION_LIFT
            p = float(np.clip(p, 0.0, 0.97))
            converted = int(RNG.random() < p)

            journeys.append(chans)
            conversions.append(converted)
            journey_customer.append(c)
            journey_index.append(j)

            n = len(chans)
            gaps = np.sort(RNG.exponential(6.0, n))[::-1]
            t = 0.0
            times = []
            for g in gaps:
                t += float(g)
                times.append(round(t, 3))
            span = max(times[-1], 1e-6)
            touch_times.append([round(30.0 * x / span, 3) for x in times])

    truth = {
        "channel_effects": {k: v["effect"] for k, v in CHANNELS.items()},
        "channel_costs": {k: round(v["cost"] * COST_SCALE, 4)
                          for k, v in CHANNELS.items()},
        "zero_effect_channel": "retargeting",
        "channel_value_tilt": {k: v["value_tilt"] for k, v in CHANNELS.items()},
        # Recorded so the report can state exactly how much of the confounding is
        # in principle unfixable. `in_market` is never written to disk.
        "unobserved_confounder": dict(
            name="in_market", rate=IN_MARKET_RATE,
            conversion_lift=IN_MARKET_CONVERSION_LIFT,
            closer_exposure_multiplier=IN_MARKET_CLOSER_EXPOSURE),
        "calibration_days": CALIBRATION_DAYS,
        "observation_days": OBSERVATION_DAYS,
    }

    np.save(os.path.join(out_dir, "transactions.npy"), txn)
    customers.to_csv(os.path.join(out_dir, "customers.csv"), index=False)
    with open(os.path.join(out_dir, "cleaning_report.json"), "w") as f:
        json.dump(cleaning, f, indent=2)
    with open(os.path.join(out_dir, "journeys.json"), "w") as f:
        json.dump(dict(journeys=journeys, conversions=conversions,
                       customer_id=journey_customer, touch_days=touch_times,
                       journey_index=journey_index), f)
    with open(os.path.join(out_dir, "TRUTH.json"), "w") as f:
        json.dump(truth, f, indent=2)

    # true share of total incremental effect, which is what a budget SHOULD track
    tot = sum(v["effect"] for v in CHANNELS.values())
    truth["true_effect_share"] = {k: v["effect"] / tot for k, v in CHANNELS.items()}
    with open(os.path.join(out_dir, "TRUTH.json"), "w") as f:
        json.dump(truth, f, indent=2)

    stats = dict(
        source="UCI Online Retail II (real transactions) + simulated touch layer",
        n_customers=N_CUSTOMERS, n_transactions=int(len(txn)),
        mean_orders_per_customer=round(len(txn) / N_CUSTOMERS, 2),
        conversion_rate=round(float(np.mean(conversions)), 4),
        n_journeys=len(journeys),
        journeys_per_customer=round(len(journeys) / N_CUSTOMERS, 3),
        n_channels=len(CHANNEL_LIST),
        in_market_share=round(float(np.mean(in_market)), 4),
        mean_touches=round(float(np.mean([len(j) for j in journeys])), 2),
        retargeting_touch_rate=round(
            float(np.mean([1.0 if "retargeting" in j else 0.0 for j in journeys])), 4),
        retargeting_conv_rate=round(
            float(np.mean([conversions[i] for i, j in enumerate(journeys)
                           if "retargeting" in j])), 4),
        no_retargeting_conv_rate=round(
            float(np.mean([conversions[i] for i, j in enumerate(journeys)
                           if "retargeting" not in j])), 4))
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(build(os.path.join(here, "data")), indent=2))
