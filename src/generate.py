"""Transactions plus multi-channel marketing journeys with KNOWN channel effects.

WHY THIS IS SIMULATED, AND WHY THAT IS THE RIGHT CHOICE HERE
------------------------------------------------------------
Real multi-touch attribution data with ground truth does not exist publicly, and
it cannot: the ground truth is a causal quantity, so establishing it requires an
experiment nobody publishes. Without truth, attribution methods can only be
ASSERTED, never validated -- which is precisely why the field is full of
confident numbers that misallocate budget.

So the touch data is generated with channel effects I choose, and every
attribution method is then SCORED against them. That converts the most BS-prone
area of marketing analytics into a controlled experiment.

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

RNG = np.random.default_rng(4711)

N_CUSTOMERS = 8000
OBSERVATION_DAYS = 730
CALIBRATION_DAYS = 511          # ~70% of the window; the rest is holdout

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


def _customers():
    """Latent heterogeneity: purchase rate, dropout, spend level, and an
    intrinsic propensity that retargeting will later chase."""
    lam = RNG.gamma(0.9, 1 / 22.0, N_CUSTOMERS)        # purchases per day
    p_drop = RNG.beta(1.2, 12.0, N_CUSTOMERS)          # per-purchase dropout
    spend_mu = RNG.gamma(6.0, 12.0, N_CUSTOMERS)       # mean order value
    propensity = RNG.beta(2.0, 5.0, N_CUSTOMERS)       # baseline intent
    disc_affinity = RNG.beta(2.0, 4.0, N_CUSTOMERS)
    cat_breadth = RNG.integers(1, 9, N_CUSTOMERS)
    in_market = RNG.random(N_CUSTOMERS) < IN_MARKET_RATE
    return (lam, p_drop, spend_mu, propensity, disc_affinity, cat_breadth,
            in_market)


def build(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    (lam, p_drop, spend_mu, propensity, disc_aff, breadth,
     in_market) = _customers()

    # ---------------- transactions: BG/NBD-shaped by construction ----------
    txns = []
    for c in range(N_CUSTOMERS):
        t = 0.0
        alive = True
        # everyone has an initial purchase at t=0 (acquisition)
        n_orders = 0
        while alive and t < OBSERVATION_DAYS:
            gap = RNG.exponential(1.0 / max(lam[c], 1e-6))
            t += gap
            if t >= OBSERVATION_DAYS:
                break
            value = float(RNG.gamma(4.0, spend_mu[c] / 4.0))
            txns.append((c, round(t, 3), round(value, 2),
                         int(RNG.integers(1, breadth[c] + 1)),
                         int(RNG.random() < disc_aff[c])))
            n_orders += 1
            if RNG.random() < p_drop[c]:
                alive = False
        if n_orders == 0:
            txns.append((c, 0.0, float(round(RNG.gamma(4.0, spend_mu[c] / 4.0), 2)),
                         1, int(RNG.random() < disc_aff[c])))

    txn = np.array(txns, dtype=np.float64)

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
        "channel_costs": {k: v["cost"] for k, v in CHANNELS.items()},
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
