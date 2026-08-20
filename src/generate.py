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
CHANNELS = {
    "display":     dict(effect=0.020, position=0.05, cost=0.004),
    "social":      dict(effect=0.045, position=0.25, cost=0.012),
    "email":       dict(effect=0.070, position=0.70, cost=0.002),
    "paid_search": dict(effect=0.090, position=0.85, cost=0.35),
    # THE PLANTED CHANNEL: touches many journeys, causes nothing.
    "retargeting": dict(effect=0.000, position=0.90, cost=0.05),
}
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
    return lam, p_drop, spend_mu, propensity, disc_affinity, cat_breadth


def build(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    lam, p_drop, spend_mu, propensity, disc_aff, breadth = _customers()

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
    journeys, conversions = [], []
    for c in range(N_CUSTOMERS):
        n_touch = int(RNG.integers(1, 9))
        base = propensity[c]
        # which channels touch this user, ordered by their position bias plus noise
        chans = list(RNG.choice(CHANNEL_LIST, n_touch,
                                replace=True,
                                p=[0.30, 0.22, 0.20, 0.16, 0.12]))
        # RETARGETING IS TARGETED AT HIGH-PROPENSITY USERS. This is the
        # confound: it appears in the journeys most likely to convert anyway.
        if base > 0.45 and RNG.random() < 0.75:
            chans.append("retargeting")
        order_key = [CHANNELS[ch]["position"] + RNG.normal(0, 0.18) for ch in chans]
        chans = [ch for _, ch in sorted(zip(order_key, chans))]

        # conversion probability: baseline + sum of TRUE incremental effects,
        # counted once per distinct channel (a second email is not a second lift)
        p = base * 0.45
        for ch in set(chans):
            p += CHANNELS[ch]["effect"]
        p = float(np.clip(p, 0.0, 0.97))
        converted = int(RNG.random() < p)
        journeys.append(chans)
        conversions.append(converted)

    truth = {
        "channel_effects": {k: v["effect"] for k, v in CHANNELS.items()},
        "channel_costs": {k: v["cost"] for k, v in CHANNELS.items()},
        "zero_effect_channel": "retargeting",
        "calibration_days": CALIBRATION_DAYS,
        "observation_days": OBSERVATION_DAYS,
    }

    np.save(os.path.join(out_dir, "transactions.npy"), txn)
    with open(os.path.join(out_dir, "journeys.json"), "w") as f:
        json.dump(dict(journeys=journeys, conversions=conversions), f)
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
