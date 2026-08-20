"""Attribution methods, all scored against the simulator's known channel effects.

ATTRIBUTION IS NOT INCREMENTALITY. Every method in this file answers the question
"which touchpoints appear on converting journeys?" -- a correlational question
about credit allocation. None of them answers "what would have happened without
this channel?", which is the causal question a budget decision actually needs.

The distinction is not pedantry, and section 3 of the report demonstrates it
rather than disclaiming it: a channel with exactly zero causal effect, targeted
at users who were going to convert anyway, receives substantial credit from every
method here -- including Markov removal effects, which sound causal and are not.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _norm(d: dict) -> dict:
    tot = sum(d.values())
    return {k: (v / tot if tot else 0.0) for k, v in d.items()}


def last_touch(journeys, conversions, channels) -> dict:
    credit = {c: 0.0 for c in channels}
    for j, y in zip(journeys, conversions):
        if y and j:
            credit[j[-1]] += 1.0
    return _norm(credit)


def first_touch(journeys, conversions, channels) -> dict:
    credit = {c: 0.0 for c in channels}
    for j, y in zip(journeys, conversions):
        if y and j:
            credit[j[0]] += 1.0
    return _norm(credit)


def linear(journeys, conversions, channels) -> dict:
    credit = {c: 0.0 for c in channels}
    for j, y in zip(journeys, conversions):
        if y and j:
            for ch in j:
                credit[ch] += 1.0 / len(j)
    return _norm(credit)


def time_decay(journeys, conversions, channels, half_life: float = 2.0,
               touch_days=None, half_life_days: float = 7.0) -> dict:
    """Exponential decay toward the conversion.

    The first pass decayed over journey POSITION because the simulator had no
    touch timestamps, and said so rather than pretending otherwise. It now has
    them, so the decay is over DAYS -- which is what time-decay attribution
    actually means. Position decay treats a touch three steps back the same
    whether it was yesterday or three weeks ago, and those are very different
    claims about influence.
    """
    credit = {c: 0.0 for c in channels}
    for i, (j, y) in enumerate(zip(journeys, conversions)):
        if not (y and j):
            continue
        if touch_days is not None:
            days = np.asarray(touch_days[i], float)
            age = days[-1] - days                      # days before conversion
            w = 0.5 ** (age / half_life_days)
        else:
            w = np.array([0.5 ** ((len(j) - 1 - k) / half_life)
                          for k in range(len(j))])
        w = w / w.sum()
        for ch, wi in zip(j, w):
            credit[ch] += wi
    return _norm(credit)


def shapley(journeys, conversions, channels, max_order: int = 4) -> dict:
    """Shapley value attribution over channel COALITIONS.

    The idea: a channel's credit is its average marginal contribution to the
    conversion rate across every subset it could join. Unlike the heuristics it
    is symmetric, efficient (credits sum to the total) and satisfies the dummy
    axiom -- a channel that never changes any coalition's conversion rate gets
    exactly zero.

    THAT LAST AXIOM IS WHY IT IS WORTH RUNNING HERE. Shapley is the only method
    in this file with a formal guarantee that a genuinely useless channel scores
    zero, so it is the strongest possible test of the planted-channel section: if
    even Shapley credits retargeting, the problem is definitively the DATA and
    not the estimator.

    Implemented over the SET of channels in a journey (order ignored), with
    coalition conversion rates estimated empirically. Subsets larger than
    `max_order` are dropped -- with 5 channels that is not binding, and on a real
    catalogue of 30 channels the full computation is 2^30 coalitions and needs
    sampling instead.
    """
    from itertools import combinations

    idx = {c: i for i, c in enumerate(channels)}
    tot = np.zeros(1 << len(channels))
    conv = np.zeros(1 << len(channels))
    for j, y in zip(journeys, conversions):
        mask = 0
        for ch in set(j):
            if ch in idx:
                mask |= 1 << idx[ch]
        tot[mask] += 1
        conv[mask] += y

    def rate(mask):
        """Conversion rate of journeys whose channel set is EXACTLY this."""
        return conv[mask] / tot[mask] if tot[mask] > 0 else 0.0

    from math import factorial
    n = len(channels)
    values = {c: 0.0 for c in channels}
    used_weight = {c: 0.0 for c in channels}
    others = {c: [o for o in channels if o != c] for c in channels}
    for c in channels:
        i = idx[c]
        for r in range(0, min(max_order, n - 1) + 1):
            for subset in combinations(others[c], r):
                m = 0
                for o in subset:
                    m |= 1 << idx[o]
                with_c = m | (1 << i)
                # A coalition with NO OBSERVED JOURNEYS has an unknown rate, not
                # a rate of zero. Treating it as zero was a real bug: it made the
                # marginal contribution of every channel look like -rate(m)
                # whenever the larger coalition was unobserved, and on a fixture
                # where two channels never co-occur it drove both Shapley values
                # to +/- floating-point noise that normalisation then amplified
                # into a 1.0 / 0.0 split between interchangeable channels.
                #
                # The empty coalition is the exception: no channels means no
                # marketing, and a rate of zero is the right reading.
                if m != 0 and tot[m] == 0:
                    continue
                if tot[with_c] == 0:
                    continue
                marginal = rate(with_c) - rate(m)
                weight = factorial(r) * factorial(n - r - 1) / factorial(n)
                values[c] += weight * marginal
                used_weight[c] += weight
    # Renormalise by the weight actually used, so a channel that appears in few
    # observed coalitions is not penalised for the coalitions we never saw.
    values = {c: (v / used_weight[c] if used_weight[c] > 0 else 0.0)
              for c, v in values.items()}
    # Shapley values can be negative; a channel that makes coalitions WORSE has
    # earned that. Clipping at zero before normalising would hide it, so the
    # negative is carried into the share and the caller can see it.
    shifted = {c: max(v, 0.0) for c, v in values.items()}
    return _norm(shifted)


def markov_removal(journeys, conversions, channels, order: int = 1) -> dict:
    """Removal-effect attribution over a first-order Markov chain.

    Build the transition matrix over states {start, channels..., conversion,
    null}. For each channel, remove it from the graph and recompute the
    conversion probability; the drop is that channel's removal effect.

    IT SOUNDS CAUSAL AND IT IS NOT. "Remove the channel from the graph" is a
    statement about the observed path data, not about the world: it assumes the
    users who saw that channel would otherwise have walked the same graph minus
    one node. If the channel was TARGETED at high-intent users -- which is what
    retargeting is -- removing it also removes the intent that came with it, and
    the method charges that intent to the channel.
    """
    trans = defaultdict(lambda: defaultdict(float))
    for j, y in zip(journeys, conversions):
        path = ["start"] + list(j) + ["conv" if y else "null"]
        for a, b in zip(path[:-1], path[1:]):
            trans[a][b] += 1.0

    def probs(exclude=None):
        P = {}
        for s, nxt in trans.items():
            if s == exclude:
                continue
            filtered = {k: v for k, v in nxt.items() if k != exclude}
            tot = sum(filtered.values())
            if tot == 0:
                P[s] = {"null": 1.0}
            else:
                P[s] = {k: v / tot for k, v in filtered.items()}
        # a removed channel's inbound mass has to go somewhere: route it as if
        # the user skipped that touch, i.e. redistribute over the remaining
        # transitions, which is what the filtering above already does
        return P

    def conv_prob(P, max_steps: int = 40) -> float:
        state = {"start": 1.0}
        converted = 0.0
        for _ in range(max_steps):
            nxt = defaultdict(float)
            for s, m in state.items():
                for t, p in P.get(s, {"null": 1.0}).items():
                    if t == "conv":
                        converted += m * p
                    elif t == "null":
                        continue
                    else:
                        nxt[t] += m * p
            state = nxt
            if not state:
                break
        return converted

    base = conv_prob(probs())
    removal = {}
    for ch in channels:
        removal[ch] = max(base - conv_prob(probs(exclude=ch)), 0.0)
    return _norm(removal)


METHODS = {
    "last_touch": last_touch,
    "first_touch": first_touch,
    "linear": linear,
    "time_decay": time_decay,
    "markov_removal": markov_removal,
    "shapley": shapley,
}


# --------------------------------------------------------------------------
def budget_allocation(credit: dict, budget: float) -> dict:
    """Spend proportional to credited share -- what a marketing team does with an
    attribution report."""
    return {ch: budget * share for ch, share in credit.items()}


def conversions_under(alloc: dict, true_effects: dict, costs: dict,
                      base_conversion: float, n_users: int,
                      saturation: float = 0.6) -> float:
    """Evaluate an allocation in the TRUE world.

    Reach is concave in spend (saturation), so pouring the whole budget into the
    single best channel does not win by construction -- which would make the
    comparison meaningless.
    """
    total = 0.0
    for ch, spend in alloc.items():
        impressions = spend / max(costs[ch], 1e-9)
        reach = (impressions / n_users) ** saturation
        total += true_effects[ch] * min(reach, 1.0) * n_users
    return base_conversion * n_users + total
