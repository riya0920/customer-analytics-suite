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


def time_decay(journeys, conversions, channels, half_life: float = 2.0) -> dict:
    """Exponential decay toward the conversion. Positions are steps, not days --
    the simulator has no touch timestamps, and pretending otherwise would be a
    fake precision."""
    credit = {c: 0.0 for c in channels}
    for j, y in zip(journeys, conversions):
        if not (y and j):
            continue
        w = np.array([0.5 ** ((len(j) - 1 - i) / half_life) for i in range(len(j))])
        w = w / w.sum()
        for ch, wi in zip(j, w):
            credit[ch] += wi
    return _norm(credit)


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
