"""Sampled Shapley, higher-order Markov, and the economics the budget table lacked.

THREE GAPS
----------
  "Shapley is exact over 5 channels. At 30 channels the 2^30 coalitions need
   sampling, which is not implemented."
  "Markov is first-order only; no higher-order path models."
  "No CAC or ROAS, so the budget table is conversions and customer value, not
   profit."

WHY SAMPLING IS JUSTIFIED HERE RATHER THAN ASSERTED
---------------------------------------------------
Exact Shapley enumerates 2^n coalitions. The generator now runs TWELVE channels,
which is 4,096 -- large enough to be interesting and small enough that the exact
answer is still computable. So the sampled estimator can be scored against the
truth it is approximating, at the scale where both exist, and the error curve
against sample count is a measurement rather than a promise.

That matters because the usual justification for sampling is "the exact version
is intractable", which is exactly the regime in which nobody checks it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations

import numpy as np


# --------------------------------------------------------------------------
# sampled Shapley
# --------------------------------------------------------------------------
def _coalition_rates(journeys, conversions, channels):
    """v(S) = observed conversion rate of journeys whose channel set is exactly S.

    Same coalition function as the exact estimator in `attribution.py`, so the
    two are approximating the same object. If they used different value functions
    the comparison below would be meaningless.
    """
    idx = {c: i for i, c in enumerate(channels)}
    hits, tot = defaultdict(float), defaultdict(int)
    for j, y in zip(journeys, conversions):
        key = frozenset(idx[c] for c in j if c in idx)
        hits[key] += y
        tot[key] += 1
    return {k: hits[k] / tot[k] for k in tot if tot[k] > 0}, tot


def shapley_sampled(journeys, conversions, channels, n_perms: int = 400,
                    seed: int = 0) -> dict:
    """Monte-Carlo Shapley over random permutations of the channel order.

    The estimator: draw a random ordering, walk it adding one channel at a time,
    and credit each channel with the increase in the coalition's value when it
    joins. Averaged over orderings this converges to the exact Shapley value, and
    the standard error falls as 1/sqrt(n_perms).

    UNOBSERVED COALITIONS ARE SKIPPED, not treated as zero -- the same fix the
    exact version needed. Treating an unobserved coalition as having conversion
    rate zero injects large spurious negative marginals, which normalisation then
    amplifies into confident nonsense (it once gave two interchangeable channels
    1.0 and 0.0).
    """
    rng = np.random.default_rng(seed)
    rates, counts = _coalition_rates(journeys, conversions, channels)
    n = len(channels)
    contrib = np.zeros((n_perms, n))

    for p in range(n_perms):
        order = rng.permutation(n)
        cur = frozenset()
        prev_val = rates.get(cur, None)
        for c in order:
            nxt = cur | {int(c)}
            val = rates.get(nxt)
            if val is not None and prev_val is not None:
                contrib[p, c] = val - prev_val
            if val is not None:
                prev_val = val
                cur = nxt
        # `cur` only advances through coalitions that were actually observed, so
        # a permutation walks the observed lattice rather than inventing rungs.

    mean = contrib.mean(axis=0)
    se = contrib.std(axis=0, ddof=1) / np.sqrt(max(n_perms, 1))
    pos = np.clip(mean, 0, None)
    total = pos.sum()
    share = pos / total if total > 0 else np.zeros(n)
    return {"credit": {c: float(share[i]) for i, c in enumerate(channels)},
            "raw": {c: float(mean[i]) for i, c in enumerate(channels)},
            "se": {c: float(se[i]) for i, c in enumerate(channels)},
            "n_perms": n_perms,
            "observed_coalitions": len(rates)}


def shapley_error_curve(journeys, conversions, channels, exact: dict,
                        perm_counts=(25, 50, 100, 200, 400, 800),
                        seed: int = 0) -> list[dict]:
    """Mean absolute deviation from the exact answer, by sample count."""
    out = []
    for m in perm_counts:
        est = shapley_sampled(journeys, conversions, channels, n_perms=m,
                              seed=seed)["credit"]
        err = float(np.mean([abs(est[c] - exact.get(c, 0.0)) for c in channels]))
        out.append(dict(n_perms=m, mean_abs_error=err,
                        max_abs_error=float(max(abs(est[c] - exact.get(c, 0.0))
                                                for c in channels))))
    return out


# --------------------------------------------------------------------------
# higher-order Markov
# --------------------------------------------------------------------------
def markov_removal_order_k(journeys, conversions, channels, order: int = 2,
                           damping: float = 1.0) -> dict:
    """Removal effects on an order-k Markov chain over touch SEQUENCES.

    A first-order chain's state is the last touch. That makes
    `display -> email -> convert` and `paid_search -> email -> convert`
    indistinguishable once you are at `email`, which throws away the thing a
    multi-touch model exists to capture: the path.

    An order-k chain's state is the last k touches, so it can represent "email
    after display" as a different state from "email after paid search". The cost
    is state explosion -- with 12 channels an order-2 chain has up to 144 states
    and order-3 has 1,728 -- and most of them are seen a handful of times, which
    is where a higher-order model starts fitting noise. The state-count and the
    share of states with thin support are both reported for that reason.
    """
    START, CONV, NULL = "<start>", "<conv>", "<null>"

    def build(exclude=None):
        trans = defaultdict(Counter)
        for j, y in zip(journeys, conversions):
            seq = [c for c in j if c != exclude]
            path = [START] + seq + [CONV if y else NULL]
            for i in range(len(path) - 1):
                lo = max(0, i - order + 1)
                state = tuple(path[lo:i + 1])
                trans[state][path[i + 1]] += 1
        return trans

    def conv_prob(trans):
        """Probability of reaching <conv> from <start>, by forward iteration.

        Iterated rather than solved as a linear system: the state space is
        ragged (only observed histories exist) and building the full transition
        matrix would mean materialising states that were never seen.
        """
        cur = {(START,): 1.0}
        total = 0.0
        for _ in range(24):                       # journeys here are <= 9 touches
            nxt = defaultdict(float)
            for state, mass in cur.items():
                row = trans.get(state)
                if not row:
                    continue
                tot = sum(row.values())
                for nxt_tok, cnt in row.items():
                    pr = mass * (cnt / tot) * damping
                    if nxt_tok == CONV:
                        total += pr
                    elif nxt_tok == NULL:
                        continue
                    else:
                        new = (state + (nxt_tok,))[-order:]
                        nxt[new] += pr
            if not nxt:
                break
            cur = nxt
        return total

    base_trans = build()
    base = conv_prob(base_trans)
    thin = sum(1 for st, row in base_trans.items() if sum(row.values()) < 10)

    # SIGNED, not clamped at zero. Clamping hides the distinction between "this
    # channel contributes nothing" and "the estimate is noise straddling zero",
    # and those are different findings -- the second one says the estimator has
    # no power here, which is the more important thing to know.
    signed = {ch: base - conv_prob(build(exclude=ch)) for ch in channels}
    effects = {ch: max(0.0, v) for ch, v in signed.items()}
    tot = sum(effects.values())
    credit = ({c: v / tot for c, v in effects.items()} if tot > 0
              else {c: 0.0 for c in channels})
    return {"credit": credit, "removal_effects": effects,
            "signed_effects": signed,
            "max_abs_signed": max(abs(v) for v in signed.values()),
            "n_states": len(base_trans),
            "thin_state_share": thin / max(len(base_trans), 1),
            "base_conversion": base, "order": order}


# --------------------------------------------------------------------------
# economics
# --------------------------------------------------------------------------
def channel_economics(journeys, conversions, channels, costs: dict,
                      customer_value: dict, journey_customer: list) -> list[dict]:
    """Impressions, spend, CAC and ROAS per channel.

    CAC HERE IS THE OBSERVATIONAL ONE and the report says so: spend divided by
    conversions the channel TOUCHED. It double-counts every conversion touched by
    more than one channel, so the CACs do not add up to the blended CAC and
    summing them is a category error. That is not a defect of this
    implementation -- it is what every channel-level CAC in every marketing deck
    is, and the reason the incremental version needs an experiment.
    """
    imp = Counter()
    conv_touched = Counter()
    value_touched = defaultdict(float)
    for j, y, cid in zip(journeys, conversions, journey_customer):
        for c in j:
            imp[c] += 1
        for c in set(j):
            if y:
                conv_touched[c] += 1
                value_touched[c] += customer_value.get(cid, 0.0)

    rows = []
    for c in channels:
        spend = imp[c] * costs.get(c, 0.0)
        n = conv_touched[c]
        rows.append(dict(
            channel=c, impressions=int(imp[c]), spend=float(spend),
            conversions_touched=int(n),
            cac=float(spend / n) if n else float("nan"),
            revenue_touched=float(value_touched[c]),
            roas=float(value_touched[c] / spend) if spend > 0 else float("nan"),
            profit=float(value_touched[c] - spend)))
    return rows


def incremental_economics(rows: list[dict], true_share: dict,
                          total_conversions: int, mean_value: float) -> list[dict]:
    """The same table with conversions allocated by TRUE incremental share.

    Only available because this is a simulator. It is the number every one of
    those decks is trying to approximate, and putting the two side by side is the
    cheapest possible demonstration of how far apart they are.
    """
    out = []
    for r in rows:
        inc = true_share.get(r["channel"], 0.0) * total_conversions
        out.append(dict(
            channel=r["channel"], spend=r["spend"],
            observational_cac=r["cac"],
            incremental_cac=float(r["spend"] / inc) if inc > 0 else float("inf"),
            observational_roas=r["roas"],
            incremental_roas=(float(inc * mean_value / r["spend"])
                              if r["spend"] > 0 else float("nan"))))
    return out
