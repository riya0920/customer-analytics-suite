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
    stalls = 0

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
            else:
                stalls += 1
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
            "observed_coalitions": len(rates),
            "stalls": stalls,
            "stall_rate": stalls / max(n_perms * n, 1)}


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
# THE FIX: a value function defined on the WHOLE lattice
# --------------------------------------------------------------------------
def subset_closure_values(journeys, conversions, channels,
                          max_channels: int = 22):
    """v(S) = conversion rate among journeys whose channel set is a SUBSET of S.

    WHY THIS EXISTS. The coalition function above is v(S) = the rate of journeys
    whose set is EXACTLY S, and it is undefined wherever a combination was never
    observed. That is not a rare corner: a permutation walking the full lattice
    hits an undefined rung constantly, stalls there, and the resulting estimator
    is biased rather than noisy -- which is what the error curve measured and
    what more permutations could never have fixed.

    This value function is defined on every coalition containing at least one
    observed journey set. It also reads better as a business question -- "what
    conversion rate is achievable using only the channels in S" rather than "what
    happened to the customers who saw exactly this combination and nothing else".

    It is a DIFFERENT ESTIMAND, not a better estimator of the same one, and that
    is the honest framing: the exact answer under this value function is not the
    exact answer under the other one, and the two should not be expected to agree.

    Computed with a subset-sum (zeta) transform, n * 2^n rather than 2^n * J.

    THIS ONE IS DENSE and therefore bounded by channel count: it materialises
    2^n values. It refuses above `max_channels` rather than trying to allocate,
    because the failure mode is an out-of-memory kill rather than a slow answer.
    `shapley_per_journey` is the estimator that does not have this bound.
    """
    idx = {c: i for i, c in enumerate(channels)}
    n = len(channels)
    if n > max_channels:
        raise ValueError(
            "subset_closure_values materialises 2^%d values; refusing above "
            "%d channels. Use shapley_per_journey, whose lattice is bounded by "
            "journey length instead." % (n, max_channels))
    size = 1 << n
    tot = np.zeros(size)
    conv = np.zeros(size)
    for j, y in zip(journeys, conversions):
        m = 0
        for ch in j:
            i = idx.get(ch)
            if i is not None:
                m |= 1 << i
        tot[m] += 1.0
        conv[m] += float(y)
    exact_observed = int((tot > 0).sum())
    for i in range(n):
        bit = 1 << i
        for m in range(size):
            if m & bit:
                tot[m] += tot[m ^ bit]
                conv[m] += conv[m ^ bit]
    v = np.zeros(size)
    nz = tot > 0
    v[nz] = conv[nz] / tot[nz]
    v[0] = 0.0          # no channels is no marketing, and that rate IS zero
    cover = dict(exact_observed=exact_observed,
                 closure_defined=int(nz.sum()),
                 total=size,
                 exact_coverage=exact_observed / size,
                 closure_coverage=int(nz.sum()) / size)
    return v, cover


def _weights(n):
    from math import factorial
    return [factorial(r) * factorial(n - r - 1) / factorial(n) for r in range(n)]


def _share(vals, channels):
    pos = np.clip(np.asarray(vals, dtype=float), 0, None)
    tot = pos.sum()
    if tot <= 0:
        return {c: 0.0 for c in channels}
    return {c: float(pos[i] / tot) for i, c in enumerate(channels)}


def shapley_closure(journeys, conversions, channels, v=None):
    """Exact Shapley over the subset-closure value function.

    Efficiency is checkable here in a way it is not for the exact-set version:
    the values must sum to v(grand coalition) - v(empty). The residual is
    returned rather than assumed, because an attribution that does not add up to
    the thing being attributed is not an allocation.
    """
    n = len(channels)
    cover = None
    if v is None:
        v, cover = subset_closure_values(journeys, conversions, channels)
    w = _weights(n)
    popcount = [bin(m).count("1") for m in range(1 << n)]
    vals = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        for m in range(1 << n):
            if m & bit:
                continue
            vals[i] += w[popcount[m]] * (v[m | bit] - v[m])
    return dict(credit=_share(vals, channels),
                raw={c: float(vals[i]) for i, c in enumerate(channels)},
                efficiency_residual=float(vals.sum() - (v[(1 << n) - 1] - v[0])),
                grand_value=float(v[(1 << n) - 1]),
                coverage=cover)


def shapley_sampled_closure(journeys, conversions, channels, n_perms: int = 400,
                            seed: int = 0, v=None) -> dict:
    """The SAME Monte-Carlo estimator, over the value function defined
    everywhere. Nothing about the sampling changed; only the game it samples.

    `stalls` is reported and should be zero by construction. A sampler that
    cannot stall is one whose error is variance -- and variance is the only kind
    of error more permutations buy anything against.
    """
    n = len(channels)
    if v is None:
        v, _ = subset_closure_values(journeys, conversions, channels)
    rng = np.random.default_rng(seed)
    contrib = np.zeros((n_perms, n))
    stalls = 0
    for p in range(n_perms):
        order = rng.permutation(n)
        m = 0
        prev = v[0]
        for c in order:
            nm = m | (1 << int(c))
            contrib[p, int(c)] = v[nm] - prev
            prev = v[nm]
            m = nm
    mean = contrib.mean(axis=0)
    sd = contrib.std(axis=0, ddof=1)
    return dict(credit=_share(mean, channels),
                raw={c: float(mean[i]) for i, c in enumerate(channels)},
                se={c: float(sd[i] / np.sqrt(n_perms))
                    for i, c in enumerate(channels)},
                n_perms=n_perms, stalls=stalls)


def shapley_per_journey(journeys, conversions, channels,
                        max_journey_channels: int = 20) -> dict:
    """Shapley computed INSIDE each journey and averaged across journeys.

    This is the second fix and it is not the same fix. It changes what the
    lattice IS: a journey with five touches has 32 sub-coalitions whether the
    catalogue holds 12 channels or 300, so cost is set by JOURNEY LENGTH rather
    than by channel count -- which is the thing that was supposed to force
    sampling in the first place.

    The value function is the same subset closure -- v(T) = the conversion rate
    among journeys whose channel set is a subset of T -- but it is built on each
    journey's OWN lattice, one zeta transform over k bits instead of one over n.
    That gives the identical numbers, because "journeys whose set is a subset of
    T" does not depend on which journey T came from, and it never allocates 2^n.

    > The first version of this called `subset_closure_values` for its value
    > function, which materialises 2^n. On the 12-channel panel that is 4,096 and
    > invisible; the test that runs it at 30 channels asked for 8 GiB. The
    > estimator whose entire claim is that it does not depend on channel count
    > was depending on channel count, and the claim was in the docstring for a
    > full run before a test disagreed with it.

    Each journey's value is divided among its own touches exactly, so the
    per-journey values sum to v(S_j) by efficiency. Memoised on the channel set,
    because 15,238 journeys share far fewer distinct sets.
    """
    idx = {c: i for i, c in enumerate(channels)}
    n = len(channels)
    exact_cnt, exact_conv = Counter(), Counter()
    set_counts = Counter()
    for j, y in zip(journeys, conversions):
        m = 0
        for ch in j:
            i = idx.get(ch)
            if i is not None:
                m |= 1 << i
        exact_cnt[m] += 1
        exact_conv[m] += float(y)
        set_counts[m] += 1

    popcount = {}

    def pc(x):
        if x not in popcount:
            popcount[x] = bin(x).count("1")
        return popcount[x]

    total = np.zeros(n)
    n_j = 0
    max_k = 0
    evaluations = 0
    cache = {}
    for S, cnt in set_counts.items():
        if S not in cache:
            members = [i for i in range(n) if S & (1 << i)]
            k = len(members)
            if k > max_journey_channels:
                raise ValueError(
                    "journey touches %d distinct channels; the exact per-journey "
                    "lattice is 2^%d. Raise max_journey_channels deliberately or "
                    "sample within the journey." % (k, k))
            max_k = max(max_k, k)
            size = 1 << k
            tc = np.zeros(size)
            vc = np.zeros(size)
            # exact-set counts for every subset of S, on LOCAL bit positions
            for local in range(size):
                g = 0
                for t, b in enumerate(members):
                    if local & (1 << t):
                        g |= 1 << b
                c_ = exact_cnt.get(g)
                if c_:
                    tc[local] += c_
                    vc[local] += exact_conv[g]
            # zeta transform over k bits -> subset-closure totals
            for t in range(k):
                bit = 1 << t
                for local in range(size):
                    if local & bit:
                        tc[local] += tc[local ^ bit]
                        vc[local] += vc[local ^ bit]
            vloc = np.zeros(size)
            nz = tc > 0
            vloc[nz] = vc[nz] / tc[nz]
            vloc[0] = 0.0
            w = _weights(k) if k else []
            out = np.zeros(n)
            for t, b in enumerate(members):
                bit = 1 << t
                for local in range(size):
                    if local & bit:
                        continue
                    out[b] += w[pc(local)] * (vloc[local | bit] - vloc[local])
                    evaluations += 1
            cache[S] = out
        total += cache[S] * cnt
        n_j += cnt
    mean = total / max(n_j, 1)
    return dict(credit=_share(mean, channels),
                raw={c: float(mean[i]) for i, c in enumerate(channels)},
                distinct_sets=len(set_counts),
                max_journey_channels=max_k,
                lattice_per_journey=1 << max_k,
                evaluations=evaluations,
                mean_value=float(mean.sum()))


def convergence_curve(estimator, exact_credit, channels,
                      perm_counts=(25, 50, 100, 200, 400, 800), seed: int = 0,
                      n_reps: int = 12):
    """Mean absolute deviation from an exact answer, by sample count.

    `estimator(n_perms, seed) -> credit dict`, so the same curve runs for either
    value function and the two are directly comparable. That comparability is the
    point: one curve alone cannot distinguish "converging slowly" from "converged
    to the wrong number".

    AVERAGED OVER `n_reps` SEEDS, and that is not a detail. A single-seed curve
    of a Monte-Carlo estimator is itself a Monte-Carlo draw. One seed came out
    non-monotone (100 permutations beat 200) with a headline ratio of 8.10x
    against a theoretical ceiling of 5.66x; six seeds still read 6.44x. A rate
    ABOVE the 1/sqrt(n) ceiling is not a fast estimator, it is an unconverged
    measurement of an estimator, and it took twelve seeds for the ratio to settle
    just underneath the ceiling where it belongs.

    The number of seeds needed to measure a convergence rate is itself a thing
    that has to be checked, which is the same lesson one level up.
    """
    out = []
    for m in perm_counts:
        errs, maxes = [], []
        for r in range(n_reps):
            est = estimator(m, seed + r)
            devs = [abs(est[c] - exact_credit.get(c, 0.0)) for c in channels]
            errs.append(float(np.mean(devs)))
            maxes.append(float(max(devs)))
        out.append(dict(n_perms=m,
                        mean_abs_error=float(np.mean(errs)),
                        max_abs_error=float(np.mean(maxes)),
                        n_reps=n_reps))
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
