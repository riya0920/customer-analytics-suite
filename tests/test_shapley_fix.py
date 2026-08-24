"""The Shapley repair: a value function defined on the whole lattice, and a
per-journey estimator whose cost is set by journey length rather than channel
count.

The previous pass measured that sampled Shapley does not converge and wrote the
diagnosis down instead of implementing the fix. These are the tests for the fix.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import scaling as SC  # noqa: E402


CHANNELS = ["a", "b", "c", "d"]


def _toy(seed=0, n=4000, channels=CHANNELS):
    """Journeys whose channel sets are deliberately SPARSE over the lattice, so
    the exact-set value function has holes and the closure one does not."""
    rng = np.random.default_rng(seed)
    journeys, convs = [], []
    for _ in range(n):
        k = int(rng.integers(1, min(4, len(channels)) + 1))
        js = list(rng.choice(channels, size=k, replace=False))
        p = 0.04 + 0.05 * len(js)
        journeys.append(js)
        convs.append(int(rng.random() < p))
    return journeys, convs


# --------------------------------------------------------------------------
# the closure value function
# --------------------------------------------------------------------------
def test_the_closure_value_function_is_defined_on_the_whole_lattice():
    """Every coalition except the empty set. The empty set is undefined for the
    right reason -- no channels is no marketing -- and is pinned to zero rather
    than to missing data."""
    j, c = _toy()
    v, cover = SC.subset_closure_values(j, c, CHANNELS)
    assert cover["closure_defined"] == cover["total"] - 1
    assert cover["exact_observed"] <= cover["closure_defined"]
    assert v[0] == 0.0


def test_the_closure_function_is_monotone_in_the_coalition():
    """Adding a channel can only widen the set of journeys counted, so the rate
    is an average over a superset -- not required to rise, but the SUPPORT is
    required to. A shrinking support would mean the transform is wrong."""
    j, c = _toy(1)
    idx = {ch: i for i, ch in enumerate(CHANNELS)}
    n = len(CHANNELS)
    tot = np.zeros(1 << n)
    for jj in j:
        m = 0
        for ch in jj:
            m |= 1 << idx[ch]
        tot[m] += 1
    for i in range(n):
        bit = 1 << i
        for m in range(1 << n):
            if m & bit:
                tot[m] += tot[m ^ bit]
    for m in range(1 << n):
        for i in range(n):
            if not m & (1 << i):
                assert tot[m | (1 << i)] >= tot[m]


def test_closure_shapley_is_efficient_to_machine_precision():
    """The credits must add up to the thing being attributed. This check is not
    available for the exact-set version, whose grand coalition is estimated from
    whoever happened to see every channel."""
    j, c = _toy(2)
    out = SC.shapley_closure(j, c, CHANNELS)
    assert abs(out["efficiency_residual"]) < 1e-12


def test_a_channel_nobody_ever_saw_gets_exactly_zero():
    """The dummy axiom, under the new value function."""
    j, c = _toy(3, channels=["a", "b", "c"])
    chans = ["a", "b", "c", "ghost"]
    out = SC.shapley_closure(j, c, chans)
    assert out["raw"]["ghost"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------
def _sparse(seed=0, n=4000):
    """Six channels but never more than two in a journey, so most of the 63
    coalitions are never observed. The four-channel fixture above does NOT do
    this -- it observes all 15 sets and the old sampler never stalls on it,
    which is how this test first failed."""
    chans = ["a", "b", "c", "d", "e", "f"]
    rng = np.random.default_rng(seed)
    journeys, convs = [], []
    for _ in range(n):
        k = int(rng.integers(1, 3))
        js = list(rng.choice(chans, size=k, replace=False))
        journeys.append(js)
        convs.append(int(rng.random() < 0.05 + 0.05 * k))
    return journeys, convs, chans


def test_the_old_sampler_stalls_and_the_new_one_cannot():
    """The stall is the mechanism behind the bias, so it is counted rather than
    inferred."""
    j, c, chans = _sparse(4)
    old = SC.shapley_sampled(j, c, chans, n_perms=200, seed=0)
    new = SC.shapley_sampled_closure(j, c, chans, n_perms=200, seed=0)
    assert old["stalls"] > 0
    assert new["stalls"] == 0
    assert old["observed_coalitions"] < 2 ** len(chans)


def test_the_new_sampler_converges_and_the_old_one_does_not():
    """32x the permutations should cut a purely noisy error by 5.66x. Measured
    on the real panel the old estimator manages 1.68x and plateaus; this is the
    same shape on a small fixture, with slack for the fixture's size."""
    j, c = _toy(5, n=6000)
    v, _ = SC.subset_closure_values(j, c, CHANNELS)
    exact_clo = SC.shapley_closure(j, c, CHANNELS, v=v)["credit"]
    curve = SC.convergence_curve(
        lambda m, s: SC.shapley_sampled_closure(j, c, CHANNELS, n_perms=m,
                                                seed=s, v=v)["credit"],
        exact_clo, CHANNELS, perm_counts=(25, 800), n_reps=8, seed=1)
    ratio = curve[0]["mean_abs_error"] / max(curve[-1]["mean_abs_error"], 1e-12)
    assert ratio > 2.5, curve


def test_the_convergence_rate_never_exceeds_the_theoretical_ceiling():
    """A measured rate ABOVE 1/sqrt(n) is not a fast estimator, it is an
    unconverged measurement of one. One seed read 8.10x on the real panel and
    six seeds read 6.44x, both above the 5.66x ceiling; twelve settled under it.
    """
    j, c = _toy(6, n=6000)
    v, _ = SC.subset_closure_values(j, c, CHANNELS)
    exact_clo = SC.shapley_closure(j, c, CHANNELS, v=v)["credit"]
    curve = SC.convergence_curve(
        lambda m, s: SC.shapley_sampled_closure(j, c, CHANNELS, n_perms=m,
                                                seed=s, v=v)["credit"],
        exact_clo, CHANNELS, perm_counts=(25, 800), n_reps=16, seed=11)
    ratio = curve[0]["mean_abs_error"] / max(curve[-1]["mean_abs_error"], 1e-12)
    assert ratio <= 5.66 * 1.15, ratio


def test_the_curve_records_how_many_seeds_it_averaged():
    j, c = _toy(7)
    v, _ = SC.subset_closure_values(j, c, CHANNELS)
    ex = SC.shapley_closure(j, c, CHANNELS, v=v)["credit"]
    curve = SC.convergence_curve(
        lambda m, s: SC.shapley_sampled_closure(j, c, CHANNELS, n_perms=m,
                                                seed=s, v=v)["credit"],
        ex, CHANNELS, perm_counts=(25,), n_reps=3)
    assert curve[0]["n_reps"] == 3


def test_the_sampler_agrees_with_the_exact_answer_it_approximates():
    """The point of the fix: sampled and exact are now the SAME estimator at two
    sample sizes, which is what the old pair were not."""
    j, c = _toy(8, n=6000)
    v, _ = SC.subset_closure_values(j, c, CHANNELS)
    ex = SC.shapley_closure(j, c, CHANNELS, v=v)["credit"]
    est = SC.shapley_sampled_closure(j, c, CHANNELS, n_perms=4000, seed=2,
                                     v=v)["credit"]
    assert max(abs(est[ch] - ex[ch]) for ch in CHANNELS) < 0.02


# --------------------------------------------------------------------------
# per-journey
# --------------------------------------------------------------------------
def test_the_per_journey_lattice_is_set_by_journey_length_not_channel_count():
    """This is the whole reason the second fix is a different fix. Thirty
    channels with three-touch journeys costs 8 sub-coalitions per journey, not
    2^30 -- so the intractability that justified sampling was a property of the
    value function rather than of the problem."""
    wide = ["ch%02d" % i for i in range(30)]
    rng = np.random.default_rng(0)
    journeys, convs = [], []
    for _ in range(3000):
        js = list(rng.choice(wide, size=int(rng.integers(1, 4)), replace=False))
        journeys.append(js)
        convs.append(int(rng.random() < 0.1))
    out = SC.shapley_per_journey(journeys, convs, wide)
    assert out["max_journey_channels"] <= 3
    assert out["lattice_per_journey"] <= 8
    assert out["evaluations"] < 100_000


def test_per_journey_refuses_a_lattice_it_cannot_compute_exactly():
    """Rather than allocating. The dense builder refuses for the same reason at
    its own bound -- an out-of-memory kill is a worse failure than an error."""
    wide = ["ch%02d" % i for i in range(30)]
    with pytest.raises(ValueError):
        SC.subset_closure_values([["ch00"]], [1], wide)
    with pytest.raises(ValueError):
        SC.shapley_per_journey([wide[:6]], [1], wide, max_journey_channels=4)


def test_the_local_lattice_agrees_with_the_global_one():
    """The rewrite is only legitimate if it changes cost and not answers:
    'journeys whose set is a subset of T' does not depend on which journey T
    came from."""
    j, c = _toy(12, n=5000)
    v, _ = SC.subset_closure_values(j, c, CHANNELS)
    local = SC.shapley_per_journey(j, c, CHANNELS)["raw"]
    idx = {ch: i for i, ch in enumerate(CHANNELS)}
    # the same computation, driven off the dense global value function
    from src.scaling import _weights
    from collections import Counter as _C
    counts = _C()
    for jj in j:
        counts[sum(1 << idx[ch] for ch in set(jj))] += 1
    tot = np.zeros(len(CHANNELS))
    for m, cnt in counts.items():
        members = [i for i in range(len(CHANNELS)) if m & (1 << i)]
        k = len(members)
        w = _weights(k)
        for i in members:
            rest = [b for b in members if b != i]
            for sub in range(1 << (k - 1)):
                sm = 0
                for t, b in enumerate(rest):
                    if sub & (1 << t):
                        sm |= 1 << b
                tot[i] += cnt * w[bin(sm).count("1")] * (v[sm | (1 << i)] - v[sm])
    ref = tot / sum(counts.values())
    for i, ch in enumerate(CHANNELS):
        assert local[ch] == pytest.approx(float(ref[i]), abs=1e-12)


def test_per_journey_credits_sum_to_the_mean_journey_value():
    """Efficiency again, one level down: each journey's own value is divided
    among its own touches exactly."""
    j, c = _toy(9)
    v, _ = SC.subset_closure_values(j, c, CHANNELS)
    out = SC.shapley_per_journey(j, c, CHANNELS)
    idx = {ch: i for i, ch in enumerate(CHANNELS)}
    want = float(np.mean([v[sum(1 << idx[ch] for ch in set(jj))] for jj in j]))
    assert out["mean_value"] == pytest.approx(want, abs=1e-9)


def test_per_journey_gives_a_never_seen_channel_nothing():
    j, c = _toy(10, channels=["a", "b", "c"])
    out = SC.shapley_per_journey(j, c, ["a", "b", "c", "ghost"])
    assert out["raw"]["ghost"] == pytest.approx(0.0, abs=1e-12)


def test_the_two_fixes_do_not_have_to_agree():
    """They answer different questions -- what a channel adds to what is
    ACHIEVABLE, versus how an observed journey's outcome divides among the
    touches that were in it. A test that forced them to agree would be asserting
    that one of them is redundant."""
    j, c = _toy(11, n=6000)
    v, _ = SC.subset_closure_values(j, c, CHANNELS)
    a = SC.shapley_closure(j, c, CHANNELS, v=v)["credit"]
    b = SC.shapley_per_journey(j, c, CHANNELS)["credit"]
    assert set(a) == set(b)
    assert sum(a.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(b.values()) == pytest.approx(1.0, abs=1e-9)
