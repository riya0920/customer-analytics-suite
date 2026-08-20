"""Guards on the CLV estimators and the attribution methods.

The parameter-recovery test is the load-bearing one: if BG/NBD cannot recover
parameters from data drawn with known parameters, every CLV number downstream is
noise wearing a Greek letter.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import attribution as A  # noqa: E402
from src import clv as CLV  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CH = ["display", "social", "email", "paid_search", "retargeting"]


# --------------------------------------------------------------------------
# BG/NBD
# --------------------------------------------------------------------------
def _simulate_bgnbd(n, r, alpha, a, b, T, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    x, t_x = np.zeros(n), np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / max(lam[i], 1e-9))
            if t > T:
                break
            x[i] += 1
            t_x[i] = t
            if rng.random() < p[i]:
                break
    return x, t_x, np.full(n, float(T))


def test_bgnbd_recovers_known_parameters():
    """Fit on data drawn from a BG/NBD process with known parameters and check
    the fitted ones are in the right neighbourhood."""
    r, alpha, a, b = 0.8, 12.0, 1.0, 4.0
    x, t_x, T = _simulate_bgnbd(4000, r, alpha, a, b, 300, seed=3)
    m = CLV.BGNBD().fit(x, t_x, T)
    # generous bands: 4000 customers is not many for a 4-parameter fit
    assert 0.4 < m.r < 1.6, m.r
    assert 5.0 < m.alpha < 28.0, m.alpha
    assert 0.3 < m.a < 3.5, m.a
    assert 1.0 < m.b < 14.0, m.b


def test_p_alive_falls_with_recency():
    """A customer who bought a lot and stopped long ago must look less alive than
    one who bought the same amount recently."""
    m = CLV.BGNBD()
    m.r, m.alpha, m.a, m.b = 0.8, 12.0, 1.0, 4.0
    recent = m.p_alive(np.array([10.0]), np.array([290.0]), np.array([300.0]))[0]
    stale = m.p_alive(np.array([10.0]), np.array([50.0]), np.array([300.0]))[0]
    assert recent > stale


def test_expected_purchases_is_monotone_in_horizon():
    m = CLV.BGNBD()
    m.r, m.alpha, m.a, m.b = 0.8, 12.0, 1.5, 4.0
    x, t_x, T = np.array([5.0]), np.array([200.0]), np.array([300.0])
    prev = -1.0
    for h in (10, 60, 180, 365):
        cur = m.expected_purchases(h, x, t_x, T)[0]
        assert cur > prev
        prev = cur


def test_zero_purchase_customers_get_low_expectations():
    m = CLV.BGNBD()
    m.r, m.alpha, m.a, m.b = 0.8, 12.0, 1.5, 4.0
    none = m.expected_purchases(180, np.array([0.0]), np.array([0.0]),
                                np.array([300.0]))[0]
    many = m.expected_purchases(180, np.array([20.0]), np.array([290.0]),
                                np.array([300.0]))[0]
    assert none < many


# --------------------------------------------------------------------------
# Gamma-Gamma
# --------------------------------------------------------------------------
def test_gamma_gamma_shrinks_toward_the_population_mean():
    """The whole reason to use it: a customer with one purchase should be pulled
    hard toward the population mean, one with fifty barely at all."""
    gg = CLV.GammaGamma()
    gg.p, gg.q, gg.v = 3.0, 5.0, 100.0
    pop = gg.p * gg.v / (gg.q - 1)
    one = gg.expected_value(np.array([1.0]), np.array([500.0]))[0]
    fifty = gg.expected_value(np.array([50.0]), np.array([500.0]))[0]
    assert abs(one - pop) < abs(fifty - pop)
    assert fifty > one


def test_gamma_gamma_uses_population_mean_for_zero_purchase_customers():
    gg = CLV.GammaGamma()
    gg.p, gg.q, gg.v = 3.0, 5.0, 100.0
    pop = gg.p * gg.v / (gg.q - 1)
    assert gg.expected_value(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(pop)


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------
def test_every_method_returns_a_distribution():
    journeys = [["display", "email"], ["social"], ["display", "social", "email"]]
    conv = [1, 1, 0]
    for name, fn in A.METHODS.items():
        credit = fn(journeys, conv, CH)
        assert set(credit) == set(CH)
        assert sum(credit.values()) == pytest.approx(1.0), name
        assert all(v >= 0 for v in credit.values()), name


def test_last_touch_credits_only_the_closer():
    journeys = [["display", "social", "email"]]
    credit = A.last_touch(journeys, [1], CH)
    assert credit["email"] == pytest.approx(1.0)
    assert credit["display"] == 0.0


def test_first_touch_credits_only_the_opener():
    journeys = [["display", "social", "email"]]
    credit = A.first_touch(journeys, [1], CH)
    assert credit["display"] == pytest.approx(1.0)


def test_linear_splits_evenly():
    credit = A.linear([["display", "social"]], [1], CH)
    assert credit["display"] == pytest.approx(0.5)
    assert credit["social"] == pytest.approx(0.5)


def test_time_decay_favours_later_touches():
    credit = A.time_decay([["display", "social", "email"]], [1], CH)
    assert credit["email"] > credit["social"] > credit["display"]


def test_non_converting_journeys_get_no_credit():
    assert A.last_touch([["display"]], [0], CH)["display"] == 0.0


def test_markov_removal_ranks_a_necessary_channel_highest():
    """A channel present on every conversion and nowhere else must dominate."""
    journeys = [["email"], ["email"], ["email"], ["display"], ["display"]]
    conv = [1, 1, 1, 0, 0]
    credit = A.markov_removal(journeys, conv, CH)
    assert credit["email"] > credit["display"]


# --------------------------------------------------------------------------
# the planted channel -- the project's central claim
# --------------------------------------------------------------------------
def test_planted_channel_has_zero_true_effect_but_is_credited():
    """If the generated data does not reproduce this, section 4 is fiction."""
    if not os.path.exists(os.path.join(DATA, "TRUTH.json")):
        pytest.skip("run `python src/generate.py` first")
    with open(os.path.join(DATA, "TRUTH.json")) as f:
        truth = json.load(f)
    with open(os.path.join(DATA, "journeys.json")) as f:
        jd = json.load(f)
    zc = truth["zero_effect_channel"]
    assert truth["channel_effects"][zc] == 0.0

    channels = list(truth["channel_effects"])
    for name, fn in A.METHODS.items():
        credit = fn(jd["journeys"], jd["conversions"], channels)
        assert credit[zc] > 0.01, \
            "%s credits the zero-effect channel %.4f; the confound is not landing" \
            % (name, credit[zc])


def test_budget_allocation_is_exhaustive():
    credit = {c: 1.0 / len(CH) for c in CH}
    alloc = A.budget_allocation(credit, 1000.0)
    assert sum(alloc.values()) == pytest.approx(1000.0)


def test_spending_on_a_zero_effect_channel_buys_nothing():
    effects = {c: 0.0 for c in CH}
    effects["email"] = 0.05
    costs = {c: 0.01 for c in CH}
    good = A.conversions_under({"email": 1000.0}, effects, costs, 0.1, 5000)
    bad = A.conversions_under({"retargeting": 1000.0}, effects, costs, 0.1, 5000)
    assert good > bad
