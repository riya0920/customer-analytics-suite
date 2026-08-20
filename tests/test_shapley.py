"""Tests for Shapley attribution, day-based time decay, and the CLV link."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import attribution as A  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CH = ["display", "social", "email", "paid_search", "retargeting"]


# --------------------------------------------------------------------------
# Shapley
# --------------------------------------------------------------------------
def test_shapley_returns_a_distribution():
    journeys = [["display"], ["display", "email"], ["email"], ["social"]]
    conv = [1, 1, 0, 1]
    credit = A.shapley(journeys, conv, CH)
    assert set(credit) == set(CH)
    assert sum(credit.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in credit.values())


def test_shapley_gives_a_dummy_channel_zero_when_the_data_is_clean():
    """THE axiom Shapley has and the heuristics do not: a channel that never
    changes any coalition's conversion rate earns exactly nothing.

    This is the CLEAN case -- the dummy channel is added at random rather than
    targeted. The report's planted channel fails this precisely because it is
    NOT added at random, which is the whole point."""
    rng = np.random.default_rng(0)
    journeys, conv = [], []
    for _ in range(6000):
        j = ["email"] if rng.random() < 0.5 else ["display"]
        p = 0.5 if j[0] == "email" else 0.2
        # dummy added independently of everything, including the outcome
        if rng.random() < 0.5:
            j = j + ["retargeting"]
        journeys.append(j)
        conv.append(int(rng.random() < p))
    credit = A.shapley(journeys, conv, ["display", "email", "retargeting"])
    assert credit["retargeting"] < 0.06, credit
    assert credit["email"] > credit["display"]


def test_shapley_credits_a_channel_that_lifts_every_coalition():
    rng = np.random.default_rng(1)
    journeys, conv = [], []
    for _ in range(6000):
        j = ["display"]
        p = 0.2
        if rng.random() < 0.5:
            j = j + ["email"]
            p += 0.3                      # email genuinely lifts
        journeys.append(j)
        conv.append(int(rng.random() < p))
    credit = A.shapley(journeys, conv, ["display", "email"])
    assert credit["email"] > credit["display"]


def test_shapley_is_symmetric_for_interchangeable_channels():
    rng = np.random.default_rng(2)
    journeys, conv = [], []
    for _ in range(4000):
        ch = "display" if rng.random() < 0.5 else "social"
        journeys.append([ch])
        conv.append(int(rng.random() < 0.3))
    credit = A.shapley(journeys, conv, ["display", "social"])
    assert credit["display"] == pytest.approx(credit["social"], abs=0.08)


def test_shapley_handles_channels_that_never_appear():
    journeys = [["email"], ["email", "display"]]
    conv = [1, 1]
    credit = A.shapley(journeys, conv, CH)
    assert credit["paid_search"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# time decay over days
# --------------------------------------------------------------------------
def test_time_decay_uses_days_when_given_them():
    """Two journeys with identical POSITIONS but different timing must get
    different credit -- that is the whole reason to carry timestamps."""
    journeys = [["display", "email"], ["display", "email"]]
    conv = [1, 1]
    recent = [[29.0, 30.0], [29.0, 30.0]]      # display just before conversion
    stale = [[0.0, 30.0], [0.0, 30.0]]         # display a month before
    c_recent = A.time_decay(journeys, conv, CH, touch_days=recent)
    c_stale = A.time_decay(journeys, conv, CH, touch_days=stale)
    assert c_recent["display"] > c_stale["display"]


def test_time_decay_falls_back_to_position_without_timestamps():
    journeys = [["display", "email"]]
    conv = [1]
    credit = A.time_decay(journeys, conv, CH)
    assert sum(credit.values()) == pytest.approx(1.0)
    assert credit["email"] > credit["display"]


def test_time_decay_still_favours_later_touches_on_days():
    journeys = [["display", "social", "email"]]
    conv = [1]
    credit = A.time_decay(journeys, conv, CH, touch_days=[[0.0, 15.0, 30.0]])
    assert credit["email"] > credit["social"] > credit["display"]


def test_all_methods_including_shapley_are_registered():
    assert "shapley" in A.METHODS
    # A fixture with enough coalitions for Shapley to be identified at all.
    # One journey is not enough: with a single observed channel set there are
    # no marginal contributions to average, and Shapley correctly returns
    # nothing rather than inventing a split.
    rng = np.random.default_rng(9)
    journeys, conv = [], []
    for _ in range(2000):
        j = [c for c in ("display", "email") if rng.random() < 0.6] or ["display"]
        journeys.append(j)
        conv.append(int(rng.random() < 0.3))
    for name, fn in A.METHODS.items():
        credit = fn(journeys, conv, CH)
        assert sum(credit.values()) == pytest.approx(1.0), name


def test_shapley_returns_nothing_when_nothing_is_identifiable():
    """One journey cannot identify a coalition function. Returning all zeros is
    the honest answer; inventing a split would not be."""
    credit = A.shapley([["display", "email"]], [1], CH)
    assert sum(credit.values()) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# the customer link
# --------------------------------------------------------------------------
def test_generated_journeys_carry_customer_ids_and_timestamps():
    if not os.path.exists(os.path.join(DATA, "journeys.json")):
        pytest.skip("run `python src/generate.py` first")
    with open(os.path.join(DATA, "journeys.json")) as f:
        jd = json.load(f)
    assert "customer_id" in jd and "touch_days" in jd
    assert len(jd["customer_id"]) == len(jd["journeys"])
    assert len(jd["touch_days"]) == len(jd["journeys"])
    for j, t in zip(jd["journeys"][:200], jd["touch_days"][:200]):
        assert len(j) == len(t)
        assert t == sorted(t), "touch timestamps must be ordered"


def test_customer_ids_are_unique_per_journey():
    if not os.path.exists(os.path.join(DATA, "journeys.json")):
        pytest.skip("run `python src/generate.py` first")
    with open(os.path.join(DATA, "journeys.json")) as f:
        jd = json.load(f)
    ids = jd["customer_id"]
    assert len(set(ids)) == len(ids), "one journey per customer in this simulator"


def test_even_shapley_credits_the_planted_channel():
    """The strongest form of the project's central finding. Shapley has a formal
    dummy-player axiom, so if IT credits the zero-effect channel, the problem is
    definitively the DATA (a targeted confound) and not the estimator."""
    if not os.path.exists(os.path.join(DATA, "TRUTH.json")):
        pytest.skip("run `python src/generate.py` first")
    with open(os.path.join(DATA, "TRUTH.json")) as f:
        truth = json.load(f)
    with open(os.path.join(DATA, "journeys.json")) as f:
        jd = json.load(f)
    zc = truth["zero_effect_channel"]
    assert truth["channel_effects"][zc] == 0.0
    credit = A.shapley(jd["journeys"], jd["conversions"],
                       list(truth["channel_effects"]))
    assert credit[zc] > 0.05, \
        "the confound is not landing; the planted-channel section has no teeth"
