"""Tests for the completion pass: the pipeline DAG, dbt's guarantees, k selection,
sampled Shapley, higher-order Markov, and the unit economics."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import attribution as A     # noqa: E402
from src import pipeline as PL       # noqa: E402
from src import scaling as SC        # noqa: E402
from src import segmentation as SEG  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")


# --------------------------------------------------------------------------
# the DAG
# --------------------------------------------------------------------------
def test_tasks_run_in_dependency_order():
    seen = []
    d = PL.DAG()
    d.add(PL.Task("c", lambda: seen.append("c"), depends_on=["b"]))
    d.add(PL.Task("a", lambda: seen.append("a")))
    d.add(PL.Task("b", lambda: seen.append("b"), depends_on=["a"]))
    d.run(verbose=False)
    assert seen == ["a", "b", "c"]


def test_a_cycle_is_reported_rather_than_hung_on():
    d = PL.DAG()
    d.add(PL.Task("a", lambda: None, depends_on=["b"]))
    d.add(PL.Task("b", lambda: None, depends_on=["a"]))
    with pytest.raises(ValueError):
        d.order()


def test_downstream_is_skipped_not_run_on_stale_inputs():
    """The alternative -- running downstream on yesterday's data -- is worse than
    an outage, because a dashboard showing stale numbers with a fresh timestamp
    does not look broken."""
    ran = []
    d = PL.DAG()
    d.add(PL.Task("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    d.add(PL.Task("after", lambda: ran.append(1), depends_on=["bad"]))
    res = {r["task"]: r for r in d.run(verbose=False)}
    assert res["bad"]["status"] == "failed"
    assert res["after"]["status"] == "skipped"
    assert ran == []


def test_an_idempotent_task_is_retried():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    d = PL.DAG().add(PL.Task("flaky", flaky, retries=3, idempotent=True))
    res = d.run(verbose=False)[0]
    assert res["status"] == "ok" and calls["n"] == 3


def test_a_non_idempotent_task_is_never_retried():
    """Retrying a task that appends rows silently doubles them, which is the most
    common way a pipeline corrupts its own output."""
    calls = {"n": 0}

    def appender():
        calls["n"] += 1
        raise RuntimeError("boom")

    d = PL.DAG().add(PL.Task("append", appender, retries=5, idempotent=False))
    res = d.run(verbose=False)[0]
    assert res["status"] == "failed" and calls["n"] == 1


# --------------------------------------------------------------------------
# dbt's guarantees
# --------------------------------------------------------------------------
def _warehouse_ready():
    return os.path.exists(PL.WAREHOUSE)


def test_the_marts_exist_and_are_keyed_by_customer():
    if not _warehouse_ready():
        pytest.skip("run `python run_complete.py` first")
    rfm = PL.query("select * from customer_rfm")
    assert len(rfm) > 0
    assert rfm.customer_id.is_unique


def test_no_holdout_row_falls_inside_the_calibration_window():
    """The leakage guard, asserted here as well as in dbt so it fails even if
    somebody runs the analysis without building."""
    if not _warehouse_ready():
        pytest.skip("run `python run_complete.py` first")
    truth = json.load(open(os.path.join(DATA, "TRUTH.json")))
    bad = PL.query("select count(*) as n from customer_holdout where "
                   "holdout_last_day <= %d" % truth["calibration_days"])
    assert int(bad.n.iloc[0]) == 0


def test_channel_daily_is_reach_and_over_counts_conversions():
    """The model claims to be a reach table. If this ever stops holding, someone
    has quietly turned it into an attribution table."""
    if not _warehouse_ready():
        pytest.skip("run `python run_complete.py` first")
    credited = PL.query(
        "select sum(touches_on_converting_journeys) as n from channel_daily")
    actual = PL.query(
        "select count(distinct journey_id) as n from stg_touches where converted")
    assert int(credited.n.iloc[0]) > int(actual.n.iloc[0])


# --------------------------------------------------------------------------
# k selection
# --------------------------------------------------------------------------
def _blobs(n=900, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [6, 6], [0, 6]], float)
    lab = rng.integers(0, 3, n)
    return centres[lab] + rng.normal(0, 0.5, (n, 2)), lab


def test_silhouette_finds_the_planted_number_of_blobs():
    X, _ = _blobs()
    rows = SEG.k_selection(X, ks=(2, 3, 4, 5), seed=0)
    assert max(rows, key=lambda r: r["silhouette"])["k"] == 3


def test_elbow_is_computed_not_eyeballed():
    X, _ = _blobs()
    rows = SEG.k_selection(X, ks=(2, 3, 4, 5, 6), seed=0)
    assert SEG.elbow_k(rows) in (2, 3, 4)


def test_stability_prefers_coarse_partitions_which_is_its_bias():
    """Reported alongside the others precisely because of this: k=2 is stable on
    almost any data because there is little to disagree about."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))               # no structure at all
    rows = SEG.stability_k(X, ks=(2, 5, 8), n_boot=5, seed=0)
    by_k = {r["k"]: r["mean_ari"] for r in rows}
    assert by_k[2] > by_k[8]


def test_forward_separation_is_adjusted_for_k():
    """Raw eta^2 rises with k mechanically, so an unadjusted table always
    recommends the largest k on offer."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    future = rng.normal(size=500)               # unrelated to X
    rows = SEG.forward_separation(X, future, ks=(2, 6), seed=0)
    raw = {r["k"]: r["eta_squared"] for r in rows}
    adj = {r["k"]: r["adjusted_eta_squared"] for r in rows}
    assert raw[6] > raw[2]
    assert adj[6] < raw[6]


def test_hdbscan_reports_a_noise_share():
    X, _ = _blobs()
    out = SEG.hdbscan_segments(X, min_cluster_size=40)
    assert 0.0 <= out["noise_share"] <= 1.0
    assert out["n_clusters"] >= 2


# --------------------------------------------------------------------------
# sampled Shapley
# --------------------------------------------------------------------------
def _sim(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    chans = ["a", "b", "c", "d"]
    eff = {"a": 0.30, "b": 0.15, "c": 0.05, "d": 0.0}
    js, ys = [], []
    for _ in range(n):
        j = [c for c in chans if rng.random() < 0.5] or ["a"]
        p = 0.05 + sum(eff[c] for c in set(j))
        js.append(j)
        ys.append(int(rng.random() < p))
    return js, ys, chans


def test_sampled_shapley_converges_to_exact():
    js, ys, chans = _sim()
    exact = A.shapley(js, ys, chans)
    curve = SC.shapley_error_curve(js, ys, chans, exact,
                                   perm_counts=(20, 400), seed=0)
    assert curve[-1]["mean_abs_error"] <= curve[0]["mean_abs_error"] + 1e-9
    assert curve[-1]["mean_abs_error"] < 0.05


def test_sampled_shapley_returns_a_distribution_with_standard_errors():
    js, ys, chans = _sim()
    out = SC.shapley_sampled(js, ys, chans, n_perms=100, seed=0)
    assert sum(out["credit"].values()) == pytest.approx(1.0)
    assert all(out["se"][c] >= 0 for c in chans)


def test_the_zero_effect_channel_gets_least_credit_when_it_is_added_at_random():
    """The dummy-player axiom, on the clean case. The report's planted channel
    fails it because it is targeted, which is the whole point."""
    js, ys, chans = _sim()
    out = SC.shapley_sampled(js, ys, chans, n_perms=300, seed=1)["credit"]
    assert out["d"] == min(out.values())


def test_unobserved_coalitions_are_skipped_not_treated_as_zero():
    """Treating them as rate zero injects spurious negative marginals that
    normalisation amplifies -- the bug that once gave two interchangeable
    channels 1.0 and 0.0."""
    js = [["a"], ["a", "b"], ["b"]] * 200
    ys = [1, 1, 0] * 200
    out = SC.shapley_sampled(js, ys, ["a", "b", "c"], n_perms=200, seed=0)
    assert out["credit"]["c"] == pytest.approx(0.0, abs=1e-9)
    assert out["observed_coalitions"] == 3


# --------------------------------------------------------------------------
# higher-order Markov
# --------------------------------------------------------------------------
def test_state_count_grows_with_order():
    js, ys, chans = _sim(n=1500)
    o1 = SC.markov_removal_order_k(js, ys, chans, order=1)
    o2 = SC.markov_removal_order_k(js, ys, chans, order=2)
    assert o2["n_states"] > o1["n_states"]


def test_thin_state_share_grows_with_order():
    """The diagnostic that says where a higher-order model starts fitting
    individual journeys -- not the error, which keeps improving on the data it
    was fitted to."""
    js, ys, chans = _sim(n=1500)
    o1 = SC.markov_removal_order_k(js, ys, chans, order=1)
    o3 = SC.markov_removal_order_k(js, ys, chans, order=3)
    assert o3["thin_state_share"] >= o1["thin_state_share"]


def test_removal_credit_is_a_distribution():
    js, ys, chans = _sim(n=1200)
    out = SC.markov_removal_order_k(js, ys, chans, order=2)
    assert sum(out["credit"].values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in out["credit"].values())


def test_journey_deletion_removal_effects_are_zero_even_for_a_necessary_channel():
    """THE FINDING OF SECTION D, pinned so it cannot silently change.

    `e` appears on every conversion and nowhere else, so the textbook removal
    effect would credit it entirely. This implementation removes a channel by
    DELETING THE TOUCH FROM THE JOURNEYS and re-estimating -- the counterfactual
    a marketer actually means -- and under that definition the conversion
    probability does not move, because in observational path data the outcome is
    attached to the journey rather than to the path.

    The textbook version gets a non-zero number by deleting the NODE FROM THE
    GRAPH, which strands its inbound mass in the null state. That number comes
    from the graph representation, not from the channel.
    """
    js = [["e"]] * 300 + [["z"]] * 300
    ys = [1] * 300 + [0] * 300
    out = SC.markov_removal_order_k(js, ys, ["e", "z"], order=1)
    assert out["max_abs_signed"] < 1e-6, out["signed_effects"]
    assert all(v == 0.0 for v in out["credit"].values())


def test_the_graph_deletion_version_does_give_a_number():
    """The control: the same data through the textbook implementation returns a
    confident non-zero credit. Two defensible definitions, completely different
    answers -- which is the point of running both."""
    js = [["e"]] * 300 + [["z"]] * 300
    ys = [1] * 300 + [0] * 300
    credit = A.markov_removal(js, ys, ["e", "z"])
    assert credit["e"] > credit["z"]


# --------------------------------------------------------------------------
# economics
# --------------------------------------------------------------------------
def test_channel_conversions_touched_over_counts_the_total():
    """Every conversion touched by three channels is counted three times, which
    is why channel CACs never reconcile to the blended figure."""
    js = [["a", "b", "c"]] * 100
    ys = [1] * 100
    rows = SC.channel_economics(js, ys, ["a", "b", "c"],
                                {"a": 1.0, "b": 1.0, "c": 1.0},
                                {i: 10.0 for i in range(100)}, list(range(100)))
    assert sum(r["conversions_touched"] for r in rows) == 300
    assert sum(ys) == 100


def test_spend_scales_with_impressions_not_with_journeys():
    js = [["a", "a", "a"]] * 10
    ys = [1] * 10
    rows = SC.channel_economics(js, ys, ["a"], {"a": 2.0},
                                {i: 0.0 for i in range(10)}, list(range(10)))
    assert rows[0]["impressions"] == 30
    assert rows[0]["spend"] == pytest.approx(60.0)


def test_incremental_cac_is_infinite_for_a_zero_effect_channel():
    """A channel that causes nothing has a defensible-looking observational CAC
    and an infinite incremental one. That gap is the business case for the
    experiment, in dollars."""
    rows = SC.channel_economics(
        [["a", "z"]] * 50, [1] * 50, ["a", "z"], {"a": 1.0, "z": 1.0},
        {i: 5.0 for i in range(50)}, list(range(50)))
    inc = SC.incremental_economics(rows, {"a": 1.0, "z": 0.0}, 50, 5.0)
    z = [r for r in inc if r["channel"] == "z"][0]
    assert np.isfinite(z["observational_cac"])
    assert not np.isfinite(z["incremental_cac"])
