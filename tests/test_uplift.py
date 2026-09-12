"""Guards on the uplift lab: the planted heterogeneity and the grading harness.

The load-bearing checks are (a) the generator really plants heterogeneity with a
near-zero average effect and a negative "sleeping dogs" segment, and (b) the
oracle that ranks by the true tau genuinely beats random — if that ceiling is not
there, no learner score below it means anything.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import uplift as U  # noqa: E402


def test_generator_plants_heterogeneity_with_near_zero_ate():
    d = U.simulate(n=20000, seed=0)
    tau = d["tau_true"]
    assert abs(tau.mean()) < 0.02, "average effect should be ~0 by design"
    assert tau.std() > 0.025, "there must be real heterogeneity to recover"
    assert tau.min() < 0.0, "the sleeping-dogs segment should have negative lift"
    assert tau.max() > 0.05, "high-intent customers should have a clear positive lift"


def test_oracle_beats_random_on_top_decile():
    d = U.simulate(n=20000, seed=0)
    pv = U.policy_value(d["tau_true"], d["tau_true"], frac=0.1)  # oracle ranks by truth
    assert pv["oracle_mean_true_tau"] > 5 * max(pv["random_mean_true_tau"], 1e-4)


@pytest.mark.skipif(U.GradientBoostingClassifier is None, reason="scikit-learn required")
def test_a_learner_recovers_the_ranking_and_models_never_beat_the_oracle():
    rep = U.run(n=20000, seed=0)
    learners = rep["learners"]
    # at least one meta-learner should clearly recover the heterogeneity ordering
    best_spear = max(m["spearman_vs_true_tau"] for m in learners.values())
    best_qini = max(m["qini_vs_oracle"] for m in learners.values())
    assert best_spear > 0.4, ("no learner recovered the ranking", learners)
    assert best_qini > 0.3, ("no learner captured a meaningful share of oracle Qini", learners)
    # the oracle is a ceiling: no model's targeted decile should exceed it
    for m in learners.values():
        pv = m["policy_top_decile"]
        assert pv["model_mean_true_tau"] <= pv["oracle_mean_true_tau"] + 1e-9
        assert pv["model_mean_true_tau"] > pv["random_mean_true_tau"]
