"""Tests for generalization across situations."""

from __future__ import annotations

from tactics import (
    Context,
    ExactEstimator,
    Goal,
    InMemoryStore,
    SimilarityEstimator,
    feature_similarity,
)


class _Tgt:  # minimal stand-in target
    name = "t"

    def observe(self):
        return {}


def _ctx(goal: Goal, features: dict) -> Context:
    return Context(target=_Tgt(), goal=goal, features=features)


def test_feature_similarity_basics():
    assert feature_similarity({}, {}) == 1.0
    assert feature_similarity({"a": 1}, {"a": 1}) == 1.0
    assert feature_similarity({"a": "x"}, {"a": "y"}) == 0.0
    # numeric closeness
    assert feature_similarity({"a": 10}, {"a": 10}) == 1.0
    assert 0.0 < feature_similarity({"a": 10}, {"a": 9}) < 1.0
    # missing key counts against similarity
    assert feature_similarity({"a": 1, "b": 1}, {"a": 1}) == 0.5


def test_exact_estimator_only_sees_identical_situation():
    g = Goal(name="g")
    m = InMemoryStore()
    here = _ctx(g, {"vol": 1.0})
    m.record("t", here.signature(), reward=1.0, success=True, features={"vol": 1.0}, goal="g")

    est = ExactEstimator()
    # identical -> known
    assert est.estimate("t", here, m).trials == 1
    # different situation -> unknown (no generalization)
    elsewhere = _ctx(g, {"vol": 0.9})
    assert est.estimate("t", elsewhere, m).trials == 0


def test_similarity_estimator_borrows_from_similar_situations():
    g = Goal(name="g")
    m = InMemoryStore()
    # Learned that tactic 't' works well when vol is high (~1.0)
    seen = _ctx(g, {"vol": 1.0})
    for _ in range(5):
        m.record("t", seen.signature(), reward=1.0, success=True,
                 features={"vol": 1.0}, goal="g")

    est = SimilarityEstimator(sharpness=2.0)
    # A brand-new but *similar* situation should inherit evidence...
    similar = _ctx(g, {"vol": 0.95})
    near = est.estimate("t", similar, m)
    assert near.trials > 0
    assert near.mean > 0.9

    # ...while a very different situation gets little or nothing.
    different = _ctx(g, {"vol": 0.0})
    far = est.estimate("t", different, m)
    assert far.trials < near.trials


def test_similarity_estimator_respects_goal_boundary():
    m = InMemoryStore()
    ga, gb = Goal(name="A"), Goal(name="B")
    ca = _ctx(ga, {"vol": 1.0})
    m.record("t", ca.signature(), reward=1.0, success=True, features={"vol": 1.0}, goal="A")

    est = SimilarityEstimator()
    # Same features but a different goal must not leak experience across goals.
    cb = _ctx(gb, {"vol": 1.0})
    assert est.estimate("t", cb, m).trials == 0
