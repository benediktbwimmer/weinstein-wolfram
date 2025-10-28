"""Unit tests for the hypergraph rewrite starter kit."""

from __future__ import annotations

from engine.hypergraph import Hypergraph
from engine.rewrite import EdgeSplit3Rule, RewriteEngine
from metrics import compute_unification_summary
from metrics.geom import mean_forman_curvature, spectral_dimension


def test_rewrite_engine_grows_hypergraph() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=0)
    engine.run(steps=10)
    assert hypergraph.node_count > 3
    assert hypergraph.edge_count > 1
    assert len(engine.events) == 10
    stats = engine.causal_graph.basic_stats()
    assert stats["event_count"] == 10


def test_geometric_metrics_return_values() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=1)
    engine.run(steps=25)
    skeleton = hypergraph.one_skeleton()
    ds = spectral_dimension(skeleton, max_time=4, trials=200, seed=2)
    curvature = mean_forman_curvature(skeleton)
    assert ds == ds  # not NaN
    assert curvature == curvature  # not NaN


def test_unification_summary_blends_metrics() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=3)
    engine.run(steps=20)
    summary = compute_unification_summary(
        engine,
        spectral_max_time=4,
        spectral_trials=80,
        spectral_seed=4,
    )
    assert summary["event_count"] == 20
    assert summary["node_count"] >= 3
    assert summary["information_density"] == summary["information_density"]
    assert summary["unity_consistency"] == summary["unity_consistency"]
    assert summary["causal_max_depth"] >= 0
