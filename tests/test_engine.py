"""Unit tests for the hypergraph rewrite starter kit."""

from __future__ import annotations

from engine.hypergraph import Hypergraph
from engine.rewrite import EdgeSplit3Rule, RewriteEngine
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
