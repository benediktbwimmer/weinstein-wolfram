"""Unit tests for the hypergraph rewrite starter kit."""

from __future__ import annotations

import pytest

from engine.hypergraph import Hypergraph
from engine.rewrite import EdgeSplit3Rule, RewriteEngine
from metrics import (
    collect_unification_dynamics,
    compute_unification_summary,
    generate_unification_certificate,
)
from metrics.geom import (
    average_clustering_coefficient,
    mean_forman_curvature,
    spectral_dimension,
)


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
    clustering = average_clustering_coefficient(skeleton)
    assert ds == ds  # not NaN
    assert curvature == curvature  # not NaN
    assert clustering == clustering  # not NaN


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
    assert summary["mean_clustering_coefficient"] == summary["mean_clustering_coefficient"]


def test_collect_unification_dynamics_tracks_growth() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=7)
    history = collect_unification_dynamics(
        engine,
        steps=4,
        spectral_max_time=4,
        spectral_trials=40,
        spectral_seed=11,
    )

    assert len(history) == 5  # initial snapshot + four new events
    event_counts = [entry["event_count"] for entry in history]
    assert event_counts == [float(i) for i in range(5)]

    node_counts = [entry["node_count"] for entry in history]
    assert node_counts[0] < node_counts[-1]
    for earlier, later in zip(node_counts, node_counts[1:]):
        assert later >= earlier

    final_unity = history[-1]["unity_consistency"]
    assert final_unity == final_unity  # not NaN
    assert history[-1]["mean_clustering_coefficient"] == history[-1][
        "mean_clustering_coefficient"
    ]


def test_collect_unification_dynamics_requires_non_negative_steps() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=9)
    with pytest.raises(ValueError):
        collect_unification_dynamics(engine, steps=-1)


def test_generate_unification_certificate_reports_bridge_metrics() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=5)
    certificate = generate_unification_certificate(
        engine,
        steps=18,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=13,
    )
    dual = certificate["dual_correlation"]
    synergy = certificate["causal_synergy"]
    strength = certificate["certificate_strength"]

    assert 0.0 <= dual <= 1.0000001
    assert synergy > 0
    assert strength > 0
