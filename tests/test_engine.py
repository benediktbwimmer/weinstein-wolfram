"""Unit tests for the hypergraph rewrite starter kit."""

from __future__ import annotations

import math

import pytest

from engine.hypergraph import Hypergraph
from engine.multiway import MultiwaySystem
from engine.rewrite import EdgeSplit3Rule, RewriteEngine
from metrics import (
    ToyModelResult,
    assess_unification_robustness,
    collect_unification_dynamics,
    compute_unification_summary,
    derive_unification_principles,
    evaluate_unification_alignment,
    generate_unification_certificate,
    run_toy_unification_model,
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


def test_multiway_system_evolution_records_branching() -> None:
    base = Hypergraph([(0, 1, 2), (0, 2, 3)])
    system = MultiwaySystem(base, [EdgeSplit3Rule()])
    evolution = system.run(max_generations=2)

    histogram = evolution.depth_histogram()
    assert histogram[0] == 1
    assert sum(histogram.values()) == evolution.state_count
    assert evolution.event_count >= histogram.get(1, 0)
    assert evolution.average_branching_factor() > 0
    assert all(
        child in evolution.states
        for children in evolution.adjacency.values()
        for child in children
    )


def test_multiway_system_validates_generations() -> None:
    system = MultiwaySystem(Hypergraph([(0, 1, 2)]), [EdgeSplit3Rule()])
    with pytest.raises(ValueError):
        system.run(max_generations=-1)


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
    assert summary["multiway_state_count"] >= 1
    assert summary["multiway_event_count"] >= 0
    assert summary["multiway_avg_branching_factor"] >= 0
    assert summary["multiway_frontier_size"] >= 1


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


def test_collect_unification_dynamics_with_sampling_interval() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=23)
    history = collect_unification_dynamics(
        engine,
        steps=5,
        sample_interval=2,
        spectral_max_time=2,
        spectral_trials=20,
        spectral_seed=29,
    )

    assert len(history) == 4  # initial + three sampled summaries (2, 4, 5 events)
    event_counts = [entry["event_count"] for entry in history]
    assert event_counts == [0.0, 2.0, 4.0, 5.0]
    assert all(later >= earlier for earlier, later in zip(event_counts, event_counts[1:]))


def test_collect_unification_dynamics_requires_non_negative_steps() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=9)
    with pytest.raises(ValueError):
        collect_unification_dynamics(engine, steps=-1)
    with pytest.raises(ValueError):
        collect_unification_dynamics(engine, steps=1, sample_interval=0)


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


def test_evaluate_unification_alignment_quantifies_interplay() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=31)
    alignment = evaluate_unification_alignment(
        engine,
        steps=10,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=37,
    )

    for key in (
        "discrete_geometric_correlation",
        "causal_geometric_correlation",
        "multiway_branching_correlation",
        "information_density_trend",
        "unity_range",
        "alignment_score",
    ):
        assert key in alignment
    assert math.isfinite(alignment["discrete_geometric_correlation"])
    assert math.isfinite(alignment["causal_geometric_correlation"])
    multiway_corr = alignment["multiway_branching_correlation"]
    assert math.isnan(multiway_corr) or -1.0 <= multiway_corr <= 1.0
    assert math.isfinite(alignment["information_density_trend"])
    assert alignment["alignment_score"] >= 0
    assert alignment["unity_range"] >= 0 or math.isnan(alignment["unity_range"])


def test_derive_unification_principles_follows_first_principles() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=15)
    principles = derive_unification_principles(
        engine,
        steps=12,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=19,
    )

    assert principles["growth_rate"] > 0
    assert principles["causal_alignment"] == principles["causal_alignment"]
    assert principles["geometric_balance"] > 0
    assert 0 < principles["unity_stability"] <= 1.0


def test_derive_unification_principles_requires_positive_steps() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=21)
    with pytest.raises(ValueError):
        derive_unification_principles(engine, steps=0)


def test_assess_unification_robustness_aggregates_replicates() -> None:
    class Factory:
        def __init__(self) -> None:
            self.seed = 30

        def __call__(self) -> RewriteEngine:
            hg = Hypergraph([(0, 1, 2)])
            engine = RewriteEngine(hg, EdgeSplit3Rule(), seed=self.seed)
            self.seed += 1
            return engine

    result = assess_unification_robustness(
        Factory(),
        steps=6,
        replicates=4,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=7,
    )

    assert result["replicates"] == 4.0
    assert result["mean_final_node_count"] > 3
    assert result["mean_final_edge_count"] > 1
    assert result["mean_final_unity"] == result["mean_final_unity"]
    assert result["unity_variance"] >= 0 or math.isnan(result["unity_variance"])
    assert 0.0 < result["discretization_stability"] <= 1.0
    assert result["mean_growth_rate"] > 0
    assert result["trajectory_coherence"] == result["trajectory_coherence"]
    assert (
        result["mean_discrete_geometric_correlation"]
        == result["mean_discrete_geometric_correlation"]
    )


def test_assess_unification_robustness_validates_parameters() -> None:
    def factory() -> RewriteEngine:
        hg = Hypergraph([(0, 1, 2)])
        return RewriteEngine(hg, EdgeSplit3Rule(), seed=0)

    with pytest.raises(ValueError):
        assess_unification_robustness(factory, steps=0)
    with pytest.raises(ValueError):
        assess_unification_robustness(factory, steps=1, replicates=0)


def test_run_toy_unification_model_produces_bridge_metrics() -> None:
    result = run_toy_unification_model(
        steps=6,
        replicates=2,
        spectral_max_time=3,
        spectral_trials=40,
        seed=42,
    )

    assert isinstance(result, ToyModelResult)
    assert len(result.history) == 7
    assert result.history[0]["event_count"] == 0.0
    assert result.history[-1]["event_count"] == float(6)
    assert result.final_summary == result.history[-1]
    assert result.certificate["certificate_strength"] > 0
    assert result.principles["growth_rate"] > 0
    assert result.alignment["alignment_score"] >= 0
    assert result.robustness["replicates"] == 2.0
    assert result.robustness["mean_final_unity"] == result.robustness["mean_final_unity"]
