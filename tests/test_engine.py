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
    compose_unification_manifest,
    construct_unification_landscape,
    compute_unification_summary,
    derive_unification_principles,
    evaluate_unification_alignment,
    generate_unification_certificate,
    analyze_unification_feedback,
    map_unification_resonance,
    synthesize_unification_attractor,
    harmonize_unification_channels,
    orchestrate_unification_symphony,
    calibrate_unification_compass,
    trace_unification_phase_portrait,
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
        multiway_generations=3,
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
        multiway_generations=2,
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
        multiway_generations=1,
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
        multiway_generations=2,
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
        multiway_generations=2,
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
        multiway_generations=2,
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
        multiway_generations=2,
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
        multiway_generations=2,
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
    assert result.landscape["history_length"] == float(len(result.history))
    assert math.isfinite(result.landscape["layered_frontier_mean"]) or math.isnan(
        result.landscape["layered_frontier_mean"]
    )
    assert "frontier_unity_correlation" in result.feedback
    assert math.isnan(result.feedback["spectral_equilibrium"]) or 0.0 < result.feedback[
        "spectral_equilibrium"
    ] <= 1.0
    assert set(result.symphony) == {
        "unity_momentum",
        "discrete_curvature_correlation",
        "causal_frontier_correlation",
        "spectral_unity_correlation",
        "multiway_resilience",
        "windowed_unity_variance",
        "symphony_score",
    }
    assert math.isnan(result.symphony["symphony_score"]) or result.symphony[
        "symphony_score"
    ] >= 0
    assert set(result.attractor) == {
        "discrete_persistence",
        "geometric_resonance",
        "causal_gradient",
        "multiway_pressure",
    }
    assert any(math.isfinite(value) for value in result.attractor.values())
    assert result.resonance["depths_evaluated"] >= 1.0
    assert math.isnan(result.resonance["depth_span"]) or result.resonance["depth_span"] >= 0
    assert "resonance_score" in result.resonance
    assert math.isnan(result.resonance["resonance_score"]) or result.resonance[
        "resonance_score"
    ] >= 0
    assert result.manifest["replicates"] == 2.0
    assert "concordance_index" in result.manifest
    assert result.manifest["mean_certificate_strength"] == result.manifest[
        "mean_certificate_strength"
    ]
    assert set(result.phase_portrait) == {
        "phase_area",
        "frontier_phase_coupling",
        "geometric_phase_coupling",
        "causal_phase_gradient",
        "unity_causal_correlation",
        "phase_coherence",
    }
    assert math.isnan(result.phase_portrait["phase_coherence"]) or result.phase_portrait[
        "phase_coherence"
    ] >= 0


def test_compute_unification_summary_respects_multiway_generations() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=43)
    engine.run(steps=5)

    shallow = compute_unification_summary(
        engine,
        spectral_max_time=2,
        spectral_trials=40,
        spectral_seed=44,
        multiway_generations=0,
    )
    deep = compute_unification_summary(
        engine,
        spectral_max_time=2,
        spectral_trials=40,
        spectral_seed=45,
        multiway_generations=3,
    )

    assert shallow["multiway_state_count"] == 1.0
    assert shallow["multiway_event_count"] == 0.0
    assert deep["multiway_state_count"] >= shallow["multiway_state_count"]
    assert deep["multiway_max_depth"] <= 3

    with pytest.raises(ValueError):
        compute_unification_summary(
            engine,
            spectral_max_time=2,
            spectral_trials=20,
            spectral_seed=46,
            multiway_generations=-1,
        )


def test_construct_unification_landscape_summarizes_multiway_interplay() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=51)
    landscape = construct_unification_landscape(
        engine,
        steps=5,
        spectral_max_time=2,
        spectral_trials=20,
        spectral_seed=53,
        multiway_generations=2,
    )

    assert landscape["history_length"] >= 1.0
    assert landscape["multiway_generations"] == 2.0
    assert math.isnan(landscape["unity_frontier_correlation"]) or -1.0 <= landscape[
        "unity_frontier_correlation"
    ] <= 1.0
    assert math.isnan(landscape["multiway_depth_range"]) or landscape[
        "multiway_depth_range"
    ] >= 0.0


def test_construct_unification_landscape_validates_inputs() -> None:
    engine = RewriteEngine(Hypergraph([(0, 1, 2)]), EdgeSplit3Rule(), seed=59)
    with pytest.raises(ValueError):
        construct_unification_landscape(engine, steps=0)
    with pytest.raises(ValueError):
        construct_unification_landscape(engine, steps=1, sample_interval=0)
    with pytest.raises(ValueError):
        construct_unification_landscape(engine, steps=1, multiway_generations=-1)


def test_map_unification_resonance_profiles_depth_response() -> None:
    def factory() -> RewriteEngine:
        hypergraph = Hypergraph([(0, 1, 2)])
        return RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=61)

    result = map_unification_resonance(
        factory,
        steps=4,
        multiway_depths=[0, 1, 2],
        spectral_max_time=2,
        spectral_trials=20,
        spectral_seed=71,
    )

    assert result["depths_evaluated"] == 3.0
    assert result["depth_span"] == 2.0
    mean_unity = result["mean_unity_consistency"]
    assert math.isnan(mean_unity) or math.isfinite(mean_unity)
    mean_frontier = result["mean_frontier_size"]
    assert math.isnan(mean_frontier) or math.isfinite(mean_frontier)
    mean_causal = result["mean_causal_depth"]
    assert math.isnan(mean_causal) or math.isfinite(mean_causal)
    resonance = result["resonance_score"]
    assert resonance >= 0 or math.isnan(resonance)


def test_map_unification_resonance_validates_inputs() -> None:
    def factory() -> RewriteEngine:
        hypergraph = Hypergraph([(0, 1, 2)])
        return RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=73)

    with pytest.raises(ValueError):
        map_unification_resonance(factory, steps=0, multiway_depths=[0])
    with pytest.raises(ValueError):
        map_unification_resonance(factory, steps=1, multiway_depths=[])
    with pytest.raises(ValueError):
        map_unification_resonance(factory, steps=1, multiway_depths=[-1])


def test_analyze_unification_feedback_measures_loops() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=107)
    feedback = analyze_unification_feedback(
        engine,
        steps=8,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=113,
        multiway_generations=2,
    )

    assert set(feedback) == {
        "frontier_unity_correlation",
        "curvature_response",
        "causal_feedback",
        "spectral_equilibrium",
        "discrete_resonance",
    }
    corr = feedback["frontier_unity_correlation"]
    assert math.isnan(corr) or -1.0000001 <= corr <= 1.0000001
    curvature = feedback["curvature_response"]
    assert math.isnan(curvature) or math.isfinite(curvature)
    spectral = feedback["spectral_equilibrium"]
    assert math.isnan(spectral) or 0.0 < spectral <= 1.0
    resonance = feedback["discrete_resonance"]
    assert math.isnan(resonance) or math.isfinite(resonance)


def test_analyze_unification_feedback_requires_positive_steps() -> None:
    engine = RewriteEngine(Hypergraph([(0, 1, 2)]), EdgeSplit3Rule(), seed=127)
    with pytest.raises(ValueError):
        analyze_unification_feedback(engine, steps=0)


def test_harmonize_unification_channels_fuses_correlated_views() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=139)
    harmony = harmonize_unification_channels(
        engine,
        steps=9,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=149,
        multiway_generations=2,
    )

    for key in (
        "unity_flux",
        "geometric_causal_ratio",
        "branching_intensity",
        "discrete_unity_correlation",
        "causal_unity_correlation",
        "geometric_frontier_correlation",
        "coherence_index",
    ):
        assert key in harmony

    flux = harmony["unity_flux"]
    assert math.isnan(flux) or math.isfinite(flux)
    intensity = harmony["branching_intensity"]
    assert math.isnan(intensity) or intensity >= 0
    coherence = harmony["coherence_index"]
    assert math.isnan(coherence) or 0.0 <= coherence <= 1.0000001


def test_harmonize_unification_channels_requires_positive_steps() -> None:
    engine = RewriteEngine(Hypergraph([(0, 1, 2)]), EdgeSplit3Rule(), seed=151)
    with pytest.raises(ValueError):
        harmonize_unification_channels(engine, steps=0)


def test_calibrate_unification_compass_tracks_directionality() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=173)
    compass = calibrate_unification_compass(
        engine,
        steps=7,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=181,
        multiway_generations=2,
    )

    expected_keys = {
        "discrete_drift",
        "unity_drift",
        "causal_depth_drift",
        "multiway_frontier_drift",
        "spectral_drift",
        "event_discrete_correlation",
        "event_unity_correlation",
        "event_causal_correlation",
        "event_frontier_correlation",
        "compass_alignment",
        "drift_magnitude",
    }
    assert expected_keys.issubset(compass.keys())

    assert math.isnan(compass["drift_magnitude"]) or compass["drift_magnitude"] >= 0
    assert math.isnan(compass["compass_alignment"]) or 0.0 <= compass["compass_alignment"] <= 1.0000001
    assert math.isnan(compass["event_unity_correlation"]) or -1.0 <= compass["event_unity_correlation"] <= 1.0


def test_calibrate_unification_compass_requires_positive_steps() -> None:
    engine = RewriteEngine(Hypergraph([(0, 1, 2)]), EdgeSplit3Rule(), seed=179)
    with pytest.raises(ValueError):
        calibrate_unification_compass(engine, steps=0)


def test_orchestrate_unification_symphony_fuses_correlations() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=187)
    symphony = orchestrate_unification_symphony(
        engine,
        steps=8,
        window=3,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=193,
        multiway_generations=2,
    )

    assert set(symphony) == {
        "unity_momentum",
        "discrete_curvature_correlation",
        "causal_frontier_correlation",
        "spectral_unity_correlation",
        "multiway_resilience",
        "windowed_unity_variance",
        "symphony_score",
    }

    assert math.isnan(symphony["unity_momentum"]) or math.isfinite(
        symphony["unity_momentum"]
    )
    assert math.isnan(symphony["windowed_unity_variance"]) or symphony[
        "windowed_unity_variance"
    ] >= 0
    assert math.isnan(symphony["multiway_resilience"]) or symphony[
        "multiway_resilience"
    ] >= 0
    assert math.isnan(symphony["symphony_score"]) or symphony["symphony_score"] >= 0


def test_orchestrate_unification_symphony_validates_inputs() -> None:
    engine = RewriteEngine(Hypergraph([(0, 1, 2)]), EdgeSplit3Rule(), seed=199)
    with pytest.raises(ValueError):
        orchestrate_unification_symphony(engine, steps=0)
    with pytest.raises(ValueError):
        orchestrate_unification_symphony(engine, steps=1, window=0)


def test_synthesize_unification_attractor_fuses_sliding_metrics() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=67)
    attractor = synthesize_unification_attractor(
        engine,
        steps=6,
        window=3,
        spectral_max_time=2,
        spectral_trials=30,
        spectral_seed=79,
        multiway_generations=2,
    )

    for key in (
        "discrete_persistence",
        "geometric_resonance",
        "causal_gradient",
        "multiway_pressure",
    ):
        assert key in attractor
        value = attractor[key]
        assert math.isnan(value) or math.isfinite(value)

    finite_values = [value for value in attractor.values() if math.isfinite(value)]
    assert finite_values  # ensure at least one channel produced a concrete value


def test_synthesize_unification_attractor_validates_inputs() -> None:
    engine = RewriteEngine(Hypergraph([(0, 1, 2)]), EdgeSplit3Rule(), seed=83)
    with pytest.raises(ValueError):
        synthesize_unification_attractor(engine, steps=0)
    with pytest.raises(ValueError):
        synthesize_unification_attractor(engine, steps=1, window=0)


def test_trace_unification_phase_portrait_links_channels() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=167)
    portrait = trace_unification_phase_portrait(
        engine,
        steps=8,
        spectral_max_time=4,
        spectral_trials=60,
        spectral_seed=173,
        multiway_generations=2,
    )

    assert set(portrait) == {
        "phase_area",
        "frontier_phase_coupling",
        "geometric_phase_coupling",
        "causal_phase_gradient",
        "unity_causal_correlation",
        "phase_coherence",
    }
    assert portrait["phase_area"] == portrait["phase_area"] or math.isnan(portrait["phase_area"])
    assert math.isnan(portrait["phase_coherence"]) or portrait["phase_coherence"] >= 0


def test_trace_unification_phase_portrait_requires_positive_steps() -> None:
    engine = RewriteEngine(Hypergraph([(0, 1, 2)]), EdgeSplit3Rule(), seed=179)
    with pytest.raises(ValueError):
        trace_unification_phase_portrait(engine, steps=0)


def test_compose_unification_manifest_consolidates_channels() -> None:
    class Factory:
        def __init__(self) -> None:
            self.seed = 90

        def __call__(self) -> RewriteEngine:
            hypergraph = Hypergraph([(0, 1, 2)])
            engine = RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=self.seed)
            self.seed += 1
            return engine

    manifest = compose_unification_manifest(
        Factory(),
        steps=4,
        replicates=2,
        spectral_max_time=2,
        spectral_trials=20,
        spectral_seed=91,
        multiway_generations=2,
        sample_interval=2,
        window=2,
    )

    assert manifest["replicates"] == 2.0
    assert "mean_unity_consistency" in manifest
    assert "mean_alignment_score" in manifest
    assert "unity_certificate_correlation" in manifest
    corr = manifest["unity_certificate_correlation"]
    assert math.isnan(corr) or -1.0 <= corr <= 1.0
    concordance = manifest["concordance_index"]
    assert math.isnan(concordance) or concordance >= 0


def test_compose_unification_manifest_validates_inputs() -> None:
    def factory() -> RewriteEngine:
        hypergraph = Hypergraph([(0, 1, 2)])
        return RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=101)

    with pytest.raises(ValueError):
        compose_unification_manifest(factory, steps=0)
    with pytest.raises(ValueError):
        compose_unification_manifest(factory, steps=1, replicates=0)
    with pytest.raises(ValueError):
        compose_unification_manifest(factory, steps=1, sample_interval=0)
    with pytest.raises(ValueError):
        compose_unification_manifest(factory, steps=1, window=0)
