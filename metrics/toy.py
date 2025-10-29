"""Construct a concrete toy model bridging Wolfram-style computation and geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from engine.hypergraph import Hypergraph
from engine.rewrite import EdgeSplit3Rule, RewriteEngine

from .unification import (
    assess_unification_robustness,
    collect_unification_dynamics,
    compose_unification_manifest,
    construct_unification_landscape,
    derive_unification_principles,
    evaluate_unification_alignment,
    generate_unification_certificate,
    analyze_unification_feedback,
    map_unification_resonance,
    synthesize_unification_attractor,
    trace_unification_phase_portrait,
)


@dataclass(frozen=True)
class ToyModelResult:
    """Container aggregating observables from the toy unification experiment."""

    history: List[Dict[str, float]]
    final_summary: Dict[str, float]
    certificate: Dict[str, float]
    principles: Dict[str, float]
    alignment: Dict[str, float]
    robustness: Dict[str, float]
    landscape: Dict[str, float]
    feedback: Dict[str, float]
    attractor: Dict[str, float]
    resonance: Dict[str, float]
    manifest: Dict[str, float]
    phase_portrait: Dict[str, float]


def run_toy_unification_model(
    *,
    steps: int = 12,
    replicates: int = 4,
    initial_edges: Iterable[Sequence[int]] | None = None,
    spectral_max_time: int = 4,
    spectral_trials: int = 80,
    seed: int = 0,
    multiway_generations: int = 2,
) -> ToyModelResult:
    """Execute a minimal experiment linking discrete rewrites to geometric cues.

    The returned :class:`ToyModelResult` highlights how a simple hypergraph
    rewrite system (standing in for Wolfram's computational models) generates
    geometric observables reminiscent of Weinstein's geometric unity.  It
    reports:

    ``history``
        The time series of unification summaries across the rewrite run.
    ``final_summary``
        The last element of ``history`` capturing the accumulated state.
    ``certificate``
        Cross-metric correlations measuring duality between discrete and
        geometric perspectives.
    ``principles``
        First-principles indicators that tie causal depth, growth, and emergent
        geometry together.
    ``robustness``
        Aggregate statistics across multiple replicas, demonstrating that the
        bridge remains stable under stochastic variation.
    ``landscape``
        A synthesized summary of the multiway/discrete interplay generated via
        :func:`metrics.unification.construct_unification_landscape`.
    ``feedback``
        Cross-channel feedback indicators computed by
        :func:`metrics.unification.analyze_unification_feedback` that quantify
        how multiway branching, causal depth, and geometric consistency respond
        to one another.
    ``attractor``
        Sliding-window fusion of discrete, causal, geometric, and multiway
        observables produced by
        :func:`metrics.unification.synthesize_unification_attractor`.
    ``resonance``
        Survey of how varying the depth of the auxiliary multiway exploration
        modulates blended observables via
        :func:`metrics.unification.map_unification_resonance`.
    ``manifest``
        Aggregate statistics across multiple bridge metrics constructed by
        :func:`metrics.unification.compose_unification_manifest`.

    The ``multiway_generations`` parameter adjusts how deeply the auxiliary
    multiway explorations probe when constructing these summaries.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    base_edges = list(initial_edges) if initial_edges is not None else [
        (0, 1, 2),
        (0, 2, 3),
        (1, 2, 3),
    ]

    def make_engine(local_seed: int) -> RewriteEngine:
        hypergraph = Hypergraph(base_edges)
        return RewriteEngine(hypergraph, EdgeSplit3Rule(), seed=local_seed)

    def make_factory_stream(start_seed: int) -> "Factory":
        class Factory:
            def __init__(self) -> None:
                self._next_seed = start_seed

            def __call__(self) -> RewriteEngine:
                engine = make_engine(self._next_seed)
                self._next_seed += 1
                return engine

        return Factory()

    next_seed = seed

    history_engine = make_engine(next_seed)
    history = collect_unification_dynamics(
        history_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    final_summary = history[-1]
    next_seed += 1

    certificate_engine = make_engine(next_seed)
    certificate = generate_unification_certificate(
        certificate_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    next_seed += 1

    principles_engine = make_engine(next_seed)
    principles = derive_unification_principles(
        principles_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    next_seed += 1

    alignment_engine = make_engine(next_seed)
    alignment = evaluate_unification_alignment(
        alignment_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    next_seed += 1

    robustness_factory = make_factory_stream(next_seed)
    robustness = assess_unification_robustness(
        robustness_factory,
        steps=steps,
        replicates=replicates,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    next_seed += replicates

    landscape_engine = make_engine(next_seed)
    landscape = construct_unification_landscape(
        landscape_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    next_seed += 1

    feedback_engine = make_engine(next_seed)
    feedback = analyze_unification_feedback(
        feedback_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    next_seed += 1

    attractor_engine = make_engine(next_seed)
    attractor = synthesize_unification_attractor(
        attractor_engine,
        steps=steps,
        window=min(4, steps),
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    next_seed += 1

    depth_candidates = {0, multiway_generations}
    if multiway_generations > 0:
        depth_candidates.add(max(1, multiway_generations // 2))
    resonance_depths = sorted(depth_candidates)

    resonance_factory = make_factory_stream(next_seed)
    resonance = map_unification_resonance(
        resonance_factory,
        steps=steps,
        multiway_depths=resonance_depths,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
    )
    next_seed += len(resonance_depths)

    phase_engine = make_engine(next_seed)
    phase_portrait = trace_unification_phase_portrait(
        phase_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )
    next_seed += 1

    manifest_factory = make_factory_stream(next_seed)
    manifest = compose_unification_manifest(
        manifest_factory,
        steps=steps,
        replicates=replicates,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=next_seed,
        multiway_generations=multiway_generations,
    )

    return ToyModelResult(
        history=history,
        final_summary=final_summary,
        certificate=certificate,
        principles=principles,
        alignment=alignment,
        robustness=robustness,
        landscape=landscape,
        feedback=feedback,
        attractor=attractor,
        resonance=resonance,
        manifest=manifest,
        phase_portrait=phase_portrait,
    )


__all__ = ["ToyModelResult", "run_toy_unification_model"]

