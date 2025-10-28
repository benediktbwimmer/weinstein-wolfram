"""Construct a concrete toy model bridging Wolfram-style computation and geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from engine.hypergraph import Hypergraph
from engine.rewrite import EdgeSplit3Rule, RewriteEngine

from .unification import (
    assess_unification_robustness,
    collect_unification_dynamics,
    derive_unification_principles,
    generate_unification_certificate,
)


@dataclass(frozen=True)
class ToyModelResult:
    """Container aggregating observables from the toy unification experiment."""

    history: List[Dict[str, float]]
    final_summary: Dict[str, float]
    certificate: Dict[str, float]
    principles: Dict[str, float]
    robustness: Dict[str, float]


def run_toy_unification_model(
    *,
    steps: int = 12,
    replicates: int = 4,
    initial_edges: Iterable[Sequence[int]] | None = None,
    spectral_max_time: int = 4,
    spectral_trials: int = 80,
    seed: int = 0,
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

    history_engine = make_engine(seed)
    history = collect_unification_dynamics(
        history_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=seed,
    )
    final_summary = history[-1]

    certificate_engine = make_engine(seed + 1)
    certificate = generate_unification_certificate(
        certificate_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=seed + 1,
    )

    principles_engine = make_engine(seed + 2)
    principles = derive_unification_principles(
        principles_engine,
        steps=steps,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=seed + 2,
    )

    def factory_generator() -> "Factory":
        class Factory:
            def __init__(self) -> None:
                self._next_seed = seed + 3

            def __call__(self) -> RewriteEngine:
                engine = make_engine(self._next_seed)
                self._next_seed += 1
                return engine

        return Factory()

    robustness = assess_unification_robustness(
        factory_generator(),
        steps=steps,
        replicates=replicates,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=seed + 3,
    )

    return ToyModelResult(
        history=history,
        final_summary=final_summary,
        certificate=certificate,
        principles=principles,
        robustness=robustness,
    )


__all__ = ["ToyModelResult", "run_toy_unification_model"]

