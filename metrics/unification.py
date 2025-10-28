"""Bridge Wolfram-style hypergraph dynamics with geometric summaries."""

from __future__ import annotations

from typing import Dict

from engine.rewrite import RewriteEngine

from .geom import mean_forman_curvature, spectral_dimension


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def compute_unification_summary(
    engine: RewriteEngine,
    *,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
) -> Dict[str, float]:
    """Return a dictionary connecting discrete rewrites to geometric structure.

    The resulting metrics pull together three complementary perspectives:

    * **Discretization** (à la Wolfram): how the hypergraph grows and stores
      information through repeated rewrites.
    * **Causality**: statistics from the causal dependency graph that encode the
      flow of updates through rewrite history.
    * **Geometric Unity** (à la Weinstein): coarse geometric observables derived
      from the 1-skeleton of the hypergraph which stand in for curvature and
      effective dimensionality.

    The dictionary values are floats for ease of downstream analysis and may be
    ``NaN`` when a metric is undefined.
    """

    hypergraph = engine.hypergraph
    skeleton = hypergraph.one_skeleton()
    ds = spectral_dimension(
        skeleton,
        max_time=spectral_max_time,
        trials=spectral_trials,
        seed=spectral_seed,
    )
    curvature = mean_forman_curvature(skeleton)

    causal_stats = engine.causal_graph.basic_stats()
    node_count = hypergraph.node_count
    edge_count = hypergraph.edge_count
    event_count = len(engine.events)

    created_nodes = sum(len(event.result.created_nodes) for event in engine.events)
    info_density = _safe_ratio(float(edge_count), float(node_count))
    avg_created = created_nodes / event_count if event_count else 0.0

    if ds == ds and curvature == curvature:
        unity_consistency = ds / (1.0 + abs(curvature))
    else:
        unity_consistency = float("nan")

    discretization_index = (
        info_density * (1.0 + avg_created)
        if info_density == info_density
        else float("nan")
    )

    summary: Dict[str, float] = {
        "node_count": float(node_count),
        "edge_count": float(edge_count),
        "event_count": float(event_count),
        "information_density": info_density,
        "average_nodes_created_per_event": float(avg_created),
        "spectral_dimension": ds,
        "mean_forman_curvature": curvature,
        "causal_max_depth": causal_stats.get("max_depth", float("nan")),
        "causal_avg_indegree": causal_stats.get("avg_indegree", float("nan")),
        "causal_avg_outdegree": causal_stats.get("avg_outdegree", float("nan")),
        "discretization_index": discretization_index,
        "unity_consistency": unity_consistency,
    }

    return summary


__all__ = ["compute_unification_summary"]
