"""Bridge Wolfram-style hypergraph dynamics with geometric summaries."""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from engine.rewrite import RewriteEngine

from .geom import (
    average_clustering_coefficient,
    mean_forman_curvature,
    spectral_dimension,
)


MetricTransform = Callable[[float, Dict[str, float]], float]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _collect_pairs(
    entries: Iterable[Dict[str, float]],
    x_key: str,
    y_key: str,
    *,
    x_transform: MetricTransform | None = None,
    y_transform: MetricTransform | None = None,
) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for entry in entries:
        x_raw = entry.get(x_key)
        y_raw = entry.get(y_key)
        if x_raw is None or y_raw is None:
            continue
        x_val = float(x_raw)
        y_val = float(y_raw)
        if not math.isfinite(x_val) or not math.isfinite(y_val):
            continue
        if x_transform is not None:
            x_val = x_transform(x_val, entry)
        if y_transform is not None:
            y_val = y_transform(y_val, entry)
        if math.isfinite(x_val) and math.isfinite(y_val):
            pairs.append((x_val, y_val))
    return pairs


def _pearson_from_pairs(pairs: Sequence[Tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


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
    clustering = average_clustering_coefficient(skeleton)

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
        "mean_clustering_coefficient": clustering,
        "causal_max_depth": causal_stats.get("max_depth", float("nan")),
        "causal_avg_indegree": causal_stats.get("avg_indegree", float("nan")),
        "causal_avg_outdegree": causal_stats.get("avg_outdegree", float("nan")),
        "discretization_index": discretization_index,
        "unity_consistency": unity_consistency,
    }

    return summary


def collect_unification_dynamics(
    engine: RewriteEngine,
    steps: int,
    *,
    include_initial: bool = True,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
) -> List[Dict[str, float]]:
    """Track how the unification summary evolves over multiple rewrite steps.

    Parameters
    ----------
    engine:
        The rewrite engine whose state will be advanced in-place.
    steps:
        Number of rewrite steps to execute.
    include_initial:
        If ``True`` (the default), the summary before any additional rewrites
        are applied is included as the first element of the returned list.
    spectral_max_time, spectral_trials, spectral_seed:
        Parameters forwarded to :func:`compute_unification_summary` for
        spectral-dimension estimation.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")

    summaries: List[Dict[str, float]] = []
    if include_initial:
        summaries.append(
            compute_unification_summary(
                engine,
                spectral_max_time=spectral_max_time,
                spectral_trials=spectral_trials,
                spectral_seed=spectral_seed,
            )
        )

    for _ in range(steps):
        engine.step()
        summaries.append(
            compute_unification_summary(
                engine,
                spectral_max_time=spectral_max_time,
                spectral_trials=spectral_trials,
                spectral_seed=spectral_seed,
            )
        )

    return summaries


def generate_unification_certificate(
    engine: RewriteEngine,
    steps: int,
    *,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
) -> Dict[str, float]:
    """Construct a software proof-of-concept for unity between formalisms.

    The returned dictionary highlights how discrete growth, causal structure,
    and emergent geometry interplay in a single scalar summary:

    * ``dual_correlation`` captures a Pearson correlation between the
      ``unity_consistency`` observable and the inverse of the
      ``discretization_index``.  A positive value indicates that as the system
      accumulates discrete complexity, the effective geometric observable reacts
      in a predictable complementary fashion.
    * ``causal_synergy`` reports the average product of ``unity_consistency``
      with a normalized causal depth (depth divided by the number of executed
      events).  This measures how geometric regularity co-varies with causal
      layering.
    * ``certificate_strength`` multiplies the first two measures, offering a
      compact scalar that is positive whenever the two modes of interaction are
      mutually reinforcing.
    """

    history = collect_unification_dynamics(
        engine,
        steps,
        include_initial=True,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=spectral_seed,
    )

    dual_pairs = _collect_pairs(
        history,
        "discretization_index",
        "unity_consistency",
        x_transform=lambda value, _: 1.0 / (1.0 + value),
    )
    dual_correlation = _pearson_from_pairs(dual_pairs)

    causal_pairs = _collect_pairs(
        history,
        "causal_max_depth",
        "unity_consistency",
        x_transform=lambda depth, entry: depth
        / (1.0 + float(entry.get("event_count", 0.0))),
    )
    causal_synergy = (
        sum(x * y for x, y in causal_pairs) / len(causal_pairs)
        if causal_pairs
        else float("nan")
    )

    if math.isfinite(dual_correlation) and math.isfinite(causal_synergy):
        certificate_strength = dual_correlation * causal_synergy
    else:
        certificate_strength = float("nan")

    return {
        "dual_correlation": dual_correlation,
        "causal_synergy": causal_synergy,
        "certificate_strength": certificate_strength,
    }


__all__ = [
    "compute_unification_summary",
    "collect_unification_dynamics",
    "generate_unification_certificate",
]
