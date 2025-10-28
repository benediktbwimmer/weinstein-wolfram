"""Bridge Wolfram-style hypergraph dynamics with geometric summaries."""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from engine.multiway import MultiwaySystem
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


def _finite_average(values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return sum(finite_values) / len(finite_values)


def _finite_variance(values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if len(finite_values) < 2:
        return float("nan")
    mean = sum(finite_values) / len(finite_values)
    return sum((value - mean) ** 2 for value in finite_values) / len(finite_values)


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
    * **Multiway structure**: branching characteristics of the rewrite system
      when all applicable updates are explored in parallel.

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

    multiway = MultiwaySystem(hypergraph.copy(), [engine.rule])
    evolution = multiway.run(max_generations=2)
    histogram = evolution.depth_histogram()
    max_depth = max(histogram.keys(), default=0)
    frontier_size = len(evolution.frontier(max_depth)) if histogram else 1

    summary.update(
        {
            "multiway_state_count": float(evolution.state_count),
            "multiway_event_count": float(evolution.event_count),
            "multiway_max_depth": float(max_depth),
            "multiway_avg_branching_factor": evolution.average_branching_factor(),
            "multiway_frontier_size": float(frontier_size),
        }
    )

    return summary


def collect_unification_dynamics(
    engine: RewriteEngine,
    steps: int,
    *,
    include_initial: bool = True,
    sample_interval: int = 1,
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
    sample_interval:
        Number of rewrite steps to perform between consecutive summaries.  Must
        be positive.  When ``steps`` is not a multiple of ``sample_interval``
        the final partial batch is still recorded.
    spectral_max_time, spectral_trials, spectral_seed:
        Parameters forwarded to :func:`compute_unification_summary` for
        spectral-dimension estimation.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

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

    remaining = steps
    while remaining > 0:
        batch = min(sample_interval, remaining)
        for _ in range(batch):
            engine.step()
        summaries.append(
            compute_unification_summary(
                engine,
                spectral_max_time=spectral_max_time,
                spectral_trials=spectral_trials,
                spectral_seed=spectral_seed,
            )
        )
        remaining -= batch

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


def derive_unification_principles(
    engine: RewriteEngine,
    steps: int,
    *,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
) -> Dict[str, float]:
    """Extract first-principles indicators that blend discrete and geometric views.

    The returned dictionary focuses on three foundational quantities:

    * ``growth_rate`` captures how quickly the hypergraph accumulates nodes per
      executed event.  This reflects the discrete substrate of the rewrite
      system.
    * ``causal_alignment`` averages how normalized causal depth correlates with
      the unity observable, revealing how causal layering resonates with
      geometry.
    * ``geometric_balance`` compares the average clustering coefficient against
      the magnitude of mean Forman curvature, estimating whether emergent
      geometry maintains coherence as the system grows.
    * ``unity_stability`` measures the inverse variance of the unity observable
      across the collected history, with values in ``(0, 1]`` indicating stable
      behaviour.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")

    history = collect_unification_dynamics(
        engine,
        steps,
        include_initial=True,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=spectral_seed,
    )

    if len(history) < 2:
        return {
            "growth_rate": float("nan"),
            "causal_alignment": float("nan"),
            "geometric_balance": float("nan"),
            "unity_stability": float("nan"),
        }

    first = history[0]
    last = history[-1]

    event_span = float(last.get("event_count", 0.0) - first.get("event_count", 0.0))
    node_span = float(last.get("node_count", 0.0) - first.get("node_count", 0.0))
    growth_rate = _safe_ratio(node_span, event_span)

    unity_values = [float(entry.get("unity_consistency", float("nan"))) for entry in history]
    unity_average = _finite_average(unity_values)

    normalized_depths: List[float] = []
    for entry in history:
        depth = float(entry.get("causal_max_depth", float("nan")))
        events = float(entry.get("event_count", 0.0))
        normalized = depth / (1.0 + events)
        normalized_depths.append(normalized if math.isfinite(normalized) else float("nan"))

    if math.isfinite(unity_average):
        products = [
            float(entry.get("unity_consistency", float("nan"))) * normalized
            for entry, normalized in zip(history, normalized_depths)
        ]
        causal_alignment = _finite_average(products)
    else:
        causal_alignment = float("nan")

    clustering_values = [float(entry.get("mean_clustering_coefficient", float("nan"))) for entry in history]
    curvature_values = [float(entry.get("mean_forman_curvature", float("nan"))) for entry in history]

    avg_clustering = _finite_average(clustering_values)
    avg_curvature = _finite_average(curvature_values)

    if math.isfinite(avg_clustering) and math.isfinite(avg_curvature):
        geometric_balance = avg_clustering / (1.0 + abs(avg_curvature))
    else:
        geometric_balance = float("nan")

    variance_unity = _finite_variance(unity_values)
    if math.isfinite(variance_unity):
        unity_stability = 1.0 / (1.0 + variance_unity)
    else:
        unity_stability = float("nan")

    return {
        "growth_rate": growth_rate,
        "causal_alignment": causal_alignment,
        "geometric_balance": geometric_balance,
        "unity_stability": unity_stability,
    }


def evaluate_unification_alignment(
    engine: RewriteEngine,
    steps: int,
    *,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
) -> Dict[str, float]:
    """Quantify how discrete, causal, and geometric metrics evolve together."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    history = collect_unification_dynamics(
        engine,
        steps,
        include_initial=True,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=spectral_seed,
    )

    if len(history) < 2:
        return {
            "discrete_geometric_correlation": float("nan"),
            "causal_geometric_correlation": float("nan"),
            "multiway_branching_correlation": float("nan"),
            "information_density_trend": float("nan"),
            "unity_range": float("nan"),
            "alignment_score": float("nan"),
        }

    discrete_pairs = _collect_pairs(
        history,
        "discretization_index",
        "unity_consistency",
    )
    discrete_corr = _pearson_from_pairs(discrete_pairs)

    causal_pairs = _collect_pairs(
        history,
        "causal_max_depth",
        "unity_consistency",
        x_transform=lambda depth, entry: depth
        / (1.0 + float(entry.get("event_count", 0.0))),
    )
    causal_corr = _pearson_from_pairs(causal_pairs)

    multiway_pairs = _collect_pairs(
        history,
        "multiway_max_depth",
        "multiway_frontier_size",
    )
    multiway_corr = _pearson_from_pairs(multiway_pairs)

    info_density_values = [
        float(entry.get("information_density", float("nan"))) for entry in history
    ]
    event_counts = [float(entry.get("event_count", 0.0)) for entry in history]

    first_density = next(
        (value for value in info_density_values if math.isfinite(value)),
        float("nan"),
    )
    last_density = next(
        (
            value
            for value in reversed(info_density_values)
            if math.isfinite(value)
        ),
        float("nan"),
    )
    event_span = event_counts[-1] - event_counts[0]
    if math.isfinite(first_density) and math.isfinite(last_density):
        information_density_trend = _safe_ratio(last_density - first_density, event_span)
    else:
        information_density_trend = float("nan")

    unity_values = [
        float(entry.get("unity_consistency", float("nan"))) for entry in history
    ]
    finite_unity = [value for value in unity_values if math.isfinite(value)]
    if finite_unity:
        unity_range = max(finite_unity) - min(finite_unity)
    else:
        unity_range = float("nan")

    alignment_components = [
        abs(value)
        for value in (discrete_corr, causal_corr, multiway_corr)
        if math.isfinite(value)
    ]
    alignment_score = _finite_average(alignment_components)

    return {
        "discrete_geometric_correlation": discrete_corr,
        "causal_geometric_correlation": causal_corr,
        "multiway_branching_correlation": multiway_corr,
        "information_density_trend": information_density_trend,
        "unity_range": unity_range,
        "alignment_score": alignment_score,
    }


def assess_unification_robustness(
    engine_factory: Callable[[], RewriteEngine],
    *,
    steps: int,
    replicates: int = 3,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
) -> Dict[str, float]:
    """Evaluate how reproducible the blended metrics are across runs.

    Parameters
    ----------
    engine_factory:
        Callable that returns a fresh :class:`~engine.rewrite.RewriteEngine` for
        each replicate.  The hypergraph inside the engine is mutated in-place
        during assessment and therefore must not be reused across replicates.
    steps:
        Number of rewrite steps to execute per replicate.  Must be positive.
    replicates:
        Number of independent experiments to execute.  Must be positive.
    spectral_max_time, spectral_trials, spectral_seed:
        Parameters forwarded to :func:`collect_unification_dynamics` to control
        the spectral-dimension estimation inside each summary.  When
        ``spectral_seed`` is provided, successive replicates increment it to
        avoid identical random walks in each run.

    Returns
    -------
    Dict[str, float]
        Aggregate statistics that highlight whether discrete growth, causal
        layering, and emergent geometry remain consistent across runs.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    final_nodes: List[float] = []
    final_edges: List[float] = []
    final_unity: List[float] = []
    final_discretization: List[float] = []
    final_depths: List[float] = []
    growth_rates: List[float] = []
    unity_means: List[float] = []
    coherence_values: List[float] = []
    correlations: List[float] = []

    for replicate in range(replicates):
        engine = engine_factory()
        seed = spectral_seed + replicate if spectral_seed is not None else None
        history = collect_unification_dynamics(
            engine,
            steps,
            include_initial=True,
            spectral_max_time=spectral_max_time,
            spectral_trials=spectral_trials,
            spectral_seed=seed,
        )
        if not history:
            continue

        initial = history[0]
        final = history[-1]

        final_nodes.append(float(final.get("node_count", float("nan"))))
        final_edges.append(float(final.get("edge_count", float("nan"))))
        final_unity.append(float(final.get("unity_consistency", float("nan"))))
        final_discretization.append(
            float(final.get("discretization_index", float("nan")))
        )
        final_depths.append(float(final.get("causal_max_depth", float("nan"))))

        node_span = float(final.get("node_count", 0.0) - initial.get("node_count", 0.0))
        event_span = float(
            final.get("event_count", 0.0) - initial.get("event_count", 0.0)
        )
        growth_rates.append(_safe_ratio(node_span, event_span))

        unity_values = [
            float(entry.get("unity_consistency", float("nan"))) for entry in history
        ]
        unity_means.append(_finite_average(unity_values))

        normalized_depths: List[float] = []
        for entry in history:
            depth = float(entry.get("causal_max_depth", float("nan")))
            events = float(entry.get("event_count", 0.0))
            normalized = depth / (1.0 + events)
            normalized_depths.append(
                normalized if math.isfinite(normalized) else float("nan")
            )

        coherence_components = [
            float(entry.get("unity_consistency", float("nan"))) * normalized
            for entry, normalized in zip(history, normalized_depths)
        ]
        coherence_values.append(_finite_average(coherence_components))

        pairs = _collect_pairs(
            history,
            "discretization_index",
            "unity_consistency",
        )
        correlations.append(_pearson_from_pairs(pairs))

    unity_variance = _finite_variance(final_unity)
    discretization_variance = _finite_variance(final_discretization)
    if math.isfinite(discretization_variance):
        discretization_stability = 1.0 / (1.0 + discretization_variance)
    else:
        discretization_stability = float("nan")

    return {
        "replicates": float(len(final_nodes)),
        "mean_final_node_count": _finite_average(final_nodes),
        "mean_final_edge_count": _finite_average(final_edges),
        "mean_final_unity": _finite_average(final_unity),
        "unity_variance": unity_variance,
        "discretization_stability": discretization_stability,
        "mean_causal_depth": _finite_average(final_depths),
        "mean_growth_rate": _finite_average(growth_rates),
        "mean_unity_integral": _finite_average(unity_means),
        "trajectory_coherence": _finite_average(coherence_values),
        "mean_discrete_geometric_correlation": _finite_average(correlations),
    }


__all__ = [
    "compute_unification_summary",
    "collect_unification_dynamics",
    "generate_unification_certificate",
    "derive_unification_principles",
    "assess_unification_robustness",
    "evaluate_unification_alignment",
]
