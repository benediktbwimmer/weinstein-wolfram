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
    multiway_generations: int = 2,
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
    ``NaN`` when a metric is undefined.  The ``multiway_generations`` parameter
    controls how deep the auxiliary multiway exploration proceeds when
    computing the branching-based observables.
    """

    if multiway_generations < 0:
        raise ValueError("multiway_generations must be non-negative")

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
    evolution = multiway.run(max_generations=multiway_generations)
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
    multiway_generations: int = 2,
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
    multiway_generations:
        Maximum depth explored when summarizing the auxiliary multiway
        evolution.  A value of ``0`` restricts the summary to the initial state.
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
                multiway_generations=multiway_generations,
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
                multiway_generations=multiway_generations,
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
    multiway_generations: int = 2,
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
    The optional ``multiway_generations`` argument mirrors the corresponding
    parameter of :func:`compute_unification_summary`, enabling deeper or shallower
    explorations of the auxiliary branching structure that informs the
    certificate.
    """

    history = collect_unification_dynamics(
        engine,
        steps,
        include_initial=True,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=spectral_seed,
        multiway_generations=multiway_generations,
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
    multiway_generations: int = 2,
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
    The ``multiway_generations`` parameter is forwarded to
    :func:`collect_unification_dynamics` and consequently influences the
    branching observables available for downstream aggregation.
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
        multiway_generations=multiway_generations,
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
    multiway_generations: int = 2,
) -> Dict[str, float]:
    """Quantify how discrete, causal, and geometric metrics evolve together."""

    # ``multiway_generations`` is threaded through to
    # :func:`collect_unification_dynamics` to allow callers to tune how deeply
    # the multiway branching structure is sampled when computing each summary.

    if steps <= 0:
        raise ValueError("steps must be positive")

    history = collect_unification_dynamics(
        engine,
        steps,
        include_initial=True,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=spectral_seed,
        multiway_generations=multiway_generations,
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


def analyze_unification_feedback(
    engine: RewriteEngine,
    steps: int,
    *,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
    multiway_generations: int = 2,
) -> Dict[str, float]:
    """Measure feedback loops between discrete, causal, and geometric channels.

    The returned dictionary provides five complementary observables:

    ``frontier_unity_correlation``
        Pearson correlation between the size of the multiway frontier and the
        unity consistency observable, highlighting how branching richness aligns
        with geometric regularity.
    ``curvature_response``
        Linear-response coefficient describing how the unity observable reacts
        to shifts in mean Forman curvature across the recorded history.
    ``causal_feedback``
        Correlation between normalized causal depth and the discretization
        index, capturing how deeply layered causality reinforces discrete
        growth.
    ``spectral_equilibrium``
        Normalized proximity between the terminal spectral dimension and its
        historical average, with values near ``1`` signalling stable geometric
        behaviour.
    ``discrete_resonance``
        Finite average of the product between information density and frontier
        size, serving as a coarse resonance indicator between discrete growth
        and multiway expansion.
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
        multiway_generations=multiway_generations,
    )

    if len(history) < 2:
        return {
            "frontier_unity_correlation": float("nan"),
            "curvature_response": float("nan"),
            "causal_feedback": float("nan"),
            "spectral_equilibrium": float("nan"),
            "discrete_resonance": float("nan"),
        }

    frontier_pairs = _collect_pairs(
        history,
        "multiway_frontier_size",
        "unity_consistency",
    )
    frontier_unity_correlation = _pearson_from_pairs(frontier_pairs)

    curvature_pairs = _collect_pairs(
        history,
        "mean_forman_curvature",
        "unity_consistency",
    )
    if len(curvature_pairs) >= 2:
        xs = [pair[0] for pair in curvature_pairs]
        ys = [pair[1] for pair in curvature_pairs]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x > 0:
            cov = sum((x - mean_x) * (y - mean_y) for x, y in curvature_pairs)
            curvature_response = cov / var_x
        else:
            curvature_response = float("nan")
    else:
        curvature_response = float("nan")

    causal_pairs = _collect_pairs(
        history,
        "causal_max_depth",
        "discretization_index",
        x_transform=lambda depth, entry: depth
        / (1.0 + float(entry.get("event_count", 0.0))),
    )
    causal_feedback = _pearson_from_pairs(causal_pairs)

    spectral_values = [
        float(entry.get("spectral_dimension", float("nan"))) for entry in history
    ]
    spectral_average = _finite_average(spectral_values)
    final_spectral = spectral_values[-1]
    if math.isfinite(final_spectral) and math.isfinite(spectral_average):
        spectral_equilibrium = 1.0 / (1.0 + abs(final_spectral - spectral_average))
    else:
        spectral_equilibrium = float("nan")

    resonance_components: List[float] = []
    for entry in history:
        frontier = float(entry.get("multiway_frontier_size", float("nan")))
        density = float(entry.get("information_density", float("nan")))
        if math.isfinite(frontier) and math.isfinite(density):
            resonance_components.append(frontier * density)
    discrete_resonance = _finite_average(resonance_components)

    return {
        "frontier_unity_correlation": frontier_unity_correlation,
        "curvature_response": curvature_response,
        "causal_feedback": causal_feedback,
        "spectral_equilibrium": spectral_equilibrium,
        "discrete_resonance": discrete_resonance,
    }


def assess_unification_robustness(
    engine_factory: Callable[[], RewriteEngine],
    *,
    steps: int,
    replicates: int = 3,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
    multiway_generations: int = 2,
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
    multiway_generations:
        The maximum depth explored when constructing the auxiliary multiway
        summaries used for each replicate.

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
            multiway_generations=multiway_generations,
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


def map_unification_resonance(
    engine_factory: Callable[[], RewriteEngine],
    *,
    steps: int,
    multiway_depths: Sequence[int],
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
) -> Dict[str, float]:
    """Survey how multiway exploration depth modulates blended observables.

    For each entry in ``multiway_depths`` the factory is invoked to produce a
    fresh engine which is then advanced for ``steps`` rewrites while recording
    the unification summary.  The resulting dictionary aggregates statistics
    that describe how the unity observable, average multiway frontier size, and
    causal depth respond to progressively deeper explorations of the auxiliary
    multiway system.  Correlation coefficients against the requested depth act
    as coarse resonance indicators between Wolfram-style growth and Weinstein's
    geometric cues.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if not multiway_depths:
        raise ValueError("multiway_depths must not be empty")
    if any(depth < 0 for depth in multiway_depths):
        raise ValueError("multiway_depths must be non-negative")

    depth_values: List[float] = []
    unity_means: List[float] = []
    frontier_means: List[float] = []
    frontier_gradients: List[float] = []
    causal_depths: List[float] = []

    for index, depth in enumerate(multiway_depths):
        engine = engine_factory()
        seed = spectral_seed + index if spectral_seed is not None else None
        history = collect_unification_dynamics(
            engine,
            steps,
            include_initial=True,
            spectral_max_time=spectral_max_time,
            spectral_trials=spectral_trials,
            spectral_seed=seed,
            multiway_generations=depth,
        )
        if not history:
            continue

        depth_values.append(float(depth))

        unity_series = [
            float(entry.get("unity_consistency", float("nan"))) for entry in history
        ]
        frontier_series = [
            float(entry.get("multiway_frontier_size", float("nan"))) for entry in history
        ]
        causal_series = [
            float(entry.get("causal_max_depth", float("nan"))) for entry in history
        ]
        event_counts = [float(entry.get("event_count", 0.0)) for entry in history]

        unity_means.append(_finite_average(unity_series))
        frontier_means.append(_finite_average(frontier_series))
        causal_depths.append(float(history[-1].get("causal_max_depth", float("nan"))))

        first_frontier = next(
            (value for value in frontier_series if math.isfinite(value)),
            float("nan"),
        )
        last_frontier = next(
            (
                value
                for value in reversed(frontier_series)
                if math.isfinite(value)
            ),
            float("nan"),
        )
        event_span = event_counts[-1] - event_counts[0]
        if math.isfinite(first_frontier) and math.isfinite(last_frontier):
            frontier_gradients.append(
                _safe_ratio(last_frontier - first_frontier, event_span)
            )
        else:
            frontier_gradients.append(float("nan"))

    def _pairs(depths: Sequence[float], values: Sequence[float]) -> List[Tuple[float, float]]:
        pairs: List[Tuple[float, float]] = []
        for depth, value in zip(depths, values):
            if math.isfinite(depth) and math.isfinite(value):
                pairs.append((float(depth), float(value)))
        return pairs

    unity_corr = _pearson_from_pairs(_pairs(depth_values, unity_means))
    frontier_corr = _pearson_from_pairs(_pairs(depth_values, frontier_means))
    causal_corr = _pearson_from_pairs(_pairs(depth_values, causal_depths))

    alignment_components = [
        abs(value)
        for value in (unity_corr, frontier_corr, causal_corr)
        if math.isfinite(value)
    ]
    resonance_score = _finite_average(alignment_components)

    depth_span = (
        max(depth_values) - min(depth_values)
        if depth_values and len(depth_values) > 1
        else 0.0 if depth_values else float("nan")
    )

    return {
        "depths_evaluated": float(len(depth_values)),
        "depth_span": float(depth_span),
        "mean_unity_consistency": _finite_average(unity_means),
        "mean_frontier_size": _finite_average(frontier_means),
        "mean_frontier_gradient": _finite_average(frontier_gradients),
        "mean_causal_depth": _finite_average(causal_depths),
        "unity_depth_correlation": unity_corr,
        "frontier_depth_correlation": frontier_corr,
        "causal_depth_correlation": causal_corr,
        "resonance_score": resonance_score,
    }


def construct_unification_landscape(
    engine: RewriteEngine,
    steps: int,
    *,
    sample_interval: int = 1,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
    multiway_generations: int = 2,
) -> Dict[str, float]:
    """Synthesize a coarse landscape describing multiway/discrete interplay.

    The returned dictionary aggregates time series statistics that highlight how
    the size of the multiway frontier evolves relative to causal depth and the
    unity observable.  The function reuses :func:`collect_unification_dynamics`
    to obtain snapshots and compresses them into a handful of descriptive
    scalars suitable for dashboards or further analysis.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    if multiway_generations < 0:
        raise ValueError("multiway_generations must be non-negative")

    history = collect_unification_dynamics(
        engine,
        steps,
        include_initial=True,
        sample_interval=sample_interval,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=spectral_seed,
        multiway_generations=multiway_generations,
    )

    if not history:
        return {
            "history_length": 0.0,
            "layered_frontier_mean": float("nan"),
            "layered_frontier_growth": float("nan"),
            "multiway_depth_range": float("nan"),
            "unity_frontier_correlation": float("nan"),
            "multiway_generations": float(multiway_generations),
        }

    frontier_values = [
        float(entry.get("multiway_frontier_size", float("nan"))) for entry in history
    ]
    depth_values = [
        float(entry.get("multiway_max_depth", float("nan"))) for entry in history
    ]
    event_counts = [float(entry.get("event_count", 0.0)) for entry in history]

    layered_frontier_mean = _finite_average(frontier_values)

    first_frontier = next(
        (value for value in frontier_values if math.isfinite(value)),
        float("nan"),
    )
    last_frontier = next(
        (
            value
            for value in reversed(frontier_values)
            if math.isfinite(value)
        ),
        float("nan"),
    )
    event_span = event_counts[-1] - event_counts[0]
    if math.isfinite(first_frontier) and math.isfinite(last_frontier):
        layered_frontier_growth = _safe_ratio(last_frontier - first_frontier, event_span)
    else:
        layered_frontier_growth = float("nan")

    finite_depths = [value for value in depth_values if math.isfinite(value)]
    if finite_depths:
        multiway_depth_range = max(finite_depths) - min(finite_depths)
    else:
        multiway_depth_range = float("nan")

    frontier_pairs = _collect_pairs(
        history,
        "multiway_frontier_size",
        "unity_consistency",
    )
    unity_frontier_correlation = _pearson_from_pairs(frontier_pairs)

    return {
        "history_length": float(len(history)),
        "layered_frontier_mean": layered_frontier_mean,
        "layered_frontier_growth": layered_frontier_growth,
        "multiway_depth_range": multiway_depth_range,
        "unity_frontier_correlation": unity_frontier_correlation,
        "multiway_generations": float(multiway_generations),
    }


def synthesize_unification_attractor(
    engine: RewriteEngine,
    steps: int,
    *,
    window: int = 4,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
    multiway_generations: int = 2,
) -> Dict[str, float]:
    """Fuse sliding-window observables into a coarse unification attractor.

    The attractor blends four complementary perspectives across a moving
    observation window of summaries obtained via
    :func:`collect_unification_dynamics`:

    ``discrete_persistence``
        Averaged gradient of the discretization index, capturing how reliably
        the discrete substrate keeps growing.
    ``geometric_resonance``
        Mean product of unity consistency with the multiway frontier size,
        standing in for a resonance between geometric cues and branching
        richness.
    ``causal_gradient``
        Finite average of normalized causal depth gradients, measuring how
        quickly causal layering thickens relative to executed events.
    ``multiway_pressure``
        Ratio of average frontier size to the average number of executed
        events, highlighting the multiway contribution to the overall dynamical
        pressure.

    Parameters
    ----------
    engine:
        Rewrite engine evolved in-place to gather summaries.
    steps:
        Number of rewrite steps to apply.  Must be positive.
    window:
        Sliding-window length when aggregating observables.  Must be positive.

    The remaining keyword parameters mirror those of
    :func:`collect_unification_dynamics`.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if window <= 0:
        raise ValueError("window must be positive")

    history = collect_unification_dynamics(
        engine,
        steps,
        include_initial=True,
        spectral_max_time=spectral_max_time,
        spectral_trials=spectral_trials,
        spectral_seed=spectral_seed,
        multiway_generations=multiway_generations,
    )

    if len(history) < 2:
        return {
            "discrete_persistence": float("nan"),
            "geometric_resonance": float("nan"),
            "causal_gradient": float("nan"),
            "multiway_pressure": float("nan"),
        }

    window_size = min(window, len(history))
    discrete_gradients: List[float] = []
    causal_gradients: List[float] = []
    geometric_products: List[float] = []
    multiway_pressures: List[float] = []

    for start in range(len(history) - window_size + 1):
        segment = history[start : start + window_size]
        discretization_series = [
            float(entry.get("discretization_index", float("nan")))
            for entry in segment
        ]
        causal_series = [
            float(entry.get("causal_max_depth", float("nan")))
            for entry in segment
        ]
        unity_series = [
            float(entry.get("unity_consistency", float("nan")))
            for entry in segment
        ]
        frontier_series = [
            float(entry.get("multiway_frontier_size", float("nan")))
            for entry in segment
        ]
        event_series = [
            float(entry.get("event_count", 0.0)) for entry in segment
        ]

        def _segment_gradient(series: Sequence[float]) -> float:
            start_value = next(
                (value for value in series if math.isfinite(value)),
                float("nan"),
            )
            end_value = next(
                (
                    value
                    for value in reversed(series)
                    if math.isfinite(value)
                ),
                float("nan"),
            )
            if math.isfinite(start_value) and math.isfinite(end_value):
                return _safe_ratio(
                    end_value - start_value,
                    float(max(1, len(series) - 1)),
                )
            return float("nan")

        discrete_gradients.append(_segment_gradient(discretization_series))

        normalized_causal = []
        for depth, events in zip(causal_series, event_series):
            if math.isfinite(depth):
                normalized = depth / (1.0 + events)
                normalized_causal.append(normalized if math.isfinite(normalized) else float("nan"))
            else:
                normalized_causal.append(float("nan"))
        causal_gradients.append(_segment_gradient(normalized_causal))

        unity_mean = _finite_average(unity_series)
        frontier_mean = _finite_average(frontier_series)
        if math.isfinite(unity_mean) and math.isfinite(frontier_mean):
            geometric_products.append(unity_mean * frontier_mean)
        else:
            geometric_products.append(float("nan"))

        frontier_average = _finite_average(frontier_series)
        event_average = _finite_average(event_series)
        multiway_pressures.append(
            _safe_ratio(frontier_average, 1.0 + event_average)
            if math.isfinite(frontier_average) and math.isfinite(event_average)
            else float("nan")
        )

    return {
        "discrete_persistence": _finite_average(discrete_gradients),
        "geometric_resonance": _finite_average(geometric_products),
        "causal_gradient": _finite_average(causal_gradients),
        "multiway_pressure": _finite_average(multiway_pressures),
    }


def compose_unification_manifest(
    engine_factory: Callable[[], RewriteEngine],
    *,
    steps: int,
    replicates: int = 3,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
    multiway_generations: int = 2,
    sample_interval: int = 1,
    window: int = 4,
) -> Dict[str, float]:
    """Fuse multiple bridge metrics into a unified manifest of observables.

    The manifest aggregates the terminal values of the unification summary,
    certificate, alignment, principles, landscape, and attractor channels over
    several independent replicates supplied by ``engine_factory``.  Each
    replicate uses a fresh :class:`~engine.rewrite.RewriteEngine` instance,
    ensuring that stochastic rewrite choices do not leak across measurements.

    Parameters
    ----------
    engine_factory:
        Callable returning a brand-new engine whenever invoked.  The hypergraph
        within the engine is mutated in-place during the computation and must
        therefore not be reused across replicates.
    steps:
        Number of rewrite steps explored for each replicate.  Must be
        positive.
    replicates:
        Number of independent replicates to aggregate.  Must be positive.
    spectral_max_time, spectral_trials, spectral_seed:
        Parameters forwarded to the spectral-dimension estimators inside the
        downstream metrics.  When ``spectral_seed`` is provided, successive
        replicates increment it to avoid identical random walks.
    multiway_generations:
        Maximum depth explored when constructing auxiliary multiway summaries.
        The value is forwarded to all downstream helper functions.
    sample_interval:
        Sampling cadence used when constructing landscape summaries.  Must be
        positive.
    window:
        Sliding-window length employed by the attractor synthesis.  Must be
        positive.

    Returns
    -------
    Dict[str, float]
        Aggregate statistics highlighting how discrete growth (Wolfram) and
        geometric regularity (Weinstein) cohere across the supplied replicates.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    if window <= 0:
        raise ValueError("window must be positive")

    unity_values: List[float] = []
    discretization_values: List[float] = []
    certificate_strengths: List[float] = []
    dual_correlations: List[float] = []
    alignment_scores: List[float] = []
    growth_rates: List[float] = []
    geometric_balances: List[float] = []
    geometric_resonances: List[float] = []
    causal_gradients: List[float] = []
    multiway_pressures: List[float] = []
    frontier_correlations: List[float] = []
    frontier_growths: List[float] = []

    for replicate in range(replicates):
        seed = spectral_seed + replicate if spectral_seed is not None else None

        history_engine = engine_factory()
        history = collect_unification_dynamics(
            history_engine,
            steps,
            include_initial=True,
            spectral_max_time=spectral_max_time,
            spectral_trials=spectral_trials,
            spectral_seed=seed,
            multiway_generations=multiway_generations,
        )
        final_summary = history[-1] if history else {}
        unity_values.append(float(final_summary.get("unity_consistency", float("nan"))))
        discretization_values.append(
            float(final_summary.get("discretization_index", float("nan")))
        )

        certificate_engine = engine_factory()
        certificate = generate_unification_certificate(
            certificate_engine,
            steps,
            spectral_max_time=spectral_max_time,
            spectral_trials=spectral_trials,
            spectral_seed=seed,
            multiway_generations=multiway_generations,
        )
        certificate_strengths.append(
            float(certificate.get("certificate_strength", float("nan")))
        )
        dual_correlations.append(float(certificate.get("dual_correlation", float("nan"))))

        alignment_engine = engine_factory()
        alignment = evaluate_unification_alignment(
            alignment_engine,
            steps,
            spectral_max_time=spectral_max_time,
            spectral_trials=spectral_trials,
            spectral_seed=seed,
            multiway_generations=multiway_generations,
        )
        alignment_scores.append(float(alignment.get("alignment_score", float("nan"))))

        principles_engine = engine_factory()
        principles = derive_unification_principles(
            principles_engine,
            steps,
            spectral_max_time=spectral_max_time,
            spectral_trials=spectral_trials,
            spectral_seed=seed,
            multiway_generations=multiway_generations,
        )
        growth_rates.append(float(principles.get("growth_rate", float("nan"))))
        geometric_balances.append(float(principles.get("geometric_balance", float("nan"))))

        attractor_engine = engine_factory()
        attractor = synthesize_unification_attractor(
            attractor_engine,
            steps,
            window=window,
            spectral_max_time=spectral_max_time,
            spectral_trials=spectral_trials,
            spectral_seed=seed,
            multiway_generations=multiway_generations,
        )
        geometric_resonances.append(float(attractor.get("geometric_resonance", float("nan"))))
        causal_gradients.append(float(attractor.get("causal_gradient", float("nan"))))
        multiway_pressures.append(float(attractor.get("multiway_pressure", float("nan"))))

        landscape_engine = engine_factory()
        landscape = construct_unification_landscape(
            landscape_engine,
            steps,
            sample_interval=sample_interval,
            spectral_max_time=spectral_max_time,
            spectral_trials=spectral_trials,
            spectral_seed=seed,
            multiway_generations=multiway_generations,
        )
        frontier_correlations.append(
            float(landscape.get("unity_frontier_correlation", float("nan")))
        )
        frontier_growths.append(float(landscape.get("layered_frontier_growth", float("nan"))))

    unity_certificate_pairs = [
        (u, c)
        for u, c in zip(unity_values, certificate_strengths)
        if math.isfinite(u) and math.isfinite(c)
    ]
    discretization_pressure_pairs = [
        (d, p)
        for d, p in zip(discretization_values, multiway_pressures)
        if math.isfinite(d) and math.isfinite(p)
    ]
    frontier_alignment_pairs = [
        (f, a)
        for f, a in zip(frontier_growths, alignment_scores)
        if math.isfinite(f) and math.isfinite(a)
    ]

    mean_certificate_strength = _finite_average(certificate_strengths)
    mean_alignment_score = _finite_average(alignment_scores)
    mean_geometric_resonance = _finite_average(geometric_resonances)
    mean_multiway_pressure = _finite_average(multiway_pressures)

    concordance_components = [
        abs(value)
        for value in (
            mean_certificate_strength,
            mean_alignment_score,
            mean_geometric_resonance,
            mean_multiway_pressure,
        )
        if math.isfinite(value)
    ]

    return {
        "replicates": float(replicates),
        "mean_unity_consistency": _finite_average(unity_values),
        "mean_discretization_index": _finite_average(discretization_values),
        "mean_certificate_strength": mean_certificate_strength,
        "mean_dual_correlation": _finite_average(dual_correlations),
        "mean_alignment_score": mean_alignment_score,
        "mean_growth_rate": _finite_average(growth_rates),
        "mean_geometric_balance": _finite_average(geometric_balances),
        "mean_geometric_resonance": mean_geometric_resonance,
        "mean_causal_gradient": _finite_average(causal_gradients),
        "mean_multiway_pressure": mean_multiway_pressure,
        "mean_unity_frontier_correlation": _finite_average(frontier_correlations),
        "mean_frontier_growth": _finite_average(frontier_growths),
        "unity_certificate_correlation": _pearson_from_pairs(unity_certificate_pairs),
        "discretization_pressure_correlation": _pearson_from_pairs(
            discretization_pressure_pairs
        ),
        "frontier_alignment_correlation": _pearson_from_pairs(frontier_alignment_pairs),
        "concordance_index": _finite_average(concordance_components),
    }


def harmonize_unification_channels(
    engine: RewriteEngine,
    steps: int,
    *,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
    multiway_generations: int = 2,
) -> Dict[str, float]:
    """Blend discrete, causal, geometric, and multiway cues into harmony metrics.

    The returned dictionary exposes coarse indicators that summarize how the
    computational (Wolfram) and geometric (Weinstein) perspectives co-evolve
    during a rewrite experiment.  Four complementary measurements are
    provided:

    ``unity_flux``
        Finite-difference estimate of the unity observable's drift per executed
        event.
    ``geometric_causal_ratio``
        Ratio comparing mean Forman curvature against the average normalized
        causal depth, highlighting the balance between curvature and causal
        layering.
    ``branching_intensity``
        Average multiway frontier size normalized by the mean number of
        executed events, representing the pressure contributed by multiway
        branching.
    ``coherence_index``
        Mean absolute correlation across three channel pairs (discrete vs
        unity, causal vs unity, geometric vs frontier) used as a global harmony
        score.

    Individual correlations backing the ``coherence_index`` are included in the
    result for downstream inspection.  The ``multiway_generations`` argument is
    forwarded to :func:`collect_unification_dynamics`, allowing callers to tune
    how deeply the auxiliary branching structure is explored while computing
    the harmony metrics.
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
        multiway_generations=multiway_generations,
    )

    if len(history) < 2:
        return {
            "unity_flux": float("nan"),
            "geometric_causal_ratio": float("nan"),
            "branching_intensity": float("nan"),
            "discrete_unity_correlation": float("nan"),
            "causal_unity_correlation": float("nan"),
            "geometric_frontier_correlation": float("nan"),
            "coherence_index": float("nan"),
        }

    first = history[0]
    last = history[-1]

    initial_unity = float(first.get("unity_consistency", float("nan")))
    final_unity = float(last.get("unity_consistency", float("nan")))
    event_span = float(last.get("event_count", 0.0) - first.get("event_count", 0.0))
    if math.isfinite(initial_unity) and math.isfinite(final_unity):
        unity_flux = _safe_ratio(final_unity - initial_unity, event_span)
    else:
        unity_flux = float("nan")

    curvature_series = [
        float(entry.get("mean_forman_curvature", float("nan"))) for entry in history
    ]
    normalized_depths: List[float] = []
    for entry in history:
        depth = float(entry.get("causal_max_depth", float("nan")))
        events = float(entry.get("event_count", 0.0))
        normalized = depth / (1.0 + events)
        normalized_depths.append(normalized if math.isfinite(normalized) else float("nan"))

    mean_curvature = _finite_average(curvature_series)
    mean_normalized_depth = _finite_average(normalized_depths)
    if math.isfinite(mean_curvature) and math.isfinite(mean_normalized_depth):
        geometric_causal_ratio = mean_curvature / (1.0 + mean_normalized_depth)
    else:
        geometric_causal_ratio = float("nan")

    frontier_series = [
        float(entry.get("multiway_frontier_size", float("nan"))) for entry in history
    ]
    event_series = [float(entry.get("event_count", 0.0)) for entry in history]
    frontier_average = _finite_average(frontier_series)
    event_average = _finite_average(event_series)
    if math.isfinite(frontier_average) and math.isfinite(event_average):
        branching_intensity = _safe_ratio(frontier_average, 1.0 + event_average)
    else:
        branching_intensity = float("nan")

    discrete_pairs = _collect_pairs(
        history,
        "discretization_index",
        "unity_consistency",
    )
    discrete_unity_correlation = _pearson_from_pairs(discrete_pairs)

    causal_pairs = _collect_pairs(
        history,
        "causal_max_depth",
        "unity_consistency",
        x_transform=lambda depth, entry: depth
        / (1.0 + float(entry.get("event_count", 0.0))),
    )
    causal_unity_correlation = _pearson_from_pairs(causal_pairs)

    geometric_pairs = _collect_pairs(
        history,
        "mean_forman_curvature",
        "multiway_frontier_size",
    )
    geometric_frontier_correlation = _pearson_from_pairs(geometric_pairs)

    coherence_components = [
        abs(value)
        for value in (
            discrete_unity_correlation,
            causal_unity_correlation,
            geometric_frontier_correlation,
        )
        if math.isfinite(value)
    ]
    coherence_index = _finite_average(coherence_components)

    return {
        "unity_flux": unity_flux,
        "geometric_causal_ratio": geometric_causal_ratio,
        "branching_intensity": branching_intensity,
        "discrete_unity_correlation": discrete_unity_correlation,
        "causal_unity_correlation": causal_unity_correlation,
        "geometric_frontier_correlation": geometric_frontier_correlation,
        "coherence_index": coherence_index,
    }


def trace_unification_phase_portrait(
    engine: RewriteEngine,
    steps: int,
    *,
    spectral_max_time: int = 6,
    spectral_trials: int = 200,
    spectral_seed: int | None = None,
    multiway_generations: int = 2,
) -> Dict[str, float]:
    """Trace a phase portrait linking discrete and geometric observables.

    The phase portrait complements the other bridge metrics by explicitly
    charting how the discretization index and unity observable co-evolve during
    a rewrite experiment.  The returned dictionary provides five derived
    quantities:

    ``phase_area``
        Absolute area swept in the (discretization, unity) plane using a
        trapezoidal estimate ordered by executed events.  This measures the
        cumulative interplay between Wolfram-style growth and Weinstein's unity
        cues.
    ``frontier_phase_coupling``
        Pearson correlation between discretization index and multiway frontier
        size across the recorded history.
    ``geometric_phase_coupling``
        Pearson correlation between the unity observable and mean Forman
        curvature.
    ``causal_phase_gradient``
        Finite-difference gradient of normalized causal depth with respect to
        executed events.
    ``unity_causal_correlation``
        Pearson correlation between normalized causal depth and the unity
        observable.
    ``phase_coherence``
        Mean absolute value of the three correlation coefficients, offering a
        compact harmony indicator for the phase portrait.

    ``multiway_generations`` mirrors the parameter used by
    :func:`collect_unification_dynamics`, allowing callers to tune the depth of
    the auxiliary multiway exploration informing each snapshot.
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
        multiway_generations=multiway_generations,
    )

    if len(history) < 2:
        return {
            "phase_area": float("nan"),
            "frontier_phase_coupling": float("nan"),
            "geometric_phase_coupling": float("nan"),
            "causal_phase_gradient": float("nan"),
            "unity_causal_correlation": float("nan"),
            "phase_coherence": float("nan"),
        }

    phase_points: List[Tuple[float, float, float]] = []
    for entry in history:
        discretization = float(entry.get("discretization_index", float("nan")))
        unity = float(entry.get("unity_consistency", float("nan")))
        events = float(entry.get("event_count", 0.0))
        if math.isfinite(discretization) and math.isfinite(unity):
            phase_points.append((events, discretization, unity))

    phase_points.sort(key=lambda item: item[0])

    if len(phase_points) >= 2:
        area = 0.0
        _, previous_discretization, previous_unity = phase_points[0]
        for _, discretization, unity in phase_points[1:]:
            area += 0.5 * (previous_unity + unity) * (discretization - previous_discretization)
            previous_discretization = discretization
            previous_unity = unity
        phase_area = abs(area)
    else:
        phase_area = float("nan")

    frontier_pairs = _collect_pairs(
        history,
        "discretization_index",
        "multiway_frontier_size",
    )
    frontier_phase_coupling = _pearson_from_pairs(frontier_pairs)

    geometric_pairs = _collect_pairs(
        history,
        "unity_consistency",
        "mean_forman_curvature",
    )
    geometric_phase_coupling = _pearson_from_pairs(geometric_pairs)

    causal_pairs = _collect_pairs(
        history,
        "causal_max_depth",
        "unity_consistency",
        x_transform=lambda depth, entry: depth
        / (1.0 + float(entry.get("event_count", 0.0))),
    )
    unity_causal_correlation = _pearson_from_pairs(causal_pairs)

    normalized_depths: List[float] = []
    event_counts: List[float] = []
    for entry in history:
        depth = float(entry.get("causal_max_depth", float("nan")))
        events = float(entry.get("event_count", 0.0))
        normalized = depth / (1.0 + events) if math.isfinite(depth) else float("nan")
        normalized_depths.append(normalized if math.isfinite(normalized) else float("nan"))
        event_counts.append(events)

    first_index = next(
        (index for index, value in enumerate(normalized_depths) if math.isfinite(value)),
        None,
    )
    last_index = next(
        (
            len(normalized_depths) - 1 - index
            for index, value in enumerate(reversed(normalized_depths))
            if math.isfinite(value)
        ),
        None,
    )

    if first_index is not None and last_index is not None and first_index != last_index:
        depth_delta = normalized_depths[last_index] - normalized_depths[first_index]
        event_delta = event_counts[last_index] - event_counts[first_index]
        causal_phase_gradient = _safe_ratio(depth_delta, event_delta) if event_delta else float("nan")
    else:
        causal_phase_gradient = float("nan")

    coherence_components = [
        abs(value)
        for value in (
            frontier_phase_coupling,
            geometric_phase_coupling,
            unity_causal_correlation,
        )
        if math.isfinite(value)
    ]
    phase_coherence = _finite_average(coherence_components)

    return {
        "phase_area": phase_area,
        "frontier_phase_coupling": frontier_phase_coupling,
        "geometric_phase_coupling": geometric_phase_coupling,
        "causal_phase_gradient": causal_phase_gradient,
        "unity_causal_correlation": unity_causal_correlation,
        "phase_coherence": phase_coherence,
    }


__all__ = [
    "compute_unification_summary",
    "collect_unification_dynamics",
    "generate_unification_certificate",
    "derive_unification_principles",
    "assess_unification_robustness",
    "evaluate_unification_alignment",
    "map_unification_resonance",
    "construct_unification_landscape",
    "synthesize_unification_attractor",
    "compose_unification_manifest",
    "analyze_unification_feedback",
    "harmonize_unification_channels",
    "trace_unification_phase_portrait",
]
