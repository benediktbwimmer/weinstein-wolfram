"""Geometric metrics for hypergraphs."""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from engine.hypergraph import Adjacency


def spectral_dimension(
    adjacency: Adjacency,
    *,
    max_time: int = 6,
    trials: int = 200,
    seed: int | None = None,
) -> float:
    """Estimate the spectral dimension via random-walk return probabilities."""

    if not adjacency:
        return float("nan")
    nodes = list(adjacency.keys())
    if not nodes:
        return float("nan")
    rng = random.Random(seed)
    samples: List[Tuple[float, float]] = []
    for t in range(1, max_time + 1):
        returns = 0
        attempts = 0
        for _ in range(trials):
            start = rng.choice(nodes)
            current = start
            valid = True
            for _ in range(t):
                neighbors = list(adjacency[current])
                if not neighbors:
                    valid = False
                    break
                current = rng.choice(neighbors)
            if not valid:
                continue
            attempts += 1
            if current == start:
                returns += 1
        if attempts == 0 or returns == 0:
            continue
        probability = returns / attempts
        samples.append((math.log(t), math.log(probability)))
    if len(samples) < 2:
        return float("nan")
    sum_x = sum(x for x, _ in samples)
    sum_y = sum(y for _, y in samples)
    sum_xx = sum(x * x for x, _ in samples)
    sum_xy = sum(x * y for x, y in samples)
    n = len(samples)
    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-9:
        return float("nan")
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return max(0.0, -2.0 * slope)


def mean_forman_curvature(adjacency: Adjacency) -> float:
    """Compute the mean Forman curvature for the 1-skeleton."""

    if not adjacency:
        return float("nan")
    degrees = {node: len(neigh) for node, neigh in adjacency.items()}
    total = 0.0
    edge_count = 0
    for u, neighbors in adjacency.items():
        for v in neighbors:
            if u < v:
                total += 4 - (degrees[u] + degrees[v])
                edge_count += 1
    if edge_count == 0:
        return float("nan")
    return total / edge_count


def average_clustering_coefficient(adjacency: Adjacency) -> float:
    """Return the mean local clustering coefficient of the graph."""

    if not adjacency:
        return float("nan")

    coefficients: List[float] = []
    for node, neighbors in adjacency.items():
        degree = len(neighbors)
        if degree < 2:
            continue
        link_count = 0
        for u, v in combinations(sorted(neighbors), 2):
            if u in adjacency and v in adjacency[u]:
                link_count += 1
        possible_links = degree * (degree - 1) / 2
        if possible_links > 0:
            coefficients.append(link_count / possible_links)

    if not coefficients:
        return float("nan")

    return sum(coefficients) / len(coefficients)


__all__ = ["spectral_dimension", "mean_forman_curvature", "average_clustering_coefficient"]
