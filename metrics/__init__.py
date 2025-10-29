"""Metric computations for hypergraph experiments."""

from .geom import (
    average_clustering_coefficient,
    mean_forman_curvature,
    spectral_dimension,
)
from .causal import summarize_causal_graph
from .unification import (
    assess_unification_robustness,
    collect_unification_dynamics,
    construct_unification_landscape,
    compute_unification_summary,
    derive_unification_principles,
    evaluate_unification_alignment,
    generate_unification_certificate,
    map_unification_resonance,
    synthesize_unification_attractor,
)
from .toy import ToyModelResult, run_toy_unification_model

__all__ = [
    "spectral_dimension",
    "mean_forman_curvature",
    "average_clustering_coefficient",
    "summarize_causal_graph",
    "compute_unification_summary",
    "collect_unification_dynamics",
    "construct_unification_landscape",
    "generate_unification_certificate",
    "derive_unification_principles",
    "assess_unification_robustness",
    "evaluate_unification_alignment",
    "map_unification_resonance",
    "synthesize_unification_attractor",
    "ToyModelResult",
    "run_toy_unification_model",
]
