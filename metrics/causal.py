"""Summaries for causal graphs."""

from __future__ import annotations

from typing import Dict

from engine.causal import CausalGraph


def summarize_causal_graph(graph: CausalGraph) -> Dict[str, float]:
    """Return a dictionary of simple causal graph statistics."""

    return graph.basic_stats()


__all__ = ["summarize_causal_graph"]
