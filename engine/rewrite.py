"""Rewrite rules and engine implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .causal import CausalGraph
from .hypergraph import Hyperedge, Hypergraph


@dataclass
class RewriteResult:
    """Outcome of applying a rewrite rule."""

    created_nodes: Tuple[int, ...]
    new_edges: Tuple[Hyperedge, ...]
    removed_edges: Tuple[Hyperedge, ...]


@dataclass
class RewriteEvent:
    """A single rewrite event with dependency metadata."""

    event_id: int
    rule_name: str
    target_edge: Hyperedge
    dependencies: Tuple[int, ...]
    result: RewriteResult


class RewriteRule:
    """Base class for hypergraph rewrite rules."""

    name: str = "base"

    def choose_target(self, hypergraph: Hypergraph, rng: random.Random) -> Hyperedge:
        edges = sorted(hypergraph.edges)
        if not edges:
            raise ValueError("Rewrite rule cannot act on an empty hypergraph")
        return rng.choice(edges)

    def apply(
        self, hypergraph: Hypergraph, edge: Hyperedge, rng: random.Random
    ) -> RewriteResult:
        raise NotImplementedError


class EdgeSplit3Rule(RewriteRule):
    """Split one 3-edge into three around a newly created node."""

    name = "edge_split_3"

    def apply(
        self, hypergraph: Hypergraph, edge: Hyperedge, rng: random.Random
    ) -> RewriteResult:
        if edge not in hypergraph.edges:
            raise ValueError("Edge must exist to be rewritten")
        hypergraph.remove_edge(edge)
        new_node = hypergraph.new_node()
        a, b, c = edge
        new_edges = (
            hypergraph.add_edge((a, b, new_node)),
            hypergraph.add_edge((b, c, new_node)),
            hypergraph.add_edge((c, a, new_node)),
        )
        return RewriteResult(
            created_nodes=(new_node,),
            new_edges=new_edges,
            removed_edges=(edge,),
        )


class RewriteEngine:
    """Apply rewrite rules while tracking causal dependencies."""

    def __init__(
        self,
        hypergraph: Hypergraph,
        rule: RewriteRule,
        seed: Optional[int] = None,
    ) -> None:
        self.hypergraph = hypergraph
        self.rule = rule
        self.rng = random.Random(seed)
        self._next_event_id = 0
        self.causal_graph = CausalGraph()
        self._edge_last_touch: dict[Hyperedge, int] = {}
        self.events: List[RewriteEvent] = []

    def step(self) -> RewriteEvent:
        target = self.rule.choose_target(self.hypergraph, self.rng)
        event_id = self._next_event_id
        self._next_event_id += 1
        result = self.rule.apply(self.hypergraph, target, self.rng)
        dependencies = set()
        for edge in result.removed_edges:
            previous = self._edge_last_touch.get(edge)
            if previous is not None:
                dependencies.add(previous)
            self._edge_last_touch.pop(edge, None)
        for edge in result.new_edges:
            previous = self._edge_last_touch.get(edge)
            if previous is not None:
                dependencies.add(previous)
            self._edge_last_touch[edge] = event_id
        event = RewriteEvent(
            event_id=event_id,
            rule_name=self.rule.name,
            target_edge=target,
            dependencies=tuple(sorted(dependencies)),
            result=result,
        )
        self.causal_graph.add_event(event_id, event.dependencies)
        self.events.append(event)
        return event

    def run(self, steps: int) -> List[RewriteEvent]:
        history: List[RewriteEvent] = []
        for _ in range(steps):
            history.append(self.step())
        return history


__all__ = [
    "RewriteEngine",
    "RewriteEvent",
    "RewriteResult",
    "RewriteRule",
    "EdgeSplit3Rule",
]
