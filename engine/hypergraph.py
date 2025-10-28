"""Core hypergraph data structures used by the rewrite engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

Hyperedge = Tuple[int, int, int]
Adjacency = Dict[int, Set[int]]


def _normalize_edge(nodes: Sequence[int]) -> Hyperedge:
    if len(nodes) != 3:
        raise ValueError("Hypergraph currently supports only 3-uniform edges")
    return tuple(sorted(int(n) for n in nodes))  # type: ignore[return-value]


@dataclass
class Hypergraph:
    """A minimal mutable 3-uniform hypergraph."""

    edges: Set[Hyperedge]
    nodes: Set[int]
    _next_node_id: int

    def __init__(self, edges: Iterable[Sequence[int]] | None = None) -> None:
        self.edges = set()
        self.nodes = set()
        self._next_node_id = 0
        if edges:
            for edge in edges:
                self.add_edge(edge)

    def copy(self) -> "Hypergraph":
        new = Hypergraph()
        new.edges = set(self.edges)
        new.nodes = set(self.nodes)
        new._next_node_id = self._next_node_id
        return new

    def new_node(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes.add(node_id)
        return node_id

    def ensure_node(self, node_id: int) -> None:
        self.nodes.add(node_id)
        if node_id >= self._next_node_id:
            self._next_node_id = node_id + 1

    def add_edge(self, nodes: Sequence[int]) -> Hyperedge:
        edge = _normalize_edge(nodes)
        for node in edge:
            self.ensure_node(node)
        self.edges.add(edge)
        return edge

    def remove_edge(self, edge: Sequence[int]) -> None:
        normalized = _normalize_edge(edge)
        self.edges.remove(normalized)

    def __len__(self) -> int:
        return len(self.edges)

    def __iter__(self) -> Iterator[Hyperedge]:
        return iter(self.edges)

    def one_skeleton(self) -> Adjacency:
        """Return the 1-skeleton (graph) adjacency map."""
        adj: Adjacency = defaultdict(set)
        for node in self.nodes:
            adj[node]  # ensure node appears even if isolated
        for edge in self.edges:
            for u, v in combinations(edge, 2):
                adj[u].add(v)
                adj[v].add(u)
        return {node: set(neigh) for node, neigh in adj.items()}

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def choose_edge(self, index: int) -> Hyperedge:
        """Deterministically choose an edge by index (for RNG-driven selection)."""
        if not self.edges:
            raise ValueError("Cannot choose an edge from an empty hypergraph")
        try:
            return list(sorted(self.edges))[index % len(self.edges)]
        except IndexError as exc:  # pragma: no cover - defensive
            raise ValueError("Edge index out of range") from exc


__all__ = ["Hypergraph", "Hyperedge", "Adjacency"]
