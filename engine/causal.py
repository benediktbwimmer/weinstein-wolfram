"""Causal graph tracking for rewrite events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Sequence, Set


@dataclass
class CausalGraph:
    """Directed acyclic graph capturing dependencies between rewrite events."""

    predecessors: Dict[int, Set[int]]
    successors: Dict[int, Set[int]]

    def __init__(self) -> None:
        self.predecessors = defaultdict(set)
        self.successors = defaultdict(set)

    def add_event(self, event_id: int, dependencies: Iterable[int]) -> None:
        deps = set(dependencies)
        self.predecessors[event_id] = deps
        for dep in deps:
            self.successors[dep].add(event_id)
        self.successors.setdefault(event_id, set())

    def __len__(self) -> int:
        return len(self.predecessors)

    def average_indegree(self) -> float:
        if not self.predecessors:
            return 0.0
        total = sum(len(deps) for deps in self.predecessors.values())
        return total / len(self.predecessors)

    def average_outdegree(self) -> float:
        if not self.successors:
            return 0.0
        total = sum(len(succ) for succ in self.successors.values())
        return total / len(self.successors)

    def max_depth(self) -> int:
        depth: Dict[int, int] = {}
        for event_id in sorted(self.predecessors.keys()):
            deps = self.predecessors[event_id]
            if not deps:
                depth[event_id] = 0
            else:
                depth[event_id] = max(depth[dep] + 1 for dep in deps)
        return max(depth.values(), default=0)

    def basic_stats(self) -> Dict[str, float]:
        return {
            "event_count": float(len(self)),
            "avg_indegree": self.average_indegree(),
            "avg_outdegree": self.average_outdegree(),
            "max_depth": float(self.max_depth()),
        }


__all__ = ["CausalGraph"]
