"""Multiway system support for hypergraph rewrite rules."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from .hypergraph import Hyperedge, Hypergraph
from .rewrite import RewriteResult, RewriteRule


def _state_signature(hypergraph: Hypergraph) -> Tuple[Hyperedge, ...]:
    """Return a canonical signature for hypergraph comparison."""

    return tuple(sorted(hypergraph.edges))


@dataclass(frozen=True)
class MultiwayEvent:
    """A single branch in the multiway evolution."""

    event_id: int
    parent_state: int
    child_state: int
    rule_name: str
    target_edge: Hyperedge
    result: RewriteResult


@dataclass(frozen=True)
class MultiwayState:
    """A node in the multiway state graph."""

    state_id: int
    depth: int
    hypergraph: Hypergraph
    parent_event: int | None
    signature: Tuple[Hyperedge, ...]


@dataclass
class MultiwayEvolution:
    """Full record of a multiway computation."""

    states: Dict[int, MultiwayState]
    events: List[MultiwayEvent]
    adjacency: Dict[int, Set[int]]

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def depth_histogram(self) -> Dict[int, int]:
        histogram: Dict[int, int] = defaultdict(int)
        for state in self.states.values():
            histogram[state.depth] += 1
        return dict(sorted(histogram.items()))

    def average_branching_factor(self) -> float:
        if not self.adjacency:
            return 0.0
        branching_sum = sum(len(children) for children in self.adjacency.values())
        return branching_sum / len(self.adjacency)

    def frontier(self, depth: int) -> List[MultiwayState]:
        return [state for state in self.states.values() if state.depth == depth]


class MultiwaySystem:
    """Enumerate the multiway evolution of a rewrite system."""

    def __init__(
        self,
        initial_hypergraph: Hypergraph,
        rules: Sequence[RewriteRule],
        *,
        seed: int | None = None,
    ) -> None:
        if not rules:
            raise ValueError("MultiwaySystem requires at least one rewrite rule")
        self._initial_template = initial_hypergraph.copy()
        self.rules = tuple(rules)
        self.seed = seed

    def run(self, max_generations: int) -> MultiwayEvolution:
        """Explore all rewrites up to ``max_generations`` layers deep."""

        if max_generations < 0:
            raise ValueError("max_generations must be non-negative")

        rng = random.Random(self.seed)
        initial_hypergraph = self._initial_template.copy()

        states: Dict[int, MultiwayState] = {}
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        events: List[MultiwayEvent] = []
        signature_index: Dict[Tuple[Hyperedge, ...], int] = {}

        initial_signature = _state_signature(initial_hypergraph)
        initial_state = MultiwayState(
            state_id=0,
            depth=0,
            hypergraph=initial_hypergraph,
            parent_event=None,
            signature=initial_signature,
        )
        states[0] = initial_state
        signature_index[initial_signature] = 0
        adjacency[0]  # ensure presence

        queue: deque[Tuple[int, int]] = deque([(0, 0)])
        next_state_id = 1
        next_event_id = 0

        while queue:
            state_id, depth = queue.popleft()
            state = states[state_id]
            adjacency.setdefault(state_id, set())
            if depth >= max_generations:
                continue

            for rule in self.rules:
                for edge in sorted(state.hypergraph.edges):
                    new_hypergraph = state.hypergraph.copy()
                    result = rule.apply(new_hypergraph, edge, rng)
                    signature = _state_signature(new_hypergraph)
                    child_state_id = signature_index.get(signature)
                    if child_state_id is None:
                        child_state_id = next_state_id
                        next_state_id += 1
                        child_state = MultiwayState(
                            state_id=child_state_id,
                            depth=depth + 1,
                            hypergraph=new_hypergraph,
                            parent_event=next_event_id,
                            signature=signature,
                        )
                        states[child_state_id] = child_state
                        signature_index[signature] = child_state_id
                        adjacency.setdefault(child_state_id, set())
                        queue.append((child_state_id, depth + 1))
                    event = MultiwayEvent(
                        event_id=next_event_id,
                        parent_state=state_id,
                        child_state=child_state_id,
                        rule_name=rule.name,
                        target_edge=edge,
                        result=result,
                    )
                    events.append(event)
                    adjacency[state_id].add(child_state_id)
                    next_event_id += 1

        return MultiwayEvolution(
            states=states,
            events=events,
            adjacency={state: set(children) for state, children in adjacency.items()},
        )


__all__ = [
    "MultiwayEvent",
    "MultiwayState",
    "MultiwayEvolution",
    "MultiwaySystem",
]
