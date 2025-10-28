"""Run a small hypergraph rewrite experiment and print summary metrics."""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.hypergraph import Hypergraph
from engine.rewrite import EdgeSplit3Rule, RewriteEngine
from metrics import (
    assess_unification_robustness,
    compute_unification_summary,
    mean_forman_curvature,
    spectral_dimension,
    summarize_causal_graph,
)


def main() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    rule = EdgeSplit3Rule()
    engine = RewriteEngine(hypergraph, rule, seed=1)
    engine.run(steps=200)

    skeleton = hypergraph.one_skeleton()
    ds = spectral_dimension(skeleton, max_time=6, trials=400, seed=2)
    curvature = mean_forman_curvature(skeleton)
    causal_stats = summarize_causal_graph(engine.causal_graph)
    unified = compute_unification_summary(
        engine,
        spectral_max_time=6,
        spectral_trials=400,
        spectral_seed=3,
    )

    print("Final hypergraph state:")
    print(f"  Nodes: {hypergraph.node_count}")
    print(f"  Hyperedges: {hypergraph.edge_count}")
    if ds == ds:
        print(f"  Spectral dimension (estimate): {ds:.3f}")
    else:
        print("  Spectral dimension: NaN")
    if curvature == curvature:
        print(f"  Mean Forman curvature: {curvature:.3f}")
    else:
        print("  Mean Forman curvature: NaN")
    print("  Causal statistics:")
    pprint(causal_stats)
    print("  Unified summary:")
    for key in sorted(unified.keys()):
        value = unified[key]
        if value == value:
            if abs(value - round(value)) < 1e-9:
                display = str(int(round(value)))
            else:
                display = f"{value:.6f}"
        else:
            display = "NaN"
        print(f"    {key}: {display}")

    class Factory:
        def __init__(self) -> None:
            self.seed = 10

        def __call__(self) -> RewriteEngine:
            hg = Hypergraph([(0, 1, 2)])
            engine = RewriteEngine(hg, EdgeSplit3Rule(), seed=self.seed)
            self.seed += 1
            return engine

    robustness = assess_unification_robustness(
        Factory(),
        steps=60,
        replicates=4,
        spectral_max_time=4,
        spectral_trials=200,
        spectral_seed=5,
    )

    print("  Robustness across replicates:")
    for key in sorted(robustness.keys()):
        value = robustness[key]
        if value == value:
            display = f"{value:.6f}"
        else:
            display = "NaN"
        print(f"    {key}: {display}")


if __name__ == "__main__":
    main()
