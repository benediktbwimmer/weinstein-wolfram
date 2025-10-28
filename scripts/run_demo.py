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
from metrics import mean_forman_curvature, spectral_dimension, summarize_causal_graph


def main() -> None:
    hypergraph = Hypergraph([(0, 1, 2)])
    rule = EdgeSplit3Rule()
    engine = RewriteEngine(hypergraph, rule, seed=1)
    engine.run(steps=200)

    skeleton = hypergraph.one_skeleton()
    ds = spectral_dimension(skeleton, max_time=6, trials=400, seed=2)
    curvature = mean_forman_curvature(skeleton)
    causal_stats = summarize_causal_graph(engine.causal_graph)

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


if __name__ == "__main__":
    main()
