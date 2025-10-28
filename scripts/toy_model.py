"""Command-line entry point for the toy Wolfram-Weinstein bridge model."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.toy import run_toy_unification_model


def main() -> None:
    """Execute the toy model and print a JSON digest of the results."""

    result = run_toy_unification_model()
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

