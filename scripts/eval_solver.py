"""Measure the symbolic pipeline on synthetic NCERT-style problems.

Runs ``MathService.solve_latex`` (parse -> classify -> SymPy solve -> steps)
on problems from ``scripts/generate_problems.py`` and reports, per topic, how
many were solved and which problem types were predicted. No network, no OCR
model and no GPU are needed, so it is cheap enough to run in CI.

Usage:
    python scripts/eval_solver.py                       # 500 problems, seed 7
    python scripts/eval_solver.py --count 1000 --min-solve-rate 0.85
    python scripts/eval_solver.py --json metrics/solver_metrics.json

Topics that are English word problems (distance between points, probability) are not solvable offline: they need the Gemini fallback, which is
not available in CI. They are reported separately and excluded from the
symbolic solve rate that ``--min-solve-rate`` is checked against.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Generated as English sentences inside \text{...}; need an LLM, not SymPy.
WORD_PROBLEM_TOPICS = {"Coordinate Geometry", "Probability"}


def evaluate(count: int, seed: int) -> dict:
    """Run the pipeline on ``count`` generated problems.

    Args:
        count: Number of problems (spread evenly over the generators).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with ``overall`` solve rate, per-topic stats and timing.
    """
    from scripts.generate_problems import GENERATORS
    from services.math_service import MathService

    logging.disable(logging.CRITICAL)
    rng = random.Random(seed)
    service = MathService()
    per_topic: dict[str, dict] = collections.defaultdict(lambda: {"n": 0, "solved": 0, "types": collections.Counter()})
    start = time.time()
    for i in range(count):
        problem = GENERATORS[i % len(GENERATORS)](rng)
        stats = per_topic[problem.topic]
        stats["n"] += 1
        try:
            result = service.solve_latex(problem.latex_problem)
        except Exception:  # noqa: BLE001 - any failure counts as unsolved
            continue
        stats["solved"] += 1
        stats["types"][result["problem_type"]] += 1

    symbolic = {t: s for t, s in per_topic.items() if t not in WORD_PROBLEM_TOPICS}
    sym_n = sum(s["n"] for s in symbolic.values())
    sym_ok = sum(s["solved"] for s in symbolic.values())
    return {
        "problems": sum(s["n"] for s in per_topic.values()),
        "symbolic_problems": sym_n,
        "symbolic_solved": sym_ok,
        "symbolic_solve_rate": sym_ok / sym_n if sym_n else 0.0,
        "seconds": round(time.time() - start, 1),
        "topics": {t: {"n": s["n"], "solve_rate": s["solved"] / s["n"], "types": dict(s["types"]),
                       "needs_llm": t in WORD_PROBLEM_TOPICS}
                   for t, s in sorted(per_topic.items())},
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the MathVision AI solver on synthetic problems.")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-solve-rate", type=float, default=0.95,
                        help="minimum SYMBOLIC solve rate (word-problem topics excluded)")
    parser.add_argument("--json", type=Path, help="optional path to write the metrics as JSON")
    args = parser.parse_args()

    report = evaluate(args.count, args.seed)
    print(f"{'topic':28}{'n':>5}{'solved':>9}  predicted types")
    for topic, s in report["topics"].items():
        note = "  (word problem: needs Gemini, not solvable offline)" if s["needs_llm"] else ""
        print(f"{topic:28}{s['n']:5}{s['solve_rate']:9.1%}  {s['types']}{note}")
    print(f"\nSYMBOLIC: {report['symbolic_solved']}/{report['symbolic_problems']} solved = "
          f"{report['symbolic_solve_rate']:.1%}  ({report['seconds']}s)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["symbolic_solve_rate"] < args.min_solve_rate:
        sys.exit(f"Symbolic solve rate {report['symbolic_solve_rate']:.1%} is below "
                 f"the {args.min_solve_rate:.0%} threshold")


if __name__ == "__main__":
    main()
