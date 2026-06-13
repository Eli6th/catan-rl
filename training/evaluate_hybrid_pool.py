"""Evaluate a pool of CTNN checkpoints with the deployable hybrid policy."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
EVALUATOR = ROOT / "training/evaluate_planner_vs_alpha.py"


def evaluate_one(net: Path, opponent: str, games: int, seed: int) -> int:
    command = [
        str(PYTHON),
        str(EVALUATOR),
        str(net),
        "--opponent",
        opponent,
        "--planner",
        "hybrid-v2",
        "--strategy-settlement-weight",
        "5",
        "--opening-production-weight",
        "1",
        "--opening-settlement-lookahead",
        "--heuristic-refinement",
        "--endgame-conversion",
        "--road-refinement",
        "--immediate-vp-min",
        "0",
        "--leader-robber-weight",
        "1",
        "--games",
        str(games),
        "--seed",
        str(seed),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(json.loads(result.stdout)["wins"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("nets", type=Path, nargs="+")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[29_900_000, 30_000_000, 30_100_000],
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    opponents = ("heuristic", "heuristic_v2")
    wins = {
        str(net): {opponent: [] for opponent in opponents}
        for net in args.nets
    }
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for net in args.nets:
            for opponent in opponents:
                for seed in args.seeds:
                    job = executor.submit(
                        evaluate_one,
                        net,
                        opponent,
                        args.games,
                        seed,
                    )
                    jobs[job] = (str(net), opponent)
        for job in as_completed(jobs):
            net, opponent = jobs[job]
            wins[net][opponent].append(job.result())

    denominator = args.games * len(args.seeds)
    results = []
    for net in args.nets:
        rates = {
            opponent: sum(wins[str(net)][opponent]) / denominator
            for opponent in opponents
        }
        results.append(
            {
                "net": str(net),
                "rates": rates,
                "worst_rate": min(rates.values()),
                "mean_rate": sum(rates.values()) / len(rates),
            }
        )
    results.sort(
        key=lambda result: (result["worst_rate"], result["mean_rate"]),
        reverse=True,
    )
    payload = {
        "games_per_opponent": denominator,
        "seeds": args.seeds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
