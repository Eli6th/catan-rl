"""Paired multi-seed search over high-impact hybrid tactical switches."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
EVALUATOR = ROOT / "training/evaluate_planner_vs_alpha.py"


@dataclass(frozen=True)
class Config:
    immediate_vp_min: int
    conversion_min_vp: int
    knight_pressure: bool
    leader_robber_weight: float


def configs() -> list[Config]:
    return [
        Config(*values)
        for values in itertools.product(
            (0, 2, 4, 5),
            (3, 4, 5),
            (False, True),
            (0.0, 1.0, 3.0),
        )
    ]


def evaluate_one(
    net: Path,
    config: Config,
    opponent: str,
    games: int,
    seed: int,
) -> int:
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
        str(config.immediate_vp_min),
        "--conversion-min-vp",
        str(config.conversion_min_vp),
        "--proposal-conversion-min-vp",
        "5",
        "--leader-robber-weight",
        str(config.leader_robber_weight),
        "--games",
        str(games),
        "--seed",
        str(seed),
    ]
    if config.knight_pressure:
        command.append("--knight-pressure")
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
    parser.add_argument("net", type=Path)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[29_400_000, 29_500_000, 29_600_000],
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = configs()
    opponents = ("heuristic", "heuristic_v2")
    wins = {
        index: {opponent: [] for opponent in opponents}
        for index in range(len(candidates))
    }
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, config in enumerate(candidates):
            for opponent in opponents:
                for seed in args.seeds:
                    job = executor.submit(
                        evaluate_one,
                        args.net,
                        config,
                        opponent,
                        args.games,
                        seed,
                    )
                    jobs[job] = (index, opponent)
        for job in as_completed(jobs):
            index, opponent = jobs[job]
            wins[index][opponent].append(job.result())

    denominator = args.games * len(args.seeds)
    results = []
    for index, config in enumerate(candidates):
        rates = {
            opponent: sum(wins[index][opponent]) / denominator
            for opponent in opponents
        }
        results.append(
            {
                "config": asdict(config),
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
        "net": str(args.net),
        "games_per_opponent": denominator,
        "seeds": args.seeds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(results[:12], indent=2))


if __name__ == "__main__":
    main()
