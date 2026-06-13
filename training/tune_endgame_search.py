"""Tune opponent-aware rollout search restricted to late-game positions."""

from __future__ import annotations

import argparse
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
    name: str
    root_k: int
    samples: int
    continuation: int
    min_vp: int
    value_weight: float
    potential_weight: float
    to_terminal: bool = False
    max_decisions: int = 64


CONFIGS = (
    Config("greedy", 1, 1, 0, 0, 1.0, 0.0),
    Config("vp4_terminal_2x4", 2, 4, 0, 4, 1.0, 0.0, True, 48),
    Config("vp4_terminal_4x4", 4, 4, 0, 4, 1.0, 0.0, True, 48),
    Config("vp4_terminal_4x8", 4, 8, 0, 4, 1.0, 0.0, True, 48),
    Config("vp5_terminal_2x4", 2, 4, 0, 5, 1.0, 0.0, True, 48),
    Config("vp5_terminal_4x4", 4, 4, 0, 5, 1.0, 0.0, True, 48),
    Config("vp5_terminal_4x8", 4, 8, 0, 5, 1.0, 0.0, True, 48),
    Config("vp5_terminal_4x16", 4, 16, 0, 5, 1.0, 0.0, True, 48),
    Config("vp5_terminal_8x8", 8, 8, 0, 5, 1.0, 0.0, True, 48),
    Config("vp6_terminal_2x4", 2, 4, 0, 6, 1.0, 0.0, True, 48),
    Config("vp6_terminal_4x4", 4, 4, 0, 6, 1.0, 0.0, True, 48),
    Config("vp6_terminal_4x8", 4, 8, 0, 6, 1.0, 0.0, True, 48),
    Config("vp6_terminal_4x16", 4, 16, 0, 6, 1.0, 0.0, True, 48),
    Config("vp6_terminal_8x8", 8, 8, 0, 6, 1.0, 0.0, True, 48),
)


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
        "0",
        "--leader-robber-weight",
        "1",
        "--hybrid-search-root-k",
        str(config.root_k),
        "--hybrid-search-samples",
        str(config.samples),
        "--hybrid-search-continuation",
        str(config.continuation),
        "--hybrid-search-value-weight",
        str(config.value_weight),
        "--hybrid-search-potential-weight",
        str(config.potential_weight),
        "--hybrid-search-min-vp",
        str(config.min_vp),
        "--hybrid-search-max-decisions",
        str(config.max_decisions),
        "--games",
        str(games),
        "--seed",
        str(seed),
    ]
    if config.to_terminal:
        command.append("--hybrid-search-to-terminal")
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
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[37_000_000, 37_100_000, 37_200_000],
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    opponents = ("heuristic", "heuristic_v2")
    wins = {
        config.name: {opponent: [] for opponent in opponents}
        for config in CONFIGS
    }
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for config in CONFIGS:
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
                    jobs[job] = (config.name, opponent)
        for job in as_completed(jobs):
            name, opponent = jobs[job]
            wins[name][opponent].append(job.result())

    denominator = args.games * len(args.seeds)
    results = []
    for config in CONFIGS:
        rates = {
            opponent: sum(wins[config.name][opponent]) / denominator
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
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
