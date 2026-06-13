"""Multi-seed random search for the deployable hybrid inference policy."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
EVALUATOR = ROOT / "training/evaluate_planner_vs_alpha.py"


@dataclass(frozen=True)
class Config:
    strategy_settlement_weight: float
    opening_production_weight: float
    road_length_weight: float
    road_settlement_weight: float
    immediate_vp_min: int
    conversion_min_vp: int
    proposal_conversion_min_vp: int


BASELINE = Config(5.0, 1.0, 5.0, 20.0, 0, 5, 5)


def sample_configs(count: int, seed: int) -> list[Config]:
    rng = random.Random(seed)
    configs = {BASELINE}
    while len(configs) < count:
        configs.add(
            Config(
                strategy_settlement_weight=rng.choice((0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)),
                opening_production_weight=rng.choice((0.0, 0.25, 0.5, 1.0, 2.0, 4.0)),
                road_length_weight=rng.choice((0.5, 1.0, 2.0, 5.0, 10.0, 20.0)),
                road_settlement_weight=rng.choice((2.0, 5.0, 10.0, 20.0, 40.0)),
                immediate_vp_min=rng.choice((0, 2, 4, 5)),
                conversion_min_vp=rng.choice((4, 5, 6)),
                proposal_conversion_min_vp=rng.choice((4, 5, 6)),
            )
        )
    return [BASELINE, *sorted(configs - {BASELINE}, key=repr)]


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
        str(config.strategy_settlement_weight),
        "--opening-production-weight",
        str(config.opening_production_weight),
        "--opening-settlement-lookahead",
        "--heuristic-refinement",
        "--endgame-conversion",
        "--road-refinement",
        "--road-length-weight",
        str(config.road_length_weight),
        "--road-settlement-weight",
        str(config.road_settlement_weight),
        "--immediate-vp-min",
        str(config.immediate_vp_min),
        "--conversion-min-vp",
        str(config.conversion_min_vp),
        "--proposal-conversion-min-vp",
        str(config.proposal_conversion_min_vp),
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
    parser.add_argument("net", type=Path)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[27_500_000, 27_700_000, 27_900_000])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configs = sample_configs(args.candidates, args.seed)
    opponents = ("heuristic", "heuristic_v2")
    wins = {
        index: {opponent: [] for opponent in opponents}
        for index in range(len(configs))
    }
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, config in enumerate(configs):
            for opponent in opponents:
                for eval_seed in args.seeds:
                    future = executor.submit(
                        evaluate_one,
                        args.net,
                        config,
                        opponent,
                        args.games,
                        eval_seed,
                    )
                    jobs[future] = (index, opponent)
        for future in as_completed(jobs):
            index, opponent = jobs[future]
            wins[index][opponent].append(future.result())

    results = []
    denominator = args.games * len(args.seeds)
    for index, config in enumerate(configs):
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
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    main()
