"""Screen focused hybrid-policy changes for late-game conversion."""

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
    endgame_road_push: bool = False
    knight_pressure: bool = False
    leader_robber_weight: float = 0.0
    blocking_settlement_weight: float = 0.0
    trade_refinement: bool = False
    resource_tactics: bool = False
    end_turn_trade_sweep: bool = False
    evolved_state_refinement: bool = False
    state_refinement_mix: float = 0.0


CONFIGS = (
    Config("baseline"),
    Config("road_push", endgame_road_push=True),
    Config("knight", knight_pressure=True),
    Config("leader_robber", leader_robber_weight=1.0),
    Config("blocking_5", blocking_settlement_weight=5.0),
    Config("blocking_20", blocking_settlement_weight=20.0),
    Config("resource", resource_tactics=True),
    Config("trade", trade_refinement=True),
    Config("trade_sweep", end_turn_trade_sweep=True),
    Config("state_005", state_refinement_mix=0.05),
    Config("state_010", state_refinement_mix=0.1),
    Config(
        "evolved_state_005",
        evolved_state_refinement=True,
        state_refinement_mix=0.05,
    ),
    Config(
        "road_push_knight",
        endgame_road_push=True,
        knight_pressure=True,
    ),
    Config(
        "road_push_blocking",
        endgame_road_push=True,
        blocking_settlement_weight=5.0,
    ),
    Config(
        "road_push_resource",
        endgame_road_push=True,
        resource_tactics=True,
    ),
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
        "--blocking-settlement-weight",
        str(config.blocking_settlement_weight),
        "--leader-robber-weight",
        str(config.leader_robber_weight),
        "--state-refinement-mix",
        str(config.state_refinement_mix),
        "--games",
        str(games),
        "--seed",
        str(seed),
    ]
    flags = (
        ("--endgame-road-push", config.endgame_road_push),
        ("--knight-pressure", config.knight_pressure),
        ("--trade-refinement", config.trade_refinement),
        ("--resource-tactics", config.resource_tactics),
        ("--end-turn-trade-sweep", config.end_turn_trade_sweep),
        ("--evolved-state-refinement", config.evolved_state_refinement),
    )
    command.extend(flag for flag, enabled in flags if enabled)
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
        default=[36_700_000, 36_800_000, 36_900_000],
    )
    parser.add_argument("--workers", type=int, default=16)
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
