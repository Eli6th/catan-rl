#!/usr/bin/env python3
"""
Main entry point for running Catan simulations.

Examples:
    # Run 1000 games with default settings
    python run_simulation.py

    # Run 10000 games with specific seed
    python run_simulation.py --games 10000 --seed 42

    # Compare strategies
    python run_simulation.py --compare

    # Run with logging enabled
    python run_simulation.py --games 100 --log

    # Run with performance profiling
    python run_simulation.py --games 1000 --profile
"""

import argparse
import sys
from typing import List, Type

from players.base import Player
from players.strategies import RandomPlayer, HeuristicPlayer
from simulation.runner import (
    SimulationRunner,
    SimulationConfig,
    run_quick_simulation,
    compare_strategies,
)


def parse_player_config(config_str: str) -> List[Type[Player]]:
    """
    Parse player configuration string.

    Format: "R,R,H,H" for Random, Random, Heuristic, Heuristic
    """
    mapping = {
        "R": RandomPlayer,
        "H": HeuristicPlayer,
        "RANDOM": RandomPlayer,
        "HEURISTIC": HeuristicPlayer,
    }

    parts = config_str.upper().split(",")
    players = []

    for part in parts:
        part = part.strip()
        if part in mapping:
            players.append(mapping[part])
        else:
            raise ValueError(f"Unknown player type: {part}")

    return players


def main():
    parser = argparse.ArgumentParser(
        description="Run Catan game simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py --games 10000
  python run_simulation.py --players R,R,H,H --games 5000
  python run_simulation.py --compare
  python run_simulation.py --games 100 --log --verbose
  python run_simulation.py --games 1000 --profile
  python run_simulation.py --games 100 --profile --profile-detailed
        """,
    )

    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=1000,
        help="Number of games to simulate (default: 1000)",
    )
    parser.add_argument(
        "--players",
        "-p",
        type=str,
        default="R,R,R,R",
        help="Player types: R=Random, H=Heuristic (default: R,R,R,R)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log", action="store_true", default=False, help="Enable game logging"
    )
    parser.add_argument(
        "--log-rate",
        type=float,
        default=0.01,
        help="Fraction of games to log (default: 0.01)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--silent", "-q", action="store_true", help="Silent mode (no progress display)"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare Random vs Heuristic strategies"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file for results (CSV)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling (memory, CPU, timing)",
    )
    parser.add_argument(
        "--profile-detailed",
        action="store_true",
        help="Store detailed per-game profiling results (more memory overhead)",
    )

    args = parser.parse_args()

    if args.compare:
        print("Comparing Random vs Heuristic strategies...")
        print("=" * 50)

        random_rate, heuristic_rate = compare_strategies(
            RandomPlayer,
            HeuristicPlayer,
            num_games=args.games,
            seed=args.seed,
        )

        print(f"\nResults:")
        print(f"  RandomPlayer win rate:    {random_rate:.2f}%")
        print(f"  HeuristicPlayer win rate: {heuristic_rate:.2f}%")
        return

    try:
        player_types = parse_player_config(args.players)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.silent:
        verbosity = "silent"
    elif args.verbose:
        verbosity = "verbose"
    else:
        verbosity = "progress"

    config = SimulationConfig(
        num_players=len(player_types),
        player_types=player_types,
        num_games=args.games,
        base_seed=args.seed,
        log_games=args.log,
        log_sample_rate=args.log_rate,
        verbosity=verbosity,
        enable_profiling=args.profile,
        profile_memory=args.profile,
        profile_detailed=args.profile_detailed,
    )

    runner = SimulationRunner(config)
    stats = runner.run()

    if not args.silent:
        stats.print_summary()

    if args.profile:
        profile = runner.get_profile()
        if profile:
            profile.print_summary()

    if args.output:
        if args.output.endswith(".csv"):
            stats.export_csv(args.output)
            print(f"\nResults exported to {args.output}")
        elif args.output.endswith(".json"):
            stats.export_json(args.output)
            print(f"\nSummary exported to {args.output}")


if __name__ == "__main__":
    main()
