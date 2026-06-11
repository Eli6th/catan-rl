"""
Statistics collection for Catan simulations.

Collects and analyzes data from simulation runs:
- Win rates by strategy
- Game length distribution
- Resource acquisition patterns
- Building patterns
"""

# Standard Library Imports
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

# Third Party Imports
import numpy as np


@dataclass
class GameResult:
    """Result of a single game."""

    seed: int
    winner: int
    winner_strategy: str
    turns: int
    player_strategies: List[str]
    final_vps: List[int]


@dataclass
class SimulationStats:
    """
    Statistics collector for simulation runs.

    Tracks:
    - Win counts and rates by strategy
    - Game length distribution
    - Victory point distribution
    """

    total_games: int = 0
    wins_by_strategy: Dict[str, int] = field(default_factory=dict)
    games_by_length: Dict[int, int] = field(default_factory=dict)
    total_turns: int = 0
    results: List[GameResult] = field(default_factory=list)

    def record_game(self, result: GameResult):
        """Record a game result."""
        self.total_games += 1
        self.total_turns += result.turns

        # Win count
        strategy = result.winner_strategy
        if strategy not in self.wins_by_strategy:
            self.wins_by_strategy[strategy] = 0
        self.wins_by_strategy[strategy] += 1

        # Game length bucket (round to nearest 10)
        bucket = (result.turns // 10) * 10
        if bucket not in self.games_by_length:
            self.games_by_length[bucket] = 0
        self.games_by_length[bucket] += 1

        # Store result
        self.results.append(result)

    def get_win_rate(self, strategy: str) -> float:
        """Get win rate for a strategy."""
        if self.total_games == 0:
            return 0.0
        wins = self.wins_by_strategy.get(strategy, 0)
        return wins / self.total_games * 100

    def get_all_win_rates(self) -> Dict[str, float]:
        """Get win rates for all strategies."""
        return {
            strategy: self.get_win_rate(strategy) for strategy in self.wins_by_strategy
        }

    @property
    def average_game_length(self) -> float:
        """Get average game length in turns."""
        if self.total_games == 0:
            return 0.0
        return self.total_turns / self.total_games

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total_games": self.total_games,
            "average_turns": self.average_game_length,
            "win_rates": self.get_all_win_rates(),
            "games_by_length": dict(sorted(self.games_by_length.items())),
        }

    def print_summary(self):
        """Print a summary to console."""
        print(f"\n{'='*50}")
        print(f"Simulation Summary")
        print(f"{'='*50}")
        print(f"Total Games: {self.total_games:,}")
        print(f"Average Game Length: {self.average_game_length:.1f} turns")

        print(f"\nWin Rates:")
        for strategy, rate in sorted(
            self.get_all_win_rates().items(), key=lambda x: -x[1]
        ):
            wins = self.wins_by_strategy[strategy]
            print(f"  {strategy}: {rate:.2f}% ({wins:,} wins)")

        print(f"\nGame Length Distribution:")
        for bucket in sorted(self.games_by_length.keys()):
            count = self.games_by_length[bucket]
            pct = count / self.total_games * 100
            bar = "#" * int(pct / 2)
            print(f"  {bucket:3d}-{bucket+9:3d} turns: {count:6,} ({pct:5.1f}%) {bar}")

    def export_csv(self, filepath: Union[str, Path]):
        """Export results to CSV."""
        filepath = Path(filepath)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "seed",
                    "winner",
                    "winner_strategy",
                    "turns",
                    "player_strategies",
                    "final_vps",
                ]
            )

            for result in self.results:
                writer.writerow(
                    [
                        result.seed,
                        result.winner,
                        result.winner_strategy,
                        result.turns,
                        ",".join(result.player_strategies),
                        ",".join(map(str, result.final_vps)),
                    ]
                )

    def export_json(self, filepath: Union[str, Path]):
        """Export summary to JSON."""
        filepath = Path(filepath)

        with open(filepath, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

    @classmethod
    def load_csv(cls, filepath: Union[str, Path]) -> "SimulationStats":
        """Load stats from CSV."""
        stats = cls()

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result = GameResult(
                    seed=int(row["seed"]),
                    winner=int(row["winner"]),
                    winner_strategy=row["winner_strategy"],
                    turns=int(row["turns"]),
                    player_strategies=row["player_strategies"].split(","),
                    final_vps=list(map(int, row["final_vps"].split(","))),
                )
                stats.record_game(result)

        return stats

    def merge(self, other: "SimulationStats"):
        """Merge another stats object into this one."""
        self.total_games += other.total_games
        self.total_turns += other.total_turns

        for strategy, wins in other.wins_by_strategy.items():
            if strategy not in self.wins_by_strategy:
                self.wins_by_strategy[strategy] = 0
            self.wins_by_strategy[strategy] += wins

        for bucket, count in other.games_by_length.items():
            if bucket not in self.games_by_length:
                self.games_by_length[bucket] = 0
            self.games_by_length[bucket] += count

        self.results.extend(other.results)
