"""
Simulation runner for mass Catan game execution.

Runs many games with progress tracking, logging, and statistics collection.
"""

# Standard Library Imports
import time
from dataclasses import dataclass, field
from typing import List, Optional, Type, Tuple, Callable, Union
from pathlib import Path

# Third Party Imports
import numpy as np

# Local Imports
from engine.game import CatanGame
from players.base import Player
from players.strategies import RandomPlayer
from .stats import SimulationStats, GameResult
from .logger import GameLogger
from .progress import ProgressTracker, QuietProgressTracker
from .profiler import Profiler, NoOpProfiler, ProfileResult


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""

    # Game settings
    num_players: int = 4

    # Player configuration (list of player class types or instances)
    player_types: List[Type[Player]] = field(default_factory=lambda: [RandomPlayer] * 4)

    # Run settings
    num_games: int = 1000
    base_seed: Optional[int] = None

    # Logging
    log_games: bool = False
    log_dir: str = "logs"
    log_sample_rate: float = 0.001  # Log 0.1% of games by default

    # Progress display
    verbosity: str = "progress"  # "silent", "progress", "verbose"

    # Performance
    max_turns_per_game: int = 1000  # Prevent infinite games

    # Profiling
    enable_profiling: bool = False  # Enable performance profiling
    profile_memory: bool = True  # Track memory usage (requires enable_profiling)
    profile_detailed: bool = False  # Store per-game detailed profile results


class SimulationRunner:
    """
    Runs mass Catan simulations.

    Features:
    - Configurable player strategies
    - Progress tracking with Rich UI
    - Optional game logging
    - Statistics collection
    - Reproducible via seeds
    - Optional performance profiling
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation runner.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.stats = SimulationStats()
        self.logger = GameLogger(config.log_dir) if config.log_games else None

        # Profiler setup
        if config.enable_profiling:
            self.profiler: Union[Profiler, NoOpProfiler] = Profiler(
                track_memory=config.profile_memory,
                detailed=config.profile_detailed,
            )
        else:
            self.profiler = NoOpProfiler()

        # Seed management
        if config.base_seed is not None:
            self.rng = np.random.default_rng(config.base_seed)
        else:
            self.rng = np.random.default_rng()

    def run(
        self, callback: Optional[Callable[[int, GameResult], None]] = None
    ) -> SimulationStats:
        """
        Run the simulation.

        Args:
            callback: Optional callback after each game (game_idx, result)

        Returns:
            SimulationStats with all collected statistics
        """
        config = self.config

        # Start profiler
        self.profiler.start()

        # Create player instances
        with self.profiler.phase("player_creation"):
            players = self._create_players()
        player_names = [p.name for p in players]

        # Create progress tracker
        if config.verbosity == "silent":
            progress = None
        else:
            try:
                progress = ProgressTracker(
                    config.num_games, player_names, config.verbosity
                )
            except ImportError:
                progress = QuietProgressTracker(config.num_games)

        if progress:
            progress.start()

        try:
            for game_idx in range(config.num_games):
                self.profiler.start_game(game_idx)

                result = self._run_single_game(players, game_idx)
                self.stats.record_game(result)

                self.profiler.end_game()

                # Sample memory periodically (every 100 games)
                if game_idx % 100 == 0:
                    self.profiler.sample_memory()

                if progress:
                    progress.update(
                        games_completed=1,
                        winner_name=result.winner_strategy,
                        turns=result.turns,
                    )

                if callback:
                    callback(game_idx, result)

        finally:
            if progress:
                progress.stop()

            # Stop profiler
            self.profiler.stop()

        return self.stats

    def _create_players(self) -> List[Player]:
        """Create player instances from config."""
        players = []

        for i, player_type in enumerate(self.config.player_types):
            if isinstance(player_type, Player):
                # Already an instance
                player = player_type
            else:
                # Create instance from type
                player = player_type(name=f"{player_type.__name__}_{i}")

            players.append(player)

        return players

    def _run_single_game(self, players: List[Player], game_idx: int) -> GameResult:
        """Run a single game and return the result."""
        config = self.config

        # Generate seed for this game
        seed = int(self.rng.integers(0, 2**31))

        # Create game
        game = CatanGame(config.num_players, seed)

        # Reset player RNGs deterministically from the game seed so that
        # simulations with the same base_seed are fully reproducible.
        player_rng = np.random.default_rng(seed)
        for player in players:
            if hasattr(player, "rng"):
                player.rng = np.random.default_rng(int(player_rng.integers(0, 2**31)))

        # set_players calls on_game_start for each player
        game.set_players(players)

        # Start logging if enabled
        should_log = (
            self.logger is not None and self.rng.random() < config.log_sample_rate
        )
        if should_log:
            self.logger.start_game(game.state)

        # Play game
        turns = 0
        while not game.is_game_over() and turns < config.max_turns_per_game:
            player_idx = game.get_current_player()
            player = players[player_idx]

            with self.profiler.phase("get_valid_actions"):
                valid_actions = game.get_valid_actions()
            if not valid_actions:
                break

            with self.profiler.phase("player_choose_action"):
                action = player.choose_action(game.state, valid_actions)

            with self.profiler.phase("execute_action"):
                game.execute_action(action)

            if should_log:
                self.logger.log_action(action)

            # Count turns (only increment on turn end)
            if game.state.turn > turns:
                turns = game.state.turn

        # End logging
        if should_log:
            self.logger.end_game(game.state)

        # Notify players of game end
        with self.profiler.phase("player_notifications"):
            winner = game.get_winner()
            for player in players:
                player.on_game_end(game.state, winner)

        # Create result
        with self.profiler.phase("result_creation"):
            winner_strategy = players[winner].name if winner >= 0 else "None"
            final_vps = [
                game.state.calculate_victory_points(p)
                for p in range(config.num_players)
            ]

        return GameResult(
            seed=seed,
            winner=winner,
            winner_strategy=winner_strategy,
            turns=turns,
            player_strategies=[p.name for p in players],
            final_vps=final_vps,
        )

    def get_stats(self) -> SimulationStats:
        """Get collected statistics."""
        return self.stats

    def get_profile(self) -> Optional[ProfileResult]:
        """Get profiling results (if profiling was enabled)."""
        return self.profiler.result


def run_quick_simulation(
    num_games: int = 1000,
    num_players: int = 4,
    player_types: Optional[List[Type[Player]]] = None,
    seed: Optional[int] = None,
    verbosity: str = "progress",
) -> SimulationStats:
    """
    Convenience function for quick simulations.

    Args:
        num_games: Number of games to run
        num_players: Number of players per game
        player_types: List of player types (defaults to RandomPlayer)
        seed: Random seed for reproducibility
        verbosity: "silent", "progress", or "verbose"

    Returns:
        SimulationStats with results
    """
    if player_types is None:
        player_types = [RandomPlayer] * num_players

    config = SimulationConfig(
        num_players=num_players,
        player_types=player_types,
        num_games=num_games,
        base_seed=seed,
        verbosity=verbosity,
    )

    runner = SimulationRunner(config)
    return runner.run()


def compare_strategies(
    strategy_a: Type[Player],
    strategy_b: Type[Player],
    num_games: int = 1000,
    num_players: int = 4,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compare two strategies head-to-head.

    Runs games with mixed player configurations and returns win rates.

    Args:
        strategy_a: First strategy to test
        strategy_b: Second strategy to test
        num_games: Number of games to run
        num_players: Number of players per game
        seed: Random seed

    Returns:
        Tuple of (strategy_a_win_rate, strategy_b_win_rate)
    """
    # Mix strategies evenly
    player_types = []
    for i in range(num_players):
        if i % 2 == 0:
            player_types.append(strategy_a)
        else:
            player_types.append(strategy_b)

    stats = run_quick_simulation(
        num_games=num_games,
        num_players=num_players,
        player_types=player_types,
        seed=seed,
        verbosity="progress",
    )

    # Calculate win rates
    a_name = strategy_a.__name__
    b_name = strategy_b.__name__

    a_wins = sum(1 for r in stats.results if a_name in r.winner_strategy)
    b_wins = sum(1 for r in stats.results if b_name in r.winner_strategy)

    total = len(stats.results)
    return (a_wins / total * 100, b_wins / total * 100)
