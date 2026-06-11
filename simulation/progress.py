"""
Progress tracking for Catan simulations.

Uses Rich library for beautiful terminal output with:
- Progress bars
- Live statistics
- Win rate tracking
- Memory usage
"""

# Standard Library Imports
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

# Third Party Imports
import psutil
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""

    games_completed: int = 0
    games_total: int = 0
    start_time: float = field(default_factory=time.time)
    wins_by_strategy: Dict[str, int] = field(default_factory=dict)
    total_turns: int = 0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def games_per_second(self) -> float:
        if self.elapsed_time == 0:
            return 0
        return self.games_completed / self.elapsed_time

    @property
    def eta_seconds(self) -> float:
        if self.games_per_second == 0:
            return float("inf")
        remaining = self.games_total - self.games_completed
        return remaining / self.games_per_second

    @property
    def average_turns(self) -> float:
        if self.games_completed == 0:
            return 0
        return self.total_turns / self.games_completed

    def get_win_rates(self) -> Dict[str, float]:
        if self.games_completed == 0:
            return {}
        return {
            name: wins / self.games_completed * 100
            for name, wins in self.wins_by_strategy.items()
        }


class ProgressTracker:
    """
    Rich terminal progress display for simulations.

    Shows:
    - Progress bar
    - Games per second
    - ETA
    - Memory usage
    - Win rates by strategy
    """

    def __init__(
        self, total_games: int, player_names: List[str], verbosity: str = "progress"
    ):
        """
        Initialize progress tracker.

        Args:
            total_games: Total number of games to simulate
            player_names: Names of player strategies
            verbosity: "silent", "progress", or "verbose"
        """
        self.console = Console()
        self.metrics = SimulationMetrics(games_total=total_games)
        self.player_names = player_names
        self.verbosity = verbosity

        # Initialize win counters
        for name in player_names:
            self.metrics.wins_by_strategy[name] = 0

        self._progress: Optional[Progress] = None
        self._task_id = None
        self._live: Optional[Live] = None

    def start(self):
        """Start the progress display."""
        if self.verbosity == "silent":
            return

        self.console.print(
            Panel.fit("[bold cyan]Catan Simulation[/bold cyan]", border_style="cyan")
        )
        self.console.print(
            f"Running [bold]{self.metrics.games_total:,}[/bold] games "
            f"with [bold]{len(self.player_names)}[/bold] players each\n"
        )

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

        self._progress.start()
        self._task_id = self._progress.add_task(
            "Simulating", total=self.metrics.games_total
        )

    def update(
        self,
        games_completed: int = 1,
        winner_name: Optional[str] = None,
        turns: int = 0,
    ):
        """
        Update progress after completing games.

        Args:
            games_completed: Number of games completed since last update
            winner_name: Name of winning strategy (if applicable)
            turns: Number of turns in the game
        """
        self.metrics.games_completed += games_completed
        self.metrics.total_turns += turns

        if winner_name and winner_name in self.metrics.wins_by_strategy:
            self.metrics.wins_by_strategy[winner_name] += 1

        if self.verbosity != "silent" and self._progress:
            self._progress.update(self._task_id, advance=games_completed)

    def stop(self):
        """Stop the progress display and show final stats."""
        if self.verbosity == "silent":
            return

        if self._progress:
            self._progress.stop()

        self._print_final_stats()

    def _print_final_stats(self):
        """Print final simulation statistics."""
        self.console.print()

        # Summary table
        table = Table(title="Simulation Results", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total Games", f"{self.metrics.games_completed:,}")
        table.add_row("Total Time", f"{self.metrics.elapsed_time:.2f}s")
        table.add_row("Games/Second", f"{self.metrics.games_per_second:,.1f}")
        table.add_row("Avg Turns/Game", f"{self.metrics.average_turns:.1f}")

        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        table.add_row("Memory Used", f"{memory_mb:.1f} MB")

        self.console.print(table)

        # Win rates table
        self.console.print()
        win_table = Table(title="Win Rates by Strategy", border_style="green")
        win_table.add_column("Strategy", style="bold")
        win_table.add_column("Wins", justify="right")
        win_table.add_column("Win Rate", justify="right")

        win_rates = self.metrics.get_win_rates()
        for name, rate in sorted(win_rates.items(), key=lambda x: -x[1]):
            wins = self.metrics.wins_by_strategy[name]
            win_table.add_row(name, f"{wins:,}", f"{rate:.2f}%")

        self.console.print(win_table)

    def get_metrics(self) -> SimulationMetrics:
        """Get current metrics."""
        return self.metrics


class QuietProgressTracker:
    """
    Minimal progress tracker that just updates periodically.

    For use when Rich is not available or when running in non-interactive mode.
    """

    def __init__(self, total_games: int, update_interval: int = 10000):
        self.total_games = total_games
        self.update_interval = update_interval
        self.games_completed = 0
        self.start_time = time.time()
        self.last_print = 0

    def start(self):
        print(f"Starting simulation of {self.total_games:,} games...")

    def update(self, games_completed: int = 1, **kwargs):
        self.games_completed += games_completed

        if self.games_completed - self.last_print >= self.update_interval:
            elapsed = time.time() - self.start_time
            rate = self.games_completed / elapsed if elapsed > 0 else 0
            pct = self.games_completed / self.total_games * 100
            print(
                f"Progress: {self.games_completed:,}/{self.total_games:,} "
                f"({pct:.1f}%) - {rate:,.0f} games/sec"
            )
            self.last_print = self.games_completed

    def stop(self):
        elapsed = time.time() - self.start_time
        rate = self.games_completed / elapsed if elapsed > 0 else 0
        print(
            f"\nCompleted {self.games_completed:,} games in {elapsed:.2f}s "
            f"({rate:,.0f} games/sec)"
        )
