"""
Performance profiler for Catan simulations.

Tracks:
- Memory usage (peak and average)
- CPU time
- Wall-clock time for different game phases
"""

# Standard Library Imports
import contextlib
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Third Party Imports

# Local Imports


@dataclass
class TimingStats:
    """Statistics for a timed operation."""

    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0

    def record(self, elapsed: float):
        """Record a timing measurement."""
        self.total_time += elapsed
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)

    @property
    def avg_time(self) -> float:
        """Average time per call."""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_time_ms": self.total_time * 1000,
            "call_count": self.call_count,
            "avg_time_ms": self.avg_time * 1000,
            "min_time_ms": self.min_time * 1000 if self.call_count > 0 else 0,
            "max_time_ms": self.max_time * 1000,
        }


@dataclass
class GameProfileResult:
    """Profile result for a single game."""

    game_idx: int
    wall_time_ms: float
    cpu_time_ms: float
    peak_memory_mb: float
    phase_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """
    Aggregated profiling results for a simulation run.

    Contains:
    - Overall timing statistics
    - Per-phase timing breakdowns
    - Memory usage statistics
    - CPU usage statistics
    """

    # Overall stats
    total_wall_time: float = 0.0
    total_cpu_time: float = 0.0

    # Memory stats
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_samples: int = 0
    _memory_sum: float = 0.0

    # Per-phase timing
    phase_timings: Dict[str, TimingStats] = field(default_factory=dict)

    # Per-game results (optional, for detailed analysis)
    game_results: List[GameProfileResult] = field(default_factory=list)

    def record_memory(self, memory_mb: float):
        """Record a memory sample."""
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        self._memory_sum += memory_mb
        self.memory_samples += 1
        self.avg_memory_mb = self._memory_sum / self.memory_samples

    def record_phase(self, phase_name: str, elapsed: float):
        """Record timing for a game phase."""
        if phase_name not in self.phase_timings:
            self.phase_timings[phase_name] = TimingStats()
        self.phase_timings[phase_name].record(elapsed)

    def add_game_result(self, result: GameProfileResult):
        """Add a per-game profile result."""
        self.game_results.append(result)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total_wall_time_s": self.total_wall_time,
            "total_cpu_time_s": self.total_cpu_time,
            "cpu_utilization_pct": (
                (self.total_cpu_time / self.total_wall_time * 100)
                if self.total_wall_time > 0
                else 0
            ),
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": self.avg_memory_mb,
            "games_profiled": len(self.game_results),
            "avg_game_time_ms": (
                (self.total_wall_time * 1000 / len(self.game_results))
                if self.game_results
                else 0
            ),
            "phase_breakdown": {
                name: stats.to_dict() for name, stats in self.phase_timings.items()
            },
        }

    def print_summary(self):
        """Print a formatted summary to console."""
        print(f"\n{'='*50}")
        print("Performance Profile")
        print(f"{'='*50}")

        print(f"\n📊 Overall Statistics:")
        print(f"  Total wall time:     {self.total_wall_time:.2f}s")
        print(f"  Total CPU time:      {self.total_cpu_time:.2f}s")
        if self.total_wall_time > 0:
            cpu_util = self.total_cpu_time / self.total_wall_time * 100
            print(f"  CPU utilization:     {cpu_util:.1f}%")

        print(f"\n💾 Memory Usage:")
        print(f"  Peak memory:         {self.peak_memory_mb:.2f} MB")
        print(f"  Average memory:      {self.avg_memory_mb:.2f} MB")

        if self.game_results:
            print(f"\n⏱️  Per-Game Statistics ({len(self.game_results)} games):")
            avg_time = self.total_wall_time * 1000 / len(self.game_results)
            print(f"  Average game time:   {avg_time:.2f} ms")

            if self.game_results:
                times = [g.wall_time_ms for g in self.game_results]
                print(f"  Min game time:       {min(times):.2f} ms")
                print(f"  Max game time:       {max(times):.2f} ms")

        if self.phase_timings:
            print(f"\n📈 Phase Breakdown:")
            # Sort by total time descending
            sorted_phases = sorted(
                self.phase_timings.items(), key=lambda x: x[1].total_time, reverse=True
            )

            total_phase_time = sum(s.total_time for _, s in sorted_phases)

            for name, stats in sorted_phases:
                pct = (
                    (stats.total_time / total_phase_time * 100)
                    if total_phase_time > 0
                    else 0
                )
                bar_len = int(pct / 5)
                bar = "█" * bar_len
                print(
                    f"  {name:25s} {stats.total_time*1000:8.1f}ms ({pct:5.1f}%) {bar}"
                )
                if stats.call_count > 1:
                    print(
                        f"    └─ {stats.call_count:,} calls, "
                        f"avg: {stats.avg_time*1000:.3f}ms, "
                        f"range: [{stats.min_time*1000:.3f}, {stats.max_time*1000:.3f}]ms"
                    )


class Profiler:
    """
    Performance profiler for Catan simulations.

    Usage:
        profiler = Profiler()
        profiler.start()

        with profiler.phase("setup"):
            # setup code

        with profiler.phase("game_loop"):
            # game code

        profiler.stop()
        profiler.result.print_summary()
    """

    def __init__(self, track_memory: bool = True, detailed: bool = False):
        """
        Initialize the profiler.

        Args:
            track_memory: Whether to track memory usage (slight overhead)
            detailed: Whether to store per-game detailed results
        """
        self.track_memory = track_memory
        self.detailed = detailed
        self.result = ProfileResult()

        self._start_wall_time: float = 0.0
        self._start_cpu_time: float = 0.0
        self._running = False

        # Current game tracking
        self._game_start_wall: float = 0.0
        self._game_start_cpu: float = 0.0
        self._game_idx: int = 0
        self._game_phase_timings: Dict[str, float] = {}

    def start(self):
        """Start the profiler."""
        self._running = True
        self._start_wall_time = time.perf_counter()
        self._start_cpu_time = time.process_time()

        if self.track_memory:
            tracemalloc.start()

    def stop(self):
        """Stop the profiler and finalize results."""
        if not self._running:
            return

        self.result.total_wall_time = time.perf_counter() - self._start_wall_time
        self.result.total_cpu_time = time.process_time() - self._start_cpu_time

        if self.track_memory:
            current, peak = tracemalloc.get_traced_memory()
            self.result.record_memory(peak / (1024 * 1024))
            tracemalloc.stop()

        self._running = False

    def sample_memory(self):
        """Take a memory sample (call periodically)."""
        if self.track_memory and self._running:
            current, peak = tracemalloc.get_traced_memory()
            self.result.record_memory(current / (1024 * 1024))

    @contextlib.contextmanager
    def phase(self, name: str):
        """
        Context manager to time a phase of execution.

        Usage:
            with profiler.phase("action_execution"):
                game.execute_action(action)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.result.record_phase(name, elapsed)

            # Track for current game if detailed
            if self.detailed:
                if name not in self._game_phase_timings:
                    self._game_phase_timings[name] = 0.0
                self._game_phase_timings[name] += elapsed

    def start_game(self, game_idx: int):
        """Mark the start of a game (for detailed profiling)."""
        self._game_idx = game_idx
        self._game_start_wall = time.perf_counter()
        self._game_start_cpu = time.process_time()
        self._game_phase_timings = {}

    def end_game(self):
        """Mark the end of a game and record results."""
        if not self.detailed:
            return

        wall_time = (time.perf_counter() - self._game_start_wall) * 1000
        cpu_time = (time.process_time() - self._game_start_cpu) * 1000

        peak_memory = 0.0
        if self.track_memory:
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / (1024 * 1024)
            self.result.record_memory(current / (1024 * 1024))

        game_result = GameProfileResult(
            game_idx=self._game_idx,
            wall_time_ms=wall_time,
            cpu_time_ms=cpu_time,
            peak_memory_mb=peak_memory,
            phase_timings=self._game_phase_timings.copy(),
        )
        self.result.add_game_result(game_result)


class NoOpProfiler:
    """
    No-op profiler that does nothing.

    Used when profiling is disabled to avoid conditionals in the main code.
    """

    def start(self):
        pass

    def stop(self):
        pass

    def sample_memory(self):
        pass

    @contextlib.contextmanager
    def phase(self, name: str):
        yield

    def start_game(self, game_idx: int):
        pass

    def end_game(self):
        pass

    @property
    def result(self) -> Optional[ProfileResult]:
        return None
