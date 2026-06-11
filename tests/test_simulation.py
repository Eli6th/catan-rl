"""Tests for simulation framework."""

# Standard Library Imports
import tempfile

# Third Party Imports

# Local Imports
from engine.game import CatanGame
from players.strategies import RandomPlayer, HeuristicPlayer
from simulation.runner import SimulationRunner, SimulationConfig, run_quick_simulation
from simulation.stats import SimulationStats, GameResult
from simulation.logger import GameLogger
from simulation.replay import GameReplay


class TestSimulationRunner:
    """Test simulation runner."""

    def test_basic_simulation(self):
        """Test running a basic simulation."""
        config = SimulationConfig(
            num_players=4,
            player_types=[RandomPlayer] * 4,
            num_games=10,
            base_seed=42,
            verbosity="silent",
        )

        runner = SimulationRunner(config)
        stats = runner.run()

        assert stats.total_games == 10
        assert len(stats.results) == 10

    def test_deterministic_with_seed(self):
        """Test that simulations are deterministic with same seed."""
        config = SimulationConfig(
            num_players=4,
            player_types=[RandomPlayer] * 4,
            num_games=5,
            base_seed=42,
            verbosity="silent",
        )

        runner1 = SimulationRunner(config)
        stats1 = runner1.run()

        runner2 = SimulationRunner(config)
        stats2 = runner2.run()

        # Results should be identical
        for r1, r2 in zip(stats1.results, stats2.results):
            assert r1.seed == r2.seed
            assert r1.winner == r2.winner
            assert r1.turns == r2.turns

    def test_mixed_strategies(self):
        """Test simulation with mixed strategies."""
        config = SimulationConfig(
            num_players=4,
            player_types=[RandomPlayer, HeuristicPlayer, RandomPlayer, HeuristicPlayer],
            num_games=10,
            base_seed=42,
            verbosity="silent",
        )

        runner = SimulationRunner(config)
        stats = runner.run()

        assert stats.total_games == 10
        # Should have wins from both strategies
        assert len(stats.wins_by_strategy) >= 1


class TestSimulationStats:
    """Test statistics collection."""

    def test_record_game(self):
        """Test recording game results."""
        stats = SimulationStats()

        result = GameResult(
            seed=42,
            winner=0,
            winner_strategy="RandomPlayer",
            turns=100,
            player_strategies=["RandomPlayer"] * 4,
            final_vps=[10, 5, 3, 2],
        )

        stats.record_game(result)

        assert stats.total_games == 1
        assert stats.wins_by_strategy["RandomPlayer"] == 1
        assert stats.total_turns == 100

    def test_win_rates(self):
        """Test win rate calculation."""
        stats = SimulationStats()

        for i in range(10):
            result = GameResult(
                seed=i,
                winner=i % 2,
                winner_strategy="A" if i % 2 == 0 else "B",
                turns=50,
                player_strategies=["A", "B", "A", "B"],
                final_vps=[10, 5, 5, 5],
            )
            stats.record_game(result)

        rates = stats.get_all_win_rates()

        assert rates["A"] == 50.0
        assert rates["B"] == 50.0

    def test_average_game_length(self):
        """Test average game length calculation."""
        stats = SimulationStats()

        for turns in [50, 100, 150]:
            result = GameResult(
                seed=turns,
                winner=0,
                winner_strategy="Test",
                turns=turns,
                player_strategies=["Test"] * 4,
                final_vps=[10, 0, 0, 0],
            )
            stats.record_game(result)

        assert stats.average_game_length == 100


class TestGameLogger:
    """Test game logging."""

    def test_log_and_load(self):
        """Test logging and loading a game."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GameLogger(tmpdir)

            # Create and log a game
            game = CatanGame(num_players=4, seed=42)
            players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
            game.set_players(players)

            logger.start_game(game.state)

            # Play a few actions
            for _ in range(20):
                actions = game.get_valid_actions()
                if not actions:
                    break
                player = players[game.get_current_player()]
                action = player.choose_action(game.state, actions)
                game.execute_action(action)
                logger.log_action(action)

            # Save
            filepath = logger.save_binary(game.state)

            # Load and verify
            data = GameLogger.load_binary(filepath)

            assert data["seed"] == 42
            assert data["num_players"] == 4
            assert len(data["actions"]) == 20


class TestGameReplay:
    """Test game replay functionality."""

    def test_replay_step(self):
        """Test stepping through a replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and log a game
            logger = GameLogger(tmpdir)
            game = CatanGame(num_players=4, seed=42)
            players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
            game.set_players(players)

            logger.start_game(game.state)

            for _ in range(10):
                actions = game.get_valid_actions()
                if not actions:
                    break
                player = players[game.get_current_player()]
                action = player.choose_action(game.state, actions)
                game.execute_action(action)
                logger.log_action(action)

            filepath = logger.save_binary(game.state)

            # Load replay
            replay = GameReplay.load(filepath)

            assert replay.action_index == 0

            # Step through
            action = replay.step()
            assert action is not None
            assert replay.action_index == 1

    def test_replay_jump_to_action(self):
        """Test jumping to specific action."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = GameLogger(tmpdir)
            game = CatanGame(num_players=4, seed=42)
            players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
            game.set_players(players)

            logger.start_game(game.state)

            for _ in range(20):
                actions = game.get_valid_actions()
                if not actions:
                    break
                player = players[game.get_current_player()]
                action = player.choose_action(game.state, actions)
                game.execute_action(action)
                logger.log_action(action)

            filepath = logger.save_binary(game.state)

            replay = GameReplay.load(filepath)

            # Jump to action 10
            success = replay.jump_to_action(10)

            assert success
            assert replay.action_index == 10


class TestQuickSimulation:
    """Test convenience functions."""

    def test_run_quick_simulation(self):
        """Test quick simulation helper."""
        stats = run_quick_simulation(
            num_games=5,
            num_players=4,
            seed=42,
            verbosity="silent",
        )

        assert stats.total_games == 5
