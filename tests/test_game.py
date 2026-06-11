"""Tests for game engine."""

# Standard Library Imports

# Third Party Imports
import pytest
import numpy as np

# Local Imports
from engine.state import GameState
from engine.game import CatanGame, ActionType, GamePhase, TurnPhase, Action
from players.strategies import RandomPlayer


class TestGameState:
    """Test game state initialization and operations."""

    def test_initialization(self):
        """Test basic game state initialization."""
        state = GameState(num_players=4, seed=42)

        assert state.num_players == 4
        assert state.seed == 42
        assert state.phase == 0
        assert state.turn == 0
        assert state.current_player == 0
        assert state.winner == -1

    def test_different_seeds_different_boards(self):
        """Test that different seeds produce different boards."""
        state1 = GameState(num_players=4, seed=1)
        state2 = GameState(num_players=4, seed=2)

        # Tile resources should be different
        assert not np.array_equal(state1.tile_resources, state2.tile_resources)

    def test_same_seed_same_board(self):
        """Test that same seed produces same board."""
        state1 = GameState(num_players=4, seed=42)
        state2 = GameState(num_players=4, seed=42)

        assert np.array_equal(state1.tile_resources, state2.tile_resources)
        assert np.array_equal(state1.tile_numbers, state2.tile_numbers)

    def test_copy(self):
        """Test state copy functionality."""
        state = GameState(num_players=4, seed=42)
        state.resources[0, 0] = 5

        copy = state.copy()

        assert copy.resources[0, 0] == 5

        # Modify copy, original should be unchanged
        copy.resources[0, 0] = 10
        assert state.resources[0, 0] == 5

    def test_player_count_limits(self):
        """Test player count validation."""
        with pytest.raises(AssertionError):
            GameState(num_players=1)

        with pytest.raises(AssertionError):
            GameState(num_players=5)

    def test_initial_resources_empty(self):
        """Test that players start with no resources."""
        state = GameState(num_players=4)

        for p in range(4):
            assert state.get_player_total_resources(p) == 0

    def test_bank_initial_resources(self):
        """Test bank has correct initial resources."""
        state = GameState(num_players=4)

        for r in range(5):
            assert state.bank[r] == 19

    def test_victory_points_initial(self):
        """Test initial victory points are zero."""
        state = GameState(num_players=4)

        for p in range(4):
            assert state.calculate_victory_points(p) == 0


class TestCatanGame:
    """Test game controller."""

    def test_initialization(self):
        """Test game initialization."""
        game = CatanGame(num_players=4, seed=42)

        assert game.state.num_players == 4
        assert game.game_phase == GamePhase.SETUP_FORWARD
        assert not game.is_game_over()

    def test_setup_phase_actions(self):
        """Test that setup phase provides settlement placement."""
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}") for i in range(4)]
        game.set_players(players)

        actions = game.get_valid_actions()

        # Should have settlement placement actions
        assert len(actions) > 0
        assert all(
            a.action_type == ActionType.PLACE_INITIAL_SETTLEMENT for a in actions
        )

    def test_play_through_setup(self):
        """Test playing through setup phase."""
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)

        max_actions = 100
        action_count = 0

        while game.game_phase in (GamePhase.SETUP_FORWARD, GamePhase.SETUP_BACKWARD):
            actions = game.get_valid_actions()
            if not actions:
                break

            player = players[game.get_current_player()]
            action = player.choose_action(game.state, actions)
            game.execute_action(action)

            action_count += 1
            if action_count > max_actions:
                break

        # Should have completed setup
        assert game.game_phase == GamePhase.PLAYING

        # Each player should have 2 settlements and 2 roads
        for p in range(4):
            assert game.state.settlements_built[p] == 2
            assert game.state.roads_built[p] == 2

    def test_dice_roll_action(self):
        """Test dice rolling during play phase."""
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)

        # Skip setup
        self._skip_setup(game, players)

        # Should be in main game, pre-roll (can play knight or roll)
        assert game.game_phase == GamePhase.PLAYING
        assert game.turn_phase == TurnPhase.PRE_ROLL

        actions = game.get_valid_actions()
        # ROLL_DICE is always available in PRE_ROLL (no knight in hand here)
        assert any(a.action_type == ActionType.ROLL_DICE for a in actions)

        # Execute roll
        game.execute_action(actions[0])

        # Should have rolled
        assert game.state.dice_roll >= 2
        assert game.state.dice_roll <= 12

    def test_complete_game(self):
        """Test playing a complete game."""
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)

        winner = game.play_game()

        # Game should be finished
        assert game.is_game_over()
        assert winner >= 0 or game.state.turn >= 1000  # Either winner or max turns

    def _skip_setup(self, game, players):
        """Helper to skip setup phase."""
        while game.game_phase in (GamePhase.SETUP_FORWARD, GamePhase.SETUP_BACKWARD):
            actions = game.get_valid_actions()
            if not actions:
                break
            player = players[game.get_current_player()]
            action = player.choose_action(game.state, actions)
            game.execute_action(action)


class TestDiceAndResources:
    """Test dice rolling and resource distribution."""

    def test_dice_range(self):
        """Test that dice rolls are in valid range."""
        from engine.resources import roll_dice

        state = GameState(num_players=4, seed=42)

        for _ in range(100):
            d1, d2, total = roll_dice(state)
            assert 1 <= d1 <= 6
            assert 1 <= d2 <= 6
            assert total == d1 + d2
            assert 2 <= total <= 12

    def test_dice_distribution(self):
        """Test that dice distribution is roughly correct."""
        from engine.resources import roll_dice

        state = GameState(num_players=4, seed=42)
        counts = {i: 0 for i in range(2, 13)}

        num_rolls = 10000
        for _ in range(num_rolls):
            _, _, total = roll_dice(state)
            counts[total] += 1

        # 7 should be most common
        assert counts[7] > counts[2]
        assert counts[7] > counts[12]

        # 2 and 12 should be least common
        assert counts[2] < counts[6]
        assert counts[12] < counts[8]
