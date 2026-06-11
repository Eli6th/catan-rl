"""Comprehensive tests for development cards."""

import numpy as np
import pytest

from engine.state import (
    GameState,
    DEV_KNIGHT,
    DEV_VICTORY_POINT,
    DEV_ROAD_BUILDING,
    DEV_YEAR_OF_PLENTY,
    DEV_MONOPOLY,
    NUM_DEV_CARD_TYPES,
)
from engine.dev_cards import (
    buy_dev_card,
    can_buy_dev_card,
    can_play_dev_card,
    play_knight,
    play_road_building,
    play_year_of_plenty,
    play_monopoly,
)
from engine.game import (
    CatanGame,
    ActionType,
    TurnPhase,
    GamePhase,
    Action,
)
from players.strategies import RandomPlayer


def make_state(num_players=4, seed=42):
    return GameState(num_players=num_players, seed=seed)


def give_dev_card(state, player, card_type):
    """Helper: give a player a dev card without drawing from deck."""
    state.dev_cards[player, card_type] += 1


def setup_to_main(game, players):
    """Skip setup phase."""
    while game.game_phase in (GamePhase.SETUP_FORWARD, GamePhase.SETUP_BACKWARD):
        acts = game.get_valid_actions()
        if not acts:
            break
        act = players[game.get_current_player()].choose_action(game.state, acts)
        game.execute_action(act)


class TestBuyDevCard:
    """Test buying development cards."""

    def test_can_buy_with_resources(self):
        state = make_state()
        state.resources[0] = np.array([1, 1, 0, 0, 1], dtype=np.int16)
        assert can_buy_dev_card(state, 0)

    def test_cant_buy_without_resources(self):
        state = make_state()
        state.resources[0] = np.zeros(5, dtype=np.int16)
        assert not can_buy_dev_card(state, 0)

    def test_cant_buy_empty_deck(self):
        state = make_state()
        state.resources[0] = np.array([1, 1, 0, 0, 1], dtype=np.int16)
        state.dev_deck_idx = len(state.dev_deck)  # exhausted
        assert not can_buy_dev_card(state, 0)

    def test_buy_deducts_resources(self):
        state = make_state()
        state.resources[0] = np.array([2, 2, 0, 0, 2], dtype=np.int16)
        buy_dev_card(state, 0)
        assert state.resources[0, 0] == 1  # wheat
        assert state.resources[0, 1] == 1  # sheep
        assert state.resources[0, 4] == 1  # stone

    def test_buy_increments_deck_index(self):
        state = make_state()
        state.resources[0] = np.array([1, 1, 0, 0, 1], dtype=np.int16)
        idx_before = state.dev_deck_idx
        buy_dev_card(state, 0)
        assert state.dev_deck_idx == idx_before + 1

    def test_buy_marks_card_bought_this_turn(self):
        state = make_state()
        state.resources[0] = np.array([1, 1, 0, 0, 1], dtype=np.int16)
        card_type = buy_dev_card(state, 0)
        assert card_type >= 0
        assert state.dev_cards_bought_this_turn[card_type] == 1


class TestCanPlayDevCard:
    """Test the can_play_dev_card restriction logic."""

    def test_cant_play_card_bought_this_turn(self):
        """If you bought a knight this turn and have no others, can't play it."""
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        state.dev_cards_bought_this_turn[DEV_KNIGHT] = 1
        assert not can_play_dev_card(state, 0, DEV_KNIGHT)

    def test_can_play_older_card_after_buying_another(self):
        """If you had 2 knights and bought 1 this turn, you can still play 1."""
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        give_dev_card(state, 0, DEV_KNIGHT)
        # One knight bought this turn, but player has 2 total
        state.dev_cards_bought_this_turn[DEV_KNIGHT] = 1
        assert can_play_dev_card(state, 0, DEV_KNIGHT)

    def test_cant_play_without_card(self):
        state = make_state()
        assert not can_play_dev_card(state, 0, DEV_KNIGHT)

    def test_cant_play_victory_point(self):
        state = make_state()
        give_dev_card(state, 0, DEV_VICTORY_POINT)
        assert not can_play_dev_card(state, 0, DEV_VICTORY_POINT)

    def test_cant_play_second_card_in_turn(self):
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        give_dev_card(state, 0, DEV_ROAD_BUILDING)
        state.dev_card_played_this_turn = True
        assert not can_play_dev_card(state, 0, DEV_KNIGHT)
        assert not can_play_dev_card(state, 0, DEV_ROAD_BUILDING)

    def test_can_play_fresh_card(self):
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        assert can_play_dev_card(state, 0, DEV_KNIGHT)


class TestKnight:
    """Test knight card mechanics."""

    def test_play_knight_removes_card(self):
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        play_knight(state, 0)
        assert state.dev_cards[0, DEV_KNIGHT] == 0

    def test_play_knight_increments_count(self):
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        play_knight(state, 0)
        assert state.knights_played[0] == 1

    def test_play_knight_marks_played_this_turn(self):
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        play_knight(state, 0)
        assert state.dev_card_played_this_turn

    def test_largest_army_at_3_knights(self):
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        give_dev_card(state, 0, DEV_KNIGHT)
        give_dev_card(state, 0, DEV_KNIGHT)
        play_knight(state, 0)
        state.dev_card_played_this_turn = False
        play_knight(state, 0)
        # Not yet at 3
        assert state.largest_army_player == -1
        # 3rd knight triggers largest army
        state.dev_card_played_this_turn = False
        play_knight(state, 0)
        assert state.largest_army_player == 0
        assert state.largest_army_size == 3

    def test_largest_army_not_at_2_knights(self):
        state = make_state()
        give_dev_card(state, 0, DEV_KNIGHT)
        give_dev_card(state, 0, DEV_KNIGHT)
        play_knight(state, 0)
        state.dev_card_played_this_turn = False
        play_knight(state, 0)
        assert state.largest_army_player == -1

    def test_largest_army_stolen_by_more_knights(self):
        state = make_state()
        # Player 0 gets largest army with 3 knights
        state.largest_army_player = 0
        state.largest_army_size = 3
        state.knights_played[0] = 3
        # Player 1 plays 4 knights
        give_dev_card(state, 1, DEV_KNIGHT)
        state.knights_played[1] = 3
        state.current_player = 1
        play_knight(state, 1)
        assert state.largest_army_player == 1
        assert state.largest_army_size == 4

    def test_largest_army_tie_original_keeps(self):
        """When tied, original holder keeps largest army."""
        state = make_state()
        state.largest_army_player = 0
        state.largest_army_size = 3
        state.knights_played[0] = 3
        state.knights_played[1] = 2
        # Player 1 plays to 3 (tie) - original holder should keep
        give_dev_card(state, 1, DEV_KNIGHT)
        state.current_player = 1
        play_knight(state, 1)
        # Player 1 has 3 knights now but player 0 had it first
        assert state.largest_army_player == 0  # Original holder keeps it


class TestKnightPreRoll:
    """Test that knight can be played before rolling dice (PRE_ROLL phase)."""

    def _make_game_at_pre_roll_with_knight(self, player_idx=0):
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)
        setup_to_main(game, players)

        # Manually advance to the right player with a knight
        game.state.current_player = player_idx
        game.turn_phase = TurnPhase.PRE_ROLL
        give_dev_card(game.state, player_idx, DEV_KNIGHT)
        return game, players

    def test_knight_available_in_pre_roll(self):
        game, _ = self._make_game_at_pre_roll_with_knight()
        actions = game.get_valid_actions()
        types = {a.action_type for a in actions}
        assert ActionType.ROLL_DICE in types
        assert ActionType.PLAY_KNIGHT in types

    def test_no_knight_in_pre_roll_without_card(self):
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)
        setup_to_main(game, players)
        game.turn_phase = TurnPhase.PRE_ROLL
        actions = game.get_valid_actions()
        types = {a.action_type for a in actions}
        assert ActionType.ROLL_DICE in types
        assert ActionType.PLAY_KNIGHT not in types

    def test_pre_roll_knight_leads_to_must_roll_after_robber(self):
        """After knight in PRE_ROLL, robber resolution returns to MUST_ROLL."""
        game, players = self._make_game_at_pre_roll_with_knight(0)
        player = game.state.current_player

        # Play knight in PRE_ROLL
        knight_action = Action(ActionType.PLAY_KNIGHT, player)
        game.execute_action(knight_action)
        assert game.turn_phase == TurnPhase.ROBBER_MOVE

        # Move robber
        robber_actions = game.get_valid_actions()
        game.execute_action(robber_actions[0])

        # After robber resolution, should be at MUST_ROLL (or ROBBER_STEAL)
        if game.turn_phase == TurnPhase.ROBBER_STEAL:
            steal_actions = game.get_valid_actions()
            game.execute_action(steal_actions[0])

        assert game.turn_phase == TurnPhase.MUST_ROLL

    def test_main_phase_knight_stays_in_main(self):
        """Knight played in MAIN phase returns to MAIN after robber."""
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)
        setup_to_main(game, players)

        player = game.state.current_player
        game.turn_phase = TurnPhase.MAIN
        give_dev_card(game.state, player, DEV_KNIGHT)

        knight_action = Action(ActionType.PLAY_KNIGHT, player)
        game.execute_action(knight_action)
        assert game.turn_phase == TurnPhase.ROBBER_MOVE

        robber_actions = game.get_valid_actions()
        game.execute_action(robber_actions[0])

        if game.turn_phase == TurnPhase.ROBBER_STEAL:
            steal_actions = game.get_valid_actions()
            game.execute_action(steal_actions[0])

        assert game.turn_phase == TurnPhase.MAIN


class TestYearOfPlenty:
    """Test year of plenty card."""

    def test_year_of_plenty_gives_two_resources(self):
        state = make_state()
        give_dev_card(state, 0, DEV_YEAR_OF_PLENTY)
        state.bank[0] = 5
        state.bank[1] = 5
        play_year_of_plenty(state, 0, 0, 1)
        assert state.resources[0, 0] == 1
        assert state.resources[0, 1] == 1

    def test_year_of_plenty_deducts_from_bank(self):
        state = make_state()
        give_dev_card(state, 0, DEV_YEAR_OF_PLENTY)
        state.bank[0] = 5
        play_year_of_plenty(state, 0, 0, 0)
        assert state.bank[0] == 3  # gave 2 of resource 0

    def test_year_of_plenty_same_resource_needs_2_in_bank(self):
        state = make_state()
        give_dev_card(state, 0, DEV_YEAR_OF_PLENTY)
        state.bank[0] = 1  # only 1 in bank
        success = play_year_of_plenty(state, 0, 0, 0)
        assert not success

    def test_year_of_plenty_same_resource_with_2_in_bank(self):
        state = make_state()
        give_dev_card(state, 0, DEV_YEAR_OF_PLENTY)
        state.bank[0] = 2
        success = play_year_of_plenty(state, 0, 0, 0)
        assert success
        assert state.resources[0, 0] == 2

    def test_year_of_plenty_actions_exclude_same_resource_when_bank_has_1(self):
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)
        setup_to_main(game, players)

        player = game.state.current_player
        give_dev_card(game.state, player, DEV_YEAR_OF_PLENTY)
        game.state.bank[0] = 1  # only 1 wheat
        game.turn_phase = TurnPhase.MAIN

        actions = game.get_valid_actions()
        yop_actions = [a for a in actions if a.action_type == ActionType.PLAY_YEAR_OF_PLENTY]

        # Should not include (0, 0) since bank only has 1 wheat
        for a in yop_actions:
            if a.data[0] == 0 and a.data[1] == 0:
                pytest.fail("Should not generate (wheat, wheat) when bank has only 1 wheat")

    def test_year_of_plenty_actions_no_duplicates(self):
        """Each (r1, r2) pair should appear once, with r1 <= r2."""
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)
        setup_to_main(game, players)

        player = game.state.current_player
        give_dev_card(game.state, player, DEV_YEAR_OF_PLENTY)
        game.turn_phase = TurnPhase.MAIN

        actions = game.get_valid_actions()
        yop_actions = [a for a in actions if a.action_type == ActionType.PLAY_YEAR_OF_PLENTY]

        # All pairs should have r1 <= r2 (no duplicates)
        for a in yop_actions:
            r1, r2 = int(a.data[0]), int(a.data[1])
            assert r1 <= r2, f"Duplicate pair ({r1}, {r2}) found"


class TestMonopoly:
    """Test monopoly card."""

    def test_monopoly_takes_all_of_resource(self):
        state = make_state()
        give_dev_card(state, 0, DEV_MONOPOLY)
        state.resources[1, 0] = 3
        state.resources[2, 0] = 2
        state.resources[3, 0] = 1
        total_stolen = play_monopoly(state, 0, 0)
        assert total_stolen == 6
        assert state.resources[0, 0] == 6
        assert state.resources[1, 0] == 0
        assert state.resources[2, 0] == 0
        assert state.resources[3, 0] == 0

    def test_monopoly_doesnt_take_other_resources(self):
        state = make_state()
        give_dev_card(state, 0, DEV_MONOPOLY)
        state.resources[1, 0] = 3
        state.resources[1, 1] = 2  # sheep - not taken
        play_monopoly(state, 0, 0)
        assert state.resources[1, 1] == 2  # sheep untouched

    def test_monopoly_on_empty_resource_steals_nothing(self):
        state = make_state()
        give_dev_card(state, 0, DEV_MONOPOLY)
        total_stolen = play_monopoly(state, 0, 0)
        assert total_stolen == 0

    def test_monopoly_invalid_resource(self):
        state = make_state()
        give_dev_card(state, 0, DEV_MONOPOLY)
        result = play_monopoly(state, 0, 5)  # invalid resource
        assert result == -1


class TestRoadBuilding:
    """Test road building card."""

    def test_road_building_places_2_free_roads(self):
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)
        setup_to_main(game, players)

        player = game.state.current_player
        roads_before = game.state.roads_built[player]
        give_dev_card(game.state, player, DEV_ROAD_BUILDING)
        game.turn_phase = TurnPhase.MAIN

        rb_action = Action(ActionType.PLAY_ROAD_BUILDING, player)
        game.execute_action(rb_action)
        assert game.turn_phase == TurnPhase.ROAD_BUILDING

        # Place road 1
        acts = game.get_valid_actions()
        assert all(a.action_type == ActionType.BUILD_ROAD for a in acts)
        game.execute_action(acts[0])

        # Should still be in ROAD_BUILDING
        assert game.turn_phase == TurnPhase.ROAD_BUILDING

        # Place road 2
        acts = game.get_valid_actions()
        game.execute_action(acts[0])

        # Now back to MAIN
        assert game.turn_phase == TurnPhase.MAIN
        assert game.state.roads_built[player] == roads_before + 2

    def test_road_building_no_resource_cost(self):
        game = CatanGame(num_players=4, seed=42)
        players = [RandomPlayer(f"P{i}", seed=i) for i in range(4)]
        game.set_players(players)
        setup_to_main(game, players)

        player = game.state.current_player
        game.state.resources[player] = np.zeros(5, dtype=np.int16)  # no resources
        give_dev_card(game.state, player, DEV_ROAD_BUILDING)
        game.turn_phase = TurnPhase.MAIN

        rb_action = Action(ActionType.PLAY_ROAD_BUILDING, player)
        game.execute_action(rb_action)

        acts = game.get_valid_actions()
        if acts:
            resources_before = game.state.resources[player].copy()
            game.execute_action(acts[0])
            # Resources should still be 0 (road was free)
            assert np.all(game.state.resources[player] >= 0)


class TestVictoryPointCards:
    """Test victory point card handling."""

    def test_vp_card_counted_in_victory_points(self):
        state = make_state()
        give_dev_card(state, 0, DEV_VICTORY_POINT)
        vp = state.calculate_victory_points(0)
        assert vp == 1

    def test_vp_card_counts_multiply(self):
        state = make_state()
        give_dev_card(state, 0, DEV_VICTORY_POINT)
        give_dev_card(state, 0, DEV_VICTORY_POINT)
        vp = state.calculate_victory_points(0)
        assert vp == 2

    def test_cant_play_vp_card_directly(self):
        state = make_state()
        give_dev_card(state, 0, DEV_VICTORY_POINT)
        assert not can_play_dev_card(state, 0, DEV_VICTORY_POINT)
