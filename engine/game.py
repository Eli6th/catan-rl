"""
Main game engine for Catan.

Handles:
- Game setup (initial placements)
- Turn management
- Win condition checking
- Action validation and execution
"""

# Standard Library Imports
from enum import IntEnum
from typing import Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

# Third Party Imports
import numpy as np

# Local Imports
from .state import (
    GameState,
    VICTORY_POINTS_TO_WIN,
    DEV_KNIGHT,
    DEV_ROAD_BUILDING,
    DEV_YEAR_OF_PLENTY,
    DEV_MONOPOLY,
)
from .board import get_board_topology
from .resources import distribute_resources, roll_dice
from .building import (
    build_road,
    build_settlement,
    build_city,
    get_valid_road_placements,
    get_valid_settlement_placements,
    get_valid_city_placements,
)
from .dev_cards import (
    buy_dev_card,
    play_knight,
    play_road_building,
    play_year_of_plenty,
    play_monopoly,
    can_buy_dev_card,
    can_play_dev_card,
)
from .robber import (
    get_players_who_must_discard,
    discard_resources,
    get_valid_robber_placements,
    get_stealable_players,
    move_robber,
    steal_random_resource,
)
from .trading import trade_with_bank, get_possible_bank_trades

if TYPE_CHECKING:
    from players.base import Player


class ActionType(IntEnum):
    """All possible action types in Catan."""

    # Setup actions
    PLACE_INITIAL_SETTLEMENT = 0
    PLACE_INITIAL_ROAD = 1

    # Turn actions
    ROLL_DICE = 2
    BUILD_ROAD = 3
    BUILD_SETTLEMENT = 4
    BUILD_CITY = 5
    BUY_DEV_CARD = 6

    # Development card actions
    PLAY_KNIGHT = 7
    PLAY_ROAD_BUILDING = 8
    PLAY_YEAR_OF_PLENTY = 9
    PLAY_MONOPOLY = 10

    # Robber actions
    MOVE_ROBBER = 11
    STEAL_RESOURCE = 12
    DISCARD_RESOURCES = 13

    # Trading actions
    TRADE_WITH_BANK = 14
    PROPOSE_TRADE = 15
    ACCEPT_TRADE = 16
    REJECT_TRADE = 17

    # End turn
    END_TURN = 18


@dataclass
class Action:
    """Represents a game action."""

    action_type: ActionType
    player: int
    data: Optional[np.ndarray] = None  # Action-specific data

    def to_bytes(self) -> bytes:
        """Serialize action for logging."""
        result = bytes([self.action_type, self.player])
        if self.data is not None:
            result += self.data.tobytes()
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "Action":
        """Deserialize action from bytes."""
        action_type = ActionType(data[0])
        player = data[1]
        action_data = np.frombuffer(data[2:], dtype=np.int8) if len(data) > 2 else None
        return cls(action_type, player, action_data)


class GamePhase(IntEnum):
    """Game phases."""

    SETUP_FORWARD = 0  # First round of initial placements
    SETUP_BACKWARD = 1  # Second round (reverse order)
    PLAYING = 2
    FINISHED = 3


class TurnPhase(IntEnum):
    """Phases within a turn."""

    PRE_ROLL = 0  # Can play dev card before rolling
    MUST_ROLL = 1  # Must roll dice
    ROBBER_DISCARD = 2  # Players must discard (on 7)
    ROBBER_MOVE = 3  # Current player moves robber
    ROBBER_STEAL = 4  # Current player steals
    MAIN = 5  # Main phase - build, trade, etc.
    ROAD_BUILDING = 6  # Placing roads from dev card


class CatanGame:
    """
    Main game controller.

    Manages game flow, validates actions, and coordinates between
    game state and player strategies.
    """

    def __init__(self, num_players: int = 4, seed: Optional[int] = None):
        """
        Initialize a new game.

        Args:
            num_players: Number of players (2-4)
            seed: Random seed for reproducibility
        """
        self.state = GameState(num_players, seed)
        self.players: List["Player"] = []

        # Game flow tracking
        self.game_phase = GamePhase.SETUP_FORWARD
        self.turn_phase = TurnPhase.MUST_ROLL
        self.setup_player_idx = 0
        self.roads_to_place = 0  # For road building card

        # Action history for logging/replay
        self.action_history: List[Action] = []

        # Pending robber state
        self.pending_discards: List[Tuple[int, int]] = []
        self.discard_idx = 0

        # Phase to return to after robber resolution (MAIN or MUST_ROLL)
        self.post_robber_phase = TurnPhase.MAIN

    def set_players(self, players: List["Player"]):
        """Set the player strategies."""
        assert len(players) == self.state.num_players
        self.players = players
        for i, player in enumerate(players):
            player.on_game_start(self.state, i)

    def get_current_player(self) -> int:
        """Get the current player index."""
        if self.game_phase in (GamePhase.SETUP_FORWARD, GamePhase.SETUP_BACKWARD):
            return self.setup_player_idx
        return self.state.current_player

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.game_phase == GamePhase.FINISHED

    def get_winner(self) -> int:
        """Get the winning player, or -1 if game not over."""
        return self.state.winner

    def get_valid_actions(self) -> List[Action]:
        """Get all valid actions for the current player."""
        player = self.get_current_player()
        actions = []

        if (
            self.game_phase == GamePhase.SETUP_FORWARD
            or self.game_phase == GamePhase.SETUP_BACKWARD
        ):
            return self._get_setup_actions(player)

        if self.turn_phase == TurnPhase.ROBBER_DISCARD:
            return self._get_discard_actions()

        if self.turn_phase == TurnPhase.ROBBER_MOVE:
            return self._get_robber_move_actions(player)

        if self.turn_phase == TurnPhase.ROBBER_STEAL:
            return self._get_steal_actions(player)

        if self.turn_phase == TurnPhase.ROAD_BUILDING:
            return self._get_road_building_actions(player)

        if self.turn_phase == TurnPhase.PRE_ROLL:
            return self._get_pre_roll_actions(player)

        if self.turn_phase == TurnPhase.MUST_ROLL:
            return [Action(ActionType.ROLL_DICE, player)]

        # Main phase
        return self._get_main_phase_actions(player)

    def _get_pre_roll_actions(self, player: int) -> List[Action]:
        """Get valid actions before rolling (can play knight)."""
        actions = [Action(ActionType.ROLL_DICE, player)]
        if can_play_dev_card(self.state, player, DEV_KNIGHT):
            actions.append(Action(ActionType.PLAY_KNIGHT, player))
        return actions

    def _get_setup_actions(self, player: int) -> List[Action]:
        """Get valid actions during setup phase."""
        actions = []

        # Check what the player has placed this round
        settlements = self.state.settlements_built[player]
        roads = self.state.roads_built[player]

        expected_settlements = 1 if self.game_phase == GamePhase.SETUP_FORWARD else 2

        if settlements < expected_settlements:
            # Place settlement
            valid_vertices = get_valid_settlement_placements(
                self.state, player, setup_phase=True
            )
            for v in valid_vertices:
                actions.append(
                    Action(
                        ActionType.PLACE_INITIAL_SETTLEMENT,
                        player,
                        np.array([v], dtype=np.int8),
                    )
                )
        elif roads < settlements:
            # Place road adjacent to last settlement
            valid_edges = self._get_initial_road_placements(player)
            for e in valid_edges:
                actions.append(
                    Action(
                        ActionType.PLACE_INITIAL_ROAD,
                        player,
                        np.array([e], dtype=np.int8),
                    )
                )

        return actions

    def _get_initial_road_placements(self, player: int) -> np.ndarray:
        """Get valid road placements for setup phase (adjacent to last settlement)."""
        topology = self.state.topology
        valid = []

        for vertex in range(54):
            if self.state.vertices[vertex] == player:
                # Only offer edges from the settlement that has no adjacent road yet
                has_road = any(
                    edge >= 0 and self.state.edges[edge] == player
                    for edge in topology.vertex_edges[vertex]
                )
                if not has_road:
                    for edge in topology.vertex_edges[vertex]:
                        if edge >= 0 and self.state.edges[edge] < 0:
                            valid.append(edge)

        return np.array(valid, dtype=np.int8)

    def _get_discard_actions(self) -> List[Action]:
        """Get discard actions when 7 is rolled."""
        if self.discard_idx >= len(self.pending_discards):
            return []

        player, count = self.pending_discards[self.discard_idx]
        actions = []

        # Generate some discard options (for AI to choose from)
        # This is simplified - in a full implementation, we'd enumerate more
        resources = self.state.resources[player].copy()

        # Simple strategy: generate a few random valid discards
        for _ in range(min(10, 2**count)):  # Limit options
            discard = np.zeros(5, dtype=np.int8)
            remaining = count
            available = resources.copy()

            while remaining > 0 and np.sum(available) > 0:
                # Pick a random available resource
                valid_types = np.where(available > 0)[0]
                if len(valid_types) == 0:
                    break
                res_type = self.state.rng.choice(valid_types)
                discard[res_type] += 1
                available[res_type] -= 1
                remaining -= 1

            if remaining == 0:
                actions.append(Action(ActionType.DISCARD_RESOURCES, player, discard))

        return actions

    def _get_robber_move_actions(self, player: int) -> List[Action]:
        """Get valid robber placement actions."""
        valid_tiles = get_valid_robber_placements(self.state)
        actions = []
        for tile in valid_tiles:
            actions.append(
                Action(ActionType.MOVE_ROBBER, player, np.array([tile], dtype=np.int8))
            )
        return actions

    def _get_steal_actions(self, player: int) -> List[Action]:
        """Get valid steal actions."""
        victims = get_stealable_players(self.state, self.state.robber_tile)
        actions = []

        if not victims:
            # No one to steal from, skip
            actions.append(
                Action(ActionType.STEAL_RESOURCE, player, np.array([-1], dtype=np.int8))
            )
        else:
            for victim in victims:
                actions.append(
                    Action(
                        ActionType.STEAL_RESOURCE,
                        player,
                        np.array([victim], dtype=np.int8),
                    )
                )

        return actions

    def _get_road_building_actions(self, player: int) -> List[Action]:
        """Get valid road placements during road building card."""
        valid_edges = get_valid_road_placements(self.state, player, free_placement=True)
        actions = []
        for edge in valid_edges:
            actions.append(
                Action(ActionType.BUILD_ROAD, player, np.array([edge], dtype=np.int8))
            )
        return actions

    def _get_main_phase_actions(self, player: int) -> List[Action]:
        """Get valid actions during main phase."""
        actions = []

        # Build road
        for edge in get_valid_road_placements(self.state, player):
            actions.append(
                Action(ActionType.BUILD_ROAD, player, np.array([edge], dtype=np.int8))
            )

        # Build settlement
        for vertex in get_valid_settlement_placements(self.state, player):
            actions.append(
                Action(
                    ActionType.BUILD_SETTLEMENT,
                    player,
                    np.array([vertex], dtype=np.int8),
                )
            )

        # Build city
        for vertex in get_valid_city_placements(self.state, player):
            actions.append(
                Action(ActionType.BUILD_CITY, player, np.array([vertex], dtype=np.int8))
            )

        # Buy development card
        if can_buy_dev_card(self.state, player):
            actions.append(Action(ActionType.BUY_DEV_CARD, player))

        # Play development cards (use can_play_dev_card to respect same-turn restriction)
        if can_play_dev_card(self.state, player, DEV_KNIGHT):
            actions.append(Action(ActionType.PLAY_KNIGHT, player))
        if can_play_dev_card(self.state, player, DEV_ROAD_BUILDING):
            actions.append(Action(ActionType.PLAY_ROAD_BUILDING, player))
        if can_play_dev_card(self.state, player, DEV_YEAR_OF_PLENTY):
            for r1 in range(5):
                for r2 in range(r1, 5):  # r2 >= r1 avoids duplicate pairs
                    if r1 == r2:
                        if self.state.bank[r1] >= 2:
                            actions.append(
                                Action(
                                    ActionType.PLAY_YEAR_OF_PLENTY,
                                    player,
                                    np.array([r1, r2], dtype=np.int8),
                                )
                            )
                    else:
                        if self.state.bank[r1] > 0 and self.state.bank[r2] > 0:
                            actions.append(
                                Action(
                                    ActionType.PLAY_YEAR_OF_PLENTY,
                                    player,
                                    np.array([r1, r2], dtype=np.int8),
                                )
                            )
        if can_play_dev_card(self.state, player, DEV_MONOPOLY):
            for r in range(5):
                actions.append(
                    Action(
                        ActionType.PLAY_MONOPOLY,
                        player,
                        np.array([r], dtype=np.int8),
                    )
                )

        # Bank trades
        for give_res, amount, recv_res in get_possible_bank_trades(self.state, player):
            actions.append(
                Action(
                    ActionType.TRADE_WITH_BANK,
                    player,
                    np.array([give_res, recv_res], dtype=np.int8),
                )
            )

        # End turn is always valid in main phase
        actions.append(Action(ActionType.END_TURN, player))

        return actions

    def execute_action(self, action: Action) -> bool:
        """
        Execute an action and update game state.

        Returns:
            True if action was successful
        """
        self.action_history.append(action)

        if action.action_type == ActionType.PLACE_INITIAL_SETTLEMENT:
            return self._execute_initial_settlement(action)
        elif action.action_type == ActionType.PLACE_INITIAL_ROAD:
            return self._execute_initial_road(action)
        elif action.action_type == ActionType.ROLL_DICE:
            return self._execute_roll_dice(action)
        elif action.action_type == ActionType.BUILD_ROAD:
            return self._execute_build_road(action)
        elif action.action_type == ActionType.BUILD_SETTLEMENT:
            return self._execute_build_settlement(action)
        elif action.action_type == ActionType.BUILD_CITY:
            return self._execute_build_city(action)
        elif action.action_type == ActionType.BUY_DEV_CARD:
            return self._execute_buy_dev_card(action)
        elif action.action_type == ActionType.PLAY_KNIGHT:
            return self._execute_play_knight(action)
        elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
            return self._execute_play_road_building(action)
        elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            return self._execute_play_year_of_plenty(action)
        elif action.action_type == ActionType.PLAY_MONOPOLY:
            return self._execute_play_monopoly(action)
        elif action.action_type == ActionType.MOVE_ROBBER:
            return self._execute_move_robber(action)
        elif action.action_type == ActionType.STEAL_RESOURCE:
            return self._execute_steal(action)
        elif action.action_type == ActionType.DISCARD_RESOURCES:
            return self._execute_discard(action)
        elif action.action_type == ActionType.TRADE_WITH_BANK:
            return self._execute_bank_trade(action)
        elif action.action_type == ActionType.END_TURN:
            return self._execute_end_turn(action)

        return False

    def _execute_initial_settlement(self, action: Action) -> bool:
        """Place initial settlement during setup."""
        vertex = action.data[0]
        success = build_settlement(self.state, action.player, vertex, free=True)

        if success and self.game_phase == GamePhase.SETUP_BACKWARD:
            # Give resources from adjacent tiles
            topology = self.state.topology
            for tile in topology.vertex_tiles[vertex]:
                if tile >= 0:
                    res_type = self.state.tile_resources[tile]
                    if res_type < 5:  # Not desert
                        self.state.resources[action.player, res_type] += 1
                        self.state.bank[res_type] -= 1

        return success

    def _execute_initial_road(self, action: Action) -> bool:
        """Place initial road during setup."""
        edge = action.data[0]
        success = build_road(self.state, action.player, edge, free=True)

        if success:
            self._advance_setup()

        return success

    def _advance_setup(self):
        """Advance to the next player in setup phase."""
        if self.game_phase == GamePhase.SETUP_FORWARD:
            self.setup_player_idx += 1
            if self.setup_player_idx >= self.state.num_players:
                self.game_phase = GamePhase.SETUP_BACKWARD
                self.setup_player_idx = self.state.num_players - 1
        elif self.game_phase == GamePhase.SETUP_BACKWARD:
            self.setup_player_idx -= 1
            if self.setup_player_idx < 0:
                self.game_phase = GamePhase.PLAYING
                self.state.phase = 1
                self.turn_phase = TurnPhase.PRE_ROLL

    def _execute_roll_dice(self, action: Action) -> bool:
        """Roll dice and distribute resources."""
        d1, d2, total = roll_dice(self.state)
        self.state.dice_roll = total
        self.state.has_rolled = True

        if total == 7:
            # Check for discards
            self.pending_discards = get_players_who_must_discard(self.state)
            if self.pending_discards:
                self.turn_phase = TurnPhase.ROBBER_DISCARD
                self.discard_idx = 0
            else:
                self.turn_phase = TurnPhase.ROBBER_MOVE
        else:
            distribute_resources(self.state, total)
            self.turn_phase = TurnPhase.MAIN

        return True

    def _execute_build_road(self, action: Action) -> bool:
        """Build a road."""
        edge = action.data[0]
        free = self.turn_phase == TurnPhase.ROAD_BUILDING
        success = build_road(self.state, action.player, edge, free=free)

        if success and self.turn_phase == TurnPhase.ROAD_BUILDING:
            self.roads_to_place -= 1
            if self.roads_to_place <= 0:
                self.turn_phase = TurnPhase.MAIN

        self._check_victory(action.player)
        return success

    def _execute_build_settlement(self, action: Action) -> bool:
        """Build a settlement."""
        vertex = action.data[0]
        success = build_settlement(self.state, action.player, vertex)
        self._check_victory(action.player)
        return success

    def _execute_build_city(self, action: Action) -> bool:
        """Build a city."""
        vertex = action.data[0]
        success = build_city(self.state, action.player, vertex)
        self._check_victory(action.player)
        return success

    def _execute_buy_dev_card(self, action: Action) -> bool:
        """Buy a development card."""
        card_type = buy_dev_card(self.state, action.player)
        return card_type >= 0

    def _execute_play_knight(self, action: Action) -> bool:
        """Play a knight card."""
        success = play_knight(self.state, action.player)
        if success:
            # After robber resolution, return to MUST_ROLL if pre-roll, else MAIN
            self.post_robber_phase = (
                TurnPhase.MUST_ROLL
                if self.turn_phase == TurnPhase.PRE_ROLL
                else TurnPhase.MAIN
            )
            self.turn_phase = TurnPhase.ROBBER_MOVE
        return success

    def _execute_play_road_building(self, action: Action) -> bool:
        """Play road building card."""
        success = play_road_building(self.state, action.player)
        if success:
            self.roads_to_place = 2
            self.turn_phase = TurnPhase.ROAD_BUILDING
        return success

    def _execute_play_year_of_plenty(self, action: Action) -> bool:
        """Play year of plenty card."""
        r1, r2 = action.data[0], action.data[1]
        return play_year_of_plenty(self.state, action.player, r1, r2)

    def _execute_play_monopoly(self, action: Action) -> bool:
        """Play monopoly card."""
        resource = action.data[0]
        result = play_monopoly(self.state, action.player, resource)
        return result >= 0

    def _execute_move_robber(self, action: Action) -> bool:
        """Move the robber."""
        tile = action.data[0]
        success = move_robber(self.state, tile)
        if success:
            victims = get_stealable_players(self.state, tile)
            if victims:
                self.turn_phase = TurnPhase.ROBBER_STEAL
            else:
                self.turn_phase = self.post_robber_phase
        return success

    def _execute_steal(self, action: Action) -> bool:
        """Steal from a player."""
        victim = action.data[0]
        if victim >= 0:
            steal_random_resource(self.state, victim)
        self.turn_phase = self.post_robber_phase
        return True

    def _execute_discard(self, action: Action) -> bool:
        """Discard resources."""
        player = action.player
        resources = action.data

        success = discard_resources(self.state, player, resources)
        if success:
            self.discard_idx += 1
            if self.discard_idx >= len(self.pending_discards):
                self.turn_phase = TurnPhase.ROBBER_MOVE
        return success

    def _execute_bank_trade(self, action: Action) -> bool:
        """Trade with bank."""
        give_res = action.data[0]
        recv_res = action.data[1]
        return trade_with_bank(self.state, action.player, give_res, recv_res)

    def _execute_end_turn(self, action: Action) -> bool:
        """End the current turn."""
        self.state.turn += 1
        self.state.current_player = (
            self.state.current_player + 1
        ) % self.state.num_players
        self.turn_phase = TurnPhase.PRE_ROLL
        self.post_robber_phase = TurnPhase.MAIN
        self.state.has_rolled = False
        self.state.dev_cards_bought_this_turn[:] = 0
        self.state.dev_card_played_this_turn = False
        return True

    def _check_victory(self, player: int):
        """Check if player has won."""
        vp = self.state.calculate_victory_points(player)
        if vp >= VICTORY_POINTS_TO_WIN:
            self.state.winner = player
            self.state.phase = 2
            self.game_phase = GamePhase.FINISHED

    def play_game(self) -> int:
        """
        Play a complete game using the assigned player strategies.

        Returns:
            Winning player index
        """
        max_turns = 1000  # Prevent infinite games

        while not self.is_game_over() and self.state.turn < max_turns:
            player_idx = self.get_current_player()
            player = self.players[player_idx]

            valid_actions = self.get_valid_actions()
            if not valid_actions:
                # No valid actions, skip (shouldn't happen)
                break

            action = player.choose_action(self.state, valid_actions)
            self.execute_action(action)

        return self.get_winner()

    def get_action_history(self) -> List[Action]:
        """Get the complete action history."""
        return self.action_history
