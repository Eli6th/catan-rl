"""
AI strategies for Catan.

Includes:
- RandomPlayer: Makes random valid moves (baseline)
- HeuristicPlayer: Uses simple heuristics to make decisions
"""

# Standard Library Imports
from typing import List, Optional

# Third Party Imports
import numpy as np

# Local Imports
from .base import Player
from engine.state import GameState
from engine.game import Action, ActionType


class RandomPlayer(Player):
    """
    A player that chooses random valid actions.

    Useful as a baseline and for fast simulations.
    """

    def __init__(self, name: str = "RandomPlayer", seed: Optional[int] = None):
        super().__init__(name)
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state: GameState, valid_actions: List[Action]) -> Action:
        """Choose a random valid action."""
        idx = self.rng.integers(0, len(valid_actions))
        return valid_actions[idx]


class HeuristicPlayer(Player):
    """
    A player that uses simple heuristics to make decisions.

    Strategy:
    - Setup: Place settlements on highest-probability intersections
    - Main game: Prioritize settlements > cities > roads
    - Prefer building on high-probability tiles
    - Use development cards strategically
    """

    def __init__(self, name: str = "HeuristicPlayer", seed: Optional[int] = None):
        super().__init__(name)
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state: GameState, valid_actions: List[Action]) -> Action:
        """Choose the best action using heuristics."""
        if not valid_actions:
            raise ValueError("No valid actions")

        # Get action by type
        actions_by_type = {}
        for action in valid_actions:
            action_type = action.action_type
            if action_type not in actions_by_type:
                actions_by_type[action_type] = []
            actions_by_type[action_type].append(action)

        # Setup phase - choose best settlement/road position
        if ActionType.PLACE_INITIAL_SETTLEMENT in actions_by_type:
            return self._choose_best_settlement(
                state, actions_by_type[ActionType.PLACE_INITIAL_SETTLEMENT]
            )

        if ActionType.PLACE_INITIAL_ROAD in actions_by_type:
            return self._choose_best_initial_road(
                state, actions_by_type[ActionType.PLACE_INITIAL_ROAD]
            )

        # Must roll dice
        if ActionType.ROLL_DICE in actions_by_type:
            return actions_by_type[ActionType.ROLL_DICE][0]

        # Must discard
        if ActionType.DISCARD_RESOURCES in actions_by_type:
            return self._choose_discard(
                state, actions_by_type[ActionType.DISCARD_RESOURCES]
            )

        # Must move robber
        if ActionType.MOVE_ROBBER in actions_by_type:
            return self._choose_robber_placement(
                state, actions_by_type[ActionType.MOVE_ROBBER]
            )

        # Must steal
        if ActionType.STEAL_RESOURCE in actions_by_type:
            return self._choose_steal_victim(
                state, actions_by_type[ActionType.STEAL_RESOURCE]
            )

        # Main phase - prioritize actions
        return self._choose_main_phase_action(state, actions_by_type)

    def _choose_best_settlement(
        self, state: GameState, actions: List[Action]
    ) -> Action:
        """Choose the best settlement location based on tile probabilities."""
        topology = state.topology

        best_action = None
        best_score = -1

        for action in actions:
            vertex = action.data[0]
            score = self._evaluate_vertex(state, vertex)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action or actions[0]

    def _evaluate_vertex(self, state: GameState, vertex: int) -> float:
        """Evaluate a vertex based on adjacent tile probabilities and resources."""
        topology = state.topology

        # Calculate probability score
        prob_score = 0
        resource_diversity = set()

        for tile in topology.vertex_tiles[vertex]:
            if tile >= 0:
                number = state.tile_numbers[tile]
                resource = state.tile_resources[tile]

                if resource < 5:  # Not desert
                    prob_score += topology.number_probabilities[number]
                    resource_diversity.add(resource)

        # Bonus for resource diversity
        diversity_bonus = len(resource_diversity) * 0.1

        # Bonus for ports
        port_bonus = 0
        if topology.vertex_port_type[vertex] >= 0:
            port_bonus = 0.2

        return prob_score + diversity_bonus + port_bonus

    def _choose_best_initial_road(
        self, state: GameState, actions: List[Action]
    ) -> Action:
        """Choose road placement during setup."""
        # For setup, just pick a random valid road
        idx = self.rng.integers(0, len(actions))
        return actions[idx]

    def _choose_discard(self, state: GameState, actions: List[Action]) -> Action:
        """Choose which resources to discard."""
        # Just pick the first valid discard option
        return actions[0]

    def _choose_robber_placement(
        self, state: GameState, actions: List[Action]
    ) -> Action:
        """Choose where to place the robber."""
        topology = state.topology
        player = state.current_player

        best_action = None
        best_score = -1

        for action in actions:
            tile = action.data[0]
            score = 0

            # Prefer tiles that hurt opponents but not self
            for vertex in topology.tile_vertices[tile]:
                owner = state.get_settlement_owner(vertex)
                if owner >= 0 and owner != player:
                    # Hurt opponent
                    score += topology.number_probabilities[state.tile_numbers[tile]]
                    # Prefer tiles where opponent has resources to steal
                    if state.get_player_total_resources(owner) > 0:
                        score += 0.1
                elif owner == player:
                    # Don't hurt self
                    score -= topology.number_probabilities[state.tile_numbers[tile]] * 2

            if score > best_score:
                best_score = score
                best_action = action

        return best_action or actions[0]

    def _choose_steal_victim(self, state: GameState, actions: List[Action]) -> Action:
        """Choose who to steal from."""
        # Prefer stealing from the leader
        best_action = None
        best_vp = -1

        for action in actions:
            victim = action.data[0]
            if victim >= 0:
                vp = state.calculate_victory_points(victim)
                resources = state.get_player_total_resources(victim)
                score = vp + resources * 0.1

                if score > best_vp:
                    best_vp = score
                    best_action = action

        return best_action or actions[0]

    def _choose_main_phase_action(
        self, state: GameState, actions_by_type: dict
    ) -> Action:
        """Choose the best action during main phase."""
        player = state.current_player

        # Priority 1: Build settlement if possible (high VP value)
        if ActionType.BUILD_SETTLEMENT in actions_by_type:
            return self._choose_best_settlement(
                state, actions_by_type[ActionType.BUILD_SETTLEMENT]
            )

        # Priority 2: Build city if possible
        if ActionType.BUILD_CITY in actions_by_type:
            actions = actions_by_type[ActionType.BUILD_CITY]
            # Choose city on highest-probability vertex
            best_action = None
            best_score = -1
            for action in actions:
                vertex = action.data[0]
                score = self._evaluate_vertex(state, vertex)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action or actions[0]

        # Priority 3: Build roads to expand
        if ActionType.BUILD_ROAD in actions_by_type:
            # Only build roads if it helps us get to a good settlement spot
            if state.settlements_built[player] < 5:
                roads = actions_by_type[ActionType.BUILD_ROAD]
                if roads:
                    return roads[self.rng.integers(0, len(roads))]

        # Priority 4: Buy dev card if we have extra resources
        if ActionType.BUY_DEV_CARD in actions_by_type:
            return actions_by_type[ActionType.BUY_DEV_CARD][0]

        # Priority 5: Play knight if we have one
        if ActionType.PLAY_KNIGHT in actions_by_type:
            return actions_by_type[ActionType.PLAY_KNIGHT][0]

        # Priority 6: Trade with bank if beneficial
        if ActionType.TRADE_WITH_BANK in actions_by_type:
            # Trade to get resources we need
            trades = actions_by_type[ActionType.TRADE_WITH_BANK]
            if trades:
                # Simple heuristic: just do a random trade
                return trades[self.rng.integers(0, len(trades))]

        # Default: End turn
        if ActionType.END_TURN in actions_by_type:
            return actions_by_type[ActionType.END_TURN][0]

        # Fallback: random action
        all_actions = []
        for actions in actions_by_type.values():
            all_actions.extend(actions)
        return all_actions[self.rng.integers(0, len(all_actions))]
