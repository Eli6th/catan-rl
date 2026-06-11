"""
Robber mechanics for Catan.

Handles:
- 7-roll discard rule (players with >7 cards discard half)
- Moving the robber
- Stealing from adjacent players
"""

# Standard Library Imports
from typing import List, Tuple

# Third Party Imports
import numpy as np

# Local Imports
from .state import GameState


def get_players_who_must_discard(state: GameState) -> List[Tuple[int, int]]:
    """
    Get list of players who must discard on a 7 roll.

    Returns:
        List of (player_idx, cards_to_discard) tuples
    """
    result = []
    for player in range(state.num_players):
        total = state.get_player_total_resources(player)
        if total > 7:
            discard_count = total // 2
            result.append((player, discard_count))
    return result


def discard_resources(state: GameState, player: int, resources: np.ndarray) -> bool:
    """
    Discard resources from a player's hand.

    Args:
        state: Current game state
        player: Player index
        resources: Array of resources to discard

    Returns:
        True if successful
    """
    # Validate player has these resources
    if not np.all(state.resources[player] >= resources):
        return False

    state.resources[player] -= resources
    state.bank += resources
    return True


def get_valid_robber_placements(state: GameState) -> np.ndarray:
    """
    Get all valid tiles where the robber can be moved.

    The robber cannot stay on its current tile.

    Returns:
        Array of valid tile indices
    """
    valid = []
    for tile in range(19):
        if tile != state.robber_tile:
            valid.append(tile)
    return np.array(valid, dtype=np.int8)


def get_stealable_players(state: GameState, tile: int) -> List[int]:
    """
    Get players who can be stolen from when robber is placed on a tile.

    Returns:
        List of player indices with settlements/cities on this tile
    """
    topology = state.topology
    players = set()

    for vertex in topology.tile_vertices[tile]:
        owner = state.get_settlement_owner(vertex)
        if owner >= 0 and owner != state.current_player:
            if state.get_player_total_resources(owner) > 0:
                players.add(owner)

    return list(players)


def move_robber(state: GameState, tile: int) -> bool:
    """
    Move the robber to a new tile.

    Args:
        state: Current game state
        tile: Target tile index

    Returns:
        True if successful
    """
    if tile < 0 or tile >= 19:
        return False

    if tile == state.robber_tile:
        return False

    state.robber_tile = tile
    return True


def steal_random_resource(state: GameState, victim: int) -> int:
    """
    Steal a random resource from a player.

    Args:
        state: Current game state
        victim: Player to steal from

    Returns:
        Resource type stolen (0-4), or -1 if victim has no resources
    """
    thief = state.current_player

    # Get victim's resources
    victim_resources = state.resources[victim]
    total = np.sum(victim_resources)

    if total == 0:
        return -1

    # Pick a random resource
    resource_idx = state.rng.integers(0, total)

    # Find which resource type this corresponds to
    cumulative = 0
    for resource_type in range(5):
        cumulative += victim_resources[resource_type]
        if resource_idx < cumulative:
            # Steal this resource
            state.resources[victim, resource_type] -= 1
            state.resources[thief, resource_type] += 1
            return resource_type

    return -1
