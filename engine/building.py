"""
Building system for Catan - placement validation and construction.

Handles:
- Settlement placement (including distance rule)
- Road placement (connectivity)
- City upgrades
- Longest road calculation
"""

# Standard Library Imports
from typing import List, Set

# Third Party Imports
import numpy as np

# Local Imports
from .state import GameState, ROAD_COST, SETTLEMENT_COST, CITY_COST


def can_afford_road(state: GameState, player: int) -> bool:
    """Check if player can afford a road."""
    return np.all(state.resources[player] >= ROAD_COST)


def can_afford_settlement(state: GameState, player: int) -> bool:
    """Check if player can afford a settlement."""
    return np.all(state.resources[player] >= SETTLEMENT_COST)


def can_afford_city(state: GameState, player: int) -> bool:
    """Check if player can afford a city."""
    return np.all(state.resources[player] >= CITY_COST)


def get_valid_road_placements(
    state: GameState, player: int, free_placement: bool = False
) -> np.ndarray:
    """
    Get all valid edge indices where a player can build a road.

    Args:
        state: Current game state
        player: Player index
        free_placement: If True, skip affordability check (for road building card)

    Returns:
        Array of valid edge indices
    """
    if not free_placement and not can_afford_road(state, player):
        return np.array([], dtype=np.int8)

    if state.roads_built[player] >= state.max_roads:
        return np.array([], dtype=np.int8)

    valid = []
    topology = state.topology

    for edge_idx in range(72):
        if state.edges[edge_idx] >= 0:  # Already has a road
            continue

        # Check if connected to player's road or settlement
        v1, v2 = topology.edge_vertices[edge_idx]

        # Check if either vertex has player's settlement/city
        connected = False
        for v in [v1, v2]:
            owner = state.get_settlement_owner(v)
            if owner == player:
                connected = True
                break

            # Check if any adjacent edge has player's road
            for adj_edge in topology.vertex_edges[v]:
                if adj_edge >= 0 and state.edges[adj_edge] == player:
                    # Make sure not blocked by opponent's settlement
                    if owner < 0 or owner == player:
                        connected = True
                        break

        if connected:
            valid.append(edge_idx)

    return np.array(valid, dtype=np.int8)


def get_valid_settlement_placements(
    state: GameState, player: int, setup_phase: bool = False
) -> np.ndarray:
    """
    Get all valid vertex indices where a player can build a settlement.

    Args:
        state: Current game state
        player: Player index
        setup_phase: If True, don't require road connectivity

    Returns:
        Array of valid vertex indices
    """
    if not setup_phase and not can_afford_settlement(state, player):
        return np.array([], dtype=np.int8)

    if state.settlements_built[player] >= state.max_settlements:
        return np.array([], dtype=np.int8)

    valid = []
    topology = state.topology

    for vertex_idx in range(54):
        if state.vertices[vertex_idx] >= 0:  # Already has a building
            continue

        # Distance rule: no adjacent settlements
        neighbors = topology.vertex_neighbors[vertex_idx]
        too_close = False
        for neighbor in neighbors:
            if neighbor >= 0 and state.vertices[neighbor] >= 0:
                too_close = True
                break

        if too_close:
            continue

        # Must be connected to player's road (unless setup phase)
        if not setup_phase:
            connected = False
            for edge in topology.vertex_edges[vertex_idx]:
                if edge >= 0 and state.edges[edge] == player:
                    connected = True
                    break

            if not connected:
                continue

        valid.append(vertex_idx)

    return np.array(valid, dtype=np.int8)


def get_valid_city_placements(state: GameState, player: int) -> np.ndarray:
    """
    Get all valid vertex indices where a player can upgrade to city.

    Returns:
        Array of valid vertex indices
    """
    if not can_afford_city(state, player):
        return np.array([], dtype=np.int8)

    if state.cities_built[player] >= state.max_cities:
        return np.array([], dtype=np.int8)

    valid = []

    for vertex_idx in range(54):
        # Must be player's settlement (not already a city)
        val = state.vertices[vertex_idx]
        if val == player:  # 0-3 = settlement by player
            valid.append(vertex_idx)

    return np.array(valid, dtype=np.int8)


def build_road(
    state: GameState, player: int, edge_idx: int, free: bool = False
) -> bool:
    """
    Build a road at the specified edge.

    Args:
        state: Current game state
        player: Player index
        edge_idx: Edge to build on
        free: If True, don't charge resources

    Returns:
        True if successful
    """
    if edge_idx < 0 or edge_idx >= 72:
        return False

    if state.edges[edge_idx] >= 0:
        return False

    if not free:
        if not can_afford_road(state, player):
            return False
        state.resources[player] -= ROAD_COST
        state.bank += ROAD_COST

    state.edges[edge_idx] = player
    state.roads_built[player] += 1

    # Update longest road
    _update_longest_road(state)

    return True


def build_settlement(
    state: GameState, player: int, vertex_idx: int, free: bool = False
) -> bool:
    """
    Build a settlement at the specified vertex.

    Args:
        state: Current game state
        player: Player index
        vertex_idx: Vertex to build on
        free: If True, don't charge resources (setup phase)

    Returns:
        True if successful
    """
    if vertex_idx < 0 or vertex_idx >= 54:
        return False

    if state.vertices[vertex_idx] >= 0:
        return False

    # Distance rule
    for neighbor in state.topology.vertex_neighbors[vertex_idx]:
        if neighbor >= 0 and state.vertices[neighbor] >= 0:
            return False

    if not free:
        if not can_afford_settlement(state, player):
            return False
        state.resources[player] -= SETTLEMENT_COST
        state.bank += SETTLEMENT_COST

    state.vertices[vertex_idx] = player
    state.settlements_built[player] += 1

    # Update longest road (settlement might break opponent's road)
    _update_longest_road(state)

    return True


def build_city(state: GameState, player: int, vertex_idx: int) -> bool:
    """
    Upgrade a settlement to a city.

    Returns:
        True if successful
    """
    if vertex_idx < 0 or vertex_idx >= 54:
        return False

    # Must be player's settlement
    if state.vertices[vertex_idx] != player:
        return False

    if not can_afford_city(state, player):
        return False

    state.resources[player] -= CITY_COST
    state.bank += CITY_COST

    state.vertices[vertex_idx] = player + 4  # 4-7 = city
    state.settlements_built[player] -= 1
    state.cities_built[player] += 1

    return True


def _calculate_longest_road_for_player(state: GameState, player: int) -> int:
    """
    Calculate the longest road length for a player using DFS.

    Returns:
        Length of longest road
    """
    topology = state.topology

    # Find all edges owned by player
    player_edges = set(np.where(state.edges == player)[0])
    if len(player_edges) < 5:  # Need at least 5 for longest road
        return len(player_edges)

    # Build adjacency for player's roads
    # Two edges are adjacent if they share a vertex not blocked by opponent
    def get_adjacent_edges(edge: int) -> List[int]:
        adjacent = []
        v1, v2 = topology.edge_vertices[edge]

        for v in [v1, v2]:
            # Check if vertex is blocked by opponent's settlement
            owner = state.get_settlement_owner(v)
            if owner >= 0 and owner != player:
                continue

            # Find other player edges at this vertex
            for adj_edge in topology.vertex_edges[v]:
                if adj_edge >= 0 and adj_edge != edge and adj_edge in player_edges:
                    adjacent.append(adj_edge)

        return adjacent

    # DFS to find longest path
    def dfs(edge: int, visited: Set[int]) -> int:
        max_length = 1
        visited.add(edge)

        for adj in get_adjacent_edges(edge):
            if adj not in visited:
                length = 1 + dfs(adj, visited)
                max_length = max(max_length, length)

        visited.remove(edge)
        return max_length

    longest = 0
    for start_edge in player_edges:
        length = dfs(start_edge, set())
        longest = max(longest, length)

    return longest


def _update_longest_road(state: GameState):
    """Update longest road holder after road building."""
    for player in range(state.num_players):
        length = _calculate_longest_road_for_player(state, player)

        if length >= 5:
            if length > state.longest_road_length:
                state.longest_road_length = length
                state.longest_road_player = player
            elif (
                length == state.longest_road_length
                and state.longest_road_player == player
            ):
                # Player extends their own longest road
                pass

    # Check if current holder still has longest road
    if state.longest_road_player >= 0:
        current_length = _calculate_longest_road_for_player(
            state, state.longest_road_player
        )

        if current_length < state.longest_road_length:
            # Someone else might have longer now
            state.longest_road_length = 0
            state.longest_road_player = -1

            for player in range(state.num_players):
                length = _calculate_longest_road_for_player(state, player)
                if length >= 5 and length > state.longest_road_length:
                    state.longest_road_length = length
                    state.longest_road_player = player
