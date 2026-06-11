"""
Resource system for Catan - distribution and bank management.

Handles:
- Resource distribution on dice rolls
- Bank limits and constraints
"""

# Standard Library Imports
from typing import Tuple

# Third Party Imports
import numpy as np

# Local Imports
from .state import GameState, RESOURCE_DESERT


def distribute_resources(state: GameState, dice_roll: int) -> np.ndarray:
    """
    Distribute resources based on dice roll.

    Args:
        state: Current game state
        dice_roll: Sum of two dice (2-12)

    Returns:
        Array of shape (num_players, 5) with resources gained per player
    """
    if dice_roll == 7:
        # No resources on 7 (robber is handled separately)
        return np.zeros((state.num_players, 5), dtype=np.int16)

    topology = state.topology
    gained = np.zeros((state.num_players, 5), dtype=np.int16)

    # Find all tiles with matching number
    matching_tiles = np.where(state.tile_numbers == dice_roll)[0]

    for tile_idx in matching_tiles:
        # Skip if robber is on this tile
        if tile_idx == state.robber_tile:
            continue

        resource_type = state.tile_resources[tile_idx]
        if resource_type == RESOURCE_DESERT:
            continue

        # Check all vertices of this tile
        for vertex in topology.tile_vertices[tile_idx]:
            val = state.vertices[vertex]
            if val < 0:
                continue

            player = val % 4
            amount = 2 if val >= 4 else 1  # City gives 2, settlement gives 1

            # Check bank availability
            available = state.bank[resource_type]
            amount = min(amount, available)

            if amount > 0:
                gained[player, resource_type] += amount

    # Apply gains (respecting bank limits)
    for resource_type in range(5):
        total_needed = np.sum(gained[:, resource_type])
        available = state.bank[resource_type]

        if total_needed > available:
            # Proportionally reduce (or just give nothing per Catan rules)
            # Standard Catan: if bank runs out, nobody gets that resource
            gained[:, resource_type] = 0
        else:
            state.bank[resource_type] -= total_needed
            state.resources[: state.num_players, resource_type] += gained[
                : state.num_players, resource_type
            ]

    return gained


def roll_dice(state: GameState) -> Tuple[int, int, int]:
    """
    Roll two dice.

    Returns:
        Tuple of (die1, die2, total)
    """
    die1 = int(state.rng.integers(1, 7))
    die2 = int(state.rng.integers(1, 7))
    return die1, die2, die1 + die2


def can_afford(state: GameState, player: int, cost: np.ndarray) -> bool:
    """Check if player can afford a cost."""
    return np.all(state.resources[player] >= cost)


def pay_cost(state: GameState, player: int, cost: np.ndarray) -> bool:
    """
    Deduct cost from player's resources.

    Returns:
        True if successful
    """
    if not can_afford(state, player, cost):
        return False

    state.resources[player] -= cost
    state.bank += cost
    return True


def give_resources(state: GameState, player: int, resources: np.ndarray) -> bool:
    """
    Give resources to a player from the bank.

    Returns:
        True if successful (bank had enough)
    """
    if not np.all(state.bank >= resources):
        return False

    state.bank -= resources
    state.resources[player] += resources
    return True


def transfer_resources(
    state: GameState, from_player: int, to_player: int, resources: np.ndarray
) -> bool:
    """
    Transfer resources between players.

    Returns:
        True if successful
    """
    if not np.all(state.resources[from_player] >= resources):
        return False

    state.resources[from_player] -= resources
    state.resources[to_player] += resources
    return True
