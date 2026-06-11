"""
Development cards for Catan.

Card types:
- Knight (14): Move robber, steal resource
- Victory Point (5): Hidden VP
- Road Building (2): Build 2 free roads
- Year of Plenty (2): Take any 2 resources from bank
- Monopoly (2): Take all of one resource type from other players

Handles:
- Buying development cards
- Playing development cards
- Largest army tracking
"""

# Standard Library Imports

# Third Party Imports
import numpy as np

# Local Imports
from .state import (
    GameState,
    DEV_CARD_COST,
    DEV_KNIGHT,
    DEV_VICTORY_POINT,
    DEV_ROAD_BUILDING,
    DEV_YEAR_OF_PLENTY,
    DEV_MONOPOLY,
)
from .resources import can_afford, pay_cost


def can_buy_dev_card(state: GameState, player: int) -> bool:
    """Check if player can buy a development card."""
    # Check affordability
    if not can_afford(state, player, DEV_CARD_COST):
        return False

    # Check if deck has cards
    if state.dev_deck_idx >= len(state.dev_deck):
        return False

    return True


def buy_dev_card(state: GameState, player: int) -> int:
    """
    Buy a development card.

    Returns:
        Card type bought, or -1 if failed
    """
    if not can_buy_dev_card(state, player):
        return -1

    pay_cost(state, player, DEV_CARD_COST)

    card_type = state.dev_deck[state.dev_deck_idx]
    state.dev_deck_idx += 1
    state.dev_cards[player, card_type] += 1
    state.dev_cards_bought_this_turn[card_type] += 1

    return card_type


def can_play_dev_card(state: GameState, player: int, card_type: int) -> bool:
    """Check if player can play a specific development card."""
    # Can only play one dev card per turn
    if state.dev_card_played_this_turn:
        return False

    # Can't play victory point cards (they're auto-scored)
    if card_type == DEV_VICTORY_POINT:
        return False

    # Must have a card of this type that wasn't bought this turn
    available = state.dev_cards[player, card_type] - state.dev_cards_bought_this_turn[card_type]
    if available <= 0:
        return False

    return True


def play_knight(state: GameState, player: int) -> bool:
    """
    Play a knight card.

    Note: Caller must handle robber movement and stealing.

    Returns:
        True if successful
    """
    if not can_play_dev_card(state, player, DEV_KNIGHT):
        return False

    state.dev_cards[player, DEV_KNIGHT] -= 1
    state.knights_played[player] += 1
    state.dev_card_played_this_turn = True

    # Update largest army
    _update_largest_army(state)

    return True


def play_road_building(state: GameState, player: int) -> bool:
    """
    Play a road building card.

    Note: Caller must handle placing the 2 roads.

    Returns:
        True if successful
    """
    if not can_play_dev_card(state, player, DEV_ROAD_BUILDING):
        return False

    state.dev_cards[player, DEV_ROAD_BUILDING] -= 1
    state.dev_card_played_this_turn = True

    return True


def play_year_of_plenty(
    state: GameState, player: int, resource1: int, resource2: int
) -> bool:
    """
    Play a Year of Plenty card and take 2 resources.

    Args:
        state: Current game state
        player: Player index
        resource1: First resource to take (0-4)
        resource2: Second resource to take (0-4)

    Returns:
        True if successful
    """
    if not can_play_dev_card(state, player, DEV_YEAR_OF_PLENTY):
        return False

    # Check bank has resources
    resources_needed = np.zeros(5, dtype=np.int16)
    resources_needed[resource1] += 1
    resources_needed[resource2] += 1

    if not np.all(state.bank >= resources_needed):
        return False

    state.dev_cards[player, DEV_YEAR_OF_PLENTY] -= 1
    state.dev_card_played_this_turn = True

    state.bank -= resources_needed
    state.resources[player] += resources_needed

    return True


def play_monopoly(state: GameState, player: int, resource_type: int) -> int:
    """
    Play a Monopoly card and take all of one resource.

    Args:
        state: Current game state
        player: Player index
        resource_type: Resource to monopolize (0-4)

    Returns:
        Total resources taken, or -1 if failed
    """
    if not can_play_dev_card(state, player, DEV_MONOPOLY):
        return -1

    if resource_type < 0 or resource_type > 4:
        return -1

    state.dev_cards[player, DEV_MONOPOLY] -= 1
    state.dev_card_played_this_turn = True

    total_stolen = 0
    for other_player in range(state.num_players):
        if other_player != player:
            amount = state.resources[other_player, resource_type]
            state.resources[other_player, resource_type] = 0
            state.resources[player, resource_type] += amount
            total_stolen += amount

    return total_stolen


def _update_largest_army(state: GameState):
    """Update largest army holder after playing a knight."""
    for player in range(state.num_players):
        knights = state.knights_played[player]

        if knights >= 3:
            if knights > state.largest_army_size:
                state.largest_army_size = knights
                state.largest_army_player = player
