"""
Trading system for Catan.

Handles:
- Bank trades (4:1 default, 3:1 with any port, 2:1 with specific port)
- Player-to-player trading
"""

import numpy as np
from typing import Optional, Tuple, List
from .state import GameState


def get_bank_trade_rate(state: GameState, player: int, resource_type: int) -> int:
    """
    Get the best trade rate for a resource type for this player.

    Returns:
        Trade rate (4, 3, or 2)
    """
    # Check for 2:1 port for this resource
    for vertex in range(54):
        if state.get_settlement_owner(vertex) == player:
            port_type = state.topology.vertex_port_type[vertex]
            if port_type == resource_type + 1:  # 2:1 for specific resource
                return 2

    # Check for 3:1 any port
    for vertex in range(54):
        if state.get_settlement_owner(vertex) == player:
            port_type = state.topology.vertex_port_type[vertex]
            if port_type == 0:  # 3:1 any
                return 3

    # Default 4:1
    return 4


def can_trade_with_bank(
    state: GameState,
    player: int,
    give_resource: int,
    give_amount: int,
    receive_resource: int,
) -> bool:
    """
    Check if a bank trade is valid.

    Args:
        state: Current game state
        player: Player index
        give_resource: Resource type to give (0-4)
        give_amount: Amount to give (should match trade rate)
        receive_resource: Resource type to receive (0-4)

    Returns:
        True if trade is valid
    """
    if give_resource == receive_resource:
        return False

    trade_rate = get_bank_trade_rate(state, player, give_resource)

    if give_amount != trade_rate:
        return False

    if state.resources[player, give_resource] < give_amount:
        return False

    if state.bank[receive_resource] < 1:
        return False

    return True


def trade_with_bank(
    state: GameState, player: int, give_resource: int, receive_resource: int
) -> bool:
    """
    Execute a bank trade at the best available rate.

    Returns:
        True if successful
    """
    trade_rate = get_bank_trade_rate(state, player, give_resource)

    if not can_trade_with_bank(
        state, player, give_resource, trade_rate, receive_resource
    ):
        return False

    state.resources[player, give_resource] -= trade_rate
    state.bank[give_resource] += trade_rate
    state.resources[player, receive_resource] += 1
    state.bank[receive_resource] -= 1

    return True


def get_possible_bank_trades(
    state: GameState, player: int
) -> List[Tuple[int, int, int]]:
    """
    Get all possible bank trades for a player.

    Returns:
        List of (give_resource, give_amount, receive_resource) tuples
    """
    trades = []

    for give_res in range(5):
        trade_rate = get_bank_trade_rate(state, player, give_res)

        if state.resources[player, give_res] >= trade_rate:
            for receive_res in range(5):
                if receive_res != give_res and state.bank[receive_res] > 0:
                    trades.append((give_res, trade_rate, receive_res))

    return trades


class TradeOffer:
    """Represents a trade offer between players."""

    def __init__(
        self,
        from_player: int,
        to_player: int,
        offering: np.ndarray,
        requesting: np.ndarray,
    ):
        """
        Create a trade offer.

        Args:
            from_player: Player making the offer
            to_player: Player receiving the offer (-1 for open offer)
            offering: Resources being offered
            requesting: Resources being requested
        """
        self.from_player = from_player
        self.to_player = to_player
        self.offering = offering.copy()
        self.requesting = requesting.copy()

    def is_valid(self, state: GameState) -> bool:
        """Check if the trade offer is valid."""
        # Check offering player has resources
        if not np.all(state.resources[self.from_player] >= self.offering):
            return False

        # If targeted, check target has resources
        if self.to_player >= 0:
            if not np.all(state.resources[self.to_player] >= self.requesting):
                return False

        return True

    def can_accept(self, state: GameState, accepting_player: int) -> bool:
        """Check if a player can accept this offer."""
        if self.to_player >= 0 and self.to_player != accepting_player:
            return False

        if accepting_player == self.from_player:
            return False

        if not np.all(state.resources[accepting_player] >= self.requesting):
            return False

        if not np.all(state.resources[self.from_player] >= self.offering):
            return False

        return True


def execute_trade(state: GameState, offer: TradeOffer, accepting_player: int) -> bool:
    """
    Execute a player-to-player trade.

    Returns:
        True if successful
    """
    if not offer.can_accept(state, accepting_player):
        return False

    # Transfer resources
    state.resources[offer.from_player] -= offer.offering
    state.resources[accepting_player] += offer.offering

    state.resources[accepting_player] -= offer.requesting
    state.resources[offer.from_player] += offer.requesting

    return True
