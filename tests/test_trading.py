"""
Tests for trading system: bank trades (4:1, 3:1, 2:1), player-to-player trades.

Port vertex assignments (fixed in BoardTopology):
  Port 0 (3:1 any):   vertices 0, 1
  Port 1 (3:1 any):   vertices 7, 10
  Port 2 (3:1 any):   vertices 12, 22
  Port 3 (3:1 any):   vertices 35, 36
  Port 4 (wheat 2:1): vertices 45, 46
  Port 5 (sheep 2:1): vertices 50, 53
  Port 6 (wood  2:1): vertices 47, 48
  Port 7 (brick 2:1): vertices 39, 40
  Port 8 (stone 2:1): vertices 27, 28
"""

import numpy as np
import pytest

from engine.state import GameState
from engine.board import (
    RESOURCE_WHEAT,
    RESOURCE_SHEEP,
    RESOURCE_WOOD,
    RESOURCE_BRICK,
    RESOURCE_STONE,
)
from engine.trading import (
    get_bank_trade_rate,
    can_trade_with_bank,
    trade_with_bank,
    get_possible_bank_trades,
    TradeOffer,
    execute_trade,
)


def make_state(num_players: int = 3, seed: int = 42) -> GameState:
    return GameState(num_players=num_players, seed=seed)


def place_settlement(state: GameState, player: int, vertex: int):
    """Place a settlement directly, bypassing normal rules."""
    state.vertices[vertex] = player
    state.settlements_built[player] += 1


# ---------------------------------------------------------------------------
# Bank trade rates
# ---------------------------------------------------------------------------

class TestBankTradeRate:
    def test_default_rate_is_4(self):
        state = make_state()
        # No port settlements
        rate = get_bank_trade_rate(state, 0, RESOURCE_WHEAT)
        assert rate == 4

    def test_any_port_gives_3(self):
        state = make_state()
        # Vertex 0 is port 0 (3:1 any)
        place_settlement(state, 0, 0)
        for resource in range(5):
            assert get_bank_trade_rate(state, 0, resource) == 3

    def test_specific_port_gives_2_for_matching_resource(self):
        state = make_state()
        # Vertex 45 is port 4 (wheat 2:1)
        place_settlement(state, 0, 45)
        assert get_bank_trade_rate(state, 0, RESOURCE_WHEAT) == 2

    def test_specific_port_still_4_for_other_resources(self):
        state = make_state()
        # Vertex 45 is port 4 (wheat 2:1) — no any-port bonus
        place_settlement(state, 0, 45)
        assert get_bank_trade_rate(state, 0, RESOURCE_SHEEP) == 4
        assert get_bank_trade_rate(state, 0, RESOURCE_WOOD) == 4

    def test_specific_port_overrides_any_port(self):
        state = make_state()
        # Both a 3:1 and a 2:1 port — 2:1 should win for the specific resource
        place_settlement(state, 0, 0)   # 3:1 any
        place_settlement(state, 0, 45)  # wheat 2:1
        assert get_bank_trade_rate(state, 0, RESOURCE_WHEAT) == 2
        # Other resources still get 3:1 from the any-port
        assert get_bank_trade_rate(state, 0, RESOURCE_SHEEP) == 3

    def test_other_players_dont_get_port(self):
        state = make_state()
        place_settlement(state, 0, 0)  # Player 0 has 3:1 any
        # Player 1 still gets 4:1
        assert get_bank_trade_rate(state, 1, RESOURCE_WHEAT) == 4

    def test_sheep_2to1_port(self):
        state = make_state()
        # Vertex 50 is port 5 (sheep 2:1)
        place_settlement(state, 0, 50)
        assert get_bank_trade_rate(state, 0, RESOURCE_SHEEP) == 2

    def test_wood_2to1_port(self):
        state = make_state()
        # Vertex 47 is port 6 (wood 2:1)
        place_settlement(state, 0, 47)
        assert get_bank_trade_rate(state, 0, RESOURCE_WOOD) == 2

    def test_brick_2to1_port(self):
        state = make_state()
        # Vertex 39 is port 7 (brick 2:1)
        place_settlement(state, 0, 39)
        assert get_bank_trade_rate(state, 0, RESOURCE_BRICK) == 2

    def test_stone_2to1_port(self):
        state = make_state()
        # Vertex 27 is port 8 (stone 2:1)
        place_settlement(state, 0, 27)
        assert get_bank_trade_rate(state, 0, RESOURCE_STONE) == 2


# ---------------------------------------------------------------------------
# can_trade_with_bank
# ---------------------------------------------------------------------------

class TestCanTradeWithBank:
    def test_valid_4to1_trade(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        assert can_trade_with_bank(state, 0, RESOURCE_WOOD, 4, RESOURCE_WHEAT)

    def test_same_resource_invalid(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        assert not can_trade_with_bank(state, 0, RESOURCE_WOOD, 4, RESOURCE_WOOD)

    def test_wrong_amount_invalid(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        assert not can_trade_with_bank(state, 0, RESOURCE_WOOD, 3, RESOURCE_WHEAT)

    def test_insufficient_resources_invalid(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 3  # need 4
        assert not can_trade_with_bank(state, 0, RESOURCE_WOOD, 4, RESOURCE_WHEAT)

    def test_bank_empty_invalid(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        state.bank[RESOURCE_WHEAT] = 0
        assert not can_trade_with_bank(state, 0, RESOURCE_WOOD, 4, RESOURCE_WHEAT)

    def test_port_trade_valid(self):
        state = make_state()
        place_settlement(state, 0, 0)  # 3:1 any
        state.resources[0, RESOURCE_WOOD] = 3
        assert can_trade_with_bank(state, 0, RESOURCE_WOOD, 3, RESOURCE_WHEAT)


# ---------------------------------------------------------------------------
# trade_with_bank
# ---------------------------------------------------------------------------

class TestTradeWithBank:
    def test_4to1_trade_deducts_resources(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        trade_with_bank(state, 0, RESOURCE_WOOD, RESOURCE_WHEAT)
        assert state.resources[0, RESOURCE_WOOD] == 0
        assert state.resources[0, RESOURCE_WHEAT] == 1

    def test_4to1_trade_updates_bank(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        initial_bank_wood = int(state.bank[RESOURCE_WOOD])
        initial_bank_wheat = int(state.bank[RESOURCE_WHEAT])
        trade_with_bank(state, 0, RESOURCE_WOOD, RESOURCE_WHEAT)
        assert state.bank[RESOURCE_WOOD] == initial_bank_wood + 4
        assert state.bank[RESOURCE_WHEAT] == initial_bank_wheat - 1

    def test_3to1_port_trade(self):
        state = make_state()
        place_settlement(state, 0, 0)  # 3:1 any
        state.resources[0, RESOURCE_WOOD] = 3
        result = trade_with_bank(state, 0, RESOURCE_WOOD, RESOURCE_WHEAT)
        assert result
        assert state.resources[0, RESOURCE_WOOD] == 0
        assert state.resources[0, RESOURCE_WHEAT] == 1

    def test_2to1_port_trade(self):
        state = make_state()
        place_settlement(state, 0, 45)  # wheat 2:1
        state.resources[0, RESOURCE_WHEAT] = 2
        result = trade_with_bank(state, 0, RESOURCE_WHEAT, RESOURCE_WOOD)
        assert result
        assert state.resources[0, RESOURCE_WHEAT] == 0
        assert state.resources[0, RESOURCE_WOOD] == 1

    def test_trade_fails_insufficient_resources(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 3  # need 4
        result = trade_with_bank(state, 0, RESOURCE_WOOD, RESOURCE_WHEAT)
        assert not result
        assert state.resources[0, RESOURCE_WOOD] == 3  # unchanged

    def test_trade_fails_bank_empty(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        state.bank[RESOURCE_WHEAT] = 0
        result = trade_with_bank(state, 0, RESOURCE_WOOD, RESOURCE_WHEAT)
        assert not result

    def test_trade_fails_same_resource(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        result = trade_with_bank(state, 0, RESOURCE_WOOD, RESOURCE_WOOD)
        assert not result


# ---------------------------------------------------------------------------
# get_possible_bank_trades
# ---------------------------------------------------------------------------

class TestGetPossibleBankTrades:
    def test_no_resources_no_trades(self):
        state = make_state()
        trades = get_possible_bank_trades(state, 0)
        assert len(trades) == 0

    def test_exact_4_resources_gives_trades(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        trades = get_possible_bank_trades(state, 0)
        # Should have 4 trades: wood for each of the other 4 resources
        assert len(trades) == 4
        for give_res, give_amt, recv_res in trades:
            assert give_res == RESOURCE_WOOD
            assert give_amt == 4
            assert recv_res != RESOURCE_WOOD

    def test_port_reduces_required_amount(self):
        state = make_state()
        place_settlement(state, 0, 0)  # 3:1 any
        state.resources[0, RESOURCE_WOOD] = 3
        trades = get_possible_bank_trades(state, 0)
        assert any(give_amt == 3 for _, give_amt, _ in trades)

    def test_bank_empty_excludes_that_resource(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 4
        state.bank[RESOURCE_WHEAT] = 0
        trades = get_possible_bank_trades(state, 0)
        # Should not include wheat as receive
        receive_resources = [recv_res for _, _, recv_res in trades]
        assert RESOURCE_WHEAT not in receive_resources


# ---------------------------------------------------------------------------
# Player-to-player trades (TradeOffer)
# ---------------------------------------------------------------------------

class TestPlayerToPlayerTrade:
    def test_valid_trade_executes(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 2
        state.resources[1, RESOURCE_WHEAT] = 2

        offering = np.zeros(5, dtype=np.int16)
        offering[RESOURCE_WOOD] = 1
        requesting = np.zeros(5, dtype=np.int16)
        requesting[RESOURCE_WHEAT] = 1

        offer = TradeOffer(0, 1, offering, requesting)
        result = execute_trade(state, offer, 1)

        assert result
        assert state.resources[0, RESOURCE_WOOD] == 1
        assert state.resources[0, RESOURCE_WHEAT] == 1
        assert state.resources[1, RESOURCE_WHEAT] == 1
        assert state.resources[1, RESOURCE_WOOD] == 1

    def test_trade_fails_offerer_lacks_resources(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 0
        state.resources[1, RESOURCE_WHEAT] = 2

        offering = np.zeros(5, dtype=np.int16)
        offering[RESOURCE_WOOD] = 1
        requesting = np.zeros(5, dtype=np.int16)
        requesting[RESOURCE_WHEAT] = 1

        offer = TradeOffer(0, 1, offering, requesting)
        result = execute_trade(state, offer, 1)

        assert not result
        assert state.resources[0, RESOURCE_WOOD] == 0  # unchanged

    def test_trade_fails_accepter_lacks_resources(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 2
        state.resources[1, RESOURCE_WHEAT] = 0

        offering = np.zeros(5, dtype=np.int16)
        offering[RESOURCE_WOOD] = 1
        requesting = np.zeros(5, dtype=np.int16)
        requesting[RESOURCE_WHEAT] = 1

        offer = TradeOffer(0, 1, offering, requesting)
        result = execute_trade(state, offer, 1)

        assert not result

    def test_targeted_trade_only_for_target(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 2
        state.resources[1, RESOURCE_WHEAT] = 2
        state.resources[2, RESOURCE_WHEAT] = 2

        offering = np.zeros(5, dtype=np.int16)
        offering[RESOURCE_WOOD] = 1
        requesting = np.zeros(5, dtype=np.int16)
        requesting[RESOURCE_WHEAT] = 1

        offer = TradeOffer(0, 1, offering, requesting)  # targeted at player 1
        # Player 2 cannot accept a trade targeted at player 1
        result = execute_trade(state, offer, 2)
        assert not result

    def test_open_trade_any_can_accept(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 2
        state.resources[2, RESOURCE_WHEAT] = 2

        offering = np.zeros(5, dtype=np.int16)
        offering[RESOURCE_WOOD] = 1
        requesting = np.zeros(5, dtype=np.int16)
        requesting[RESOURCE_WHEAT] = 1

        offer = TradeOffer(0, -1, offering, requesting)  # open offer
        result = execute_trade(state, offer, 2)
        assert result

    def test_cant_trade_with_self(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 2
        state.resources[0, RESOURCE_WHEAT] = 2

        offering = np.zeros(5, dtype=np.int16)
        offering[RESOURCE_WOOD] = 1
        requesting = np.zeros(5, dtype=np.int16)
        requesting[RESOURCE_WHEAT] = 1

        offer = TradeOffer(0, -1, offering, requesting)
        result = execute_trade(state, offer, 0)  # try to trade with self
        assert not result

    def test_trade_offer_validity(self):
        state = make_state()
        state.resources[0, RESOURCE_WOOD] = 2

        offering = np.zeros(5, dtype=np.int16)
        offering[RESOURCE_WOOD] = 1
        requesting = np.zeros(5, dtype=np.int16)
        requesting[RESOURCE_WHEAT] = 1

        offer = TradeOffer(0, -1, offering, requesting)
        assert offer.is_valid(state)

    def test_invalid_trade_offer_no_resources(self):
        state = make_state()
        # No wood to offer

        offering = np.zeros(5, dtype=np.int16)
        offering[RESOURCE_WOOD] = 1
        requesting = np.zeros(5, dtype=np.int16)
        requesting[RESOURCE_WHEAT] = 1

        offer = TradeOffer(0, -1, offering, requesting)
        assert not offer.is_valid(state)
