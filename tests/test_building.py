"""Tests for building system."""

# Standard Library Imports

# Third Party Imports
import pytest
import numpy as np

# Local Imports
from engine.state import GameState, ROAD_COST, SETTLEMENT_COST, CITY_COST
from engine.building import (
    can_afford_road,
    can_afford_settlement,
    can_afford_city,
    get_valid_road_placements,
    get_valid_settlement_placements,
    get_valid_city_placements,
    build_road,
    build_settlement,
    build_city,
)


class TestAffordability:
    """Test resource affordability checks."""

    def test_cant_afford_empty_hand(self):
        """Test that empty hand can't afford anything."""
        state = GameState(num_players=4, seed=42)

        assert not can_afford_road(state, 0)
        assert not can_afford_settlement(state, 0)
        assert not can_afford_city(state, 0)

    def test_can_afford_road(self):
        """Test road affordability."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = ROAD_COST.copy()

        assert can_afford_road(state, 0)

    def test_can_afford_settlement(self):
        """Test settlement affordability."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = SETTLEMENT_COST.copy()

        assert can_afford_settlement(state, 0)

    def test_can_afford_city(self):
        """Test city affordability."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = CITY_COST.copy()

        assert can_afford_city(state, 0)


class TestSettlementPlacement:
    """Test settlement placement rules."""

    def test_setup_phase_placements(self):
        """Test that all vertices are available in setup phase."""
        state = GameState(num_players=4, seed=42)

        valid = get_valid_settlement_placements(state, 0, setup_phase=True)

        # Most vertices should be valid (54 total)
        assert len(valid) > 40

    def test_distance_rule(self):
        """Test that distance rule is enforced."""
        state = GameState(num_players=4, seed=42)

        # Place a settlement
        build_settlement(state, 0, 10, free=True)

        # Adjacent vertices should be invalid
        valid = get_valid_settlement_placements(state, 1, setup_phase=True)

        for neighbor in state.topology.vertex_neighbors[10]:
            if neighbor >= 0:
                assert neighbor not in valid

    def test_no_double_placement(self):
        """Test that can't build on occupied vertex."""
        state = GameState(num_players=4, seed=42)

        build_settlement(state, 0, 10, free=True)

        # Same player shouldn't be able to build there
        valid = get_valid_settlement_placements(state, 0, setup_phase=True)
        assert 10 not in valid

        # Other player shouldn't be able to build there
        valid = get_valid_settlement_placements(state, 1, setup_phase=True)
        assert 10 not in valid


class TestRoadPlacement:
    """Test road placement rules."""

    def test_road_needs_connection(self):
        """Test that roads must connect to player's buildings/roads."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = ROAD_COST.copy() * 5

        # Without any buildings, can't place roads
        valid = get_valid_road_placements(state, 0)
        assert len(valid) == 0

    def test_road_from_settlement(self):
        """Test that roads can be placed from settlements."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = ROAD_COST.copy() * 5

        # Place a settlement first
        build_settlement(state, 0, 10, free=True)

        valid = get_valid_road_placements(state, 0)

        # Should be able to place roads adjacent to settlement
        assert len(valid) > 0

        # All valid placements should be adjacent to vertex 10
        for edge in valid:
            v1, v2 = state.topology.edge_vertices[edge]
            assert v1 == 10 or v2 == 10


class TestCityUpgrade:
    """Test city upgrade rules."""

    def test_city_requires_settlement(self):
        """Test that cities can only upgrade settlements."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = CITY_COST.copy()

        # Without settlements, can't upgrade
        valid = get_valid_city_placements(state, 0)
        assert len(valid) == 0

    def test_city_upgrade_own_settlement(self):
        """Test can upgrade own settlement."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = CITY_COST.copy()

        build_settlement(state, 0, 10, free=True)

        valid = get_valid_city_placements(state, 0)
        assert 10 in valid

    def test_cant_upgrade_other_settlement(self):
        """Test can't upgrade opponent's settlement."""
        state = GameState(num_players=4, seed=42)
        state.resources[1] = CITY_COST.copy()

        # Player 0 builds settlement
        build_settlement(state, 0, 10, free=True)

        # Player 1 shouldn't be able to upgrade it
        valid = get_valid_city_placements(state, 1)
        assert 10 not in valid


class TestBuildingExecution:
    """Test building execution."""

    def test_build_road_costs_resources(self):
        """Test that building a road costs resources."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = ROAD_COST.copy()

        build_settlement(state, 0, 10, free=True)

        # Get a valid edge
        valid = get_valid_road_placements(state, 0)
        edge = valid[0]

        # Build road
        success = build_road(state, 0, edge)

        assert success
        assert np.all(state.resources[0] == 0)

    def test_build_settlement_updates_state(self):
        """Test that building settlement updates game state."""
        state = GameState(num_players=4, seed=42)

        initial_count = state.settlements_built[0]

        build_settlement(state, 0, 10, free=True)

        assert state.settlements_built[0] == initial_count + 1
        assert state.vertices[10] == 0

    def test_build_city_updates_state(self):
        """Test that building city updates game state."""
        state = GameState(num_players=4, seed=42)
        state.resources[0] = CITY_COST.copy()

        build_settlement(state, 0, 10, free=True)

        success = build_city(state, 0, 10)

        assert success
        assert state.cities_built[0] == 1
        assert state.settlements_built[0] == 0  # Settlement converted
        assert state.vertices[10] == 4  # City marker for player 0
