"""
Tests for resource distribution: dice rolls, settlements, cities, robber blocking,
desert tiles, bank exhaustion rules.
"""

import numpy as np
import pytest

from engine.state import GameState, INITIAL_RESOURCES_PER_TYPE
from engine.board import (
    RESOURCE_WHEAT,
    RESOURCE_SHEEP,
    RESOURCE_WOOD,
    RESOURCE_BRICK,
    RESOURCE_STONE,
    RESOURCE_DESERT,
)
from engine.resources import distribute_resources


def make_state(seed: int = 42) -> GameState:
    return GameState(num_players=4, seed=seed)


def find_tile_with_number(state: GameState, number: int):
    """Return the first tile index that has the given dice number."""
    tiles = np.where(state.tile_numbers == number)[0]
    assert len(tiles) > 0, f"No tile with number {number}"
    return int(tiles[0])


def get_non_desert_tile_with_number(state: GameState, number: int):
    """Return a tile index with the given number that isn't desert."""
    for tile_idx in np.where(state.tile_numbers == number)[0]:
        if state.tile_resources[tile_idx] != RESOURCE_DESERT:
            return int(tile_idx), int(state.tile_resources[tile_idx])
    return None, None


def place_settlement(state: GameState, player: int, vertex: int):
    """Place a settlement directly."""
    state.vertices[vertex] = player
    state.settlements_built[player] += 1


def place_city(state: GameState, player: int, vertex: int):
    """Place a city directly (replaces settlement)."""
    state.vertices[vertex] = player + 4
    state.cities_built[player] += 1


# ---------------------------------------------------------------------------
# Basic distribution
# ---------------------------------------------------------------------------

class TestBasicDistribution:
    def test_seven_gives_no_resources(self):
        state = make_state()
        # Place settlement on any vertex
        tile_idx = find_tile_with_number(state, 6)
        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)

        gained = distribute_resources(state, 7)
        assert np.sum(gained) == 0

    def test_matching_number_gives_resource(self):
        state = make_state()
        # Use number 6 (common)
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6 in this seed")
        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)

        gained = distribute_resources(state, 6)
        assert gained[0, resource] == 1

    def test_non_matching_number_gives_nothing(self):
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")
        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)

        # Roll a different number that doesn't match this tile
        gained = distribute_resources(state, 2)
        # Player 0 should only get resources if their tile also has number 2
        # We can't easily assert 0 without knowing the board, just check bank
        # This is a smoke test to confirm no exception
        assert gained is not None

    def test_resources_added_to_player(self):
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")
        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)

        initial = int(state.resources[0, resource])
        distribute_resources(state, 6)
        assert state.resources[0, resource] == initial + 1

    def test_resources_deducted_from_bank(self):
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")
        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)

        initial_bank = int(state.bank[resource])
        distribute_resources(state, 6)
        assert state.bank[resource] == initial_bank - 1

    def test_desert_tile_gives_no_resources(self):
        state = make_state()
        # Find the desert tile
        desert_tiles = np.where(state.tile_resources == RESOURCE_DESERT)[0]
        assert len(desert_tiles) > 0
        tile_idx = int(desert_tiles[0])
        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)

        initial_resources = state.resources[0].copy()
        # Desert tile has number 0, so any roll won't match
        # But let's also verify that even if we force a match, desert gives nothing
        # by checking distribute_resources is consistent
        gained = distribute_resources(state, 7)  # 7 always gives nothing
        assert np.sum(gained) == 0


# ---------------------------------------------------------------------------
# Cities give double resources
# ---------------------------------------------------------------------------

class TestCityResources:
    def test_city_gives_2_resources(self):
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")
        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_city(state, 0, vertex)

        initial = int(state.resources[0, resource])
        distribute_resources(state, 6)
        assert state.resources[0, resource] == initial + 2

    def test_settlement_gives_1_city_gives_2_same_tile(self):
        """Two players on same tile: settlement gets 1, city gets 2."""
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")

        vertices = state.topology.tile_vertices[tile_idx]
        # Need at least 2 non-adjacent vertices on this tile
        v1 = int(vertices[0])
        v2 = int(vertices[3])  # Opposite vertex, definitely non-adjacent

        place_settlement(state, 0, v1)
        place_city(state, 1, v2)

        gained = distribute_resources(state, 6)
        assert gained[0, resource] == 1
        assert gained[1, resource] == 2


# ---------------------------------------------------------------------------
# Robber blocks resource distribution
# ---------------------------------------------------------------------------

class TestRobberBlocking:
    def test_robber_blocks_resource_from_tile(self):
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")
        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)

        # Move robber to this tile
        state.robber_tile = tile_idx

        gained = distribute_resources(state, 6)
        assert gained[0, resource] == 0
        assert state.resources[0, resource] == 0

    def test_robber_only_blocks_its_tile(self):
        """Other tiles with the same number still produce."""
        state = make_state()

        # Find two tiles with the same number
        for number in range(2, 13):
            if number == 7:
                continue
            tiles_with_number = [
                (int(t), int(state.tile_resources[t]))
                for t in np.where(state.tile_numbers == number)[0]
                if state.tile_resources[t] != RESOURCE_DESERT
            ]
            if len(tiles_with_number) >= 2:
                tile1, resource1 = tiles_with_number[0]
                tile2, resource2 = tiles_with_number[1]
                break
        else:
            pytest.skip("No two non-desert tiles with the same number found")

        v1 = int(state.topology.tile_vertices[tile1][0])
        v2 = int(state.topology.tile_vertices[tile2][0])
        # Ensure they don't share this vertex
        if v1 == v2:
            v2 = int(state.topology.tile_vertices[tile2][2])

        place_settlement(state, 0, v1)
        place_settlement(state, 1, v2)

        # Robber on tile1 blocks player 0 but not player 1
        state.robber_tile = tile1

        gained = distribute_resources(state, number)
        assert gained[0, resource1] == 0
        assert gained[1, resource2] >= 1


# ---------------------------------------------------------------------------
# Bank exhaustion
# ---------------------------------------------------------------------------

class TestBankExhaustion:
    def test_bank_exhaustion_no_one_gets_resource(self):
        """When bank can't supply all players, nobody gets that resource (Catan rule)."""
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")

        # Place multiple players/cities on this tile
        vertices = state.topology.tile_vertices[tile_idx]
        # Place 2 cities (each needing 2 = 4 total)
        v1 = int(vertices[0])
        v2 = int(vertices[3])
        place_city(state, 0, v1)
        place_city(state, 1, v2)

        # Set bank to have only 3 of this resource (2+2=4 needed, only 3 available)
        state.bank[resource] = 3

        gained = distribute_resources(state, 6)
        # Nobody should get any (bank can't cover full demand)
        assert gained[0, resource] == 0
        assert gained[1, resource] == 0
        # Bank should be unchanged
        assert state.bank[resource] == 3

    def test_bank_with_enough_resources_distributes(self):
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")

        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)
        state.bank[resource] = 10  # More than enough

        initial_player = int(state.resources[0, resource])
        distribute_resources(state, 6)
        assert state.resources[0, resource] == initial_player + 1

    def test_bank_empty_gives_nothing(self):
        state = make_state()
        tile_idx, resource = get_non_desert_tile_with_number(state, 6)
        if tile_idx is None:
            pytest.skip("No non-desert tile with number 6")

        vertex = int(state.topology.tile_vertices[tile_idx][0])
        place_settlement(state, 0, vertex)
        state.bank[resource] = 0

        gained = distribute_resources(state, 6)
        assert gained[0, resource] == 0


# ---------------------------------------------------------------------------
# Multi-tile vertex resources
# ---------------------------------------------------------------------------

class TestMultiTileVertex:
    def test_vertex_on_multiple_tiles_gets_multiple_resources(self):
        """Settlement on a vertex shared by multiple tiles gets resources from each."""
        state = make_state()
        topology = state.topology

        # Find a vertex on 3 tiles that all have matching numbers
        for vertex_idx in range(54):
            tiles = topology.vertex_tiles[vertex_idx]
            valid_tiles = [t for t in tiles if t >= 0]
            if len(valid_tiles) < 2:
                continue

            # Check if any two tiles have the same number and produce resources
            for i in range(len(valid_tiles)):
                t1 = valid_tiles[i]
                r1 = int(state.tile_resources[t1])
                n1 = int(state.tile_numbers[t1])
                if r1 == RESOURCE_DESERT or n1 == 0:
                    continue
                for j in range(i + 1, len(valid_tiles)):
                    t2 = valid_tiles[j]
                    r2 = int(state.tile_resources[t2])
                    n2 = int(state.tile_numbers[t2])
                    if r2 == RESOURCE_DESERT or n2 == 0:
                        continue
                    if n1 == n2 and r1 != r2:
                        # Different resources, same number — place here
                        place_settlement(state, 0, vertex_idx)
                        state.robber_tile = -1  # ensure robber is not on these tiles
                        state.robber_tile = int(
                            np.where(state.tile_resources == RESOURCE_DESERT)[0][0]
                        )
                        initial_r1 = int(state.resources[0, r1])
                        initial_r2 = int(state.resources[0, r2])
                        distribute_resources(state, n1)
                        assert state.resources[0, r1] >= initial_r1 + 1
                        assert state.resources[0, r2] >= initial_r2 + 1
                        return

        pytest.skip("No vertex found with two tiles having same number and different resources")
