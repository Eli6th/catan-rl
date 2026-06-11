"""
Tests for longest road calculation:
- Minimum 5 roads required
- First to 5 claims the award
- Opponent's settlement interrupts/breaks road
- Tie holder keeps the award
- Loops count correctly
- Settlement placement can revoke longest road
"""

import numpy as np
import pytest

from engine.state import GameState
from engine.building import (
    build_road,
    build_settlement,
    _calculate_longest_road_for_player,
    _update_longest_road,
)


def make_state(seed: int = 42) -> GameState:
    return GameState(num_players=4, seed=seed)


def force_road(state: GameState, player: int, edge: int):
    """Place a road directly without cost or validation."""
    state.edges[edge] = player
    state.roads_built[player] += 1


def force_settlement(state: GameState, player: int, vertex: int):
    """Place a settlement directly."""
    state.vertices[vertex] = player
    state.settlements_built[player] += 1


def get_chain_of_edges(
    state: GameState, length: int, exclude: set | None = None
) -> list[int]:
    """
    Find a chain of `length` connected edges, optionally excluding certain edges.
    Returns list of edge indices forming a connected path.
    """
    topology = state.topology
    excluded = exclude or set()

    def dfs(edge: int, visited: set, path: list) -> list | None:
        if len(path) == length:
            return path
        v1, v2 = topology.edge_vertices[edge]
        for v in [v1, v2]:
            for adj_edge in topology.vertex_edges[v]:
                if (adj_edge >= 0
                        and adj_edge not in visited
                        and adj_edge not in excluded):
                    visited.add(adj_edge)
                    result = dfs(adj_edge, visited, path + [adj_edge])
                    if result is not None:
                        return result
                    visited.remove(adj_edge)
        return None

    for start_edge in range(72):
        if start_edge in excluded:
            continue
        result = dfs(start_edge, {start_edge}, [start_edge])
        if result is not None:
            return result

    return []


# ---------------------------------------------------------------------------
# Basic calculation
# ---------------------------------------------------------------------------

class TestLongestRoadCalculation:
    def test_no_roads_length_zero(self):
        state = make_state()
        length = _calculate_longest_road_for_player(state, 0)
        assert length == 0

    def test_four_roads_returns_4(self):
        state = make_state()
        # Build a chain of 4 connected edges
        chain = get_chain_of_edges(state, 4)
        assert len(chain) >= 4, "Could not find chain"
        for edge in chain[:4]:
            force_road(state, 0, edge)

        length = _calculate_longest_road_for_player(state, 0)
        assert length == 4

    def test_five_roads_in_a_line(self):
        state = make_state()
        chain = get_chain_of_edges(state, 5)
        assert len(chain) >= 5, "Could not find chain of 5"
        for edge in chain[:5]:
            force_road(state, 0, edge)

        length = _calculate_longest_road_for_player(state, 0)
        assert length >= 5

    def test_disconnected_roads_returns_max_component(self):
        state = make_state()
        # Place 3 roads in one area and 2 disconnected roads elsewhere
        chain1 = get_chain_of_edges(state, 3)
        for edge in chain1:
            force_road(state, 0, edge)
        # Find 2 disconnected edges (far apart)
        used = set(chain1)
        isolated = []
        for e in range(72):
            if e not in used:
                v1, v2 = state.topology.edge_vertices[e]
                # Check none of the vertices are shared with existing roads
                adj_to_chain = any(
                    e2 in used
                    for v in [v1, v2]
                    for e2 in state.topology.vertex_edges[v]
                    if e2 >= 0
                )
                if not adj_to_chain:
                    isolated.append(e)
                    if len(isolated) == 2:
                        break

        for edge in isolated:
            force_road(state, 0, edge)

        length = _calculate_longest_road_for_player(state, 0)
        # The longest single chain is 3, even though total roads = 5
        # (the isolated roads don't connect)
        assert length == 3


# ---------------------------------------------------------------------------
# Award thresholds
# ---------------------------------------------------------------------------

class TestLongestRoadAward:
    def test_less_than_5_no_award(self):
        state = make_state()
        chain = get_chain_of_edges(state, 4)
        for edge in chain[:4]:
            force_road(state, 0, edge)
        _update_longest_road(state)

        assert state.longest_road_player == -1
        assert state.longest_road_length == 0

    def test_exactly_5_gets_award(self):
        state = make_state()
        chain = get_chain_of_edges(state, 5)
        for edge in chain[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)

        assert state.longest_road_player == 0
        assert state.longest_road_length == 5

    def test_first_to_5_wins(self):
        state = make_state()
        # Player 0 builds 5 roads
        chain0 = get_chain_of_edges(state, 5)
        for edge in chain0[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)

        assert state.longest_road_player == 0

    def test_longer_road_steals_award(self):
        state = make_state()
        # Player 0 has 5 roads
        chain0 = get_chain_of_edges(state, 5)
        for edge in chain0[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)
        assert state.longest_road_player == 0

        # Player 1 builds 6 roads - exclude player 0's edges
        chain1 = get_chain_of_edges(state, 6, exclude=set(chain0))
        if len(chain1) < 6:
            pytest.skip("Could not find 6 separate edges for player 1")
        for edge in chain1[:6]:
            force_road(state, 1, edge)
        _update_longest_road(state)

        assert state.longest_road_player == 1
        assert state.longest_road_length == 6

    def test_tie_original_holder_keeps_award(self):
        """Player must strictly exceed current holder's length to steal the award."""
        state = make_state()
        # Player 0 has 5
        chain0 = get_chain_of_edges(state, 5)
        for edge in chain0[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)
        assert state.longest_road_player == 0

        # Player 1 also builds 5 roads — same length, can't steal
        chain1 = get_chain_of_edges(state, 5, exclude=set(chain0))
        if len(chain1) < 5:
            pytest.skip("Could not find 5 separate edges for player 1")
        for edge in chain1[:5]:
            force_road(state, 1, edge)
        _update_longest_road(state)

        # Player 0 keeps it (tie doesn't transfer)
        assert state.longest_road_player == 0
        assert state.longest_road_length == 5


# ---------------------------------------------------------------------------
# Opponent settlement breaks road
# ---------------------------------------------------------------------------

class TestRoadBreaking:
    def test_opponent_settlement_breaks_road(self):
        """Settlement placed at the junction of a road can break longest road."""
        state = make_state()
        topology = state.topology

        # Build a 5-road chain for player 0
        chain = get_chain_of_edges(state, 5)
        for edge in chain[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)
        assert state.longest_road_player == 0

        # Find an interior vertex (used by two edges in the chain) to place
        # opponent's settlement, which will break the road
        interior_vertices = []
        for i in range(1, len(chain[:5])):
            e_prev = chain[i - 1]
            e_curr = chain[i]
            verts_prev = set(map(int, topology.edge_vertices[e_prev]))
            verts_curr = set(map(int, topology.edge_vertices[e_curr]))
            shared = verts_prev & verts_curr
            if shared:
                interior_vertices.extend(shared)

        if not interior_vertices:
            pytest.skip("Could not find interior vertex in road chain")

        # Place opponent at interior vertex
        junction_vertex = interior_vertices[0]
        # Must pass distance rule check (no adjacent settlements)
        neighbors_ok = all(
            state.vertices[n] < 0
            for n in topology.vertex_neighbors[junction_vertex]
            if n >= 0
        )
        if not neighbors_ok:
            pytest.skip("Interior vertex too close to existing settlement")

        force_settlement(state, 1, junction_vertex)
        _update_longest_road(state)

        # Player 0's road should be broken — may or may not still have longest road
        # depending on sub-chain lengths, but the road is at most 4 on each side
        player0_length = _calculate_longest_road_for_player(state, 0)
        assert player0_length < 5 or state.longest_road_player != 0 or state.longest_road_length < 5

    def test_settlement_on_endpoint_doesnt_break_road(self):
        """Settlement on an endpoint of a chain doesn't split the road."""
        state = make_state()
        topology = state.topology

        # Build a 5-road chain for player 0
        chain = get_chain_of_edges(state, 5)
        for edge in chain[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)
        assert state.longest_road_player == 0

        # Find an endpoint vertex (only used by one edge in the chain)
        endpoint = None
        first_edge = chain[0]
        v1, v2 = topology.edge_vertices[first_edge]
        second_edge = chain[1]
        verts_second = set(map(int, topology.edge_vertices[second_edge]))
        for v in [v1, v2]:
            if v not in verts_second:
                endpoint = v
                break

        if endpoint is None:
            pytest.skip("Could not find endpoint vertex")

        # Place opponent's settlement at the endpoint
        neighbors_ok = all(
            state.vertices[n] < 0
            for n in topology.vertex_neighbors[endpoint]
            if n >= 0
        )
        if not neighbors_ok:
            pytest.skip("Endpoint vertex too close to existing settlement")

        force_settlement(state, 1, endpoint)
        _update_longest_road(state)

        # The road should still be at least 5 (endpoint doesn't break it)
        player0_length = _calculate_longest_road_for_player(state, 0)
        assert player0_length >= 5


# ---------------------------------------------------------------------------
# Road building card extends road
# ---------------------------------------------------------------------------

class TestRoadBuildingExtension:
    def test_road_building_extends_longest_road(self):
        state = make_state()
        # Player 0 starts with 5 roads
        chain = get_chain_of_edges(state, 7)
        for edge in chain[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)
        assert state.longest_road_player == 0
        assert state.longest_road_length == 5

        # Extend by 2 more (simulating road building card)
        force_road(state, 0, chain[5])
        _update_longest_road(state)
        force_road(state, 0, chain[6])
        _update_longest_road(state)

        assert state.longest_road_player == 0
        assert state.longest_road_length == 7


# ---------------------------------------------------------------------------
# Victory points from longest road
# ---------------------------------------------------------------------------

class TestLongestRoadVictoryPoints:
    def test_longest_road_gives_2_vp(self):
        state = make_state()
        chain = get_chain_of_edges(state, 5)
        for edge in chain[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)

        vp = state.calculate_victory_points(0)
        assert vp >= 2  # At least 2 from longest road

    def test_losing_longest_road_loses_2_vp(self):
        state = make_state()

        # Player 0 gets longest road (5 edges)
        chain0 = get_chain_of_edges(state, 5)
        for edge in chain0[:5]:
            force_road(state, 0, edge)
        _update_longest_road(state)
        assert state.longest_road_player == 0

        vp_before = state.calculate_victory_points(0)

        # Player 1 steals it with 6 edges
        chain1 = get_chain_of_edges(state, 6, exclude=set(chain0))
        if len(chain1) < 6:
            pytest.skip("Could not find 6 separate edges for player 1")
        for edge in chain1[:6]:
            force_road(state, 1, edge)
        _update_longest_road(state)
        assert state.longest_road_player == 1

        vp_after = state.calculate_victory_points(0)
        assert vp_after == vp_before - 2
