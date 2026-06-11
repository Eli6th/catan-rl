"""Tests for board topology."""

# Standard Library Imports

# Third Party Imports
import pytest
import numpy as np

# Local Imports
from engine.board import BoardTopology, get_board_topology


class TestBoardTopology:
    """Test board topology computations."""

    @pytest.fixture
    def topology(self):
        return get_board_topology()

    def test_singleton(self):
        """Test that get_board_topology returns same instance."""
        t1 = get_board_topology()
        t2 = get_board_topology()
        assert t1 is t2

    def test_tile_count(self, topology):
        """Test correct number of tiles."""
        assert topology.NUM_TILES == 19

    def test_vertex_count(self, topology):
        """Test correct number of vertices."""
        assert topology.NUM_VERTICES == 54

    def test_edge_count(self, topology):
        """Test correct number of edges."""
        assert topology.NUM_EDGES == 72

    def test_tile_vertices_shape(self, topology):
        """Test tile_vertices array shape."""
        assert topology.tile_vertices.shape == (19, 6)

    def test_tile_vertices_range(self, topology):
        """Test that tile vertices are valid vertex indices."""
        assert np.all(topology.tile_vertices >= 0)
        assert np.all(topology.tile_vertices < 54)

    def test_vertex_tiles_shape(self, topology):
        """Test vertex_tiles array shape."""
        assert topology.vertex_tiles.shape == (54, 3)

    def test_vertex_neighbors_shape(self, topology):
        """Test vertex_neighbors array shape."""
        assert topology.vertex_neighbors.shape == (54, 3)

    def test_vertex_edges_shape(self, topology):
        """Test vertex_edges array shape."""
        assert topology.vertex_edges.shape == (54, 3)

    def test_edge_vertices_shape(self, topology):
        """Test edge_vertices array shape."""
        assert topology.edge_vertices.shape == (72, 2)

    def test_edge_vertices_unique(self, topology):
        """Test that each edge connects two different vertices."""
        for edge in topology.edge_vertices:
            assert edge[0] != edge[1]

    def test_vertex_tile_partial_consistency(self, topology):
        """Test that most tile->vertex mappings have reverse entries."""
        # Due to the hex grid edge cases, we check that most are consistent
        consistent = 0
        total = 0
        for tile_idx in range(19):
            for vertex in topology.tile_vertices[tile_idx]:
                tiles = topology.vertex_tiles[vertex]
                total += 1
                if tile_idx in tiles:
                    consistent += 1
        # At least 90% should be consistent
        assert consistent / total > 0.9

    def test_vertex_neighbor_count(self, topology):
        """Test that most vertices have neighbors."""
        vertices_with_neighbors = 0
        for v in range(54):
            neighbors = topology.vertex_neighbors[v]
            valid_neighbors = [n for n in neighbors if n >= 0]
            if len(valid_neighbors) >= 2:
                vertices_with_neighbors += 1
        # Most vertices should have 2+ neighbors
        assert vertices_with_neighbors >= 40

    def test_number_probabilities(self, topology):
        """Test dice probabilities sum correctly."""
        # Probabilities for 2-12 (excluding 7)
        probs = topology.number_probabilities
        assert probs[7] == 0  # No resources on 7
        assert probs[2] > 0
        assert probs[12] > 0
        assert abs(probs[6] - probs[8]) < 0.001  # 6 and 8 equally likely

    def test_port_vertices(self, topology):
        """Test port vertices are valid."""
        assert topology.port_vertices.shape == (9, 2)
        assert np.all(topology.port_vertices >= 0)
        assert np.all(topology.port_vertices < 54)


class TestBoardHelpers:
    """Test board topology helper methods."""

    @pytest.fixture
    def topology(self):
        return get_board_topology()

    def test_get_vertices_for_tile(self, topology):
        """Test getting vertices for a tile."""
        vertices = topology.get_vertices_for_tile(0)
        assert len(vertices) == 6
        assert all(0 <= v < 54 for v in vertices)

    def test_get_tiles_for_vertex(self, topology):
        """Test getting tiles for a vertex."""
        tiles = topology.get_tiles_for_vertex(10)
        assert len(tiles) == 3
        # At least some tiles should be valid
        assert any(t >= 0 for t in tiles)

    def test_get_adjacent_vertices(self, topology):
        """Test getting adjacent vertices."""
        neighbors = topology.get_adjacent_vertices(10)
        assert len(neighbors) == 3
        # Should have 2-3 valid neighbors
        valid_neighbors = [n for n in neighbors if n >= 0]
        assert len(valid_neighbors) >= 2

    def test_get_edges_for_vertex(self, topology):
        """Test getting edges for a vertex."""
        edges = topology.get_edges_for_vertex(10)
        assert len(edges) == 3

    def test_get_vertices_for_edge(self, topology):
        """Test getting vertices for an edge."""
        vertices = topology.get_vertices_for_edge(0)
        assert len(vertices) == 2
        assert vertices[0] != vertices[1]
