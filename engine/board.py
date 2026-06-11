"""
Board topology for Catan - pre-computed adjacency matrices for fast lookups.

The standard Catan board has:
- 19 hex tiles (resource tiles)
- 54 vertices (intersection points where settlements/cities go)
- 72 edges (paths where roads go)
- 9 ports (harbors for trading)

Coordinate system:
- Tiles are numbered 0-18 row by row (3-4-5-4-3 layout)
- Vertices are numbered 0-53 in the order they are first encountered
  when iterating tiles then their 6 vertices (pointy-top hex, clockwise
  from top: Top, Top-right, Bottom-right, Bottom, Bottom-left, Top-left)
- Edges are numbered 0-71 in sorted order

The adjacency matrices allow O(1) lookups for:
- Which vertices touch a tile (for resource distribution)
- Which edges connect to a vertex (for road building)
- Which vertices are adjacent (for distance rule)
- Which tiles touch a vertex (for resource collection)
"""

# Standard Library Imports
import math
from typing import Tuple, List, Set

# Third Party Imports
import numpy as np

# Local Imports

# Constants
RESOURCE_WHEAT = 0
RESOURCE_SHEEP = 1
RESOURCE_WOOD = 2
RESOURCE_BRICK = 3
RESOURCE_STONE = 4
RESOURCE_DESERT = 5

RESOURCE_NAMES = ["wheat", "sheep", "wood", "brick", "stone", "desert"]

TILE_RESOURCES = np.array(
    [
        RESOURCE_WHEAT,
        RESOURCE_WHEAT,
        RESOURCE_WHEAT,
        RESOURCE_WHEAT,
        RESOURCE_SHEEP,
        RESOURCE_SHEEP,
        RESOURCE_SHEEP,
        RESOURCE_SHEEP,
        RESOURCE_WOOD,
        RESOURCE_WOOD,
        RESOURCE_WOOD,
        RESOURCE_WOOD,
        RESOURCE_BRICK,
        RESOURCE_BRICK,
        RESOURCE_BRICK,
        RESOURCE_STONE,
        RESOURCE_STONE,
        RESOURCE_STONE,
        RESOURCE_DESERT,
    ],
    dtype=np.int8,
)

# Standard Catan number tokens (desert has 0)
TILE_NUMBERS = np.array(
    [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12, 0], dtype=np.int8
)

PORT_TYPE_ANY = 0
PORT_TYPES = np.array(
    [
        PORT_TYPE_ANY,
        PORT_TYPE_ANY,
        PORT_TYPE_ANY,
        PORT_TYPE_ANY,
        RESOURCE_WHEAT + 1,
        RESOURCE_SHEEP + 1,
        RESOURCE_WOOD + 1,
        RESOURCE_BRICK + 1,
        RESOURCE_STONE + 1,
    ],
    dtype=np.int8,
)


class BoardTopology:
    """
    Pre-computed adjacency information for the Catan board.
    All data is stored in NumPy arrays for fast vectorized operations.

    Uses pointy-top hexagon geometry (vertices at top and bottom):
      col_spacing = R * sqrt(3)
      row_spacing = R * 1.5

    Tile layout (row by row, 3-4-5-4-3):
              0   1   2
            3   4   5   6
          7   8   9  10  11
            12  13  14  15
              16  17  18
    """

    # Number of components
    NUM_TILES = 19
    NUM_VERTICES = 54
    NUM_EDGES = 72
    NUM_PORTS = 9

    def __init__(self):
        """Initialize all adjacency matrices."""
        # Compute tile vertices and centers together from hex geometry
        self.tile_vertices, self.tile_centers = self._compute_geometry()

        # Vertex -> Tiles: which tiles (up to 3) touch each vertex
        # Shape: (54, 3), -1 means no tile
        self.vertex_tiles = self._compute_vertex_tiles()

        # Vertex -> Vertices: adjacent vertices (for distance rule)
        # Shape: (54, 3), -1 means no neighbor
        self.vertex_neighbors = self._compute_vertex_neighbors()

        # Edge -> Vertices: which 2 vertices each edge connects
        # Shape: (72, 2) - computed before vertex_edges
        self.edge_vertices = self._compute_edge_vertices()

        # Vertex -> Edges: which edges (up to 3) connect to each vertex
        # Shape: (54, 3), -1 means no edge
        self.vertex_edges = self._compute_vertex_edges()

        # Port -> Vertices: which 2 vertices have access to each port
        # Shape: (9, 2)
        self.port_vertices = self._compute_port_vertices()

        # Lookup table: vertex -> port type (-1 if no port)
        # Shape: (54,)
        self.vertex_port_type = self._compute_vertex_port_types()

        # Number probability weights (for heuristic AI)
        # 7 has 0 since it triggers robber
        self.number_probabilities = (
            np.array([0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1], dtype=np.float32) / 36.0
        )

    def _compute_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute tile vertices and tile centers from pointy-top hex geometry.

        Pointy-top hex with radius R=1:
          h = R * sqrt(3) / 2  (half-width)
          Vertex order (clockwise from top):
            0: Top         (cx,     cy - R)
            1: Top-right   (cx + h, cy - R/2)
            2: Bottom-right (cx + h, cy + R/2)
            3: Bottom      (cx,     cy + R)
            4: Bottom-left (cx - h, cy + R/2)
            5: Top-left    (cx - h, cy - R/2)

        Vertices are assigned IDs in the order first encountered,
        iterating tiles left-to-right, top-to-bottom, vertices clockwise.
        Shared vertices between tiles get the same ID.

        Returns:
            tile_vertices: (19, 6) int8 array
            tile_centers:  (19, 2) float32 array of (x, y) center coords
        """
        R = 1.0
        h = R * math.sqrt(3) / 2  # half-width of hex
        col_spacing = R * math.sqrt(3)  # horizontal distance between centers
        row_spacing = R * 1.5  # vertical distance between row centers
        PREC = 6  # decimal places for deduplication

        def hex_verts(cx: float, cy: float) -> List[Tuple[float, float]]:
            return [
                (cx,      cy - R),       # Top
                (cx + h,  cy - R / 2),   # Top-right
                (cx + h,  cy + R / 2),   # Bottom-right
                (cx,      cy + R),       # Bottom
                (cx - h,  cy + R / 2),   # Bottom-left
                (cx - h,  cy - R / 2),   # Top-left
            ]

        # Row layout: [3, 4, 5, 4, 3] tiles per row, row 2 (5 tiles) at y=0
        row_configs = [3, 4, 5, 4, 3]

        coord_to_vertex = {}  # (rx, ry) -> vertex_id
        next_vertex_id = 0

        centers = []
        tile_verts = []

        for row_idx, num_tiles in enumerate(row_configs):
            x_start = -((num_tiles - 1) / 2.0) * col_spacing
            y = (row_idx - 2) * row_spacing  # row 2 is centered at y=0
            for col_idx in range(num_tiles):
                cx = x_start + col_idx * col_spacing
                centers.append((cx, y))
                tile_v = []
                for x, y_v in hex_verts(cx, y):
                    key = (round(x, PREC), round(y_v, PREC))
                    if key not in coord_to_vertex:
                        coord_to_vertex[key] = next_vertex_id
                        next_vertex_id += 1
                    tile_v.append(coord_to_vertex[key])
                tile_verts.append(tile_v)

        assert next_vertex_id == self.NUM_VERTICES, (
            f"Expected {self.NUM_VERTICES} unique vertices, got {next_vertex_id}"
        )

        tile_vertices = np.array(tile_verts, dtype=np.int8)
        tile_centers_arr = np.array(centers, dtype=np.float32)
        return tile_vertices, tile_centers_arr

    def _compute_vertex_tiles(self) -> np.ndarray:
        """Compute which tiles (up to 3) touch each vertex."""
        vertex_tiles = np.full((self.NUM_VERTICES, 3), -1, dtype=np.int8)

        for tile_idx in range(self.NUM_TILES):
            for vertex_idx in self.tile_vertices[tile_idx]:
                # Find first empty slot
                for slot in range(3):
                    if vertex_tiles[vertex_idx, slot] == -1:
                        vertex_tiles[vertex_idx, slot] = tile_idx
                        break

        return vertex_tiles

    def _compute_vertex_neighbors(self) -> np.ndarray:
        """Compute adjacent vertices for each vertex (for distance rule)."""
        neighbors: List[Set[int]] = [set() for _ in range(self.NUM_VERTICES)]

        for tile_idx in range(self.NUM_TILES):
            vertices = self.tile_vertices[tile_idx]
            for i in range(6):
                v1 = int(vertices[i])
                v2 = int(vertices[(i + 1) % 6])
                neighbors[v1].add(v2)
                neighbors[v2].add(v1)

        # Convert to numpy array (max 3 neighbors per vertex)
        result = np.full((self.NUM_VERTICES, 3), -1, dtype=np.int8)
        for v, nbrs in enumerate(neighbors):
            for i, n in enumerate(sorted(nbrs)[:3]):
                result[v, i] = n

        return result

    def _compute_edge_vertices(self) -> np.ndarray:
        """Compute which 2 vertices each edge connects. Returns (72, 2) array."""
        edges_set: Set[Tuple[int, int]] = set()

        for tile_idx in range(self.NUM_TILES):
            vertices = self.tile_vertices[tile_idx]
            for i in range(6):
                v1 = int(vertices[i])
                v2 = int(vertices[(i + 1) % 6])
                edges_set.add((min(v1, v2), max(v1, v2)))

        edges_list = sorted(edges_set)
        assert len(edges_list) == self.NUM_EDGES, (
            f"Expected {self.NUM_EDGES} edges, got {len(edges_list)}"
        )
        return np.array(edges_list, dtype=np.int8)

    def _compute_vertex_edges(self) -> np.ndarray:
        """Compute which edges connect to each vertex. Uses self.edge_vertices."""
        vertex_edges = np.full((self.NUM_VERTICES, 3), -1, dtype=np.int8)

        for edge_idx in range(self.NUM_EDGES):
            v1, v2 = self.edge_vertices[edge_idx]
            for v in [v1, v2]:
                for slot in range(3):
                    if vertex_edges[v, slot] == -1:
                        vertex_edges[v, slot] = edge_idx
                        break

        return vertex_edges

    def _compute_port_vertices(self) -> np.ndarray:
        """
        Compute which 2 vertices have access to each port.

        Ports are on outer (coastal) edges of the board, spread clockwise
        from the top. The 9 port positions in the new geometric numbering:

        Perimeter (30 outer edges, clockwise from vertex 0):
          (0,1),(1,6),(6,7),(7,10),(10,11),(11,12),(12,22),(22,23),(23,35),
          (35,36),(36,37),(37,45),(45,46),(46,52),(52,53),(50,53),(50,51),
          (47,51),(47,48),(48,49),(39,49),(39,40),(26,40),(26,27),(27,28),
          (16,28),(16,17),(4,17),(4,5),(0,5)

        Port positions chosen at perimeter edges 0, 3, 6, 9, 12, 15, 18, 21, 24
        (every 3rd edge, starting from edge 0):
        """
        return np.array(
            [
                [0,  1],   # Port 0: top edge of tile 0 (perimeter edge 0)
                [7,  10],  # Port 1: between tiles 1 and 2  (perimeter edge 3)
                [12, 22],  # Port 2: top-right of tile 6    (perimeter edge 6)
                [35, 36],  # Port 3: right of tile 11       (perimeter edge 9)
                [45, 46],  # Port 4: bottom-right of tile 15 (perimeter edge 12)
                [50, 53],  # Port 5: bottom-left of tile 18  (perimeter edge 15)
                [47, 48],  # Port 6: bottom of tile 16       (perimeter edge 18)
                [39, 40],  # Port 7: bottom of tile 12       (perimeter edge 21)
                [27, 28],  # Port 8: bottom of tile 7        (perimeter edge 24)
            ],
            dtype=np.int8,
        )

    def _compute_vertex_port_types(self) -> np.ndarray:
        """Create lookup table from vertex to port type (-1 if no port)."""
        result = np.full(self.NUM_VERTICES, -1, dtype=np.int8)

        for port_idx in range(self.NUM_PORTS):
            port_type = PORT_TYPES[port_idx]
            for vertex in self.port_vertices[port_idx]:
                result[vertex] = port_type

        return result

    def get_vertices_for_tile(self, tile_idx: int) -> np.ndarray:
        """Get the 6 vertices that touch a tile."""
        return self.tile_vertices[tile_idx]

    def get_tiles_for_vertex(self, vertex_idx: int) -> np.ndarray:
        """Get the tiles (up to 3) that touch a vertex. -1 means no tile."""
        return self.vertex_tiles[vertex_idx]

    def get_adjacent_vertices(self, vertex_idx: int) -> np.ndarray:
        """Get adjacent vertices (for distance rule). -1 means no neighbor."""
        return self.vertex_neighbors[vertex_idx]

    def get_edges_for_vertex(self, vertex_idx: int) -> np.ndarray:
        """Get edges connected to a vertex. -1 means no edge."""
        return self.vertex_edges[vertex_idx]

    def get_vertices_for_edge(self, edge_idx: int) -> np.ndarray:
        """Get the 2 vertices an edge connects."""
        return self.edge_vertices[edge_idx]

    def get_port_type(self, vertex_idx: int) -> int:
        """Get port type for a vertex. -1 if no port."""
        return self.vertex_port_type[vertex_idx]

    def get_vertex_probability(
        self, vertex_idx: int, tile_numbers: np.ndarray
    ) -> float:
        """Calculate the total probability of getting resources from a vertex."""
        tiles = self.vertex_tiles[vertex_idx]
        total = 0.0
        for tile in tiles:
            if tile >= 0:
                number = tile_numbers[tile]
                total += self.number_probabilities[number]
        return total


# Singleton instance for the standard board
_BOARD_TOPOLOGY: BoardTopology = None


def get_board_topology() -> BoardTopology:
    """Get the singleton board topology instance."""
    global _BOARD_TOPOLOGY
    if _BOARD_TOPOLOGY is None:
        _BOARD_TOPOLOGY = BoardTopology()
    return _BOARD_TOPOLOGY
