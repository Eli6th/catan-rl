"""
Game state for Catan - all data stored in NumPy arrays for performance.

The GameState class holds all mutable game state:
- Board configuration (tiles, numbers)
- Buildings (settlements, cities, roads)
- Player resources and development cards
- Robber position
- Special achievements (longest road, largest army)
"""

# Standard Library Imports
from typing import Optional, Tuple

# Third Party Imports
import numpy as np

# Local Imports
from .board import (
    get_board_topology,
    TILE_RESOURCES,
    TILE_NUMBERS,
    PORT_TYPES,
    RESOURCE_DESERT,
)

# Constants
NUM_RESOURCES = 5  # wheat, sheep, wood, brick, stone
MAX_PLAYERS = 4
INITIAL_RESOURCES_PER_TYPE = 19
VICTORY_POINTS_TO_WIN = 10

# Building costs
ROAD_COST = np.array([0, 0, 1, 1, 0], dtype=np.int8)  # wood, brick
SETTLEMENT_COST = np.array([1, 1, 1, 1, 0], dtype=np.int8)  # wheat, sheep, wood, brick
CITY_COST = np.array([2, 0, 0, 0, 3], dtype=np.int8)  # 2 wheat, 3 stone
DEV_CARD_COST = np.array([1, 1, 0, 0, 1], dtype=np.int8)  # wheat, sheep, stone

# Development card types
DEV_KNIGHT = 0
DEV_VICTORY_POINT = 1
DEV_ROAD_BUILDING = 2
DEV_YEAR_OF_PLENTY = 3
DEV_MONOPOLY = 4
NUM_DEV_CARD_TYPES = 5

# Development card counts in deck
DEV_CARD_COUNTS = np.array([14, 5, 2, 2, 2], dtype=np.int8)


class GameState:
    """
    Complete game state using NumPy arrays for fast operations.

    All arrays use int8 or int16 to minimize memory footprint.
    A full game state is approximately 500 bytes.
    """

    def __init__(self, num_players: int = 4, seed: Optional[int] = None):
        """
        Initialize a new game state.

        Args:
            num_players: Number of players (2-4)
            seed: Random seed for reproducibility
        """
        assert 2 <= num_players <= 4, "Catan supports 2-4 players"

        self.num_players = num_players
        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        self.rng = np.random.default_rng(self.seed)
        self.topology = get_board_topology()

        # Game phase: 0=setup, 1=playing, 2=finished
        self.phase = 0
        self.turn = 0
        self.current_player = 0
        self.winner = -1

        # Board state
        self._init_board()

        # Building state
        self._init_buildings()

        # Player state
        self._init_players()

        # Development cards
        self._init_dev_cards()

        # Special achievements
        self.longest_road_player = -1
        self.longest_road_length = 0
        self.largest_army_player = -1
        self.largest_army_size = 0

        # Robber position (starts on desert)
        self.robber_tile = np.where(self.tile_resources == RESOURCE_DESERT)[0][0]

        # Dice roll for current turn (set when rolled)
        self.dice_roll = 0
        self.has_rolled = False

    def _init_board(self):
        """Initialize board tiles with random placement."""
        # Shuffle tile resources
        self.tile_resources = TILE_RESOURCES.copy()
        self.rng.shuffle(self.tile_resources)

        # Shuffle numbers (excluding desert which gets 0)
        numbers = TILE_NUMBERS[TILE_NUMBERS > 0].copy()
        self.rng.shuffle(numbers)

        # Assign numbers to non-desert tiles
        self.tile_numbers = np.zeros(19, dtype=np.int8)
        number_idx = 0
        for i in range(19):
            if self.tile_resources[i] != RESOURCE_DESERT:
                self.tile_numbers[i] = numbers[number_idx]
                number_idx += 1

        # Shuffle port types
        self.port_types = PORT_TYPES.copy()
        self.rng.shuffle(self.port_types)

    def _init_buildings(self):
        """Initialize building arrays."""
        # Vertices: -1 = empty, 0-3 = settlement by player, 4-7 = city by player
        self.vertices = np.full(54, -1, dtype=np.int8)

        # Edges: -1 = empty, 0-3 = road by player
        self.edges = np.full(72, -1, dtype=np.int8)

        # Building counts per player
        self.settlements_built = np.zeros(MAX_PLAYERS, dtype=np.int8)
        self.cities_built = np.zeros(MAX_PLAYERS, dtype=np.int8)
        self.roads_built = np.zeros(MAX_PLAYERS, dtype=np.int8)

        # Building limits
        self.max_settlements = 5
        self.max_cities = 4
        self.max_roads = 15

    def _init_players(self):
        """Initialize player resources."""
        # Resources: (num_players, 5) for wheat, sheep, wood, brick, stone
        self.resources = np.zeros((MAX_PLAYERS, NUM_RESOURCES), dtype=np.int16)

        # Bank resources
        self.bank = np.full(NUM_RESOURCES, INITIAL_RESOURCES_PER_TYPE, dtype=np.int16)

        # Victory points (public only, not including hidden VP cards)
        self.victory_points = np.zeros(MAX_PLAYERS, dtype=np.int8)

    def _init_dev_cards(self):
        """Initialize development card deck and player hands."""
        # Create deck: list of card types
        deck = []
        for card_type in range(NUM_DEV_CARD_TYPES):
            deck.extend([card_type] * DEV_CARD_COUNTS[card_type])
        self.dev_deck = np.array(deck, dtype=np.int8)
        self.rng.shuffle(self.dev_deck)
        self.dev_deck_idx = 0

        # Player dev cards: (num_players, num_card_types)
        self.dev_cards = np.zeros((MAX_PLAYERS, NUM_DEV_CARD_TYPES), dtype=np.int8)

        # Knights played per player
        self.knights_played = np.zeros(MAX_PLAYERS, dtype=np.int8)

        # Per-type count of dev cards bought this turn (can't play same-turn card)
        self.dev_cards_bought_this_turn = np.zeros(NUM_DEV_CARD_TYPES, dtype=np.int8)
        self.dev_card_played_this_turn = False

    def copy(self) -> "GameState":
        """Create a deep copy of the game state for simulation branching."""
        new_state = GameState.__new__(GameState)

        # Copy scalars
        new_state.num_players = self.num_players
        new_state.seed = self.seed
        new_state.rng = np.random.default_rng(self.rng.integers(0, 2**31))
        new_state.topology = self.topology  # Shared reference (immutable)
        new_state.phase = self.phase
        new_state.turn = self.turn
        new_state.current_player = self.current_player
        new_state.winner = self.winner
        new_state.robber_tile = self.robber_tile
        new_state.dice_roll = self.dice_roll
        new_state.has_rolled = self.has_rolled
        new_state.longest_road_player = self.longest_road_player
        new_state.longest_road_length = self.longest_road_length
        new_state.largest_army_player = self.largest_army_player
        new_state.largest_army_size = self.largest_army_size
        new_state.dev_deck_idx = self.dev_deck_idx
        new_state.dev_cards_bought_this_turn = self.dev_cards_bought_this_turn.copy()
        new_state.dev_card_played_this_turn = self.dev_card_played_this_turn
        new_state.max_settlements = self.max_settlements
        new_state.max_cities = self.max_cities
        new_state.max_roads = self.max_roads

        # Copy arrays
        new_state.tile_resources = self.tile_resources.copy()
        new_state.tile_numbers = self.tile_numbers.copy()
        new_state.port_types = self.port_types.copy()
        new_state.vertices = self.vertices.copy()
        new_state.edges = self.edges.copy()
        new_state.settlements_built = self.settlements_built.copy()
        new_state.cities_built = self.cities_built.copy()
        new_state.roads_built = self.roads_built.copy()
        new_state.resources = self.resources.copy()
        new_state.bank = self.bank.copy()
        new_state.victory_points = self.victory_points.copy()
        new_state.dev_deck = self.dev_deck.copy()
        new_state.dev_cards = self.dev_cards.copy()
        new_state.knights_played = self.knights_played.copy()

        return new_state

    def get_player_total_resources(self, player: int) -> int:
        """Get total resource count for a player."""
        return int(np.sum(self.resources[player]))

    def get_settlement_owner(self, vertex: int) -> int:
        """Get the player who owns the settlement/city at a vertex, or -1."""
        val = self.vertices[vertex]
        if val < 0:
            return -1
        return val % 4  # 0-3 = settlement, 4-7 = city (player = val % 4)

    def is_city(self, vertex: int) -> bool:
        """Check if the building at a vertex is a city."""
        return self.vertices[vertex] >= 4

    def get_road_owner(self, edge: int) -> int:
        """Get the player who owns the road at an edge, or -1."""
        return self.edges[edge]

    def get_port_type_for_player(self, player: int) -> np.ndarray:
        """Get array of port types the player has access to."""
        ports = []
        for vertex in range(54):
            if self.get_settlement_owner(vertex) == player:
                port_type = self.topology.vertex_port_type[vertex]
                if port_type >= 0:
                    ports.append(port_type)
        return np.array(ports, dtype=np.int8)

    def calculate_victory_points(self, player: int) -> int:
        """Calculate total victory points for a player."""
        vp = 0

        # Settlements (1 VP each)
        vp += self.settlements_built[player]

        # Cities (2 VP each)
        vp += 2 * self.cities_built[player]

        # Victory point cards
        vp += self.dev_cards[player, DEV_VICTORY_POINT]

        # Longest road (2 VP)
        if self.longest_road_player == player:
            vp += 2

        # Largest army (2 VP)
        if self.largest_army_player == player:
            vp += 2

        return vp

    def to_dict(self) -> dict:
        """Serialize state to dictionary for logging/debugging."""
        return {
            "seed": self.seed,
            "num_players": self.num_players,
            "phase": self.phase,
            "turn": self.turn,
            "current_player": self.current_player,
            "winner": self.winner,
            "tile_resources": self.tile_resources.tolist(),
            "tile_numbers": self.tile_numbers.tolist(),
            "vertices": self.vertices.tolist(),
            "edges": self.edges.tolist(),
            "resources": self.resources[: self.num_players].tolist(),
            "bank": self.bank.tolist(),
            "victory_points": [
                self.calculate_victory_points(p) for p in range(self.num_players)
            ],
            "robber_tile": self.robber_tile,
            "longest_road_player": self.longest_road_player,
            "largest_army_player": self.largest_army_player,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameState":
        """Deserialize state from dictionary."""
        state = cls(data["num_players"], data["seed"])
        state.phase = data["phase"]
        state.turn = data["turn"]
        state.current_player = data["current_player"]
        state.winner = data["winner"]
        state.tile_resources = np.array(data["tile_resources"], dtype=np.int8)
        state.tile_numbers = np.array(data["tile_numbers"], dtype=np.int8)
        state.vertices = np.array(data["vertices"], dtype=np.int8)
        state.edges = np.array(data["edges"], dtype=np.int8)
        state.resources[: state.num_players] = np.array(
            data["resources"], dtype=np.int16
        )
        state.bank = np.array(data["bank"], dtype=np.int16)
        state.robber_tile = data["robber_tile"]
        state.longest_road_player = data["longest_road_player"]
        state.largest_army_player = data["largest_army_player"]
        return state
