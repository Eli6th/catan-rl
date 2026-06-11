/**
 * TypeScript interfaces for the Catan game data structures.
 * These mirror the Python data structures from the engine.
 */

// Resource types (matching engine/board.py)
export enum ResourceType {
  WHEAT = 0,
  SHEEP = 1,
  WOOD = 2,
  BRICK = 3,
  STONE = 4,
  DESERT = 5,
}

export const RESOURCE_NAMES: Record<ResourceType, string> = {
  [ResourceType.WHEAT]: "Wheat",
  [ResourceType.SHEEP]: "Sheep",
  [ResourceType.WOOD]: "Wood",
  [ResourceType.BRICK]: "Brick",
  [ResourceType.STONE]: "Stone",
  [ResourceType.DESERT]: "Desert",
};

export const RESOURCE_COLORS: Record<ResourceType, number> = {
  [ResourceType.WHEAT]: 0xf4d03f, // Golden yellow
  [ResourceType.SHEEP]: 0x7dcea0, // Pasture green
  [ResourceType.WOOD]: 0x196f3d, // Forest green
  [ResourceType.BRICK]: 0xcb4335, // Terracotta
  [ResourceType.STONE]: 0x7f8c8d, // Gray
  [ResourceType.DESERT]: 0xd4ac0d, // Sandy tan
};

// Action types (matching engine/game.py ActionType enum)
export enum ActionType {
  PLACE_INITIAL_SETTLEMENT = 0,
  PLACE_INITIAL_ROAD = 1,
  ROLL_DICE = 2,
  BUILD_ROAD = 3,
  BUILD_SETTLEMENT = 4,
  BUILD_CITY = 5,
  BUY_DEV_CARD = 6,
  PLAY_KNIGHT = 7,
  PLAY_ROAD_BUILDING = 8,
  PLAY_YEAR_OF_PLENTY = 9,
  PLAY_MONOPOLY = 10,
  MOVE_ROBBER = 11,
  STEAL_RESOURCE = 12,
  DISCARD_RESOURCES = 13,
  TRADE_WITH_BANK = 14,
  PROPOSE_TRADE = 15,
  ACCEPT_TRADE = 16,
  REJECT_TRADE = 17,
  END_TURN = 18,
}

export const ACTION_NAMES: Record<ActionType, string> = {
  [ActionType.PLACE_INITIAL_SETTLEMENT]: "Place Initial Settlement",
  [ActionType.PLACE_INITIAL_ROAD]: "Place Initial Road",
  [ActionType.ROLL_DICE]: "Roll Dice",
  [ActionType.BUILD_ROAD]: "Build Road",
  [ActionType.BUILD_SETTLEMENT]: "Build Settlement",
  [ActionType.BUILD_CITY]: "Build City",
  [ActionType.BUY_DEV_CARD]: "Buy Dev Card",
  [ActionType.PLAY_KNIGHT]: "Play Knight",
  [ActionType.PLAY_ROAD_BUILDING]: "Play Road Building",
  [ActionType.PLAY_YEAR_OF_PLENTY]: "Play Year of Plenty",
  [ActionType.PLAY_MONOPOLY]: "Play Monopoly",
  [ActionType.MOVE_ROBBER]: "Move Robber",
  [ActionType.STEAL_RESOURCE]: "Steal Resource",
  [ActionType.DISCARD_RESOURCES]: "Discard Resources",
  [ActionType.TRADE_WITH_BANK]: "Trade with Bank",
  [ActionType.PROPOSE_TRADE]: "Propose Trade",
  [ActionType.ACCEPT_TRADE]: "Accept Trade",
  [ActionType.REJECT_TRADE]: "Reject Trade",
  [ActionType.END_TURN]: "End Turn",
};

// Player colors
export const PLAYER_COLORS: number[] = [
  0xef4444, // Red
  0x3b82f6, // Blue
  0xf5f5f5, // White
  0xf97316, // Orange
];

export const PLAYER_COLOR_NAMES: string[] = ["Red", "Blue", "White", "Orange"];

// Canonical action payload from the service layer
export interface Action {
  type: keyof typeof ActionType;
  player: number;
  payload: number[] | null;
}

// Game state interface (matching engine/state.py to_dict output)
export interface GameState {
  seed: number;
  num_players: number;
  phase: number;
  turn: number;
  current_player: number;
  winner: number;
  tile_resources: number[];
  tile_numbers: number[];
  vertices: number[];
  edges: number[];
  resources: number[][];
  bank: number[];
  victory_points: number[];
  robber_tile: number;
  longest_road_player: number;
  largest_army_player: number;
}

// Log data interface (from GameLogger)
export interface LogData {
  seed: number;
  num_players: number;
  tile_resources: number[];
  tile_numbers: number[];
  port_types: number[];
  actions: Action[];
  initial_state?: GameState;
  final_state?: GameState;
}

export interface ReplaySummary {
  id: string;
  source: string;
  num_players: number;
  action_count: number;
  seed: number;
}

export interface ReplayRecord {
  summary: ReplaySummary;
  board: {
    tile_resources: number[];
    tile_numbers: number[];
    port_types: number[];
    vertices: number[];
    edges: number[];
    robber_tile: number;
  };
  actions: Action[];
}

// Board topology constants
export const NUM_TILES = 19;
export const NUM_VERTICES = 54;
export const NUM_EDGES = 72;
export const NUM_PORTS = 9;

// Tile layout rows (number of tiles per row)
export const TILE_ROWS = [3, 4, 5, 4, 3];

// Pre-computed tile vertices (which 6 vertices touch each tile)
// Copied from engine/board.py _compute_tile_vertices
export const TILE_VERTICES: number[][] = [
  // Row 0 (3 tiles)
  [0, 1, 2, 10, 9, 8], // Tile 0
  [2, 3, 4, 12, 11, 10], // Tile 1
  [4, 5, 6, 14, 13, 12], // Tile 2
  // Row 1 (4 tiles)
  [7, 8, 9, 18, 17, 16], // Tile 3
  [9, 10, 11, 20, 19, 18], // Tile 4
  [11, 12, 13, 22, 21, 20], // Tile 5
  [13, 14, 15, 24, 23, 22], // Tile 6
  // Row 2 (5 tiles)
  [16, 17, 18, 28, 27, 26], // Tile 7
  [18, 19, 20, 30, 29, 28], // Tile 8
  [20, 21, 22, 32, 31, 30], // Tile 9
  [22, 23, 24, 34, 33, 32], // Tile 10
  [24, 25, 15, 36, 35, 34], // Tile 11 (fixed)
  // Row 3 (4 tiles)
  [28, 29, 30, 39, 38, 37], // Tile 12
  [30, 31, 32, 41, 40, 39], // Tile 13
  [32, 33, 34, 43, 42, 41], // Tile 14
  [34, 35, 36, 45, 44, 43], // Tile 15
  // Row 4 (3 tiles)
  [39, 40, 41, 48, 47, 46], // Tile 16
  [41, 42, 43, 50, 49, 48], // Tile 17
  [43, 44, 45, 52, 51, 50], // Tile 18
];
