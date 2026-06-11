"""
Game replay system for Catan.

Allows stepping through logged games action by action for debugging.
"""

# Standard Library Imports
import json
from pathlib import Path
from typing import Optional, Union, List

# Third Party Imports

# Local Imports
from engine.state import GameState
from engine.game import CatanGame, Action, ActionType, GamePhase
from engine.board import RESOURCE_NAMES
from .logger import GameLogger


class GameReplay:
    """
    Replay a logged Catan game step by step.

    Features:
    - Step forward/backward through actions
    - Jump to specific turns
    - Inspect game state at any point
    - Export state snapshots
    """

    def __init__(self, log_data: dict):
        """
        Initialize replay from log data.

        Args:
            log_data: Dictionary from GameLogger.load_binary/load_json
        """
        self.seed = log_data["seed"]
        self.num_players = log_data["num_players"]
        self.tile_resources = log_data["tile_resources"]
        self.tile_numbers = log_data["tile_numbers"]
        self.port_types = log_data.get("port_types")
        self.actions = log_data["actions"]

        # Replay state
        self.action_index = 0
        self.game: Optional[CatanGame] = None
        self._reset_game()

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "GameReplay":
        """
        Load a game replay from a log file.

        Args:
            filepath: Path to .clog or .json file

        Returns:
            GameReplay instance
        """
        filepath = Path(filepath)

        if filepath.suffix == ".clog":
            log_data = GameLogger.load_binary(filepath)
        elif filepath.suffix == ".json":
            log_data = GameLogger.load_json(filepath)
        else:
            raise ValueError(f"Unknown file type: {filepath.suffix}")

        return cls(log_data)

    @classmethod
    def from_seed(cls, seed: int, log_dir: str = "logs") -> "GameReplay":
        """
        Load a game replay by seed.

        Args:
            seed: Game seed to find
            log_dir: Directory to search

        Returns:
            GameReplay instance
        """
        from .logger import find_log_by_seed

        filepath = find_log_by_seed(seed, log_dir)
        if filepath is None:
            raise FileNotFoundError(f"No log found for seed {seed}")

        return cls.load(filepath)

    def _reset_game(self):
        """Reset game to initial state."""
        self.game = CatanGame(self.num_players, self.seed)

        # Override board setup to match logged game
        self.game.state.tile_resources = self.tile_resources.copy()
        self.game.state.tile_numbers = self.tile_numbers.copy()
        if self.port_types is not None:
            self.game.state.port_types = self.port_types.copy()

        self.action_index = 0

    def step(self) -> Optional[Action]:
        """
        Execute the next action.

        Returns:
            The action executed, or None if at end
        """
        if self.action_index >= len(self.actions):
            return None

        action = self.actions[self.action_index]
        self.game.execute_action(action)
        self.action_index += 1

        return action

    def step_back(self) -> bool:
        """
        Step back one action by replaying from start.

        Returns:
            True if successful
        """
        if self.action_index <= 0:
            return False

        target_idx = self.action_index - 1
        self._reset_game()

        while self.action_index < target_idx:
            self.step()

        return True

    def jump_to_action(self, action_idx: int) -> bool:
        """
        Jump to a specific action index.

        Args:
            action_idx: Target action index

        Returns:
            True if successful
        """
        if action_idx < 0 or action_idx > len(self.actions):
            return False

        if action_idx < self.action_index:
            self._reset_game()

        while self.action_index < action_idx:
            self.step()

        return True

    def jump_to_turn(self, turn: int) -> bool:
        """
        Jump to the start of a specific turn.

        Args:
            turn: Target turn number

        Returns:
            True if successful
        """
        self._reset_game()

        while self.action_index < len(self.actions):
            if self.game.state.turn >= turn:
                break
            self.step()

        return self.game.state.turn == turn

    def play_all(self):
        """Play through all remaining actions."""
        while self.step() is not None:
            pass

    @property
    def current_turn(self) -> int:
        """Get current turn number."""
        return self.game.state.turn

    @property
    def current_player(self) -> int:
        """Get current player index."""
        return self.game.get_current_player()

    @property
    def is_finished(self) -> bool:
        """Check if replay is at the end."""
        return self.action_index >= len(self.actions)

    @property
    def total_actions(self) -> int:
        """Get total number of actions."""
        return len(self.actions)

    def get_state(self) -> GameState:
        """Get current game state."""
        return self.game.state

    def get_remaining_actions(self) -> List[Action]:
        """Get actions not yet executed."""
        return self.actions[self.action_index :]

    def print_state(self):
        """Print current game state to console."""
        state = self.game.state

        print(f"\n{'='*60}")
        print(f"Turn {state.turn} | Player {state.current_player}'s turn")
        print(f"Phase: {self.game.game_phase.name} / {self.game.turn_phase.name}")
        print(f"Action {self.action_index}/{len(self.actions)}")
        print(f"{'='*60}")

        # Victory points
        print("\nVictory Points:")
        for p in range(state.num_players):
            vp = state.calculate_victory_points(p)
            marker = " <--" if p == state.current_player else ""
            print(f"  Player {p}: {vp} VP{marker}")

        # Resources
        print("\nResources:")
        for p in range(state.num_players):
            resources = state.resources[p]
            res_str = ", ".join(
                f"{RESOURCE_NAMES[i][:1].upper()}:{resources[i]}" for i in range(5)
            )
            print(f"  Player {p}: [{res_str}]")

        # Buildings
        print("\nBuildings:")
        for p in range(state.num_players):
            s = state.settlements_built[p]
            c = state.cities_built[p]
            r = state.roads_built[p]
            print(f"  Player {p}: {s} settlements, {c} cities, {r} roads")

        # Special
        if state.longest_road_player >= 0:
            print(
                f"\nLongest Road: Player {state.longest_road_player} "
                f"({state.longest_road_length} roads)"
            )
        if state.largest_army_player >= 0:
            print(
                f"Largest Army: Player {state.largest_army_player} "
                f"({state.largest_army_size} knights)"
            )

        # Winner
        if state.winner >= 0:
            print(f"\n*** WINNER: Player {state.winner} ***")

    def print_next_action(self):
        """Print details of the next action."""
        if self.action_index >= len(self.actions):
            print("No more actions")
            return

        action = self.actions[self.action_index]
        action_name = ActionType(action.action_type).name
        data_str = action.data.tolist() if action.data is not None else "None"

        print(
            f"Next action [{self.action_index}]: "
            f"Player {action.player} - {action_name} {data_str}"
        )

    def export_json(self, filepath: Optional[str] = None) -> dict:
        """
        Export current state to JSON.

        Args:
            filepath: Optional path to save JSON file

        Returns:
            State dictionary
        """
        data = {
            "seed": self.seed,
            "action_index": self.action_index,
            "turn": self.current_turn,
            "current_player": self.current_player,
            "state": self.game.state.to_dict(),
        }

        if filepath:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        return data

    def __repr__(self) -> str:
        return (
            f"GameReplay(seed={self.seed}, "
            f"action={self.action_index}/{len(self.actions)}, "
            f"turn={self.current_turn})"
        )
