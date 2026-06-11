"""
Game logging for Catan simulations.

Provides compact binary format for storing game logs that can be replayed.
Also supports JSON export for human-readable debugging.

Log format (binary):
- Header (17 bytes):
  - Magic bytes: "CLOG" (4 bytes)
  - Version: 1 (1 byte)
  - Seed: int64 (8 bytes)
  - Num players: int8 (1 byte)
  - Num actions: int16 (2 bytes)
  - Reserved: 1 byte

- Board setup (56 bytes):
  - Tile resources: 19 bytes
  - Tile numbers: 19 bytes
  - Port types: 9 bytes
  - Reserved: 9 bytes

- Actions (variable):
  - Each action: action_type (1 byte) + player (1 byte) + data_len (1 byte) + data
"""

# Standard Library Imports
import json
import struct
from pathlib import Path
from typing import List, Optional, Union, BinaryIO
from datetime import datetime

# Third Party Imports
import numpy as np

# Local Imports
from engine.state import GameState
from engine.game import Action, ActionType


MAGIC_BYTES = b"CLOG"
LOG_VERSION = 1


class GameLogger:
    """
    Logger for recording Catan game actions.

    Can write to binary format for compact storage or JSON for debugging.
    """

    def __init__(self, log_dir: Union[str, Path] = "logs"):
        """
        Initialize the logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Current game being logged
        self.current_seed: Optional[int] = None
        self.current_state: Optional[GameState] = None
        self.actions: List[Action] = []
        self.enabled = True

    def start_game(self, state: GameState):
        """Start logging a new game."""
        if not self.enabled:
            return

        self.current_seed = state.seed
        self.current_state = state.copy()
        self.actions = []

    def log_action(self, action: Action):
        """Log a game action."""
        if not self.enabled:
            return

        self.actions.append(action)

    def end_game(self, final_state: GameState, save: bool = True) -> Optional[Path]:
        """
        End the current game log.

        Args:
            final_state: Final game state
            save: Whether to save the log file

        Returns:
            Path to saved log file, or None if not saved
        """
        if not self.enabled or self.current_state is None:
            return None

        if save:
            filepath = self.save_binary(final_state)
            return filepath

        return None

    def save_binary(self, final_state: GameState) -> Path:
        """
        Save game log in compact binary format.

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{self.current_seed}_{timestamp}.clog"
        filepath = self.log_dir / filename

        with open(filepath, "wb") as f:
            self._write_binary(f, final_state)

        return filepath

    def _write_binary(self, f: BinaryIO, final_state: GameState):
        """Write binary log format."""
        state = self.current_state

        # Header
        f.write(MAGIC_BYTES)
        f.write(struct.pack("b", LOG_VERSION))
        f.write(struct.pack("q", state.seed))
        f.write(struct.pack("b", state.num_players))
        f.write(struct.pack("h", len(self.actions)))
        f.write(b"\x00")  # Reserved

        # Board setup
        f.write(state.tile_resources.tobytes())
        f.write(state.tile_numbers.tobytes())
        f.write(state.port_types.tobytes())
        f.write(b"\x00" * 9)  # Reserved

        # Actions
        for action in self.actions:
            f.write(struct.pack("b", action.action_type))
            f.write(struct.pack("b", action.player))
            if action.data is not None:
                data_bytes = action.data.tobytes()
                f.write(struct.pack("b", len(data_bytes)))
                f.write(data_bytes)
            else:
                f.write(struct.pack("b", 0))

    def save_json(self, final_state: GameState) -> Path:
        """
        Save game log in JSON format for debugging.

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{self.current_seed}_{timestamp}.json"
        filepath = self.log_dir / filename

        data = {
            "seed": self.current_seed,
            "num_players": self.current_state.num_players,
            "initial_state": self.current_state.to_dict(),
            "final_state": final_state.to_dict(),
            "actions": [
                {
                    "type": ActionType(a.action_type).name,
                    "player": a.player,
                    "data": a.data.tolist() if a.data is not None else None,
                }
                for a in self.actions
            ],
            "winner": final_state.winner,
            "total_turns": final_state.turn,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    @classmethod
    def load_binary(cls, filepath: Union[str, Path]) -> dict:
        """
        Load a binary game log.

        Returns:
            Dictionary with game data
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            # Header
            magic = f.read(4)
            if magic != MAGIC_BYTES:
                raise ValueError(f"Invalid log file: bad magic bytes")

            version = struct.unpack("b", f.read(1))[0]
            if version != LOG_VERSION:
                raise ValueError(f"Unsupported log version: {version}")

            seed = struct.unpack("q", f.read(8))[0]
            num_players = struct.unpack("b", f.read(1))[0]
            num_actions = struct.unpack("h", f.read(2))[0]
            f.read(1)  # Reserved

            # Board setup
            tile_resources = np.frombuffer(f.read(19), dtype=np.int8)
            tile_numbers = np.frombuffer(f.read(19), dtype=np.int8)
            port_types = np.frombuffer(f.read(9), dtype=np.int8)
            f.read(9)  # Reserved

            # Actions
            actions = []
            for _ in range(num_actions):
                action_type = struct.unpack("b", f.read(1))[0]
                player = struct.unpack("b", f.read(1))[0]
                data_len = struct.unpack("b", f.read(1))[0]

                if data_len > 0:
                    data = np.frombuffer(f.read(data_len), dtype=np.int8)
                else:
                    data = None

                actions.append(Action(ActionType(action_type), player, data))

        return {
            "seed": seed,
            "num_players": num_players,
            "tile_resources": tile_resources,
            "tile_numbers": tile_numbers,
            "port_types": port_types,
            "actions": actions,
        }

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> dict:
        """Load a JSON game log."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert action dicts back to Action objects
        actions = []
        for a in data["actions"]:
            action_type = ActionType[a["type"]]
            player = a["player"]
            action_data = np.array(a["data"], dtype=np.int8) if a["data"] else None
            actions.append(Action(action_type, player, action_data))

        data["actions"] = actions
        return data


def get_log_files(
    log_dir: Union[str, Path] = "logs", pattern: str = "*.clog"
) -> List[Path]:
    """Get all log files matching pattern."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob(pattern))


def find_log_by_seed(seed: int, log_dir: Union[str, Path] = "logs") -> Optional[Path]:
    """Find a log file by game seed."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None

    for filepath in log_dir.glob(f"game_{seed}_*.clog"):
        return filepath

    for filepath in log_dir.glob(f"game_{seed}_*.json"):
        return filepath

    return None
