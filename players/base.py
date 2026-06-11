"""
Base player interface for Catan.

All strategies must implement the Player abstract class.
"""

# Standard Library Imports
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

# Third Party Imports
import numpy as np

# Local Imports
if TYPE_CHECKING:
    from engine.state import GameState
    from engine.game import Action


class Player(ABC):
    """
    Abstract base class for Catan players.

    Subclasses must implement choose_action() to select from valid actions.
    """

    def __init__(self, name: str = "Player"):
        """
        Initialize a player.

        Args:
            name: Player name for logging/display
        """
        self.name = name
        self.player_idx = -1  # Set when added to game

    @abstractmethod
    def choose_action(
        self, state: "GameState", valid_actions: List["Action"]
    ) -> "Action":
        """
        Choose an action from the list of valid actions.

        Args:
            state: Current game state
            valid_actions: List of valid actions to choose from

        Returns:
            The chosen action
        """
        pass

    def on_game_start(self, state: "GameState", player_idx: int):
        """
        Called when a game starts.

        Args:
            state: Initial game state
            player_idx: This player's index (0-3)
        """
        self.player_idx = player_idx

    def on_game_end(self, state: "GameState", winner: int):
        """
        Called when a game ends.

        Args:
            state: Final game state
            winner: Winning player index
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}')"
