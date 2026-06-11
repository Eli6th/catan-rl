"""Replay listing and loading helpers for the canonical API surface."""

from __future__ import annotations

from pathlib import Path

from engine.game import Action
from simulation.logger import GameLogger, get_log_files

from .contracts import BoardView, ReplayData, ReplaySummary, ValidActionView


def _action_to_view(action: Action) -> ValidActionView:
    return ValidActionView(
        type=action.action_type.name,
        player=action.player,
        payload=action.data.tolist() if action.data is not None else None,
    )


class ReplayService:
    """Service wrapper around on-disk replay logs."""

    def __init__(self, log_dir: Path | str = "logs"):
        self.log_dir = Path(log_dir)

    def list_replays(self) -> list[ReplaySummary]:
        replays: list[ReplaySummary] = []
        for filepath in get_log_files(self.log_dir):
            replay = GameLogger.load_binary(filepath)
            replays.append(
                ReplaySummary(
                    id=filepath.name,
                    source=str(filepath),
                    num_players=int(replay["num_players"]),
                    action_count=len(replay["actions"]),
                    seed=int(replay["seed"]),
                )
            )
        return replays

    def load_replay(self, replay_id: str) -> ReplayData:
        filepath = self.log_dir / replay_id
        if not filepath.exists():
            raise FileNotFoundError(replay_id)

        replay = GameLogger.load_binary(filepath)
        board = BoardView(
            tile_resources=replay["tile_resources"].tolist(),
            tile_numbers=replay["tile_numbers"].tolist(),
            port_types=replay["port_types"].tolist(),
            vertices=[],
            edges=[],
            robber_tile=self._initial_robber_tile(replay["tile_resources"].tolist()),
        )
        summary = ReplaySummary(
            id=filepath.name,
            source=str(filepath),
            num_players=int(replay["num_players"]),
            action_count=len(replay["actions"]),
            seed=int(replay["seed"]),
        )
        return ReplayData(
            summary=summary,
            board=board,
            actions=[_action_to_view(action) for action in replay["actions"]],
        )

    @staticmethod
    def _initial_robber_tile(tile_resources: list[int]) -> int:
        for index, resource in enumerate(tile_resources):
            if resource == 5:
                return index
        return 0
