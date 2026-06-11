"""Transport-neutral contracts for the canonical Catan service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class PlayerView:
    """Canonical player payload shared by live play and replays."""

    id: int
    name: str
    resources: list[int]
    dev_cards: list[int] = field(default_factory=list)
    knights_played: int = 0
    victory_points: int = 0


@dataclass(slots=True)
class AchievementView:
    """Current special-achievement ownership."""

    longest_road_player: int
    longest_road_length: int
    largest_army_player: int
    largest_army_size: int


@dataclass(slots=True)
class BoardView:
    """Board payload stable across transports."""

    tile_resources: list[int]
    tile_numbers: list[int]
    port_types: list[int]
    vertices: list[int]
    edges: list[int]
    robber_tile: int


@dataclass(slots=True)
class ValidActionView:
    """Canonical action representation returned by the service."""

    type: str
    player: int
    payload: list[int] | None


@dataclass(slots=True)
class GameSessionSummary:
    """High-level session metadata."""

    id: str
    mode: Literal["human", "bot-ready"] = "human"
    player_names: list[str] = field(default_factory=list)
    num_players: int = 4
    status: Literal["active", "finished"] = "active"
    current_player: int = 0
    winner: int = -1
    version: int = 0


@dataclass(slots=True)
class GameStateView:
    """Canonical state shape for both UI and future bot integrations."""

    session: GameSessionSummary
    phase: str
    turn_phase: str
    turn: int
    dice_roll: int
    has_rolled: bool
    board: BoardView
    players: list[PlayerView]
    achievements: AchievementView
    bank: list[int]
    valid_actions: list[ValidActionView] = field(default_factory=list)


@dataclass(slots=True)
class ActionRequest:
    """Transport-neutral action request."""

    type: str
    player: int
    payload: list[int] | None = None


@dataclass(slots=True)
class ActionResult:
    """Outcome of executing an action."""

    accepted: bool
    reason: str | None
    resulting_version: int | None
    state: GameStateView | None = None


@dataclass(slots=True)
class ReplaySummary:
    """Metadata for a replay file."""

    id: str
    source: str
    num_players: int
    action_count: int
    seed: int


@dataclass(slots=True)
class ReplayData:
    """Replay payload compatible with the canonical board model."""

    summary: ReplaySummary
    board: BoardView
    actions: list[ValidActionView]


@dataclass(slots=True)
class BotTurnRequest:
    """Future-facing remote bot callback contract."""

    session: GameSessionSummary
    bot_player_id: int
    visible_state: GameStateView
    valid_actions: list[ValidActionView]
    deadline_ms: int


@dataclass(slots=True)
class BotTurnResponse:
    """Future-facing remote bot callback result."""

    action: ActionRequest | None
    error: str | None = None


def contract_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass contract to a JSON-friendly dict."""

    return asdict(value)
