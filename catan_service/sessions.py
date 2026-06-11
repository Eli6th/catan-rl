"""Session lifecycle and canonical state/action mapping."""

from __future__ import annotations

import time
import uuid
from typing import Optional

import numpy as np

from engine.game import Action, ActionType, CatanGame, GamePhase, TurnPhase

from .contracts import (
    AchievementView,
    ActionRequest,
    ActionResult,
    BoardView,
    GameSessionSummary,
    GameStateView,
    PlayerView,
    ValidActionView,
)

TURN_PHASE_NAMES = {
    TurnPhase.PRE_ROLL: "pre_roll",
    TurnPhase.MUST_ROLL: "must_roll",
    TurnPhase.ROBBER_DISCARD: "robber_discard",
    TurnPhase.ROBBER_MOVE: "robber_move",
    TurnPhase.ROBBER_STEAL: "robber_steal",
    TurnPhase.MAIN: "main",
    TurnPhase.ROAD_BUILDING: "road_building",
}

GAME_PHASE_NAMES = {
    GamePhase.SETUP_FORWARD: "setup_forward",
    GamePhase.SETUP_BACKWARD: "setup_backward",
    GamePhase.PLAYING: "playing",
    GamePhase.FINISHED: "finished",
}


class GameSession:
    """In-memory live game session."""

    def __init__(self, session_id: str, game: CatanGame, player_names: list[str]):
        self.id = session_id
        self.game = game
        self.player_names = player_names
        self.created_at = time.time()
        self.last_action_at = time.time()
        self.version = 0


class SessionService:
    """Manages live sessions using canonical API contracts."""

    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        self._sessions: dict[str, GameSession] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout

    def create_game(
        self, num_players: int, player_names: Optional[list[str]] = None
    ) -> GameStateView:
        self._cleanup_old_sessions()
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError("maximum sessions reached")

        session_id = self._new_session_id()
        game = CatanGame(num_players=num_players)
        if player_names is None:
            default_colors = ["Red", "Blue", "White", "Orange"]
            player_names = [
                f"Player {index + 1} ({default_colors[index]})"
                for index in range(num_players)
            ]
        session = GameSession(session_id, game, player_names[:num_players])
        self._sessions[session_id] = session
        return self.get_state(session_id)

    def list_games(self) -> list[GameSessionSummary]:
        return [self._summary(session) for session in self._sessions.values()]

    def get_summary(self, session_id: str) -> GameSessionSummary:
        session = self._require_session(session_id)
        return self._summary(session)

    def get_state(self, session_id: str) -> GameStateView:
        session = self._require_session(session_id)
        session.last_action_at = time.time()
        return self._state_view(session)

    def get_actions(self, session_id: str) -> list[ValidActionView]:
        session = self._require_session(session_id)
        return self._valid_actions(session.game.get_valid_actions())

    def execute_action(self, session_id: str, request: ActionRequest) -> ActionResult:
        session = self._require_session(session_id)
        action = self._request_to_action(request)
        if action is None:
            return ActionResult(
                accepted=False,
                reason="invalid action request",
                resulting_version=None,
            )

        valid_actions = session.game.get_valid_actions()
        if not self._action_in_list(action, valid_actions):
            return ActionResult(
                accepted=False,
                reason="action is not currently valid",
                resulting_version=session.version,
            )

        success = session.game.execute_action(action)
        if not success:
            return ActionResult(
                accepted=False,
                reason="engine rejected action",
                resulting_version=session.version,
            )

        session.version += 1
        session.last_action_at = time.time()
        state = self._state_view(session)
        return ActionResult(
            accepted=True,
            reason=None,
            resulting_version=session.version,
            state=state,
        )

    def delete_game(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def poll_state(self, session_id: str, version: int) -> dict[str, object]:
        session = self._require_session(session_id)
        changed = session.version > version
        payload: dict[str, object] = {"changed": changed}
        if changed:
            payload["state"] = self._state_view(session)
        return payload

    def _new_session_id(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        while session_id in self._sessions:
            session_id = str(uuid.uuid4())[:8]
        return session_id

    def _cleanup_old_sessions(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_action_at > self.session_timeout
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def _require_session(self, session_id: str) -> GameSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def _summary(self, session: GameSession) -> GameSessionSummary:
        game = session.game
        return GameSessionSummary(
            id=session.id,
            player_names=list(session.player_names),
            num_players=game.state.num_players,
            status="finished" if game.is_game_over() else "active",
            current_player=game.get_current_player(),
            winner=int(game.get_winner()),
            version=session.version,
            mode="bot-ready",
        )

    def _state_view(self, session: GameSession) -> GameStateView:
        game = session.game
        state = game.state
        players = [
            PlayerView(
                id=player_index,
                name=session.player_names[player_index],
                resources=state.resources[player_index].tolist(),
                dev_cards=state.dev_cards[player_index].tolist(),
                knights_played=int(state.knights_played[player_index]),
                victory_points=int(state.calculate_victory_points(player_index)),
            )
            for player_index in range(state.num_players)
        ]
        achievements = AchievementView(
            longest_road_player=int(state.longest_road_player),
            longest_road_length=int(state.longest_road_length),
            largest_army_player=int(state.largest_army_player),
            largest_army_size=int(state.largest_army_size),
        )
        board = BoardView(
            tile_resources=state.tile_resources.tolist(),
            tile_numbers=state.tile_numbers.tolist(),
            port_types=state.port_types.tolist(),
            vertices=state.vertices.tolist(),
            edges=state.edges.tolist(),
            robber_tile=int(state.robber_tile),
        )
        return GameStateView(
            session=self._summary(session),
            phase=GAME_PHASE_NAMES[game.game_phase],
            turn_phase=TURN_PHASE_NAMES.get(game.turn_phase, "unknown"),
            turn=int(state.turn),
            dice_roll=int(state.dice_roll),
            has_rolled=bool(state.has_rolled),
            board=board,
            players=players,
            achievements=achievements,
            bank=state.bank.tolist(),
            valid_actions=self._valid_actions(game.get_valid_actions()),
        )

    @staticmethod
    def _valid_actions(actions: list[Action]) -> list[ValidActionView]:
        return [
            ValidActionView(
                type=action.action_type.name,
                player=action.player,
                payload=action.data.tolist() if action.data is not None else None,
            )
            for action in actions
        ]

    @staticmethod
    def _request_to_action(request: ActionRequest) -> Action | None:
        try:
            action_type = ActionType[request.type]
        except KeyError:
            return None

        payload = None
        if request.payload is not None:
            payload = np.array(request.payload, dtype=np.int8)
        return Action(action_type=action_type, player=request.player, data=payload)

    @staticmethod
    def _action_in_list(action: Action, valid_actions: list[Action]) -> bool:
        for valid in valid_actions:
            if action.action_type != valid.action_type:
                continue
            if action.player != valid.player:
                continue
            if action.data is None and valid.data is None:
                return True
            if action.data is not None and valid.data is not None:
                if np.array_equal(action.data, valid.data):
                    return True
        return False
