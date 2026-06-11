"""Canonical Flask transport for the Catan service."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

from .contracts import ActionRequest
from .replays import ReplayService
from .serialization import to_jsonable
from .sessions import SessionService


def create_app(
    *,
    frontend_dir: Path | None = None,
    log_dir: Path | str = "logs",
    session_service: SessionService | None = None,
    replay_service: ReplayService | None = None,
) -> Flask:
    """Create the canonical service app."""

    app = Flask(__name__)
    CORS(app)

    sessions = session_service or SessionService()
    replays = replay_service or ReplayService(log_dir)

    def payload(data, status: int = 200):
        return jsonify(to_jsonable(data)), status

    @app.route("/api/health", methods=["GET"])
    def health():
        return payload(
            {
                "service": "catan-service",
                "frontend": "visualizer",
                "bot_integration": "remote-http-first",
            }
        )

    @app.route("/api/contracts", methods=["GET"])
    def contracts():
        return payload(
            {
                "game_session_summary": {
                    "id": "string",
                    "mode": "human|bot-ready",
                    "player_names": ["string"],
                    "num_players": 4,
                    "status": "active|finished",
                    "current_player": 0,
                    "winner": -1,
                    "version": 0,
                },
                "action_request": {
                    "type": "BUILD_ROAD",
                    "player": 0,
                    "payload": [12],
                },
                "bot_turn_request": {
                    "session": "GameSessionSummary",
                    "bot_player_id": 0,
                    "visible_state": "GameStateView",
                    "valid_actions": ["ValidActionView"],
                    "deadline_ms": 5000,
                },
                "bot_turn_response": {
                    "action": "ActionRequest|null",
                    "error": "string|null",
                },
            }
        )

    @app.route("/api/games", methods=["GET"])
    def list_games():
        return payload({"games": sessions.list_games()})

    @app.route("/api/games", methods=["POST"])
    def create_game():
        data = request.get_json(silent=True) or {}
        num_players = int(data.get("num_players", 4))
        player_names = data.get("player_names")
        if not 2 <= num_players <= 4:
            return payload({"error": "num_players must be between 2 and 4"}, 400)
        try:
            state = sessions.create_game(num_players, player_names)
        except RuntimeError as exc:
            return payload({"error": str(exc)}, 503)
        return payload({"state": state}, 201)

    @app.route("/api/games/<session_id>", methods=["GET"])
    def get_game(session_id: str):
        try:
            return payload({"game": sessions.get_summary(session_id)})
        except KeyError:
            return payload({"error": "game not found"}, 404)

    @app.route("/api/games/<session_id>/state", methods=["GET"])
    def get_state(session_id: str):
        try:
            return payload({"state": sessions.get_state(session_id)})
        except KeyError:
            return payload({"error": "game not found"}, 404)

    @app.route("/api/games/<session_id>/actions", methods=["GET"])
    def get_actions(session_id: str):
        try:
            return payload({"actions": sessions.get_actions(session_id)})
        except KeyError:
            return payload({"error": "game not found"}, 404)

    @app.route("/api/games/<session_id>/actions", methods=["POST"])
    def submit_action(session_id: str):
        data = request.get_json(silent=True) or {}
        try:
            action_request = ActionRequest(
                type=str(data["type"]),
                player=int(data["player"]),
                payload=data.get("payload"),
            )
        except (KeyError, TypeError, ValueError):
            return payload({"error": "invalid action payload"}, 400)
        try:
            result = sessions.execute_action(session_id, action_request)
        except KeyError:
            return payload({"error": "game not found"}, 404)
        status = 200 if result.accepted else 400
        return payload(result, status)

    @app.route("/api/games/<session_id>/poll", methods=["GET"])
    def poll_state(session_id: str):
        version = request.args.get("version", default=-1, type=int)
        try:
            return payload(sessions.poll_state(session_id, version))
        except KeyError:
            return payload({"error": "game not found"}, 404)

    @app.route("/api/games/<session_id>", methods=["DELETE"])
    def delete_game(session_id: str):
        deleted = sessions.delete_game(session_id)
        if not deleted:
            return payload({"error": "game not found"}, 404)
        return payload({"deleted": True})

    @app.route("/api/replays", methods=["GET"])
    def list_replays():
        return payload({"replays": replays.list_replays()})

    @app.route("/api/replays/<replay_id>", methods=["GET"])
    def get_replay(replay_id: str):
        try:
            return payload({"replay": replays.load_replay(replay_id)})
        except FileNotFoundError:
            return payload({"error": "replay not found"}, 404)

    @app.route("/api/logs", methods=["GET"])
    def legacy_list_logs():
        return payload({"replays": replays.list_replays()})

    @app.route("/api/logs/<replay_id>", methods=["GET"])
    def legacy_get_log(replay_id: str):
        try:
            return payload({"replay": replays.load_replay(replay_id)})
        except FileNotFoundError:
            return payload({"error": "replay not found"}, 404)

    if frontend_dir is not None:
        frontend_dir = Path(frontend_dir)

        @app.route("/", defaults={"path": "index.html"})
        @app.route("/<path:path>")
        def serve_frontend(path: str):
            if path.startswith("api/"):
                abort(404)
            target = frontend_dir / path
            if target.exists():
                return send_from_directory(frontend_dir, path)
            index = frontend_dir / "index.html"
            if index.exists():
                return send_from_directory(frontend_dir, "index.html")
            return (
                "Visualizer frontend not built. Run 'npm install' and 'npm run build' in visualizer/.",
                404,
            )

    return app
