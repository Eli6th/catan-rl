from pathlib import Path

from catan_service.contracts import ActionRequest
from catan_service.flask_app import create_app
from catan_service.replays import ReplayService
from catan_service.sessions import SessionService
from engine.game import Action, ActionType, CatanGame
from simulation.logger import GameLogger


def test_session_service_creates_canonical_state():
    service = SessionService()

    state = service.create_game(4, ["A", "B", "C", "D"])

    assert state.session.mode == "bot-ready"
    assert state.session.status == "active"
    assert state.session.player_names == ["A", "B", "C", "D"]
    assert len(state.players) == 4
    assert all(action.type for action in state.valid_actions)


def test_session_service_rejects_invalid_action():
    service = SessionService()
    state = service.create_game(4)

    result = service.execute_action(
        state.session.id,
        request=ActionRequest(type="BUILD_CITY", player=0, payload=[0]),
    )

    assert result.accepted is False
    assert result.reason == "action is not currently valid"


def test_replay_service_lists_and_loads_replays(tmp_path: Path):
    log_dir = tmp_path / "logs"
    logger = GameLogger(log_dir)
    game = CatanGame(4, seed=7)
    logger.start_game(game.state)
    logger.log_action(Action(ActionType.ROLL_DICE, 0))
    path = logger.end_game(game.state)

    assert path is not None

    service = ReplayService(log_dir)
    replays = service.list_replays()

    assert len(replays) == 1
    replay = service.load_replay(replays[0].id)

    assert replay.summary.seed == 7
    assert replay.summary.action_count == 1
    assert replay.board.tile_resources
    assert replay.actions[0].type == "ROLL_DICE"


def test_flask_app_exposes_canonical_routes():
    app = create_app(log_dir="logs")
    client = app.test_client()

    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["service"] == "catan-service"

    create_response = client.post("/api/games", json={"num_players": 4})
    assert create_response.status_code == 201
    state = create_response.get_json()["state"]

    actions_response = client.get(f"/api/games/{state['session']['id']}/actions")
    assert actions_response.status_code == 200
    assert "actions" in actions_response.get_json()
