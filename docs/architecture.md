# Architecture Direction

This repository now treats the codebase as four layers:

1. `engine/`
   Pure game rules, state transitions, and simulation logic. No web or transport code. This is the reference implementation.
2. `rust/`
   Rust port of the engine (`catan-core`), verified against the Python engine by differential replay tests, plus the parallel simulation CLI (`catan-sim`). All bulk simulation runs here.
3. `catan_service/`
   Canonical service layer. Owns contracts, state serialization, replay loading, live session management, and the Flask app factory.
4. `visualizer/`
   Canonical frontend surface. This is the long-term base for 3D live play and replay visualization.

The legacy `interactive/` and `web/` prototypes and the Python `benchmarks/`
package have been removed; the Rust simulator supersedes them.

## Canonical API Direction

The cleanup establishes stable internal contracts for:

- `GameSessionSummary`
- `GameStateView`
- `ActionRequest`
- `ActionResult`
- `ReplaySummary`
- `ReplayData`
- `BotTurnRequest`
- `BotTurnResponse`

These live in `catan_service/contracts.py` and are transport-neutral. The current Flask transport is in `catan_service/flask_app.py`.

## Bot Integration Direction

The service cleanup assumes remote HTTP bots first:

- bots are hosted outside the service
- the service will eventually send turn payloads to registered endpoints
- uploaded sandboxed bot execution is explicitly deferred

## Current Supported Entrypoints

- Bulk simulation (fast): `rust/target/release/catan-sim`
- Simulation CLI (Python reference): `python run_simulation.py`
- Canonical service + 3D frontend: `python -m visualizer.server`
