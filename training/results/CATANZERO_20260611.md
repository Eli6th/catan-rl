# CatanZero bounded run: 2026-06-11

## Configuration

- Six wall-clock minutes on laptop CPU.
- 868 training games.
- Checkpoints after games 582, 790, and 868.
- Official hidden information with root re-determinization.
- Progressive 4/8/16-simulation search curriculum.
- Legacy-policy anchor weights 0.50/0.25/0.10.
- Greedy forward-pass evaluation; MCTS is the training teacher.

Artifacts are under `training/runs/20260611-catanzero-v2/`. The selected
model is `best.pt`, copied from `checkpoints/game_0582.pt`.

## Checkpoints

The checkpoint gate weights heuristic v1 20%, heuristic v2 20%, legacy
PPO 40%, and AlphaBot 20%.

| Checkpoint | H-v1 | H-v2 | Legacy PPO | AlphaBot | Previous | Score |
|---|---:|---:|---:|---:|---:|---:|
| game 582 | 58.3% | 50.0% | 16.7% | 50.0% | n/a | 0.383 |
| game 790 | 41.7% | 58.3% | 25.0% | 16.7% | 25.0% | 0.333 |
| game 868 | 58.3% | 66.7% | 8.3% | 50.0% | 41.7% | 0.383 |

Checkpoint evaluations used 12 games per ordinary opponent and six
against AlphaBot, so they are selection signals rather than final claims.

## Seed-matched confirmation

The selected checkpoint was rerun on the same seed blocks used for the
legacy baseline.

| Agent | H-v1 (24) | H-v2 (24) | AlphaBot (12) |
|---|---:|---:|---:|
| Legacy PPO | 62.5% | 66.7% | 8.3% |
| CatanZero best | 58.3% | 66.7% | 25.0% |

CatanZero best won 5/24 (20.8%) direct games against legacy PPO, 6/24
(25.0%) against checkpoint 790, and 2/24 (8.3%) against checkpoint 868.
The league is non-transitive: the broad fixed-opponent gate chose game 582,
while later policies beat it directly.

## Verdict

The system implementation is functional and numerically stable after
fixing illegal-action cross-entropy. This short run did not produce a
strict replacement for the legacy PPO policy: it matched the stronger
heuristic and improved the AlphaBot matchup, but regressed slightly
against heuristic v1 and substantially against legacy PPO.

Current search is root-determinized IS-MCTS with a small, unbatched budget.
It does not re-determinize at every acting player's node, and its policy
model approximates every search actor. Those are the first constraints to
remove before spending substantially more training time.
