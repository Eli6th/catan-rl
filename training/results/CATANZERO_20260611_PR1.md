# Compact CatanZero rerun with performance PR #1

## Configuration

- 10 training minutes on laptop CPU.
- 192-wide student, four parallel self-play actors.
- 24 AlphaBot bootstrap games across eight workers.
- 4,454 bootstrap decisions generated in 5.3 seconds.
- Official hidden information and root re-determinization.
- Progressive 4/8/16-simulation MCTS curriculum.
- PR #1 transposed sparse-column Rust inference.

Artifacts are under
`training/runs/20260611-210828-catanzero-pr1/`.

## Inference

The 512-wide CTNN trunk benchmark measured 13.7 microseconds per forward
pass on a realistic observation. The original layout was approximately
430 microseconds, so the isolated network path is about 31 times faster.

## Checkpoints

| Checkpoint | H-v1 | H-v2 | Legacy PPO | AlphaBot | Previous | Score |
|---|---:|---:|---:|---:|---:|---:|
| bootstrap | 0/8 | 0/8 | 0/8 | 0/4 | n/a | 0.000 |
| game 376 | 0/8 | 0/8 | 0/8 | 0/4 | 2/8 | 0.000 |
| game 608 | 0/8 | 0/8 | 0/8 | 0/4 | 3/8 | 0.000 |
| game 740 | 0/8 | 0/8 | 0/8 | 0/4 | 2/8 | 0.000 |
| game 756 | 1/8 | 0/8 | 0/8 | 1/4 | 2/8 | 0.075 |

The run completed 756 games, 352,819 decisions, 285,258 learner samples,
and 84 optimizer updates.

## Fresh holdout

The selected game-756 checkpoint was evaluated on new seeds:

| Opponent | Wins | Average VP |
|---|---:|---:|
| Heuristic v1 | 0/24 | 3.33 |
| Heuristic v2 | 1/24 | 2.96 |
| Legacy PPO | 0/24 | 2.71 |
| AlphaBot | 0/12 | 2.42 |
| Prior CatanZero best | 1/24 | 2.71 |

## Verdict

The systems changes worked: bootstrap generation, parallel actor rounds,
hidden-information search, checkpoint evaluation, and optimized Rust
inference all ran successfully. The compact model did not retain the
strength of the previous 512-wide CatanZero model. Its small checkpoint
win against AlphaBot did not reproduce on the larger holdout.

The next run should not promote this checkpoint. The highest-leverage
change is a substantially larger AlphaBot demonstration set before
self-play, followed by a 256- or 384-wide student if the 192-wide model
still cannot imitate the teacher.
