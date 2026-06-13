# CatanZero 512-wide bootstrap v2

## Configuration

- 512-wide model warm-started from the prior CatanZero champion.
- 96 initial demonstration games and 48 DAgger games.
- Eight parallel data-generation workers.
- Four AlphaBot hidden-state re-determinizations per labeled state.
- Teacher, champion, heuristic, legacy, and student trajectory coverage.
- Policy-only bootstrap with trajectory-disjoint validation and early stopping.
- Gradual auxiliary-loss scale of 0.10, 0.50, and 1.00.
- Four parallel frozen self-play actors.
- 10 minutes of self-play after bootstrap.

Artifacts are under `training/runs/20260611-512-bootstrap-v2/`.
The selected checkpoint is `checkpoints/game_1080.pt`.

## Bootstrap

| Phase | States | Best epoch | Validation agreement | Validation CE |
|---|---:|---:|---:|---:|
| Initial | 34,116 | 15 | 52.5% | 1.001 |
| Initial + DAgger | 45,130 | 3 | 58.9% | 0.917 |

The previous single-determinization bootstrap used 4,454 states and
produced only about 45-46% agreement on those same training states.

The bootstrap checkpoint already scored 6/8 against heuristic v1 and
4/8 against heuristic v2. It was correctly rejected by the champion gate
after scoring 0/8 against the prior champion.

## Training

- 1,096 games.
- 250,788 decisions.
- 217,622 learner samples.
- 83 optimizer updates.
- 41.2% learner-seat win rate across the mixed curriculum.

| Checkpoint | H-v1 | H-v2 | PPO | AlphaBot | Champion | Score | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| game 768 | 4/8 | 5/8 | 1/8 | 2/4 | 3/8 | 0.375 | yes |
| game 984 | 7/8 | 7/8 | 1/8 | 0/4 | 4/8 | 0.400 | yes |
| game 1080 | 5/8 | 4/8 | 2/8 | 2/4 | 3/8 | 0.425 | yes |
| game 1096 | 5/8 | 4/8 | 1/8 | 2/4 | 1/8 | 0.375 | no |

The final checkpoint was rejected despite reasonable fixed-opponent
scores because it fell below fair share against the champion.

## Fresh holdout

| Opponent or mode | Wins | Average VP |
|---|---:|---:|
| Heuristic v1, greedy | 12/24 | 6.00 |
| Heuristic v2, greedy | 15/24 | 6.21 |
| Legacy PPO, greedy | 2/24 | 4.33 |
| AlphaBot, greedy | 4/12 | 5.33 |
| Prior champion, greedy | 9/24 | 5.54 |
| Heuristic v2, search-8 | 4/12 | 5.58 |
| Prior champion, search-8 | 4/12 | 4.75 |

The one-candidate-versus-three-opponents format has a 25% fair share.
The selected checkpoint's 37.5% result against the prior champion
therefore clears the promotion threshold on fresh seeds.

## Expanded confirmation

An additional independent seed block produced:

| Opponent | Wins | Average VP |
|---|---:|---:|
| Prior champion | 12/48 | 4.46 |
| Legacy PPO | 14/48 | 4.65 |
| AlphaBot | 11/24 | 5.79 |

The new model is exactly fair share against the champion on this larger
block. Across both fresh champion blocks it scored 21/72, or 29.2%.
This supports competitiveness with the champion, but not a decisive
strength claim.

## Verdict

The original bootstrap was not merely undersized; its teacher signal was
diluted and it lacked a held-out stopping criterion. Multi-information-set
labels, policy-only fitting, DAgger coverage, champion warm-starting, and
the promotion gate fixed the failure.

The new checkpoint is competitive with the prior champion and clearly
stronger than the failed 192- and 1024-wide bootstrap runs. The larger
confirmation also shows meaningful PPO and AlphaBot strength. Search-8
does not consistently improve greedy inference, so search calibration
should be handled separately rather than increasing rollout budget
blindly.
