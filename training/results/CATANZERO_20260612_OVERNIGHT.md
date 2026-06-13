# CatanZero 512 overnight experiment campaign

## Verdict

The existing bootstrap-v2 checkpoint remains the champion:

`training/runs/20260611-512-bootstrap-v2/best.pt`

No new checkpoint produced a repeatable general-policy improvement. Several
small evaluations appeared positive, but larger paired-seat holdouts rejected
those promotions. The campaign did produce a materially stronger evaluation
and experiment system, and it identified the next useful algorithmic change.

## What ran

All students were 512-wide warm starts. The campaign tested:

- league-heavy historical and incumbent sampling;
- stronger PPO anchoring;
- reduced auxiliary-loss weight;
- a 20% uniform-exploration reserve;
- increased MCTS simulations;
- league-only continuation with a lower learning rate;
- a combined league/exploration continuation;
- a 12k-sample AlphaBot subset plus 64-game, eight-determinization DAgger;
- a final half-learning-rate league/exploration run.

Artifacts are under:

- `training/runs/20260612-overnight-512/`
- `training/runs/20260612-policy-explore-refine/`
- `training/runs/20260612-league-explore-refine/`
- `training/runs/20260612-league-explore-low-lr/`

## Evaluation correction

The original evaluator changed both board seed and candidate seat each game.
That accidentally correlated seat assignment with board difficulty. Identical
models scored as low as 10/96 instead of the expected 24/96.

The corrected evaluator:

1. reuses each board seed with the candidate in all four seats;
2. runs all six two-seat candidate lineups for balanced 2-vs-2 comparison;
3. requires at least 50% balanced performance for champion promotion.

Identical-policy controls now produce exactly 25% in 1-vs-3 and 50% in 2-vs-2.

## Experiment results

The initial small-sample matrix looked promising but was not stable:

| profile | initial 1-vs-3 | reverse | initial fixed delta |
|---|---:|---:|---:|
| league focus | 16/48 | 7/48 | -0.033 |
| PPO anchor | 10/48 | 14/48 | +0.008 |
| policy heavy | 15/48 | 10/48 | -0.008 |
| exploration reserve | 15/48 | 11/48 | +0.033 |
| more search | 11/48 | 11/48 | -0.033 |
| league-only low LR | 13/48 | 20/48 | +0.050 |
| targeted reanalysis | 10/48 | 19/48 | +0.050 |

Those numbers were generated before the paired-seat correction and are retained
only as evidence of why the evaluator needed to change.

The strongest corrected challenger was the half-learning-rate
league/exploration checkpoint at game 114. On one independent holdout it scored:

| opponent | challenger | bootstrap-v2 |
|---|---:|---:|
| heuristic v1 | 36/48 | 30/48 |
| heuristic v2 | 20/48 | 26/48 |
| legacy PPO | 7/48 | 11/48 |
| AlphaBot | 14/24 | 9/24 |
| heuristic v2, search-8 | 14/24 | 16/24 |

It also scored 51/96 in balanced 2-vs-2 and 15/48 against the pre-bootstrap
CatanZero model, versus 11/48 in reverse. A larger fresh direct test did not
confirm promotion:

| direct test | result |
|---|---:|
| balanced 2-vs-2 | 370/768 (48.2%) |
| challenger as singleton | 100/384 (26.0%) |
| bootstrap-v2 as singleton | 91/384 (23.7%) |

The mixed result is not strong enough to replace the incumbent.

## What the experiments say

- EM rapidly saturates at a 0.90 policy mixture. Holding back 20% uniform
  exploration can improve some short holdouts, but it is not sufficient alone.
- League pressure produces useful early checkpoints, then performance often
  degrades. Smaller learning rates delay but do not remove the overshoot.
- More MCTS simulations reduced throughput without producing stronger policy
  targets in this budget.
- Reanalysis against the old AlphaBot teacher is now counterproductive. The
  12k subset reached about 63% validation agreement, but the resulting game
  policy collapsed and never recovered its incumbent matchup.
- Fixed bots and self-play measure different capabilities. AlphaBot and
  heuristic gains can coexist with regression against the current population,
  so both gates are required.

## Research implications

The current code adds Gumbel noise after search, but it does not implement the
low-budget policy-improvement operator from
[Gumbel AlphaZero](https://openreview.net/forum?id=bERaNdoegnO). The next
search change should be root action sampling without replacement plus
sequential halving.

Population training should move from uniform historical sampling toward a
loss-rate-weighted meta-solver, following the motivation behind
[PSRO](https://arxiv.org/abs/1711.00832) and
[Neural Fictitious Self-Play](https://arxiv.org/abs/1603.01121). For hidden
information, [ReBeL](https://arxiv.org/abs/2007.13544) supports tracking a
public belief representation rather than relying only on independent root
re-determinizations.

The practical next experiment is therefore not a larger network or more stale
teacher imitation. It is proper low-simulation Gumbel search, paired-seat
evaluation from the start, and opponent sampling weighted toward checkpoints
the current policy actually loses to.
