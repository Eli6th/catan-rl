# Catan RL — Experiment Ledger

All experiments 2026-06-10/11, run locally (Apple Silicon, CPU training at
~16–20k policy-steps/sec). Game: 4-player, first-to-7 VP, perfect
information. Primary metric: **fixed-seed gate** = win rate over 192 games
(seed 777, greedy policy seat 0) vs three frozen Heuristic-v1 — fair share
25%. The in-run eval curves use varying seeds (±5pt noise at 96 games).

Raw logs: `logs/` (gzipped stdout of every run; runB2 also has its full
metrics JSONL stream — earlier runs' streams were overwritten in the shared
dashboard file and only survive as the per-run stdout).
Checkpoints/replays/configs: `../runs/<run>/`. Elo snapshots: `../runs/elo.json`.

## Reference agents (no training)

| Agent | vs 3×Heuristic-v1 | Notes |
|---|---|---|
| Random | ~0% | never finishes before a heuristic |
| Heuristic-v1 (4th seat) | 25.7% | fair-share baseline, frozen anchor |
| Heuristic-v2 (GA-evolved) | 39.9–40.5% | `catan-sim --evolve`, 12 gens; frozen anchor |
| RolloutBot 12 rollouts / 40-turn horizon | 43.3% | flat Monte Carlo, random playouts |
| **RolloutBot 48 / 80** | **72.5%** | proves ceiling ≥73%; planning scales |

## Training runs (chronological)

| Run | Config | Steps | Final eval | Fixed-seed gate |
|---|---|---|---|---|
| stage1 | 256 net, pure self-play, vp_delta 0.05, 10 min | 9.3M | 92% vs random / 14.1% vs heur | — |
| stage2 | resumed, self-play, dual eval, 12 min | +13.1M | 99% / 28.1% | — |
| stage2b | resumed, mixed seats (2×heuristic), 12 min | +12.5M | 100% / 55.7% | 47.7% |
| overnight | resumed, mixed, anneal planned, stopped at 1h22m (plateau) | +79.3M | ~66% flat | **65.1%** |
| runA | resumed, entropy 0.01→0.02, opponents heuristic+v2, 35 min | +39.1M | 69.8% | **64.2%** |
| runB | fresh 2×512, anneal 0.05→0, entropy 0.02, 50 min | 43.4M | 57.0% | 54.2% (still climbing) |
| runB2 | runB resumed, 60 min | +46.5M | 67.2% | **65.6%** |

## The plateau investigation

Symptom: overnight run flat at 66.0% (final-quarter slope literally 0.0).

Hypotheses tested, each by controlled experiment:

1. **More training** → overnight run: slope 0.0 over its final quarter. ✗
2. **Exploration / local optimum** → runA entropy 2×: gate 64.2 vs 65.1. ✗
3. **Opponent overfitting** → runA added evolved-v2 seats: (same run) no change. ✗
4. **Network width** → runB/B2 4× params: gate 65.6 vs 65.1. ✗
5. **Headroom exists at all?** → deep RolloutBot 72.5% ≫ 65%; loss anatomy:
   44% of policy losses end ≤4 VP (outplayed, not out-lucked), losses LONGER
   than wins (76 vs 66 turns — failed grinds, not opponent racing). ✓✓

Supporting signal: explained variance 0.75–0.87 in every run — the value
head evaluates positions well throughout. The deficit is action selection.

**Conclusion: the flat-MLP reactive policy converges to a ~65% gate against
Heuristic-v1 regardless of training time, exploration, opponent mix, or
width. An untrained planner exceeds it. The binding constraints are
structural: (1) one forward pass cannot do the multi-step tactical
reasoning that search demonstrably converts into wins; (2) the flat
1,350-float observation discards board topology that a graph-structured
encoder would provide.** This mirrors the historical TD-Gammon → AlphaGo
arc: reactive policies plateau; planning breaks through.

## Elo ladder (first tournament, 576 games)

stage2b 1539 > stage2 1524 > Heuristic-v1 1429 > stage1 1213 > Random 1000
(fixed anchor). Later checkpoints not yet laddered — rerun
`elo.py tournament` with current best.pt + anchors (incl. heuristic_v2).

## AlphaZero-lite (built + measured, 2026-06-11 ~06:00)

AlphaBot (catan-env/src/alpha.rs): root policy-prior pruning (one trunk
forward; top-K candidates by policy logit) + per-candidate random rollouts.
Net runs in pure Rust (net.rs, CTNN self-verifying export via
training/export_net.py). 512 net exported from runB2 best.pt.

Design iterations (120-150 games vs 3x Heuristic-v1 each):

| Variant | Result | Lesson |
|---|---|---|
| prior + 12-turn rollouts + value leaves | 38.3% | value head off-distribution = poison |
| prior + value on immediate after-state | 15.8% | worse — pure value judgment fails |
| prior + value at MY next decision (in-dist) | 25.8% | still fails: a PPO critic is a baseline, not an outcome regressor |
| prior + FULL win/loss rollouts (12) | 43.3% | = RolloutBot: prior neutral at small budget |
| prior(10) + 48 full rollouts | 74.2% | prior pays at depth (vs 72.5% unpruned) |
| **prior(8) + 96 full rollouts** | **82.0%** | breakthrough — wall broken |

Headline ladder vs 3x Heuristic-v1: raw policy 65% (gate) -> deep search
72.5% -> **policy-guided search 82.0%**. The plateau diagnosis was correct:
planning + learned priors break the reactive-policy wall.

Key insight for the next phase: the value head must be RETRAINED as a true
outcome regressor on (state, final result) pairs from search games before
it can replace rollouts (the proper AlphaZero loop). The PPO critic's EV
0.75 is aggregate calibration, not per-state precision.

## Next (in order)

1. Search-as-teacher: generate AlphaBot games; retrain value head on
   (state, outcome); distill policy on search choices. Then value leaves
   replace most rollout depth (faster AND stronger).
2. Graph-structured encoder (board topology as structure, not data).
3. Global best-ledger for promotion gate (current gate compares within-run).
4. First-to-10 graduation; realistic visibility (POMDP).
5. PettingZoo adapter; play-vs-agent in the visualizer.
