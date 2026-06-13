# Hill-Climb Report: CatanZero Competitive Policy

## Baseline

- Policy: `training/runs/20260611-catanzero-v2/best.ctnn`
- Inference: `hybrid-v2`, strategy settlement weight 5, opening production
  weight 1, heuristic refinement, endgame conversion, road refinement.
- Scope: four fresh 96-game paired-seat seeds per opponent.

| Opponent | Wins | Games | Win rate |
|---|---:|---:|---:|
| heuristic-v1 | 301 | 384 | 78.4% |
| heuristic-v2 | 291 | 384 | 75.8% |
| AlphaBot | 290 | 384 | 75.5% |
| aggregate | 882 | 1152 | 76.6% |

The 90% target was not reached. Individual 32-game slices sometimes exceeded
90%, but no improvement retained that rate across fresh 96-game holdouts.

## Strengths

- Development-card engine: wins average roughly six dev-card purchases and
  usually convert Largest Army.
- Winning positions convert cleanly: average winning score is above 7 VP with
  margins around 2.6 VP.
- Opening production is not the primary bottleneck. Win/loss opening-income
  differences are small and inconsistent.
- Road-location refinement is a real improvement over neural road selection.
- The policy is strong from all seats relative to a 25% four-player baseline,
  although seat-specific variance remains material.

## Weaknesses

- Award control collapses in losses. Against heuristic-v2, the winner holds
  Longest Road in 45.8% and Largest Army in 62.5% of losses; the candidate
  holds them in only 8.3% and 16.7%.
- AlphaBot exploits Longest Road. It holds the award in 65.2% of candidate
  losses.
- Expansion and city conversion lag in losses. Against heuristic-v2 losses
  average 4.38 VP and 0.71 cities; against AlphaBot losses average 4.22 VP and
  0.43 cities.
- The policy is seed-sensitive. A fine-tuned checkpoint beat the incumbent by
  6/288 on one paired holdout and lost by 9/288 on another.
- Training and deployment are misaligned. Self-play optimizes the raw neural
  policy plus training MCTS, while evaluation uses deterministic opening,
  conversion, and road refinements. The learner does not train on the actions
  its deployed policy actually takes.

## Changes Made

- Fixed fine-tuning anchors to use the incumbent CatanZero champion instead of
  pulling updates toward the older PPO model.
- Added `heuristic_mix`, a conservative curriculum covering both scripted
  heuristic versions with stronger incumbent anchoring.
- Fixed terminal reinforcement and elite weighting to apply to all fixed
  opponents, not only AlphaBot.
- Added `training/tune_hybrid_inference.py` for reproducible multi-seed hybrid
  policy tuning.
- Retained road-location refinement.

## Validation

- `cargo check --release --manifest-path rust/catan-py/Cargo.toml`
- `.venv/bin/pytest -q tests/test_catanzero.py`
- Multi-seed paired-seat evaluations through
  `training/evaluate_planner_vs_alpha.py`

No new checkpoint was promoted. The original champion remains the most robust
policy.

## Decision Log

| Decision | Rationale | Alternatives | Tradeoff | Status |
|---|---|---|---|---|
| Keep the original champion | Fine-tuned candidates failed fresh paired holdouts | Promote checkpoint 186 | Gives up isolated gains, preserves robustness | architectural |
| Anchor to incumbent champion | PPO anchoring caused policy drift during fine-tuning | Lower learning rate alone | Requires a compatible champion | architectural |
| Retain balanced heuristic curriculum | Single-opponent training moved wins between opponents | Heuristic-v2-only curriculum | Slower specialization | experimental |
| Reject road denial | Improved one 32-game screen but regressed a 96-game holdout | Larger denial weights | No retained complexity | experimental |
| Reject forced award conversion | Mostly neutral across opponents | Force all immediate VP moves | Avoids brittle tactical overrides | experimental |
| Reject competitor-mix checkpoint | AlphaBot training reduced heuristic robustness | Promote by internal gate | Internal gate remains noisy | experimental |
| Require multi-seed promotion | Single-seed variance is several percentage points | One 96-game gate | More evaluation cost, much lower false promotion risk | architectural |

## Next Hill

Generate self-play and fixed-opponent trajectories with the exact deployed
hybrid policy, then train the network on those decisions with terminal/VP
advantages and incumbent KL anchoring. Promotion should require at least four
fresh paired 96-game seeds per opponent and a lower confidence bound above the
incumbent, rather than a single noisy checkpoint gate.
