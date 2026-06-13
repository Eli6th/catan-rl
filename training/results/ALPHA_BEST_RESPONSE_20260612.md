# AlphaBot best-response promotion - 2026-06-12

## Promoted policy

- Network: `training/runs/20260611-catanzero-v2/best.ctnn`
- Width: 512
- Visibility: official realistic hidden information
- Initial settlement policy:
  - Heuristic-v2 production score
  - Second-settlement coherent-resource bonus with weight `5.0`
- Tactical refinement:
  - The network chooses the action type.
  - Heuristic-v2 chooses among legal actions of that type for settlements,
    cities, robber movement, steals, discards, and trade responses.
  - Roads remain neural because the shipped heuristic road choice is random.

## Promotion result

All AlphaBot evaluations use three shipped AlphaBots at `8x96d300`, realistic
visibility, first-to-7, and paired boards across all four candidate seats.

| Policy | Wins | Games | Win rate | Average VP |
|---|---:|---:|---:|---:|
| Previous settlement-only inference, same network and boards | 56 | 96 | 58.3% | 5.95 |
| Promoted inference policy, same network and boards | 70 | 96 | 72.9% | 6.44 |
| PPO policy-head checkpoint plus promoted inference | 69 | 96 | 71.9% | 6.44 |

The original 512 checkpoint with the new inference policy won the direct
same-board comparison, so the PPO checkpoint was not promoted.

## Frozen-anchor results

| Opponent | Wins | Games | Win rate | Average VP |
|---|---:|---:|---:|---:|
| AlphaBot | 70 | 96 | 72.9% | 6.44 |
| Heuristic-v1 | 69 | 96 | 71.9% | 6.51 |
| Heuristic-v2 | 65 | 96 | 67.7% | 6.41 |

## Reproduce

```bash
.venv/bin/python training/evaluate_planner_vs_alpha.py \
  training/runs/20260611-catanzero-v2/best.ctnn \
  --planner hybrid-v2 \
  --strategy-settlement-weight 5 \
  --heuristic-refinement \
  --games 96 \
  --seed 23900000
```

Promotion metadata is stored in
`training/runs/20260612-alpha-best-response-promoted.json`.
