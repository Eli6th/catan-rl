# Competitive Checkpoint: Strengths and Weaknesses

All final results use realistic hidden information, a 7 VP target, all four
candidate seats, and independent per-episode rollout seeds. Independent seeds
are required because shared policy counters can make one game's action count
change the rollout randomness used by later games.

## Best retained policies

| Opponent | Deployed policy | Fair result | Status |
|---|---|---:|---|
| Heuristic-v1 | `20260613-alphafocus-h2-elites-v1/iteration_01.ctnn` | 349/384 (90.9%) | Above target |
| Heuristic-v2 | recency-v2 base + independent-losses-v5 + masked conversion refinements | 347/384 (90.4%) | Above target |
| AlphaBot | terminal base + independent-v5 after 5 VP | 158/192 (82.3%) | Best retained specialist |

H2 uses:

- Base: `training/runs/20260613-h2-elites-recency-v2/iteration_01.ctnn`
- Late policy: `training/runs/20260613-h2-counterfactual-independent-losses-v5/counterfactual.ctnn`
- Opening rollout prior weight: `0.04`
- City-directed conversion: seats 1 and 3 (`seat mask 10`)
- Knight pressure: 4+ VP in seats 0, 2, and 3 (`seat mask 13`)

AlphaBot uses:

- Base: `training/runs/20260612-2244-h2-policy-head-continuation-5m/screening/base.ctnn`
- Late policy: `training/runs/20260613-alpha-counterfactual-independent-v5/counterfactual.ctnn`

The evaluator originally keyed independent rollout counters by board seed only.
Because all four candidate seats reuse a board seed, later seats inherited the
rollout position reached by earlier seats. The corrected evaluator keys by both
board seed and candidate seat. All numbers in this report use that correction.

## Strengths

- Heuristic-v1 is solved under the fair-seeding gate, including all four seats.
- H2 openings are not the main problem. Total opening income is `0.543` in wins
  and `0.534` in losses; the larger gap appears after the opening.
- Winning positions convert decisively, usually by combining cities, hidden VP,
  and Largest Army.
- Largest Army is the most reliable learned award. It appears in 228/347 H2
  wins and 116/158 retained AlphaBot wins.
- The promoted H2 configuration is above 90% from seats 0 and 1 and improves
  the weaker seats 2 and 3 to 87.5% each.
- Loss-only counterfactual training gives a small reproducible gain without
  changing the simple VP-heavy environment reward.
- A lower opening-rollout prior plus narrowly seat-routed city and knight
  refinements raises H2 from 341/384 to 347/384.
- The 512-wide model is large enough for the current policy quality. Increasing
  width did not address the observed failure mode.

## Weaknesses

- AlphaBot losses are dominated by Longest Road. The winner holds it in 24/34
  losses (71%), while the candidate holds it in none of those losses.
- AlphaBot also exposes a real opening-quality gap. Opening strategy score is
  `0.089` in wins versus `0.073` in losses, larger than the H2 gap.
- Across the full three-seed Alpha gate, losses average only `0.62` cities
  versus `1.38` in wins. Nineteen of 34 losses build no city.
- Losses average `5.82` End Turn actions versus `3.32` in wins, consistent with
  stalled hands that fail to turn production into VP.
- The candidate holds no award in 27/34 AlphaBot losses. It still averages road
  length `3.65`, higher than the `3.25` in wins, so raw road construction is not
  translating into Longest Road control.
- AlphaBot strength is not isolated to one candidate seat. Seats 0, 2, and 3
  are each 40/48; seat 1 is 38/48.
- The policy still over-invests in development cards when an opponent is
  converting a road or city path. The fair loss-only specialist mostly found
  extra Largest Army wins, not a reliable Longest Road response.
- H2 remains board-sensitive, although it now passes the aggregate gate.
- H2 losses remain concentrated at the conversion boundary: 10/37 finish at
  6 VP and another 10/37 finish at 5 VP. Twenty-two of 37 losses build no city.
- H2 seats 2 and 3 remain the positional weakness at 87.5% each, versus 91.7%
  and 94.8% from seats 0 and 1.
- H2 losses are not caused by too few roads. Road length is nearly unchanged
  between wins and losses (`3.84` versus `3.95`), while cities fall from `1.30`
  to `0.57`.
- AlphaBot remains far from the requested 90% gate. The promoted policy improves
  the corrected three-seed aggregate only marginally over prior policies.
- Learned-value and shallow-potential search are not calibrated well enough to
  provide a stable MCTS improvement.

## Decision log

| Experiment | Result | Decision |
|---|---:|---|
| H2 all-games terminal counterfactual | 223/256; VP5 router 220/256 | Reject |
| Alpha all-games terminal counterfactual | 70/96 vs 74/96 baseline | Reject |
| Award-aware shallow search | Tied across two seeds with high seed variance | Reject |
| H2 seat-conditioned award pressure | 113/128 vs 116/128 fresh control | Remove |
| Alpha loss-only v3 with shared episode counters | Apparent +6/128 fresh | Invalid |
| Same Alpha v3 under independent seeds | 105/128 vs 107/128 | Reject |
| Alpha independent loss-only v5, corrected seeds | 158/192 | Retain |
| Alpha second independent bootstrap v6 | No corrected improvement | Reject |
| Stronger Longest Road potential teacher v7 | 82/96 vs 83/96 retained | Reject |
| Loss-only terminal Longest Road auxiliary v8 | 82/96 vs 83/96 retained | Reject |
| All-game road auxiliary + loss-only policy v9 | 83/96 screen; 104/128 vs 106/128 extended | Reject |
| Alpha conversion-credit specialist v10 | 159/192, identical to retained v5 | Reject |
| Alpha VP6 conversion specialist v11 | 159/192, identical to retained v5 | Reject |
| Alpha VP3-4 growth specialist v12 | 81/96 vs 83/96 retained | Reject |
| Alpha end-turn trade sweep, uncapped | 158/192 vs 159/192 retained | Reject |
| Alpha end-turn trade sweep through 4 VP, corrected seeds | 157/192 vs 158/192 | Reject |
| H2 independent loss-only v5, corrected seeds | 341/384 | Retain |
| H2 opening rollout prior 0.04 | 342/384 vs 341/384 | Retain |
| H2 opening city bias, seats 2-3 | 84/96 vs 86/96 screen | Reject |
| H2 city-directed trades, all seats | 339/384 vs 342/384 | Reject globally |
| H2 city-directed trades, seats 1 and 3 | Exact paired composite 344/384 | Retain |
| H2 unrestricted knight pressure | 338/384 vs 344/384 | Reject globally |
| H2 knight pressure at 4+ VP, seats 0, 2, and 3 | Direct gate 347/384 | Retain |
| H2 early conversion trading | 82/96 vs 86/96 screen | Reject |
| H2 save-before-dev-card rule | 73/96 vs 86/96 screen | Reject |
| H2 trade refinement | 338/384 vs 344/384 | Reject |
| H2 second independent bootstrap v6 | No corrected improvement | Reject |
| H2 VP6 conversion specialist v13 | Identical 83/96 screen outcomes | Reject |
| H2 VP6 conversion-aware search | 83/96 with one gained and one lost trajectory | Reject |
| H2 stronger VP6 specialist v14 | 83/96 with one gained and one lost trajectory | Reject |

## Implication

The current policy is good at development-card and Largest Army lines, but weak
at committing production to a city and at converting roads into award control.
H2 needs a midgame city-conversion specialist for seats 2-3, not a stronger
opening city prior. AlphaBot needs both stronger opening selection and a
zero-city specialist that explicitly values Longest Road control and penalizes
stalled End Turn loops.
