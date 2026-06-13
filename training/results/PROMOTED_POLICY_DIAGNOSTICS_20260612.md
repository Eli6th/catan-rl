# Promoted policy strengths and weaknesses - 2026-06-12

## Evaluation

The promoted 512-wide policy was replayed on the exact 96-game AlphaBot
promotion set:

- Candidate: `training/runs/20260611-catanzero-v2/best.ctnn`
- Inference: heuristic-v2 opening, coherent-resource weight `5.0`, heuristic
  tactical refinement
- Opponents: three AlphaBots at `8x96d300`
- Visibility: realistic hidden information
- Victory target: 7
- Seed: `23900000`, paired across all four candidate seats

The diagnostic rerun reproduced the promotion result exactly: `70/96`
(`72.9%`) with `6.44` average VP.

## Strengths

### Development cards are the primary winning engine

- Largest Army: `80.0%` of wins versus `30.8%` of losses.
- Hidden VP: `1.69` per win versus `0.73` per loss.
- Dev purchases per estimated candidate turn: `0.63` in wins versus `0.47`
  in losses.
- Games with Largest Army but not Longest Road won `47/55` (`85.5%`).
- Games with neither award won only `6/23` (`26.1%`).

This is not only an AlphaBot pattern. Against heuristic-v1 and heuristic-v2,
wins also had more dev purchases, hidden VP, cities, and awards.

### The policy has two viable award paths

- Largest Army without Longest Road: `85.5%` win rate.
- Longest Road without Largest Army: `8/9` wins (`88.9%`).
- Both awards: `9/9` wins.

The policy is strongest when it converts production into at least one award,
not when it accumulates buildings alone.

### It is robust across seats and boards

AlphaBot win rate by candidate seat:

| Seat | Wins | Games | Win rate |
|---:|---:|---:|---:|
| 0 | 18 | 24 | 75.0% |
| 1 | 17 | 24 | 70.8% |
| 2 | 16 | 24 | 66.7% |
| 3 | 19 | 24 | 79.2% |

No one seat collapses. Across the 24 paired boards, the candidate won at
least two of four seats on every board: nine boards had two wins, eight had
three, and seven had four.

### The promoted inference policy is materially better than the old policy

On the same boards and network:

- Old inference: `56/96`, `5.95` average VP.
- Promoted inference: `70/96`, `6.44` average VP.
- Paired flips: 29 losses became wins; 15 wins became losses.
- Average road length rose from `3.74` to `4.25`.
- Longest Road rate rose from `7.3%` to `18.8%`.
- Games finished about two turns earlier on average.

## Weaknesses

### The main failure is the 6-VP conversion stall

Eight of 26 AlphaBot losses ended at 6 VP. Seven of those eight already had
Largest Army. The policy successfully builds a development-card engine, but
often cannot find the final settlement, city, road award, or VP card before
an opponent wins.

Four of those eight close losses had zero opening wood production. This is a
small sample, but it matches the observed failure: the army route reaches
6 VP and then lacks a reliable building conversion.

### It concedes Longest Road in losses

- Candidate Longest Road: `3.8%` of losses.
- Winning opponent Longest Road: `73.1%` of losses.
- Seventeen of 26 losses ended with neither candidate award.

The model does not need to win Longest Road every game, but it needs to
recognize when an opponent's road route is the fastest remaining VP path and
either contest or block it.

### Opening strategy coherence is overvalued relative to raw production

Within the promoted AlphaBot run:

- Winning opening production: `0.581` expected cards per roll.
- Losing opening production: `0.536`.
- Opening coherent-strategy score was effectively identical:
  `0.096` in wins versus `0.095` in losses.
- Openings below `0.50` production won `4/9`.
- Openings at or above `0.58` won `38/46`.

The promotion changed the opening on 71 of 96 games. It increased the
coherent-strategy score from `0.057` to `0.096`, but reduced average raw
production from `0.584` to `0.569`. Raising the strategy weight again is not
supported by these results. A production floor, with wheat as the first
tiebreaker, is safer than more weight on the current coherence proxy.

### The policy is award-dependent

Against AlphaBot, games with neither award won only `26.1%`. The same
no-award pattern explains most losses against both heuristic opponents.
Buildings alone are not being converted efficiently enough into 7 VP.

### AlphaBot results include a trade-response exploit

The candidate proposes about `2.6` trades per estimated turn. Its
confirm-to-proposal ratio is about `48%` against AlphaBot, versus `10%`
against heuristic-v1 and `16%` against heuristic-v2.

This is a real AlphaBot strength, but it is also a generalization risk. The
opponent pool should include stricter and no-trade responders so the policy
cannot rely on one opponent's acceptance behavior.

## Recommended next experiments

1. Add a phase-aware 5-6 VP conversion feature to inference and training.
   After Largest Army is secured, reduce non-VP dev spending and explicitly
   compare the fastest reachable settlement, city, Longest Road, and VP-card
   paths.
2. Replace the opening coherence bonus with a constrained score: require a
   raw-production floor, favor wheat, then use coherent strategy as a
   tiebreaker. Do not increase the current weight.
3. Train more heavily against heuristic-v2, road-contesting agents, strict
   trade responders, and no-trade agents. This targets the remaining failure
   modes better than increasing model width.
4. Add a small high-leverage search only at 5-6 VP and when an opponent is
   threatening Longest Road. Full-game MCTS is unnecessary for this
   diagnosis; conversion decisions are the highest-value search points.
5. Track these diagnostics as checkpoint gates: 6-VP loss rate, no-award
   loss rate, opponent Longest Road in losses, raw opening production, and
   trade-confirm ratio by opponent.

Raw diagnostics:

- `training/runs/20260612-promoted-alpha-diagnostics-96.json`
- `training/runs/20260612-settlement-only-alpha-diagnostics-96.json`
- `training/runs/20260612-promoted-heuristic-v1-diagnostics-96.json`
- `training/runs/20260612-promoted-heuristic-v2-diagnostics-96.json`
