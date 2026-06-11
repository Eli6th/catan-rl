# Teaching a Machine to Play Catan — the full story

Talk outline with speaker notes and a live-demo script. Everything here is
reproducible from this repo; numbers cite
[training/results/EXPERIMENTS.md](../training/results/EXPERIMENTS.md).

---

## 1. The problem (3 min)

Catan is a hostile target for game AI, and that's the point:

- **4 players** — no minimax, no clean two-player zero-sum theory.
- **Dice** — every turn forks the future eleven ways. Plans must be robust,
  not optimal-on-one-line.
- **Hidden information & trading** — opponents' hands and intentions.
- **Long horizon** — ~70–150 turns, ~600 decisions; the move that wins the
  game might be settlement placement on turn 1.

The classical techniques that cracked chess (minimax) and poker (CFR/GTO)
*structurally don't apply*. This is the gap RL + search fills.

## 2. Foundation: the engine is the product (5 min)

You cannot train against a buggy simulator — an RL agent is an exploit
search engine; it WILL find your bugs and learn them as strategy. So the
build order was deliberately: **tests first, engine second, learning last**.

- Rust rules engine, full base game incl. player trading. ~500-byte state,
  fixed arrays, bitmask caches. **~88 ns per game step**, zero heap
  allocations in steady state (proven by a counting global allocator in CI).
- The testing fortress: rulebook scenario tests, property-based tests,
  a brute-force longest-road oracle (caught a real direction-aware bug
  inherited from the Python engine), illegal-action fuzzing (~1,700
  parameterizations fired at sampled states: mask and executor cannot
  drift), self-golden replays as the optimization drift-net.
- Speed measured in **ns/step, never games/sec** (games vary in length;
  steps are fixed work). Criterion benchmarks with saved baselines.

*The before*: the original Python engine answered every rules question
with interpreted loops — "where can I build a road?" walked all 72 edges,
both endpoints each, all adjacent edges per endpoint, a method call per
probe, then allocated a fresh list for the answer. ~50–100 ns of
interpreter dispatch per loop iteration before any work happens; the GIL
caps it at one core. Net: ~300,000 ns/step, ~5 games/s. Structural, not
fixable in place — the fix is changing what a rule check physically *is*.

*The after*, in three ideas (full walkthroughs in
[PERFORMANCE.md](PERFORMANCE.md)):

1. **The state is a value** — a ~500-byte Copy struct, no heap, fits in
   L1; cloning a game for a search rollout is a memcpy. (This is the line
   that quietly makes AlphaBot possible five sections from now.)
2. **The board is a u64** — 54 vertices, 64 bits. The distance rule plus
   occupancy plus connectivity is three AND ops:
   `occupied & bit == 0 && occupied & neighbors[v] == 0 && my_roads & bit != 0`
   — and one mask expression enumerates every legal settlement at once.
   Mask generation −52%.
3. **Cache the pathological rule** — longest-road is a longest-trail
   search recomputed whole-board on every build; the invalidation insight
   (a new piece only affects players whose network touches it) makes it a
   per-player cache with adjacency-filtered recomputes. Full games −31%.

| Stage | ns/step (heuristic games) |
|---|---|
| Python engine | ~300,000 |
| Rust port (data layout) | ~820 |
| + bitboards & road cache | **221** (87 random) |

~3,400x per core, 37M steps/s across 10 threads. Enforced forever by a
counting global allocator (zero heap allocations per step — it caught 15
allocs/800 steps on day one), criterion baselines, and the iron rule that
goldens stay green through every optimization: fast-but-wrong loses to
slow, because the agent WILL learn your bugs. Payoff line for the room:
one AlphaBot decision = ~768 full-game rollouts = ~30 seconds on the
Python engine, ~50 ms here.

**DEMO 1:** `cargo run -p catan-sim --release -- --games 100000 --players H,R,H,R`
(100k full games in ~seconds, live win-rate table.)

## 3. The RL environment: design decisions (5 min)

- **Action codec**: every possible move maps to one of **299 discrete ids**,
  with an exact legality mask. Trading is a bounded menu (1–2 of one
  resource for 1 of another, max 3 offers/turn) — deliberate scope cut to
  keep the action space learnable.
- **Observation**: 1,350 floats, seat-relative (the acting player is always
  "player 0" — one policy plays all seats), dice encoded as production
  probabilities, Perfect/Realistic visibility modes differing only in the
  opponent-private block.
- **AEC semantics**: `step()` returns the *next seat that must decide*;
  forced moves (exactly one legal action) resolve inside the env so the
  policy only ever sees real decisions. Dice stay internal chance nodes.
- **Versioned contracts**: codec/obs versions are stored in every
  checkpoint and hard-fail on mismatch — trained weights can never silently
  desync from the engine.
- **Bot seats**: any seat can be played engine-side by a scripted player at
  engine speed — this one feature later enabled fixed-opponent evaluation,
  mixed-opponent training, and Elo tournaments as pure config.

Python sees the whole thing through PyO3 as a batched VecEnv: one call per
training step for 256 games, GIL released, ~370k policy-steps/s through
Python. Engine never the bottleneck (~3.4M steps/s in pure Rust).

## 4. Training arc: 25% → 65% (8 min)

PPO self-play, one network for all seats, γ=1.0 (episodic — don't discount
a 600-step game away), GAE λ=0.95, masked categorical policy, shaped
reward (small bonus per VP) annealed to pure win/loss.

The arc, all on a laptop CPU (~19k steps/s):

| Stage | Recipe | vs 3 random | vs 3 heuristic |
|---|---|---|---|
| untrained | — | 25% | 25% |
| 10 min | pure self-play | 92% | **14%** |
| +12 min | more self-play | 99% | 28% |
| +12 min | mixed: 2 seats vs scripted bots | 100% | 56% |
| +8 h | longer mixed training | 100% | 66% → flat |

Three teachable moments:
1. **Self-play beats chaos, not competence** — 92% vs random was 14% vs a
   simple priority list. Fixed reference opponents are non-negotiable.
2. **Mixed-opponent training >> pure self-play** for beating a fixed
   opponent class — direct pressure tripled the win rate in 12 minutes.
3. **The dashboard predicted everything**: avg game length falling =
   learning to close; entropy decaying gently = healthy exploration;
   explained variance ~0.8 = the critic sees the game.

**DEMO 2:** live dashboard — start a 5-min training run, watch win rate,
entropy, game length move in real time:
`python training/ppo.py --name demo --minutes 5 --vp-delta 0.05 --train-seats policy,heuristic,policy,heuristic --metrics /tmp/m.jsonl`
+ `cargo run -p catan-web --release -- --metrics-file /tmp/m.jsonl`

## 5. The wall, and the scientific method (8 min)

At 66% vs the heuristics, the curve went *exactly* flat. Four hypotheses,
four controlled experiments, one night:

| Hypothesis | Experiment | Verdict |
|---|---|---|
| needs more training | 8h run | slope literally 0.0 ✗ |
| stuck in local optimum | entropy ×2 | gate unchanged ✗ |
| overfit to one opponent | added GA-evolved v2 opponents | unchanged ✗ |
| out of capacity | 4× larger network | same plateau ✗ |

Meanwhile two diagnostics said headroom existed: 44% of losses ended ≤4 VP
(outplayed, not unlucky — and losses were LONGER than wins: failed grinds,
not opponent racing), and a brute-force Monte Carlo searcher with zero
training hit **72.5%**.

**The diagnosis**: the network evaluates fine (EV ~0.8) but a single
forward pass cannot *compute multi-step tactical lines*. Reactive policies
plateau. It's not a tuning problem — it's a structure problem.

Side quest that paid off: a **genetic algorithm** over the heuristic's six
magic numbers (common-random-numbers fitness, 12 generations, 2 minutes)
produced Heuristic-v2 at 40% — and its evolved genes are readable strategy
critiques (v1 overvalues ports; v1 is too timid with the robber).

## 6. The breakthrough: AlphaZero-lite (6 min)

Put the trained network *inside* the search:

- Export the net to a self-verifying binary; reimplement inference in ~100
  lines of pure Rust (it's 2 matmuls — no framework needed).
- **Policy head prunes the root**: rank legal moves, search only the top 8.
  Pruning is free strength: a third of the candidates funds 4× the rollouts.
- Full win/loss rollouts score each candidate.

Result: **82%** — through the policy's 65% wall and past the searcher's
72.5%, at the same compute budget.

The honest failure inside the win: every attempt to use the **value head**
as a leaf evaluator made things WORSE (38%, 16%, 26%). Lesson: *a PPO
critic is a baseline, not an outcome regressor* — calibrated on average,
coarse per state. AlphaGo's value net was trained on (position → final
outcome) pairs; that retraining is exactly the next phase.

**DEMO 3:** `cargo run -p catan-sim --release -- --games 100 --players A,H,H,H --net models/catan-512.ctnn --alpha-config 8,96,300`

## 7. Design decisions worth defending (Q&A ammo)

- **Rust + tests-first**: an RL agent is an adversarial fuzzer; fortress
  before features. The oracle caught a real inherited bug.
- **First-to-7 curriculum**: shorter games = denser reward = faster early
  learning; graduate to 10 later. (Also: short games are luckier — skill
  expresses more at 10.)
- **Restricted trade menu**: full free-form trading explodes the action
  space; a bounded menu keeps it learnable while preserving the mechanic.
- **Steps not games** for perf; **fixed-seed gates** not friendly-seed
  evals for promotion; **Elo with a pinned Random=1000 anchor** so ratings
  compare across time.
- **Everything versioned**: codec v1 / obs v1 frozen the moment a network
  trained on them.

## 8. Where this goes (3 min)

1. **Search-as-teacher** (the real AlphaZero loop): search games → retrain
   value head on true outcomes → value leaves replace rollout depth →
   faster AND stronger → repeat.
2. **Graph network**: the board is a graph; the encoder should be too.
3. **First-to-10, then realistic visibility** (the POMDP, where poker-style
   randomized strategies become relevant).
4. **Video**: every eval game is a 4.5-bytes/action replay; the archive
   already holds the agent's full evolution for rendering.
5. **Play vs the agent in the browser** — the play-test UI already drives
   the engine; wiring AlphaBot into it is config, not code.

## Demo cheat-sheet (run before the talk)

```bash
cd rust && cargo build --release -p catan-sim -p catan-web   # warm builds
cargo run -p catan-sim --release -- --games 100000 --players H,R,H,R
cargo run -p catan-sim --release -- --games 100 --players A,H,H,H \
    --net ../models/catan-512.ctnn --alpha-config 8,96,300
# dashboard demo needs: maturin develop --release in catan-py, torch installed
```

Fallbacks if live demos misbehave: the experiment ledger tables, the
dashboard screenshots, and `training/results/logs/` have every number.
