# Teaching a Machine to Play Catan, the full story

Talk outline with speaker notes and a live-demo script. Everything here is
reproducible from this repo; numbers cite
[training/results/EXPERIMENTS.md](../training/results/EXPERIMENTS.md).

---

## 1. The problem (3 min)

Catan is a hostile target for game AI, and that's the point:

- **4 players**, no minimax, no clean two-player zero-sum theory.
- **Dice**, every turn forks the future eleven ways. Plans must be robust,
  not optimal-on-one-line.
- **Hidden information & trading**, opponents' hands and intentions.
- **Long horizon**, ~70–150 turns, ~600 decisions; the move that wins the
  game might be settlement placement on turn 1.

The classical techniques that cracked chess (minimax) and poker (CFR/GTO)
*structurally don't apply*. This is the gap RL + search fills.

## 2. Foundation: the engine is the product (5 min)

You cannot train against a buggy simulator, an RL agent is an exploit
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
with interpreted loops. "Where can I build a road?" walked all 72 edges,
both endpoints each, all adjacent edges per endpoint, with a method call
per probe, then allocated a fresh list for the answer. Each loop
iteration costs 50 to 100 ns of interpreter dispatch before any work
happens, and the GIL caps the engine at one core. Net: ~300,000 ns/step,
~5 games/s. The costs are structural; the fix is changing what a rule
check physically is.

*The after*, in three ideas (full walkthroughs in
[PERFORMANCE.md](PERFORMANCE.md)):

1. **The state is a value**: a ~500-byte Copy struct, no heap, fits in
   L1, and cloning a game for a search rollout is a memcpy. This is what
   makes AlphaBot affordable later in the talk.
2. **Bitboards**: 54 vertices, 64 bits, so one u64 records a yes/no fact
   about every vertex at once. The distance rule plus occupancy plus
   connectivity becomes three AND operations, and one mask expression
   enumerates every legal settlement simultaneously. Mask generation
   sped up 52%.
3. **Cache the expensive rule**: longest-road is a longest-trail search
   that was recomputed whole-board on every build. A new piece can only
   affect players whose network touches it, so a per-player cache with
   adjacency-filtered recomputes suffices. Full games sped up 31%.

| Stage | ns/step (heuristic games) |
|---|---|
| Python engine | ~300,000 |
| Rust port (data layout) | ~820 |
| + bitboards & road cache | **221** (87 random) |

~3,400x per core, 37M steps/s across 10 threads. Kept true by a counting
global allocator (zero heap allocations per step; it caught 15 allocs per
800 steps the day it was introduced), criterion baselines, and the rule
that goldens stay green through every optimization, because the agent
will learn any bug as strategy. Closing comparison for the room: one
AlphaBot decision is ~768 full-game rollouts, which is ~30 seconds on the
Python engine and ~50 ms here.

**DEMO 1:** `cargo run -p catan-sim --release -- --games 100000 --players H,R,H,R`
(100k full games in ~seconds, live win-rate table.)

## 3. The RL environment: design decisions (5 min)

A neural network is a function from a fixed-size vector of numbers to a
fixed-size vector of numbers. A Catan game is none of those things: the
set of legal moves changes every turn, the state is structured, four
players act in rotation, and dice interrupt everything. The environment
layer is the translation between the two, and every design decision below
answers one mismatch.

*The moves problem.* The network's output layer has a fixed width, but
"what can I do?" in Catan varies from 2 options to dozens. The answer is
a codec: every move the game can ever offer maps to one of **299
discrete ids** (build X at vertex v, offer trade T, play card C, and so
on), and alongside the observation the environment emits a legality
mask, a 299-long yes/no vector. The network outputs a score for all 299;
illegal entries are erased before sampling. Two properties were tested
rather than assumed: the mapping is total (every id decodes to some
action, every legal action encodes to exactly one id), and the mask is
fuzzed against the engine so "masked legal" and "engine accepts" can
never drift apart. The deliberate scope cut lives here: trading is a
bounded menu (1 or 2 of one resource for 1 of another, at most 3 offers
a turn). Free-form trades would multiply the action space by orders of
magnitude; a bounded menu keeps the mechanic and keeps the space
learnable.

*The state problem.* The observation is 1,350 floats covering the full
board (every tile, vertex, and edge), public player state, the acting
player's hand, the bank, and any pending trade. Two encoding decisions
matter. First, dice numbers are encoded as production probabilities (a
6 enters as 5/36), because the numeral is irrelevant and the probability
is what the network must learn anyway; encode the meaning, not the
symbol. Second, the observation is seat-relative: whoever is acting sees
themselves as "player 0" and opponents in turn order from their seat.
One network therefore plays all four seats, every game produces four
seats' worth of experience, and the network never wastes capacity
learning "seat 2's strategy". The assumption baked in (strategy is
seat-symmetric up to turn order) holds, because turn order itself is in
the observation.

*The turns problem.* Four players alternate, and many "moves" are not
decisions at all: rolling at the start of a turn is mandatory, and a
player with one legal option has no choice. The environment uses
turn-based multi-agent semantics: `step(action)` executes the move and
fast-forwards to the next seat that faces a real choice, resolving
forced moves internally and treating dice as internal chance events
rather than actions. A tested invariant: every observation the policy
ever sees has at least two legal actions. The trainer never spends a
forward pass on a non-decision.

*The trust problem.* Trained weights are only meaningful against the
exact action and observation layouts they were trained on. Both layouts
are versioned, every checkpoint stores the versions, and every loader
refuses a mismatch. A network can never silently play a game it does not
understand.

*The opponent problem.* Any seat can be assigned to a scripted player
that acts inside the environment at engine speed, invisible to the
policy. This one feature, added for evaluation, turned out to be the
workhorse of the whole project: fixed-opponent benchmarks, mixed-opponent
training, and Elo tournaments all became configuration changes rather
than new code.

*The boundary problem.* Python orchestrates training but must never sit
inside the hot loop. The PyO3 bindings move an entire batch per call:
256 games step in parallel Rust with the GIL released, and NumPy arrays
come back, ~370k policy-steps/s through Python against ~3.4M in pure
Rust. The boundary is designed to disappear: training profiles show only
PyTorch.

## 4. Training arc: 25% → 65% (8 min)

PPO self-play, one network for all seats, γ=1.0 (episodic, don't discount
a 600-step game away), GAE λ=0.95, masked categorical policy, shaped
reward (small bonus per VP) annealed to pure win/loss.

The arc, all on a laptop CPU (~19k steps/s):

| Stage | Recipe | vs 3 random | vs 3 heuristic |
|---|---|---|---|
| untrained |, | 25% | 25% |
| 10 min | pure self-play | 92% | **14%** |
| +12 min | more self-play | 99% | 28% |
| +12 min | mixed: 2 seats vs scripted bots | 100% | 56% |
| +8 h | longer mixed training | 100% | 66% → flat |

Three teachable moments:
1. **Self-play beats chaos, not competence**, 92% vs random was 14% vs a
   simple priority list. Fixed reference opponents are non-negotiable.
2. **Mixed-opponent training >> pure self-play** for beating a fixed
   opponent class, direct pressure tripled the win rate in 12 minutes.
3. **The dashboard predicted everything**: avg game length falling =
   learning to close; entropy decaying gently = healthy exploration;
   explained variance ~0.8 = the critic sees the game.

**DEMO 2:** live dashboard, start a 5-min training run, watch win rate,
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
(outplayed, not unlucky, and losses were LONGER than wins: failed grinds,
not opponent racing), and a brute-force Monte Carlo searcher with zero
training hit **72.5%**.

**The diagnosis**: the network evaluates fine (EV ~0.8) but a single
forward pass cannot *compute multi-step tactical lines*. Reactive policies
plateau. It's not a tuning problem, it's a structure problem.

Side quest that paid off: a **genetic algorithm** over the heuristic's six
magic numbers (common-random-numbers fitness, 12 generations, 2 minutes)
produced Heuristic-v2 at 40%, and its evolved genes are readable strategy
critiques (v1 overvalues ports; v1 is too timid with the robber).

## 6. The breakthrough: AlphaZero-lite (6 min)

Put the trained network *inside* the search:

- Export the net to a self-verifying binary; reimplement inference in ~100
  lines of pure Rust (it's 2 matmuls, no framework needed).
- **Policy head prunes the root**: rank legal moves, search only the top 8.
  Pruning is free strength: a third of the candidates funds 4× the rollouts.
- Full win/loss rollouts score each candidate.

Result: **82%**, through the policy's 65% wall and past the searcher's
72.5%, at the same compute budget.

The honest failure inside the win: every attempt to use the **value head**
as a leaf evaluator made things WORSE (38%, 16%, 26%). Lesson: *a PPO
critic is a baseline, not an outcome regressor*, calibrated on average,
coarse per state. AlphaGo's value net was trained on (position → final
outcome) pairs; that retraining is exactly the next phase.

**DEMO 3:** `cargo run -p catan-sim --release -- --games 100 --players A,H,H,H --net models/catan-512.ctnn --alpha-config 8,96,300`

## 7. Design decisions worth defending (Q&A ammo)

- **Rust + tests-first**: an RL agent is an adversarial fuzzer; fortress
  before features. The oracle caught a real inherited bug.
- **First-to-7 curriculum**: shorter games = denser reward = faster early
  learning; graduate to 10 later. (Also: short games are luckier, skill
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
5. **Play vs the agent in the browser**, the play-test UI already drives
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
