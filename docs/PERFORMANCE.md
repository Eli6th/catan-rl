# The performance story: 300,000 ns/step → 87 ns/step

How the engine got ~3,400× faster per core than the Python original, why
each layer was built the way it was, and the guardrails that keep it fast.
Every number here is measured (criterion benchmarks with saved baselines,
or `catan-sim --profile-steps`), not estimated.

## Why speed was the strategy, not a nicety

Everything downstream multiplies by step cost:

- PPO training consumed ~500M engine steps across the experiment campaign.
- The AlphaZero-lite agent runs **96 rollouts per candidate move** — tens of
  thousands of engine steps per single decision. At Python speeds one
  AlphaBot *decision* would take ~30 seconds; at 87 ns it takes ~50 ms.
- The GA evolved a heuristic through ~25,000 games in two minutes; the Elo
  harness rates agents over 576 games in 12 seconds.

None of those tools would exist as practical artifacts without the engine
being effectively free.

## Where we started

The legacy Python engine (`engine/`): **~5 games/s, roughly 300,000
ns/step** — with fewer rules implemented than the final Rust engine. The
costs were structural, not algorithmic: interpreter dispatch on every rule
check, dict/list state with allocation churn on every action, copies for
safety, and the GIL capping parallelism.

## Methodology first (the part that made the rest safe)

Three rules, adopted before optimizing anything:

1. **Measure in ns/step, never games/sec.** Games vary in length — a
   "faster" run may just be playing shorter games. A step is fixed work.
   `catan-sim --profile-steps` also splits engine time from agent time so
   you optimize the right side.
2. **Criterion benchmarks with saved baselines.** `--save-baseline before`,
   make the change, `--baseline before` → statistical pass/fail per bench,
   covering mask generation (typical/rich/dense), roll+distribution, road
   builds through the longest-road award, the 15-road-snake worst case
   (search tail latency), and full games.
3. **Behavior locks before speed work.** The self-golden suite replays 89k
   recorded actions to field-exact final states; the property/oracle/fuzz
   harness must stay green through every optimization. Fast-but-wrong is
   worse than slow: an RL agent will find and exploit any rule drift.

## The optimization ladder

| Stage | Heuristic-game step | What changed |
|---|---|---|
| Python engine | ~300,000 ns | baseline (fewer rules) |
| Rust port, data-oriented | ~820 ns | ~370× from language + layout |
| + bitmask pass | **221 ns** | another 3.7× on the hot paths |
| (random-agent games) | **87 ns** | 11.5M steps/s single thread |
| 10 threads (rayon) | — | **37M steps/s aggregate** |

**Stage 1 — the port was data layout, not just language.** The state is a
~500-byte `Copy` struct of fixed-size arrays — no `HashMap`, no `Vec`, no
pointers, no heap in the hot state at all. Board topology (vertex
adjacency, edge incidence, tile-vertex membership) is precomputed `const`
tables. Actions are small `Copy` enums. The whole game fits in L1 cache;
cloning a game for a search rollout is a memcpy.

**Stage 2 — the bitmask pass** (every claim baseline-verified, goldens
green):

- *Occupancy / adjacency / road-touch bitboards*: connectivity and
  distance-rule checks became single AND/OR operations instead of
  neighbor-list walks. Legal-mask generation **−52%** on rich mid-game
  states — masks are generated twice per decision (validate + RL mask), so
  this is the hottest path in the engine.
- *Cached per-player road lengths*: longest-road was recomputed
  whole-board on every build; now lengths are cached and only
  adjacency-affected segments recompute. Full games **−31%**; combined
  heuristic step time **820 → 221 ns**.
- `tiles_by_number` index: dice resolution looks up affected tiles
  directly instead of scanning the board.

**Stage 3 — zero allocations, enforced forever.** A counting global
allocator wraps the test binary and asserts that steady-state play (mask →
decide → execute, including 7-roll discards, trades, robber, steals)
performs **literally zero heap allocations**. The test caught 15
allocations per ~800 steps the day it was introduced (a discard buffer
growing); buffers are now pre-sized and reused as scratch. Any future
per-step `Vec` fails CI. (Hard-won detail: allocator-counter test binaries
must contain exactly ONE `#[test]` — cargo runs tests in parallel threads
and a shared global counter bleeds between measurement windows.)

## The RL environment layer kept the discipline

- One full RL decision — legality mask + step + auto-resolution of forced
  moves + 1,350-float observation encode — costs **346 ns** (criterion:
  `env_step_mask_obs`).
- A 1,024-environment batched step costs **~296 µs ≈ 3.4M policy-steps/s**,
  and is memory-bandwidth-bound on observation writes — meaning the
  *compute* is done; only RAM throughput limits it.
- The same zero-allocation standard applies: zero allocations per decision
  in steady state, bounded per-batch allocations (lane setup + rayon
  plumbing only — never proportional to env count).

## The Python boundary: make it irrelevant

PyO3 bindings move the **whole batch in one Python call** — NumPy buffers,
GIL released while Rust steps 256 games in parallel: **~370k
policy-steps/s through Python**. That is 10–100× more than the neural
network forward pass consumes, which is the design goal: the boundary and
the engine should never appear in a training-loop profile. They don't —
training runs at ~19k steps/s end-to-end, fully bounded by PyTorch on CPU.

## Payoffs you can point at

- 11,500 self-play games during a 10-minute training stage.
- 100,000 full games in seconds on the demo command.
- AlphaBot's 82% win rate is *bought* with engine speed: ~768 full-game
  rollouts per decision, ~5 ms each thread-parallel.
- Every experiment in [training/results/EXPERIMENTS.md](../training/results/EXPERIMENTS.md)
  ran on a laptop, same-day, because iteration cost ~minutes.

## What we did NOT do (and why)

- No SIMD intrinsics, no unsafe, no custom allocators: autovectorized safe
  Rust plus good data layout got within sight of memory bandwidth. The
  next 2× would cost more correctness risk than it buys.
- No GPU for the engine: the state machine is branchy and tiny; GPUs want
  the *network*, which is the next phase's problem (batched leaf
  evaluation for search).
- No micro-optimizing before the fortress existed: every stage above
  shipped with goldens + property tests green, which is why three days of
  aggressive optimization produced zero behavioral regressions.
