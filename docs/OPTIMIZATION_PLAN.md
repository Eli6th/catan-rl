# Making it faster: the remaining headroom, tiered

Current measured state: ~160 ns/step engine-only single thread, 27 to 69M
steps/s across cores, 393 ns per full RL decision, 321 µs per 1,024-env
batch (memory-bandwidth-bound), ~50 ms per AlphaBot decision (99% of
which is rollout engine steps, 1% net inference). LTO, codegen-units=1,
and target-cpu=native are already enabled.

The first question is which clock matters. Three different consumers,
three different bottlenecks:

| Consumer | Bound by | So the lever is |
|---|---|---|
| PPO training loop | PyTorch forward/backward | not the engine at all |
| Batched env stepping | RAM bandwidth (obs writes) | write less, not compute faster |
| AlphaBot search | rollout step count x ns/step | engine micro + smarter search |

Every tier below states its consumer, expected gain, effort, and risk.
House rules apply throughout: save a criterion baseline before each
change, keep the goldens and the zero-allocation tests green, and reject
any win that costs correctness confidence.

## Tier 0: free or nearly free (hours)

- **Profile-guided optimization (PGO)**: compile with instrumentation,
  run a representative workload, recompile using the profile. Typically
  5 to 15% on branchy code like rule logic, which ordinary optimization
  cannot reorder well because it cannot know which branches are hot.
  `cargo-pgo` automates it. Zero code changes, zero risk.
- **Re-save criterion baselines** after the bot-seat and episode-stat
  additions (the env benches drifted ~12% for known reasons; the
  baseline should reflect the current intended state).

## Tier 1: the env decision path (FIRST RESULTS on perf/tier1-2)

Measured on the branch: per-seat cached board blocks cut the single-env
decision 24% (339 to 257 ns). Lesson learned in the process: the same
cache REGRESSED the 1,024-env batch by 14%, because that path is
bandwidth-bound and the cache adds a copy stream; the batch path now
encodes directly (net -7% there from the encoder split alone). Compute
optimizations and bandwidth optimizations are different problems.

Remaining items:

- **Incremental observation encoding.** The 1,350-float observation is
  rebuilt from scratch every decision, but most of it did not change.
  The tile block (152 floats) is constant for an entire episode; vertex
  and edge blocks change only on builds; player blocks change only when
  resources move. Keep the encoded observation cached per env, mark
  blocks dirty as actions execute, and rewrite only dirty blocks. The
  tile block alone is 11% of the vector for zero recurring cost.
  Expected: 2 to 3x on the encode portion, which is the largest share of
  the 393 ns.
- **Bitmask-native action masks.** `fill_valid_actions` materializes a
  `Vec<Action>` (about 24 bytes per action), and the env then encodes
  each to an id. For the RL mask we only need the 299 bits. Generate a
  `[u64; 5]` action-id bitmask directly from the engine's bitboards,
  skipping Action materialization entirely on the mask path. Expected:
  20 to 30% of mask time.

The same changes raise the batched ceiling: the 1,024-env path is
bandwidth-bound on obs writes, so writing only dirty blocks attacks the
actual constraint. Estimated combined effect on the batch: 1.5 to 3x.

## Tier 2: search throughput, the AlphaBot clock (days)

Rollouts dominate the decision, so the wins are fewer wasted rollouts
and more rollouts per second:

- **Thread-parallel rollouts within one decision.** Today one decision
  runs its ~768 rollouts on one thread (parallelism lives across games).
  For interactive play or evaluation at fixed game counts, fan rollouts
  across cores with rayon: near-linear, so ~8x wall-clock on a 10-core
  machine. Straightforward because rollouts share nothing but the
  read-only root state and the net.
- **Early termination (racing). DONE on perf/tier1-2.** Sequential
  halving: the budget is spent in rounds and the worse half of the
  candidates is dropped after each. Measured at equal budget config:
  1.6x faster wall-clock at unchanged strength (81.7% vs 84.2%, within
  noise). Reinvesting the savings (samples 96 to 240): **92.5% and
  95.8% on two 120-game seeds (94.2% combined)**, a new best agent, up
  from 82%. About 60% of the flat scheme's rollouts were spent
  confirming losers.
- **Shorter rollouts once the value head is retrained.** This is the
  search-as-teacher milestone from the main roadmap: a value head
  trained on true (state, outcome) pairs lets rollouts truncate after a
  few turns and be scored by the net. Cuts rollout length ~10x, at
  which point net inference becomes the new bottleneck, which is what
  Tier 3 is for.

## Tier 3: net inference, matters after value leaves land (days)

Status update: the first two rungs landed within hours of the repo going
public, via PR #1 (joshdchang). The naive `iter().zip().sum()` dot never
auto-vectorized at all (float addition is not associative, so LLVM kept
the reduction serial; verified in the disassembly). The PR accumulates
into 16 independent lanes (429 to 54 µs per forward pass) and stores the
trunk weights input-major so whole columns accumulate per nonzero input,
skipping the ~90% zero inputs outright, the NNUE trick from chess
engines (54 to 14 µs, about 30x total; 18.7 µs verified independently).
Strength re-verified at 84% over 120 games. Remaining rungs:

- **Batched leaf evaluation.** Evaluating leaves one at a time makes the
  forward pass memory-bound: every evaluation re-streams all 4.4 MB of
  weights for one vector. Collect 16 to 64 leaf observations and
  evaluate them together (matrix-matrix instead of matrix-vector): the
  weights are read once per batch instead of once per leaf. Near-batch-
  size speedup until compute-bound. This is also the natural on-ramp to
  GPU inference later.
- **Apple Accelerate / AMX.** On Apple Silicon, `cblas_sgemm` dispatches
  to the AMX matrix coprocessor, which outruns NEON by roughly an order
  of magnitude on matmul. One FFI call, platform-gated behind a feature
  flag with the portable path as fallback. Alternatively int8 or f16
  quantization halves or quarters weight traffic on the memory-bound
  side. These are the "fastest possible" moves for inference; batching
  first, then AMX, then quantization if still needed.

## Tier 4: the deep rewrites (weeks to months, only for AlphaZero scale)

Honest assessment: these are only worth it if the goal becomes massive
self-play data generation (millions of search games), where today's
batch path would be the limit.

- **Structure-of-arrays, cross-game vectorization.** Today one game is
  one struct and SIMD happens within a step. The inversion: store 8 or
  16 games' fields interleaved (all robber positions adjacent, all
  resource counts adjacent) and step all of them with the same
  instruction stream, using lane masks where games diverge. This is how
  the fastest modern sim engines work (vectorized Atari emulators,
  EnvPool, JAX environments). Realistic gain: 5 to 20x on the batch
  path. Realistic cost: a parallel implementation of the rules with the
  full fortress re-validated against the scalar engine as the oracle,
  plus permanent divergence-handling complexity. Catan's branchiness
  (trades, robber interactions) makes lane utilization mediocre, which
  caps the win well below the 16x lane count.
- **GPU-resident environment.** Port the whole step function to the GPU
  (Madrona-style batch simulation) so self-play never crosses the
  PCIe/unified-memory boundary. 100M+ steps/s is plausible. Same
  correctness bill as SoA, larger. Only rational once a GPU training
  loop exists and is starved for data, which is not the current state
  (training is PyTorch-bound on CPU today).

## What "fastest possible" actually means here

The single-game scalar engine has perhaps 1.5 to 2x left (Tier 0 + the
mask path) before it is pinned against memory latency on a ~500-byte
working set; chasing beyond that trades correctness risk for noise. The
real multipliers are architectural: dirty-block observations (Tier 1),
racing + parallel rollouts (Tier 2), batched/AMX inference once value
leaves land (Tier 3), and cross-game vectorization only when the
AlphaZero loop demands data at a scale the current path cannot feed
(Tier 4). Recommended order: 0, 1, 2, then 3 gated on the value-head
retrain, then re-measure and decide whether Tier 4's cost is justified
by an actual data-starvation problem rather than a benchmark number.
