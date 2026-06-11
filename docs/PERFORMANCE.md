# The performance story: 300,000 ns/step → 87 ns/step

How the engine got ~3,400× faster per core than the Python original. Every
number here is measured (criterion benchmarks with saved baselines, or
`catan-sim --profile-steps`), not estimated.

## Why speed was the strategy, not a nicety

Everything downstream multiplies by step cost:

- PPO training consumed ~500M engine steps across the experiment campaign.
- The AlphaZero-lite agent runs **96 full-game rollouts per candidate
  move** — tens of thousands of engine steps per single decision. At
  Python speeds one AlphaBot *decision* would take ~30 seconds; at 87 ns
  it takes ~50 ms. The 82% agent is *purchased* with engine speed.
- The GA evolved a heuristic through ~25,000 games in two minutes; the
  Elo harness rates agents over 576 games in 12 seconds.

## How the Python engine worked — and where the time went

The original engine (`engine/`, still in this repo) is a perfectly
reasonable Python design: a `GameState` object holding numpy arrays and
attributes (`self.tile_resources`, `self.edges`, `self.phase`, ...), with
rules implemented as functions that loop over the board asking questions.

Here is real code from its road-placement check, the shape of everything:

```python
for edge_idx in range(NUM_EDGES):              # 72 edges
    v1, v2 = topology.edge_vertices[edge_idx]
    for v in [v1, v2]:                          # both endpoints
        owner = state.get_settlement_owner(v)   # method call
        for adj_edge in topology.vertex_edges[v]:   # 3 neighbors each
            if state.edges[adj_edge] == player:     # array index + compare
                ...
    if connected:
        valid.append(edge_idx)                  # list append
return np.array(valid, dtype=np.int8)           # fresh allocation
```

Triple-nested interpreted loops, a method call per probe, and a fresh
list + numpy array allocated *per legality query* — and legality queries
happen for every player decision, hundreds of times per game. Each loop
iteration costs ~50–100 ns of interpreter dispatch before doing any actual
work. Multiply out: ~300,000 ns per game step, ~5 games/s. The numpy
arrays don't save you — the board is too small for vectorization to
amortize, so `np.where` scans and array constructions are pure overhead.
And the GIL means one core, period.

None of this is a criticism of the original code — it's idiomatic Python.
The lesson is that *the costs are structural*: no amount of in-place
optimization fixes interpreter dispatch inside O(board) loops. The fix is
to change what a "rule check" physically is.

## The five ideas that made the difference

### 1. The state is a value, not an object graph

The Rust `GameState` is a **~500-byte `Copy` struct of fixed-size
arrays** — no `HashMap`, no `Vec`, no pointers, no heap anywhere in the
hot state. Board topology (vertex adjacency, edge incidence, tile-vertex
membership) is precomputed into `const` tables, including a
`tiles_by_number[2..12]` index so a dice roll resolves straight to its
affected tiles with no board scan.

Two consequences, one obvious and one that paid off weeks later:

- The whole game fits in L1 cache; every rule check is arithmetic on data
  that is already in registers or L1.
- **Cloning a game is a memcpy.** That single property is what made
  Monte Carlo search practical: AlphaBot clones the game ~768 times per
  decision and plays each copy to completion. In the Python engine a
  defensive deep-copy of the state was itself milliseconds.

### 2. The board is a u64 (bitboards)

Catan has 54 vertices. A `u64` has 64 bits. So the entire occupancy state
of the board is **one integer**, and three of them describe everything
mask generation needs:

```rust
occupied_mask: u64,                  // bit v set = any building on vertex v
vertex_road_mask: [u64; 4],          // per player: bit v = my road touches v
neighbor_mask: [u64; 54],            // const: bits of v's adjacent vertices
```

Now watch Catan's most-frequently-checked rule — the distance rule
("no settlement adjacent to any other") plus occupancy plus connectivity —
become three AND operations (real code, `building.rs`):

```rust
state.occupied_mask & bit == 0                            // vertex free
    && state.occupied_mask & topo.neighbor_mask[v] == 0   // distance rule
    && state.vertex_road_mask[player] & bit != 0          // my road reaches
```

The Python version of that same check walks neighbor lists and calls
`get_settlement_owner` per neighbor. The Rust version is ~3 cycles, branch-
free, and — the flashy part — *enumerates every legal settlement at once*:

```rust
let candidates = state.vertex_road_mask[player]   // all vertices I reach
    & !state.occupied_mask                        // ...that are free
    & VERTEX_BITS;                                // ...on the board
// then iterate set bits; check neighbor_mask per survivor
```

One expression replaces the entire outer loop. Mask generation dropped
**−52% on rich mid-game states** — and masks are generated twice per
decision (once to validate, once for the RL legality mask), making this
the hottest path in the engine.

### 3. Cache what the rules recompute (longest road)

Longest road is the engine's pathological rule: it's a longest-trail
search (directed walk, no edge reuse) — exponential-ish in road density,
and the original design recomputed it **for the whole board on every
single build**.

The insight that fixes it is about *invalidation*, not the search itself:
placing a road or settlement can only change road lengths for players
**whose network touches the placed piece**. Everyone else's cached length
is still true. So:

```rust
road_lengths: [u8; 4],   // cached per player
```

…and on each build, only adjacency-affected players recompute (usually
one, often zero on settlement builds that touch no rival road). The award
logic (holder-keeps-ties, set-aside on non-holder ties) reads the cache.
Full games got **−31%**, and combined with the bitboards, heuristic-game
step time fell **820 → 221 ns**. The 15-road-snake worst case is pinned
in criterion (~1.9 µs) because that's search tail latency — the case
MCTS will hit.

### 4. Zero heap allocations — as an enforced invariant, not a habit

Allocation in a hot loop is death by a thousand cuts: malloc costs more
than an entire game step at these speeds. The discipline: every buffer
the engine needs during play (legal-action scratch, discard queues) is
**pre-sized once and reused**.

The flashy part is how it's *enforced*: a counting global allocator wraps
a test binary and asserts that steady-state play — including the messy
paths: 7-roll multi-player discards, trades, robber moves, steals —
performs **literally zero heap allocations**:

```rust
#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;  // counts every malloc
// ... play 600 steps after warmup ...
assert_eq!(allocations, 0);
```

You're testing the *absence* of a property, which grep can't find and
code review misses. The day it was introduced it caught 15 allocations
per ~800 steps (a discard buffer quietly growing); it later caught the
same class of bug in the RL env layer. Any future per-step `Vec` fails
CI. (Hard-won detail: allocator-counter test binaries must contain
exactly ONE `#[test]` — cargo runs tests in parallel threads, and a
shared global counter bleeds between measurement windows.)

### 5. Measure like a scientist (the meta-idea that made 1–4 safe)

- **ns/step, never games/sec.** Games vary in length — a "faster" run may
  just be playing shorter games. A step is fixed work.
  `catan-sim --profile-steps` splits engine time from agent time so you
  optimize the right side.
- **Criterion saved baselines**: `--save-baseline before` → change →
  `--baseline before` gives statistical pass/fail per benchmark.
- **Behavior locks before speed work**: the self-golden suite replays 89k
  recorded actions to field-exact final states, and the
  property/oracle/fuzz harness must stay green through every
  optimization. Fast-but-wrong is worse than slow — an RL agent is an
  exploit search engine and will learn your bugs as strategy. Three days
  of aggressive optimization, zero behavioral regressions.

## The ladder, end to end

| Stage | Heuristic-game step | What changed |
|---|---|---|
| Python engine | ~300,000 ns | idiomatic Python; structural costs |
| Rust port | ~820 ns | idea 1: data layout (≈370×) |
| + bitboards + road cache | **221 ns** | ideas 2–3 (another 3.7×) |
| (random-agent games) | **87 ns** | 11.5M steps/s single thread |
| 10 threads (rayon) | — | **37M steps/s aggregate** |

## Then: the RL environment layer

Speed without the right *interface* trains nothing. What an RL trainer
actually requires from an environment, and how each requirement was met:

**A fixed action space.** Neural networks output a fixed-size vector, but
Catan's legal moves vary wildly by state. The codec maps every possible
move to one of **299 discrete ids** (build/trade/dev-card/robber/respond,
all parameterizations) plus an exact legality mask the policy multiplies
into its logits. Two non-obvious properties: the mapping is *total* (every
id decodes to something, every legal action encodes uniquely — fuzzed
against the engine so mask and executor cannot drift), and trade actions
are **seat-relative** so the same id means "steal from the player to my
left" regardless of which seat the network occupies.

**A fixed observation.** 1,350 floats: tiles (resource one-hot ×
production probability), all 54 vertices, all 72 edges, public player
state, own hand, bank, game context, pending trade. Everything normalized
to ~[0,1]; dice numbers encoded as *production probabilities* rather than
raw values (a "6" matters because it's 5/36, not because it's six).

**Seat symmetry.** The observation is always from the acting player's
perspective — "I am player 0" — so ONE network plays all four seats and
every game yields 4× the experience. This bakes in the assumption that
strategy is seat-symmetric (true in Catan up to turn order, which the
context block encodes).

**Decisions only, chance internalized.** The env follows turn-based
multi-agent (AEC) semantics: `step(action)` advances to the *next seat
that must make a real decision*. Forced moves — exactly one legal action —
resolve inside the env, and dice are internal chance nodes, not actions.
A tested invariant: every observation the policy ever sees has ≥2 legal
actions. The trainer never wastes a forward pass on a non-decision.

**Credit assignment plumbing.** Rewards are terminal win/loss (±1) with
an optional shaped VP-delta for early training (annealed to zero — the
shaping is a bootstrap crutch, and the accounting is tested to be exact:
total shaped reward delivered = coefficient × final VP). Each seat's
reward accrues privately and is delivered at *its own next decision*, the
AEC analogue of the standard transition tuple.

**Throughput without Python in the loop.** `VecCatanEnv` steps N games in
one rayon-parallel call with auto-reset and deterministic per-episode seed
streams (splitmix64 — same base seed, same games, forever). The criterion
numbers: one full RL decision (mask + step + auto-resolve + obs encode)
costs **346 ns**; a 1,024-env batched step costs **~296 µs ≈ 3.4M
policy-steps/s**, *memory-bandwidth-bound on observation writes* — the
compute is finished; only RAM throughput limits it. The same
counting-allocator standard applies: zero allocations per decision,
bounded per-batch.

**A boundary designed to vanish.** PyO3 bindings move the whole batch in
ONE Python call — NumPy buffers, GIL released while Rust steps 256 games:
~370k policy-steps/s *through Python*, 10–100× more than the network
forward pass consumes. Training profiles show only PyTorch, which is the
design goal. Bot seats (scripted opponents played engine-side during the
env's internal advance) made fixed-opponent evaluation, mixed-opponent
training, and Elo tournaments pure configuration.

**Assumptions made deliberately** (each one a scope decision, not an
accident): trading is a bounded menu (1–2 of one resource for 1 of
another, max 3 offers/turn) to keep the action space learnable; perfect
information first (the POMDP visibility mode exists but is phase 2);
victory target is configurable (first-to-7 curriculum before first-to-10);
and the codec/obs layouts are **versioned and frozen** — every checkpoint
stores them and refuses to run against a mismatch, so trained weights can
never silently desync from the engine.

## Reproduce every number yourself

All of these run from a fresh clone (Rust toolchain; Python side needs
`torch numpy maturin`). Numbers below are from an Apple M-series laptop —
expect the same order of magnitude anywhere.

```bash
cd rust

# --- Bulk throughput: full games, parallel across cores ----------------
cargo run -p catan-sim --release -- --games 20000 --players H,H,H,H
#   elapsed: ~0.5s | games/sec: ~37,000 | steps/sec: ~23,000,000
cargo run -p catan-sim --release -- --games 100000 --players R,R,R,R
#   random games are cheaper per step; watch steps/sec climb

# --- The ns/step breakdown: engine vs agent, by phase ------------------
cargo run -p catan-sim --release -- --games 2000 --players H,H,H,H --profile-steps
#   engine: legal mask   ... ns/step (..%)     <- the hottest path
#   engine: execute      ... ns/step (..%)
#   agent:  choose       ... ns/step (..%)
#   total                ~221 ns/step
#   engine-only ceiling  ... ns/step -> steps/sec if the agent were free
#   (per-step timers add ~100 ns to bucket totals; proportions are the signal)

# --- Microbenchmarks (criterion; saved baselines) -----------------------
cargo bench -p catan-core --bench engine
#   mask_main_midgame_heuristic   ~0.3 µs   | roll+distribute  ~0.2 µs
#   longest_road_15_road_snake    ~1.9 µs   | full random game ~365 µs
cargo bench -p catan-env --bench env
#   env/step_mask_obs            ~346 ns    <- one full RL decision
#   env/vec1024_step_batch       ~296 µs    <- 1,024 envs = 3.4M steps/s
# Workflow for your own changes:
#   cargo bench -p catan-core --bench engine -- --save-baseline before
#   <edit code>
#   cargo bench -p catan-core --bench engine -- --baseline before   # pass/fail

# --- The zero-allocation proofs ----------------------------------------
cargo test --release -p catan-core --test zero_alloc
cargo test --release -p catan-env  --test env_zero_alloc

# --- The Python boundary ------------------------------------------------
cd catan-py && maturin develop --release && cd ../..
python training/smoke_env.py
#   ~370,000 policy-steps/s through Python at batch 256

# --- The legacy Python engine, for the before/after yourself ------------
pip install -r requirements.txt
python run_simulation.py --games 100 --seed 42
#   ~5 games/s — bring a book

# --- What the speed buys: search-based play ------------------------------
cd rust
cargo run -p catan-sim --release -- --games 100 --players A,H,H,H \
    --net ../models/catan-512.ctnn --alpha-config 8,96,300
#   ~768 full-game rollouts per decision, ~80s for 100 games, ~82% wins
```

## What we did NOT do (and why)

Performance docs usually list what was done. The skipped optimizations are
just as informative — each was considered and declined for a reason.

**No hand-written SIMD.** SIMD (Single Instruction, Multiple Data) is the
CPU's built-in vector hardware: special wide registers that apply one
instruction to many values at once — 4–16 floats per cycle instead of one
(NEON on Apple Silicon, AVX on x86). You can program it directly with
"intrinsics" (per-architecture pseudo-assembly), and for raw matmul
throughput that's often worth it. We declined, for three reasons:

1. *The compiler already does it when the data lets it.* LLVM
   auto-vectorizes simple loops over contiguous arrays — our dot products
   (net inference) and array sweeps compile to NEON automatically. The
   prerequisite is layout (fixed contiguous arrays, no pointer chasing),
   which is idea #1. Write the layout, get the SIMD free.
2. *The hot paths that remain don't vectorize.* Rule logic is branchy
   ("if the robber is here and that player has cards and..."); SIMD hates
   branches — all lanes must take the same path.
3. *Bitboards already are SIMD, secretly.* A `u64` AND processes **64
   board positions in one ordinary instruction** — one-bit lanes in a
   general-purpose register. The distance-rule check is effectively a
   64-wide vector op without a single intrinsic. This is the oldest trick
   in computer chess, and it's why the engine gets vector-class speedups
   while staying portable, safe, and readable.

Intrinsics would buy maybe 2× on inference at the cost of
per-architecture code, `unsafe` blocks, and a maintenance tax — and the
batch path is already memory-bandwidth-bound, where more compute buys
nothing.

**No `unsafe`.** Rust lets you opt out of bounds/aliasing checks in
`unsafe` blocks; classic engine territory ("skip the bounds check in the
inner loop"). But the compiler elides almost all bounds checks here
anyway — indices into fixed-size arrays with provable ranges — and this
codebase's entire premise is that *correctness is the product* (the agent
exploits any bug it finds). Trading proven-safe for a low single-digit
percent was never on the table.

**No custom allocators.** Arena/bump allocators make allocation cheap.
We made allocation *nonexistent* instead — the steady state performs
zero heap allocations, enforced by test. The fastest allocator is the one
you never call. (This is also why the counting-allocator test matters
more than any allocator choice: it keeps the count at zero forever.)

**No GPU for the engine.** GPUs win when thousands of identical
computations run in lockstep (big matmuls — which is why the *network*
trains on one). A Catan step is the opposite: a tiny, branchy state
machine where every game is at a different phase taking different paths —
divergent control flow leaves GPU lanes idle. The right split, and the
one we built: engine on CPU cores (embarrassingly parallel at *game*
granularity via rayon — no locks, games share nothing), network on the
accelerator. The future batched-leaf-evaluation search keeps that split.

**No fine-grained parallelism.** No threads *inside* a game, no locks, no
atomics in the hot path. Parallelism lives at the game level where it's
free. The one shared structure anywhere near the hot loop (replay/stat
harvesting in the vec env) is a mutex touched only on episode *end* —
once per ~600 steps.

**No optimizing before the test fortress existed.** Order of operations
was the whole game: lock behavior (goldens + property tests + oracle),
save a baseline, then make it fast. Three days of aggressive optimization,
zero behavioral regressions — that's not luck, it's sequencing.
