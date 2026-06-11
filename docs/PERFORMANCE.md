# The performance story: 300,000 ns/step to 87 ns/step

How the engine became roughly 3,400 times faster per core than the Python
original. Every number here is measured, either by criterion benchmarks
with saved baselines or by `catan-sim --profile-steps`.

## Why speed was the strategy

Everything downstream multiplies by step cost:

- PPO training consumed about 500M engine steps across the experiment
  campaign.
- The AlphaZero-lite agent runs 96 full-game rollouts per candidate move,
  which is tens of thousands of engine steps for a single decision. At
  Python speeds one AlphaBot decision would take about 30 seconds; at
  87 ns it takes about 50 ms. The 82% win rate is paid for with engine
  speed.
- The genetic algorithm evaluated about 25,000 games in two minutes. The
  Elo harness rates agents over 576 games in 12 seconds.

## How the Python engine worked, and where the time went

The original engine (`engine/`, still in this repo) is a reasonable
Python design: a `GameState` object holding numpy arrays and attributes
(`self.tile_resources`, `self.edges`, `self.phase`), with rules
implemented as functions that loop over the board.

This is real code from its road-placement check, and it is the shape of
everything in that engine:

```python
for edge_idx in range(NUM_EDGES):              # 72 edges
    v1, v2 = topology.edge_vertices[edge_idx]
    for v in [v1, v2]:                          # both endpoints
        owner = state.get_settlement_owner(v)   # method call
        for adj_edge in topology.vertex_edges[v]:   # 3 neighbors each
            if state.edges[adj_edge] == player:     # index + compare
                ...
    if connected:
        valid.append(edge_idx)                  # list append
return np.array(valid, dtype=np.int8)           # fresh allocation
```

Triple-nested interpreted loops, a method call per probe, and a new list
plus a new numpy array allocated per legality query. Legality queries run
for every decision, hundreds of times per game. Each loop iteration costs
roughly 50 to 100 ns of interpreter dispatch before doing any useful
work. The numpy arrays do not help: the board is too small for
vectorization to amortize, so `np.where` scans and array constructions
are pure overhead. The GIL limits the whole engine to one core.

None of this is a criticism of the original code; it is idiomatic Python.
The point is that the costs are structural. No in-place optimization
removes interpreter dispatch from an O(board) loop. The fix is to change
what a rule check physically is.

## The five ideas that made the difference

### 1. The state is a value, not an object graph

The Rust `GameState` is a roughly 500-byte `Copy` struct of fixed-size
arrays. No `HashMap`, no `Vec`, no pointers, no heap anywhere in the hot
state. Board topology (vertex adjacency, edge incidence, tile-vertex
membership) is precomputed into `const` tables, including a
`tiles_by_number` index so a dice roll resolves directly to its affected
tiles without scanning the board.

Two consequences, one immediate and one that paid off later:

- The whole game fits in L1 cache, so every rule check operates on data
  that is already in registers or L1.
- Cloning a game is a memcpy. This single property is what made Monte
  Carlo search practical: AlphaBot clones the game roughly 768 times per
  decision and plays each copy to completion. In the Python engine, a
  defensive deep copy of the state cost milliseconds by itself.

### 2. Bitboards: the board as integers

This is the technique that transformed mask generation, so it deserves a
full explanation.

**The core idea.** Give each of the 54 vertices on the board a fixed
index, 0 through 53. A `u64` integer has 64 bits, so one integer can
record a yes/no fact about every vertex simultaneously: bit `v` answers
the question for vertex `v`. In other words, *a set of board locations
becomes a single integer*. A useful mental model is a panel of 54 labeled
light switches: one integer is one panel, and each switch is on or off.

Once sets are integers, set operations become single CPU instructions:

| Set question | Bit operation |
|---|---|
| Is vertex v in the set? | `mask & (1 << v) != 0` |
| Intersection of two sets | `a & b` |
| Union of two sets | `a \| b` |
| Everything not in the set | `!a` |
| Is the set empty? | `mask == 0` |

The engine maintains three such structures:

- `occupied_mask: u64`. Bit `v` is 1 if any player has a settlement or
  city on vertex `v`. One integer describes the occupancy of the entire
  board.
- `vertex_road_mask: [u64; 4]`, one per player. Bit `v` is 1 if player
  `p` has at least one road with vertex `v` as an endpoint. Concretely:
  every road sits on an edge, and every edge connects exactly two
  vertices. When player `p` builds a road on the edge between vertices
  `a` and `b`, the engine sets bits `a` and `b` in
  `vertex_road_mask[p]`. So "bit v is set" means "player p's road
  network reaches vertex v", which is exactly the condition Catan's
  rules require for p to build a settlement there.
- `neighbor_mask: [u64; 54]`, a `const` table built once from the board
  layout. Entry `neighbor_mask[v]` is the set of vertices directly
  adjacent to `v` (sharing an edge with it). Each vertex has two or
  three neighbors, so each entry has two or three bits set. This table
  never changes; it is the board's geometry, precomputed.

**A worked example.** Suppose vertex 12 has neighbors 11, 13, and 30.
Then `neighbor_mask[12]` is the integer with exactly bits 11, 13, and 30
set. Player 2 wants to build a settlement on vertex 12. Catan requires
three things, and each is one bit operation:

1. Vertex 12 must be empty:
   `occupied_mask & (1 << 12) == 0`.
2. The distance rule: no settlement or city on any adjacent vertex. This
   is an intersection test between two sets, "occupied vertices" and
   "neighbors of 12":
   `occupied_mask & neighbor_mask[12] == 0`.
   If, say, someone holds vertex 13, then bit 13 is set in both masks,
   the AND is nonzero, and the placement is illegal. One instruction
   checks all neighbors at once.
3. Connectivity: one of player 2's roads must reach vertex 12:
   `vertex_road_mask[2] & (1 << 12) != 0`.

The Python engine answered the same three questions by walking neighbor
lists and calling `get_settlement_owner` per neighbor. The Rust engine
answers them in about three cycles with no branches.

**Enumerating all legal placements at once.** The same algebra works in
the other direction. Instead of asking "is vertex 12 legal?", ask "which
vertices are legal?":

```rust
let candidates = state.vertex_road_mask[player]   // vertices my roads reach
    & !state.occupied_mask                        // that are unoccupied
    & VERTEX_BITS;                                // and on the board
```

That one expression intersects "reachable by my roads" with "not
occupied" for all 54 vertices simultaneously. The engine then iterates
the set bits of `candidates` (a hardware instruction finds each one) and
applies the neighbor check to the few survivors. The entire outer loop of
the Python version is replaced by one line.

Mask generation dropped 52% on rich mid-game states. Masks are generated
twice per decision (once to validate the action, once to produce the RL
legality mask), which made this the hottest path in the engine.

### 3. Cache what the rules recompute (longest road)

Longest road is the engine's most expensive rule. Computing a player's
longest road is a longest-trail search: a walk along the player's road
network that may revisit vertices but never reuses an edge. Its cost
grows quickly with road density, and the original design recomputed it
for the whole board after every single build.

The fix is an observation about invalidation rather than a faster search:
placing a road or settlement can only change the road length of players
whose network touches the placed piece. Everyone else's previous answer
is still correct. So the engine keeps a cache:

```rust
road_lengths: [u8; 4],   // longest road per player, kept current
```

On each build, only the players whose networks touch the new piece are
recomputed. Usually that is one player; for settlement builds that touch
no rival road, it is zero. The award logic (current holder keeps the
title on ties; the title is set aside on ties between non-holders) reads
the cache. Full games sped up 31%, and combined with the bitboards,
heuristic-game step time fell from 820 ns to 221 ns. The worst case, a
15-road snake, is pinned in criterion at about 1.9 µs because that is
the tail latency a search algorithm will hit.

### 4. Zero heap allocations, enforced by a test

At these speeds a single malloc costs more than an entire game step, so
the discipline is: every buffer the engine needs during play (the
legal-action scratch list, discard queues) is allocated once and reused.

What keeps this true permanently is a test, not a convention. A counting
global allocator wraps a test binary and asserts that steady-state play,
including the messy paths (multi-player discards on a 7, trades, robber
moves, steals), performs exactly zero heap allocations:

```rust
#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;  // counts every malloc
// ... play 600 steps after warmup ...
assert_eq!(allocations, 0);
```

This tests the absence of a property, which neither grep nor code review
reliably catches. The day the test was introduced it found 15 allocations
per 800 steps (a discard buffer quietly growing), and it later caught the
same class of bug in the RL environment layer. Any future per-step `Vec`
fails the suite. One implementation note: a test binary using a counting
allocator must contain exactly one `#[test]`, because cargo runs tests in
parallel threads and a shared global counter bleeds between measurement
windows.

### 5. Measurement discipline

- Measure in ns/step, never games/sec. Games vary in length, so a
  "faster" run may simply be playing shorter games. A step is fixed
  work. `catan-sim --profile-steps` also splits engine time from agent
  time so the right side gets optimized.
- Criterion saved baselines: `--save-baseline before`, make the change,
  then `--baseline before` for a statistical pass/fail per benchmark.
- Behavior locks before speed work: the self-golden suite replays 89k
  recorded actions to field-exact final states, and the property, oracle,
  and fuzz harnesses must stay green through every optimization. A wrong
  fast engine is worse than a slow correct one, because an RL agent will
  find the bug and learn it as strategy. Three days of optimization
  produced zero behavioral regressions.

## The ladder, end to end

| Stage | Heuristic-game step | What changed |
|---|---|---|
| Python engine | ~300,000 ns | idiomatic Python; structural costs |
| Rust port | ~820 ns | idea 1: data layout (~370x) |
| + bitboards + road cache | **221 ns** | ideas 2 and 3 (another 3.7x) |
| (random-agent games) | **87 ns** | 11.5M steps/s single thread |
| 10 threads (rayon) | | **37M steps/s aggregate** |

## Then: the RL environment layer

Speed without the right interface trains nothing. What an RL trainer
requires from an environment, and how each requirement was met:

**A fixed action space.** Neural networks output a fixed-size vector,
but Catan's legal moves vary by state. The codec maps every possible
move to one of 299 discrete ids (build, trade, dev card, robber,
respond, with all parameterizations), plus an exact legality mask the
policy applies to its logits. Two properties matter: the mapping is
total (every id decodes to something, every legal action encodes
uniquely, fuzzed against the engine so the mask and the executor cannot
drift), and opponent-directed actions are seat-relative, so the same id
means "steal from the player to my left" regardless of which seat the
network occupies.

**A fixed observation.** 1,350 floats: tiles (resource one-hot and
production probability), all 54 vertices, all 72 edges, public player
state, the player's own hand, bank, game context, and any pending trade.
Values are normalized to roughly [0, 1]. Dice numbers are encoded as
production probabilities rather than raw values, because a "6" matters
through its 5/36 chance of rolling, not through the numeral.

**Seat symmetry.** The observation is always from the acting player's
perspective, so one network plays all four seats, and every game yields
four seats' worth of experience. The assumption baked in is that strategy
is seat-symmetric, which holds in Catan up to turn order, and turn order
is encoded in the context block.

**Decisions only, chance internalized.** The environment follows
turn-based multi-agent (AEC) semantics: `step(action)` advances to the
next seat that must make a real decision. Forced moves, where exactly
one action is legal, resolve inside the environment, and dice are
internal chance nodes rather than actions. A tested invariant: every
observation the policy sees has at least two legal actions. The trainer
never spends a forward pass on a non-decision.

**Credit assignment plumbing.** Rewards are terminal win/loss (+1/-1)
with an optional shaped per-VP bonus for early training, annealed to
zero. The shaping is a bootstrap aid, and its accounting is tested to be
exact: total shaped reward delivered equals the coefficient times final
VP. Each seat's reward accrues privately and is delivered at that seat's
next decision, the AEC analogue of the standard transition tuple.

**Throughput without Python in the loop.** `VecCatanEnv` steps N games
in one rayon-parallel call with auto-reset and deterministic per-episode
seed streams (splitmix64: the same base seed reproduces the same games).
One full RL decision (mask, step, auto-resolve, observation encode)
costs 346 ns. A 1,024-environment batched step costs about 296 µs, or
3.4M policy-steps/s, and is memory-bandwidth-bound on observation
writes: the computation is done, and only RAM throughput limits it. The
same counting-allocator standard applies here: zero allocations per
decision, bounded allocations per batch.

**A boundary designed to disappear.** PyO3 bindings move the whole batch
in one Python call: NumPy buffers, GIL released while Rust steps 256
games in parallel, about 370k policy-steps/s through Python. That is 10
to 100 times more than the network forward pass consumes, which is the
design goal: training profiles show only PyTorch. Bot seats (scripted
opponents played engine-side during the environment's internal advance)
made fixed-opponent evaluation, mixed-opponent training, and Elo
tournaments pure configuration.

**Assumptions made deliberately.** Each of these is a scope decision:
trading is a bounded menu (1 or 2 of one resource for 1 of another, at
most 3 offers per turn) to keep the action space learnable; perfect
information first (a realistic-visibility mode exists but is a later
phase); the victory target is configurable (a first-to-7 curriculum
before first-to-10); and the codec and observation layouts are versioned
and frozen. Every checkpoint stores the versions and refuses to run
against a mismatch, so trained weights can never silently desync from
the engine.

## Reproduce every number yourself

All of these run from a fresh clone (Rust toolchain; the Python side
needs `torch numpy maturin`). The numbers come from an Apple M-series
laptop; expect the same order of magnitude anywhere.

```bash
cd rust

# --- Bulk throughput: full games, parallel across cores ----------------
cargo run -p catan-sim --release -- --games 20000 --players H,H,H,H
#   elapsed: ~0.5s | games/sec: ~37,000 | steps/sec: ~23,000,000
cargo run -p catan-sim --release -- --games 100000 --players R,R,R,R
#   random games are cheaper per step; steps/sec rises further

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

# --- The legacy Python engine, to see the before/after yourself ---------
pip install -r requirements.txt
python run_simulation.py --games 100 --seed 42
#   ~5 games/s

# --- What the speed buys: search-based play ------------------------------
cd rust
cargo run -p catan-sim --release -- --games 100 --players A,H,H,H \
    --net ../models/catan-512.ctnn --alpha-config 8,96,300
#   ~768 full-game rollouts per decision, ~80s for 100 games, ~82% wins
```

## What we did not do, and why

The skipped optimizations are as informative as the implemented ones.
Each was considered and declined for a reason.

**No hand-written SIMD.** SIMD (Single Instruction, Multiple Data) is
the CPU's built-in vector hardware: wide registers that apply one
instruction to several values at once, typically 4 to 16 floats per
cycle instead of one (NEON on Apple Silicon, AVX on x86). It can be
programmed directly through "intrinsics", which are per-architecture,
near-assembly function calls. For raw matrix-multiply throughput that is
often worthwhile. We declined, for three reasons:

1. The compiler already produces SIMD when the data layout allows it.
   LLVM auto-vectorizes simple loops over contiguous arrays, so the dot
   products in the net inference and the array sweeps compile to vector
   instructions on their own. The prerequisite is layout (fixed
   contiguous arrays, no pointer chasing), which is idea 1. Get the
   layout right and the SIMD comes free.
2. The remaining hot paths do not vectorize. Rule logic is branchy ("if
   the robber is here and that player has cards and..."), and SIMD
   requires all lanes to take the same path.
3. Bitboards already provide the same effect for board logic. A `u64`
   AND processes 64 board positions in one ordinary instruction:
   one-bit lanes in a general-purpose register. The distance-rule check
   is effectively a 64-wide vector operation with no intrinsics, no
   `unsafe`, and no per-architecture code. This is a long-established
   technique from computer chess, and it is the main reason the engine
   reaches vector-class speed while staying portable and readable.

Intrinsics might buy another 2x on net inference, at the cost of
per-architecture code paths and a maintenance burden, and the batched
path is already memory-bandwidth-bound, where extra compute buys
nothing.

**No `unsafe`.** Rust allows opting out of bounds and aliasing checks
inside `unsafe` blocks, and skipping bounds checks in inner loops is
classic game-engine practice. Two reasons not to: the compiler already
elides almost all bounds checks here, because the indices go into
fixed-size arrays with provable ranges; and this codebase's premise is
that correctness is the product, since the agent exploits any bug it
finds. Trading proven safety for a low single-digit percent was not
worth it.

**No custom allocators.** Arena and bump allocators make allocation
cheap. We made allocation nonexistent instead: the steady state performs
zero heap allocations, enforced by test. An allocator that is never
called costs nothing, and the counting-allocator test matters more than
any allocator choice because it keeps the count at zero permanently.

**No GPU for the engine.** GPUs excel when thousands of identical
computations run in lockstep, which is why the network trains on one. A
Catan step is the opposite case: a small, branchy state machine where
every game is in a different phase taking different code paths, and
divergent control flow leaves GPU lanes idle. The split we built is the
appropriate one: engine on CPU cores, parallel at game granularity
through rayon (games share nothing, so there are no locks), and the
network on the accelerator. The planned batched-leaf-evaluation search
keeps the same split.

**No fine-grained parallelism.** No threads inside a game, no locks, no
atomics in the hot path. Parallelism lives at the game level, where it
is free. The one shared structure near the hot loop (replay and stat
harvesting in the vectorized environment) is a mutex touched only at
episode end, once per ~600 steps.

**No optimizing before the test fortress existed.** The order of
operations mattered most: lock behavior with goldens, property tests,
and the oracle; save a baseline; then make it fast. Zero behavioral
regressions across the entire optimization campaign is a result of that
sequencing, not luck.
