# Training: PPO self-play

`ppo.py` is the trainer: masked PPO over `catan_py.VecEnv`, one policy
playing all seats, per-(env, seat) trajectory chains for the AEC reward
semantics, GAE with gamma=1.0 (episodic — don't discount 600-step credit),
lambda=0.95. It emits `train`/`eval`/`game` events to the metrics stream
(watch on the catan-web dashboard) and saves checkpoints per the contract
below. Stage-1 smoke run:

```bash
python training/ppo.py --name stage1 --minutes 10 \
    --victory-target 7 --vp-delta 0.05 --metrics /tmp/catan-metrics.jsonl
rust/target/release/catan-web --metrics-file /tmp/catan-metrics.jsonl
```

`smoke_env.py` is the bindings contract test — run it after rebuilding
catan-py.

## Artifact storage and checkpoint gating

## Directory layout

```
training/
├── configs/                  # versioned run configs (IN git)
├── ppo.py, eval.py, ...      # training code (IN git)
└── runs/                     # all run artifacts (NOT in git)
    └── <date>-<name>/        # e.g. 2026-06-20-first-to-7-baseline
        ├── config.yaml       # frozen copy of the exact config used
        ├── meta.json         # git commit, codec/obs versions, seeds, host
        ├── tensorboard/      # curves (entropy, EV, clip frac, win rates)
        ├── eval/             # eval results per checkpoint (CSV)
        ├── replays/          # .ctrp game records from eval (video source)
        └── checkpoints/
            ├── step_00010000.pt
            ├── latest.pt -> ...      # symlink, updated every save
            ├── best.pt -> ...        # symlink, updated ONLY via the gate
            └── pool/                 # frozen self-play opponents
```

## Checkpoint contract

Every checkpoint `.pt` bundles (single torch.save dict):

- `model_state`, `optimizer_state`, `global_step`, `rng_states`
- `config`: the full run config (not a path — the actual values)
- `codec_version`: must equal `catan_env::CODEC_VERSION` (currently 1)
- `num_actions`: must equal `catan_env::NUM_ACTIONS` (currently 299)
- `obs_version` / `obs_dim`: must match the observation encoder
- `engine_commit`: `git rev-parse HEAD` at save time

**Loading MUST hard-fail on any version mismatch.** A checkpoint trained
against action layout v1 run against layout v2 doesn't crash — it plays
confidently wrong moves. The version check turns a silent corruption into
a loud error.

Saves are atomic: write to `step_X.pt.tmp`, fsync, rename. A crash mid-save
must never corrupt the latest checkpoint.

## Promotion gates

- `latest.pt`: every save. No gate.
- `pool/`: every Nth checkpoint (default: 1 per 30 min of training),
  kept forever — self-play opponents and the Elo ladder.
- `best.pt`: promoted ONLY when a fixed evaluation passes:
  400 games vs Heuristic-v1 on fixed seeds, low temperature, mixed
  seating; promote if win rate beats the current best.pt's recorded
  score by more than 2 points (outside seed noise). The eval result is
  stored next to the checkpoint in `eval/`.

Anything that ships (the video, a demo, the visualizer bot) loads
`best.pt`, never `latest.pt`.

## Game replays (video pipeline)

Eval games are recorded as `.ctrp` binary records (~2-6 KB/game; format in
`rust/catan-core/src/replay.rs`, written via `GameRecord`). Inspect any
replay with:

```
rust/target/release/catan-sim --dump-replay runs/<run>/replays/<file>.ctrp
```

Recording per checkpoint eval (a few hundred games) costs ~1-2 MB — record
every eval, forever. The video shows the same fixed seeds played by
checkpoints across training time.

## Live metrics stream (dashboard contract)

All producers append JSON lines to one metrics file; `catan-web` tails it
and serves the live dashboard at `http://127.0.0.1:5050/dashboard`:

```
rust/target/release/catan-web --metrics-file <run>/metrics.jsonl
```

Event types (one JSON object per line, `unix_ms` on everything):

- `{"t":"run", "source":, "games":, "players":[...], "seed":}` — once, at start.
- `{"t":"game", "i":, "winner":, "turns":, "steps":, "vp":[...], "cap":bool}`
  — per finished game (`catan-sim --metrics` emits these today; the env's
  rollout loop will too).
- `{"t":"train", "step":, "entropy":, "explained_variance":, "clip_frac":,
  "policy_loss":, "value_loss":, "lr":, "sps":}` — per PPO update.
- `{"t":"eval", "step":, "vs":"heuristic-v1", "win_rate":, "games":,
  "avg_vp":, "cap_rate":}` — per checkpoint evaluation, one line per opponent.

The dashboard's health panel turns these into OK/WATCH/BAD verdicts with
explanations (turn-cap rate, throughput collapse, entropy cliff, explained
variance bands, clip-fraction bands, eval-regression detection) — the curve
reading rules, automated.
