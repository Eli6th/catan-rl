"""Smoke test for the catan_py bindings: the full trainer-facing contract.

Run after `maturin develop --release` in rust/catan-py:
    python training/smoke_env.py
"""

import time

import numpy as np

import catan_py


def masked_random_actions(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform-random legal action per row: argmax of random scores on the
    legal entries (the standard vectorized masked-sampling trick)."""
    scores = rng.random(masks.shape, dtype=np.float32)
    scores[~masks] = -1.0
    return scores.argmax(axis=1).astype(np.uint32)


def main() -> None:
    n = 256
    env = catan_py.VecEnv(n, seed=42)

    # Version/shape contract (what checkpoints must store and verify).
    assert env.codec_version == catan_py.CODEC_VERSION == 1
    assert env.obs_version == catan_py.OBS_VERSION == 1
    assert env.num_actions == catan_py.NUM_ACTIONS == 299
    assert env.obs_dim == catan_py.OBS_DIM == 1350

    obs, masks, seats = env.observe()
    assert obs.shape == (n, env.obs_dim) and obs.dtype == np.float32
    assert masks.shape == (n, env.num_actions) and masks.dtype == np.bool_
    assert seats.shape == (n,) and seats.dtype == np.uint32
    assert masks.sum(axis=1).min() >= 2, "auto-resolve property violated"

    rng = np.random.default_rng(0)
    episodes = 0
    reward_check = 0.0
    steps = 0
    start = time.perf_counter()
    while episodes < 200:
        actions = masked_random_actions(masks, rng)
        obs, masks, seats, rewards, dones, terminals = env.step(actions)
        steps += n
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        assert masks.sum(axis=1).min() >= 2
        assert (seats < 4).all()
        if dones.any():
            episodes += int(dones.sum())
            rows = terminals[dones]
            sums = rows.sum(axis=1)
            decisive = np.isclose(sums, -2.0)  # +1 winner, -1 x3
            truncated = np.isclose(np.abs(rows).sum(axis=1), 0.0)
            assert (decisive | truncated).all(), f"bad terminal rows: {rows[~(decisive | truncated)]}"
            reward_check += float(sums[decisive].sum())
    elapsed = time.perf_counter() - start
    print(f"{episodes} episodes, {steps:,} batched policy-steps in {elapsed:.2f}s "
          f"({steps/elapsed:,.0f} steps/s through Python)")

    # Determinism across processes-worth of state: same seed, same outcome.
    def fingerprint(seed: int) -> float:
        e = catan_py.VecEnv(16, seed=seed)
        o, m, s = e.observe()
        r = np.random.default_rng(1)
        total = 0.0
        for _ in range(300):
            o, m, s, rew, d, t = e.step(masked_random_actions(m, r))
            total += float(o.sum()) + float(t.sum())
        return total

    assert fingerprint(7) == fingerprint(7), "seeded runs must reproduce"
    assert fingerprint(7) != fingerprint(8), "different seeds must differ"

    # Replay harvesting -> CTRP bytes (the video pipeline hook).
    rec = catan_py.VecEnv(8, seed=11)
    rec.enable_recording()
    o, m, s = rec.observe()
    r = np.random.default_rng(2)
    harvested = 0
    while harvested < 5:
        o, m, s, rew, d, t = rec.step(masked_random_actions(m, r))
        if d.any():
            replays = rec.take_replays()
            harvested += len(replays)
            for blob in replays:
                assert blob[:4] == b"CTRP", "replay bytes must be CTRP format"
    print(f"harvested {harvested} CTRP replays from the env")

    # First-to-7 curriculum + realistic visibility construct cleanly.
    catan_py.VecEnv(4, victory_target=7, visibility="realistic", vp_delta=0.05, seed=1)

    # Single-env handle for eval loops.
    e = catan_py.Env(seed=3)
    mask = e.mask()
    legal = np.flatnonzero(mask)
    seat, reward, done, winner, terminal = e.step(int(legal[0]))
    assert not done and e.obs().shape == (catan_py.OBS_DIM,)

    print("smoke test passed: bindings honor the full contract")


if __name__ == "__main__":
    main()
