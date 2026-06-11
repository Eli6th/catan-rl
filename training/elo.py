"""Elo tournament + best.pt promotion for Catan checkpoints.

Tournament: random 4-seat tables drawn from checkpoints + scripted anchors
(random, heuristic). Scripted anchors play as engine bot seats; checkpoint
seats are routed to their own network by seat index. Each finished game
yields pairwise Elo updates (winner beats each opponent). RandomPlayer is
the fixed 1000-Elo anchor.

    python training/elo.py tournament ckptA.pt ckptB.pt --games-per-table 16
    python training/elo.py promote training/runs/<run>   # best.pt gate

Promotion follows training/README.md: fixed-seed eval vs 3 Heuristic-v1;
promote latest.pt to best.pt only if it beats the incumbent's recorded win
rate by more than 2 points; eval games are recorded as CTRP replays.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import catan_py
from ppo import PolicyValueNet

VICTORY_TARGET = 7
ELO_K = 24


class Participant:
    def __init__(self, name: str, kind: str, net=None):
        self.name = name
        self.kind = kind  # "policy" | "random" | "heuristic"
        self.net = net
        self.elo = 1000.0
        self.games = 0
        self.wins = 0


def load_policy(path: str) -> Participant:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["codec_version"] == catan_py.CODEC_VERSION, f"{path}: codec mismatch"
    assert ck["obs_version"] == catan_py.OBS_VERSION, f"{path}: obs version mismatch"
    net = PolicyValueNet(catan_py.OBS_DIM, catan_py.NUM_ACTIONS, ck["config"]["hidden"])
    net.load_state_dict(ck["model_state"])
    net.eval()
    name = Path(path).parent.parent.name + "/" + Path(path).stem
    return Participant(name, "policy", net)


@torch.no_grad()
def play_table(table: list, games: int, seed: int) -> list:
    """Play `games` 4-seat games with table[i] in seat i. Returns winner
    participant indices (into `table`) for decisive games."""
    seats_cfg = [p.kind if p.kind != "policy" else "policy" for p in table]
    env = catan_py.VecEnv(
        min(16, games), victory_target=VICTORY_TARGET, seed=seed, seats=seats_cfg
    )
    obs, masks, seats = env.observe()
    winners = []
    while len(winners) < games:
        actions = np.zeros(env.num_envs, dtype=np.uint32)
        for seat_idx, participant in enumerate(table):
            if participant.kind != "policy":
                continue
            rows = seats == seat_idx
            if not rows.any():
                continue
            o = torch.as_tensor(obs[rows])
            m = torch.as_tensor(masks[rows])
            logits, _ = participant.net(o, m)
            actions[rows] = logits.argmax(dim=1).numpy().astype(np.uint32)
        obs, masks, seats, _, dones, terminals = env.step(actions)
        for turns, winner, vps, cap in env.take_episode_stats():
            if winner >= 0 and len(winners) < games:
                winners.append(int(winner))
    return winners


def update_elo(table, winner_seat: int):
    winner = table[winner_seat]
    for seat, loser in enumerate(table):
        if seat == winner_seat:
            continue
        expected = 1.0 / (1.0 + 10 ** ((loser.elo - winner.elo) / 400.0))
        delta = ELO_K * (1.0 - expected)
        if winner.name != "random":
            winner.elo += delta
        if loser.name != "random":  # RandomPlayer is the fixed anchor
            loser.elo -= delta
    winner.wins += 1
    for p in table:
        p.games += 1


def tournament(args):
    participants = [Participant("random", "random"), Participant("heuristic", "heuristic"),
                    Participant("heuristic_v2", "heuristic_v2")]
    participants += [load_policy(p) for p in args.checkpoints]
    assert len(participants) >= 4, "need at least 4 participants (incl. anchors)"
    rng = random.Random(args.seed)

    start = time.time()
    total_games = 0
    for table_idx in range(args.tables):
        table = rng.sample(participants, 4)
        rng.shuffle(table)  # seat rotation across tables
        winners = play_table(table, args.games_per_table, seed=args.seed + table_idx)
        for w in winners:
            update_elo(table, w)
        total_games += len(winners)

    participants.sort(key=lambda p: -p.elo)
    print(f"\n{total_games} games across {args.tables} tables "
          f"in {time.time()-start:.0f}s\n")
    print(f"{'participant':<44} {'elo':>6} {'games':>6} {'win%':>6}")
    for p in participants:
        anchor = " (anchor)" if p.name == "random" else ""
        print(f"{p.name:<44} {p.elo:>6.0f} {p.games:>6} "
              f"{100*p.wins/max(1,p.games):>5.1f}%{anchor}")

    out = {
        p.name: {"elo": round(p.elo), "games": p.games, "wins": p.wins}
        for p in participants
    }
    out_path = Path(__file__).parent / "runs" / "elo.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {out_path}")


@torch.no_grad()
def promote(args):
    run_dir = Path(args.run_dir)
    ck_path = run_dir / "checkpoints" / "latest.pt"
    candidate = load_policy(str(ck_path))
    step = torch.load(ck_path, map_location="cpu", weights_only=False)["global_step"]

    # Fixed-seed eval vs 3 Heuristic-v1, with CTRP recording.
    env = catan_py.VecEnv(
        48, victory_target=VICTORY_TARGET, seed=777,
        seats=["policy", "heuristic", "heuristic", "heuristic"],
    )
    env.enable_recording()
    obs, masks, seats = env.observe()
    wins = done_eps = 0
    replays = []
    while done_eps < args.games:
        o = torch.as_tensor(obs)
        m = torch.as_tensor(masks)
        logits, _ = candidate.net(o, m)
        actions = logits.argmax(dim=1).numpy().astype(np.uint32)
        obs, masks, seats, _, dones, terminals = env.step(actions)
        for turns, winner, vps, cap in env.take_episode_stats():
            done_eps += 1
            wins += int(winner == 0)
        replays.extend(env.take_replays())
    win_rate = wins / done_eps

    replay_dir = run_dir / "replays" / f"step_{step:010d}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    for i, blob in enumerate(replays[:60]):
        (replay_dir / f"eval_{i:03d}.ctrp").write_bytes(blob)

    best_file = run_dir / "checkpoints" / "best_eval.json"
    incumbent = json.loads(best_file.read_text())["win_rate"] if best_file.exists() else None
    promoted = incumbent is None or win_rate > incumbent + 0.02
    print(f"candidate step {step:,}: {win_rate*100:.1f}% vs heuristic-v1 over "
          f"{done_eps} fixed-seed games "
          f"(incumbent: {'-' if incumbent is None else f'{incumbent*100:.1f}%'})")
    if promoted:
        best = run_dir / "checkpoints" / "best.pt"
        if best.is_symlink() or best.exists():
            best.unlink()
        best.symlink_to(ck_path.resolve().name)
        best_file.write_text(json.dumps(
            {"win_rate": win_rate, "step": step, "games": done_eps}))
        print(f"PROMOTED -> best.pt ({len(replays[:60])} eval replays saved to "
              f"{replay_dir})")
    else:
        print("not promoted (within noise of incumbent)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tournament")
    t.add_argument("checkpoints", nargs="+")
    t.add_argument("--tables", type=int, default=30)
    t.add_argument("--games-per-table", type=int, default=16)
    t.add_argument("--seed", type=int, default=0)
    pr = sub.add_parser("promote")
    pr.add_argument("run_dir")
    pr.add_argument("--games", type=int, default=192)
    args = p.parse_args()
    if args.cmd == "tournament":
        tournament(args)
    else:
        promote(args)


if __name__ == "__main__":
    main()
