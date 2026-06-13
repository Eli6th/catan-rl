"""Masked PPO self-play on the Catan VecEnv.

One policy plays all seats (seat-relative obs). Trajectories are stitched
per (env, seat): a transition completes when that seat next acts — at that
moment we know both its accrued reward and its bootstrap value — or at
episode end via terminal_rewards.

Stage-1 smoke run (first-to-7, perfect info, shaped reward):
    python training/ppo.py --name smoke1 --minutes 10 \
        --victory-target 7 --vp-delta 0.05 \
        --metrics /tmp/catan-metrics.jsonl
Watch it live: catan-web --metrics-file <metrics path> -> /dashboard
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import catan_py

GAMMA = 1.0  # episodic, bounded: do NOT discount 600-step credit away
GAE_LAMBDA = 0.95
CLIP = 0.2
DEFAULT_ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


@dataclass(frozen=True)
class AlphaBudget:
    root_k: int
    samples: int
    depth: int


def parse_alpha_budget_cycle(spec: str | None, args) -> list[AlphaBudget]:
    if not spec:
        return [
            AlphaBudget(
                args.train_alpha_root_k,
                args.train_alpha_samples,
                args.train_alpha_depth,
            )
        ]
    budgets = []
    for item in spec.split(","):
        budget_text, _, repeat_text = item.strip().partition("x")
        parts = budget_text.split(":")
        if len(parts) != 3:
            raise ValueError(
                "alpha budget entries must be root_k:samples:depth[xrepeat]"
            )
        budget = AlphaBudget(*(int(part) for part in parts))
        repeat = int(repeat_text) if repeat_text else 1
        if budget.root_k <= 0 or budget.samples <= 0 or repeat <= 0:
            raise ValueError("alpha root_k, samples, and repeat must be positive")
        budgets.extend([budget] * repeat)
    return budgets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="run")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--victory-target", type=int, default=7)
    p.add_argument("--vp-delta", type=float, default=0.05)
    p.add_argument("--potential-scale", type=float, default=0.0)
    p.add_argument("--visibility", default="perfect")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--policy-head-only",
        action="store_true",
        help="Freeze the trunk and value head; optimize only policy parameters.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=8)
    p.add_argument("--eval-episodes", type=int, default=96)
    p.add_argument("--final-eval-episodes", type=int, default=192)
    p.add_argument(
        "--eval-opponents",
        default="random,heuristic",
        help="Comma-separated engine opponents: random, heuristic, heuristic_v2, alpha.",
    )
    p.add_argument("--metrics", default=None, help="metrics JSONL path override")
    p.add_argument("--resume", default=None, help="checkpoint to warm-start from")
    p.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Warm-start model weights without restoring optimizer state.",
    )
    p.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    p.add_argument(
        "--reference-kl-coef",
        type=float,
        default=0.0,
        help="KL penalty toward the frozen warm-start policy.",
    )
    p.add_argument("--vp-delta-final", type=float, default=None,
                   help="anneal shaped reward toward this over 3 run phases")
    p.add_argument("--train-seats", default=None,
                   help="comma list, e.g. policy,heuristic,policy,heuristic — "
                        "mixed-opponent training (default: all policy self-play)")
    p.add_argument(
        "--rotate-train-seat",
        action="store_true",
        help="Rotate a single policy seat across all four positions each update.",
    )
    p.add_argument(
        "--train-opening-heuristic",
        action="store_true",
        help=(
            "Auto-play learner initial settlements with Heuristic-v2 so PPO "
            "trains on the same continuation states as OpeningHybridPolicy."
        ),
    )
    p.add_argument("--alpha-net", default="models/catan-512.ctnn")
    p.add_argument("--train-alpha-root-k", type=int, default=8)
    p.add_argument("--train-alpha-samples", type=int, default=96)
    p.add_argument("--train-alpha-depth", type=int, default=300)
    p.add_argument(
        "--alpha-budget-cycle",
        default=None,
        help=(
            "Per-update Alpha budgets as root_k:samples:depth[xrepeat], "
            "for example 1:1:0x8,2:4:0x2,8:96:300."
        ),
    )
    return p.parse_args()


def rotate_single_policy_seat(
    train_seats: list[str] | None,
    rotation: int,
) -> list[str] | None:
    """Rotate one learner seat while preserving the opponent lineup."""
    if train_seats is None:
        return None
    if train_seats.count("policy") != 1:
        raise ValueError("--rotate-train-seat requires exactly one policy seat")
    shift = rotation % len(train_seats)
    if shift == 0:
        return train_seats.copy()
    return train_seats[-shift:] + train_seats[:-shift]


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.trunk(obs)
        logits = self.policy(h)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits, self.value(h).squeeze(-1)


def configure_trainable_parameters(
    net: PolicyValueNet,
    policy_head_only: bool,
) -> list[nn.Parameter]:
    if policy_head_only:
        for parameter in net.trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in net.value.parameters():
            parameter.requires_grad_(False)
    return [parameter for parameter in net.parameters() if parameter.requires_grad]


class Metrics:
    def __init__(self, path: Path, header: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", buffering=1)
        self.emit({"t": "run", **header})

    def emit(self, event: dict):
        event["unix_ms"] = int(time.time() * 1000)
        self.f.write(json.dumps(event) + "\n")


def save_checkpoint(run_dir: Path, net, optimizer, step: int, config: dict):
    ck_dir = run_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": step,
        "config": config,
        "codec_version": catan_py.CODEC_VERSION,
        "num_actions": catan_py.NUM_ACTIONS,
        "obs_version": catan_py.OBS_VERSION,
        "obs_dim": catan_py.OBS_DIM,
        "engine_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    path = ck_dir / f"step_{step:010d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic
    latest = ck_dir / "latest.pt"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(path.name)
    return path


def make_train_env(args, train_seats, vp_delta: float, budget: AlphaBudget, seed: int):
    return catan_py.VecEnv(
        args.num_envs,
        victory_target=args.victory_target,
        visibility=args.visibility,
        vp_delta=vp_delta,
        potential_scale=args.potential_scale,
        seed=seed,
        seats=train_seats,
        alpha_net=args.alpha_net if train_seats and "alpha" in train_seats else None,
        alpha_root_k=budget.root_k,
        alpha_samples=budget.samples,
        alpha_depth=budget.depth,
        policy_opening_heuristic=args.train_opening_heuristic,
    )


@torch.no_grad()
def evaluate_vs(net, device, opponent: str, episodes: int, seed: int, args) -> dict:
    """Greedy policy against three engine bots, paired across all seats."""
    wins = done_eps = vp_sum = caps = 0
    per_seat = max(1, episodes // 4)
    for candidate_seat in range(4):
        seats_config = [opponent] * 4
        seats_config[candidate_seat] = "policy"
        env = catan_py.VecEnv(
            min(24, per_seat),
            victory_target=args.victory_target,
            visibility=args.visibility,
            seed=seed,
            seats=seats_config,
            alpha_net=args.alpha_net if opponent == "alpha" else None,
            policy_opening_heuristic=args.train_opening_heuristic,
        )
        obs, masks, seats = env.observe()
        seat_done = 0
        while seat_done < per_seat:
            o = torch.as_tensor(obs, device=device)
            m = torch.as_tensor(masks, device=device)
            logits, _ = net(o, m)
            actions = logits.argmax(dim=1).cpu().numpy().astype(np.uint32)
            obs, masks, seats, rewards, dones, terminals = env.step(actions)
            for turns, winner, vps, cap in env.take_episode_stats():
                if seat_done >= per_seat:
                    break
                seat_done += 1
                done_eps += 1
                wins += int(winner == candidate_seat)
                vp_sum += vps[candidate_seat]
                caps += int(cap)
    return {
        "win_rate": wins / done_eps,
        "games": done_eps,
        "avg_vp": vp_sum / max(1, done_eps),
        "cap_rate": caps / max(1, done_eps),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)

    run_dir = Path(__file__).parent / "runs" / f"{time.strftime('%Y%m%d-%H%M')}-{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    metrics_path = Path(args.metrics) if args.metrics else run_dir / "metrics.jsonl"
    seat_names = (args.train_seats.split(",") if args.train_seats else ["Policy"] * 4)
    metrics = Metrics(metrics_path, {"source": "ppo", "games": 0,
                                     "players": [s.capitalize() for s in seat_names],
                                     "seed": args.seed, "run_dir": str(run_dir)})

    train_seats = args.train_seats.split(",") if args.train_seats else None
    alpha_budgets = parse_alpha_budget_cycle(args.alpha_budget_cycle, args)
    current_budget = alpha_budgets[0]
    current_train_seats = (
        rotate_single_policy_seat(train_seats, 0)
        if args.rotate_train_seat
        else train_seats
    )
    env = make_train_env(
        args,
        current_train_seats,
        args.vp_delta,
        current_budget,
        args.seed,
    )
    net = PolicyValueNet(env.obs_dim, env.num_actions, args.hidden).to(device)
    trainable_parameters = configure_trainable_parameters(net, args.policy_head_only)
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, eps=1e-5)
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        assert ck["codec_version"] == catan_py.CODEC_VERSION, "codec version mismatch"
        assert ck["obs_version"] == catan_py.OBS_VERSION, "obs version mismatch"
        if ck.get("catanzero_version") == 1:
            net.trunk.load_state_dict(
                {
                    key.removeprefix("trunk."): value
                    for key, value in ck["model_state"].items()
                    if key.startswith("trunk.")
                }
            )
            net.policy.load_state_dict(
                {
                    key.removeprefix("policy."): value
                    for key, value in ck["model_state"].items()
                    if key.startswith("policy.")
                }
            )
            with torch.no_grad():
                net.value.weight.copy_(ck["model_state"]["outcome.weight"][0:1])
                net.value.bias.copy_(ck["model_state"]["outcome.bias"][0:1])
            print(f"warm-started PPO from CatanZero checkpoint {args.resume}")
        else:
            net.load_state_dict(ck["model_state"])
            if not args.policy_head_only and not args.reset_optimizer:
                optimizer.load_state_dict(ck["optimizer_state"])
            print(f"resumed from {args.resume} at step {ck['global_step']:,}")
    reference_net = copy.deepcopy(net).eval()
    for parameter in reference_net.parameters():
        parameter.requires_grad_(False)
    print(
        f"net params: {sum(p.numel() for p in net.parameters()):,} "
        f"({sum(p.numel() for p in trainable_parameters):,} trainable) | "
        f"device {device}"
    )
    print(f"run dir: {run_dir}\nmetrics: {metrics_path}")

    obs, masks, seats = env.observe()
    # Per-(env, seat) chains: pending transition awaiting reward/next value.
    pending = {}
    chains = defaultdict(list)  # (env, seat) -> completed transitions
    global_step = 0
    update = 0
    start_time = time.time()
    deadline = start_time + args.minutes * 60
    # Shaped-reward annealing: 3 equal phases interpolating vp_delta toward
    # --vp-delta-final (the honest objective takes over by the last phase).
    anneal = None
    if args.vp_delta_final is not None:
        anneal = [args.vp_delta,
                  (args.vp_delta + args.vp_delta_final) / 2,
                  args.vp_delta_final]
    current_phase = 0
    best_alpha = -1.0

    while time.time() < deadline:
        update += 1
        next_budget = alpha_budgets[(update - 1) % len(alpha_budgets)]
        next_train_seats = (
            rotate_single_policy_seat(train_seats, update - 1)
            if args.rotate_train_seat
            else train_seats
        )
        phase = current_phase
        if anneal:
            phase = min(int((time.time() - start_time) / (args.minutes * 60 / 3)), 2)
        if (
            phase != current_phase
            or next_budget != current_budget
            or next_train_seats != current_train_seats
        ):
            phase_changed = phase != current_phase
            current_phase = phase
            current_budget = next_budget
            current_train_seats = next_train_seats
            if phase_changed:
                print(f"=== anneal phase {phase}: vp_delta -> {anneal[phase]} ===")
                metrics.emit({"t": "run", "source": "ppo",
                              "players": [s.capitalize() for s in seat_names],
                              "seed": args.seed, "note": f"anneal vp_delta={anneal[phase]}"})
            active_vp_delta = anneal[phase] if anneal else args.vp_delta
            env = make_train_env(
                args,
                current_train_seats,
                active_vp_delta,
                current_budget,
                args.seed + update,
            )
            pending.clear()
            chains.clear()
            obs, masks, seats = env.observe()
        t0 = time.time()
        # ----------------------------------------------------- rollout
        with torch.no_grad():
            for _ in range(args.rollout):
                o = torch.as_tensor(obs, device=device)
                m = torch.as_tensor(masks, device=device)
                logits, values = net(o, m)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
                acts_np = acts.cpu().numpy().astype(np.uint32)
                values_np = values.cpu().numpy()
                logps_np = logps.cpu().numpy()

                # Open a transition per row; complete the seat's previous one
                # (its reward was assigned when this decision point arrived).
                for i in range(args.num_envs):
                    key = (i, int(seats[i]))
                    rec = {
                        "obs": obs[i].copy(), "mask": masks[i].copy(),
                        "action": int(acts_np[i]), "logp": float(logps_np[i]),
                        "value": float(values_np[i]),
                        "reward": 0.0, "done": False, "next_value": 0.0,
                    }
                    prev = pending.get(key)
                    if prev is not None and prev.get("reward_set"):
                        prev["next_value"] = rec["value"]
                        chains[key].append(prev)
                    pending[key] = rec

                obs, masks, seats, rewards, dones, terminals = env.step(acts_np)
                global_step += args.num_envs

                for i in range(args.num_envs):
                    if dones[i]:
                        for s in range(4):
                            prev = pending.pop((i, s), None)
                            if prev is not None:
                                prev["reward"] = float(terminals[i, s])
                                prev["reward_set"] = True
                                prev["done"] = True
                                prev["next_value"] = 0.0
                                chains[(i, s)].append(prev)
                    else:
                        key = (i, int(seats[i]))
                        prev = pending.get(key)
                        if prev is not None:
                            prev["reward"] = float(rewards[i])
                            prev["reward_set"] = True

                for turns, winner, vps, cap in env.take_episode_stats():
                    metrics.emit({"t": "game", "i": 0, "winner": int(winner),
                                  "turns": int(turns), "steps": 0,
                                  "vp": [int(v) for v in vps], "cap": bool(cap)})

        # ------------------------------------------- GAE per chain + flush
        batch = {k: [] for k in ("obs", "mask", "action", "logp", "value", "adv", "ret")}
        for key, chain in chains.items():
            if not chain:
                continue
            boot = pending.get(key)
            next_adv = 0.0
            for idx in range(len(chain) - 1, -1, -1):
                tr = chain[idx]
                nv = tr["next_value"]
                if idx == len(chain) - 1 and not tr["done"] and boot is not None:
                    nv = boot["value"]  # bootstrap from the seat's open decision
                nonterminal = 0.0 if tr["done"] else 1.0
                delta = tr["reward"] + GAMMA * nv * nonterminal - tr["value"]
                adv = delta + GAMMA * GAE_LAMBDA * nonterminal * next_adv
                next_adv = adv
                batch["obs"].append(tr["obs"]); batch["mask"].append(tr["mask"])
                batch["action"].append(tr["action"]); batch["logp"].append(tr["logp"])
                batch["value"].append(tr["value"]); batch["adv"].append(adv)
                batch["ret"].append(adv + tr["value"])
            chain.clear()

        n = len(batch["action"])
        if n == 0:
            continue
        b_obs = torch.as_tensor(np.array(batch["obs"]), device=device)
        b_mask = torch.as_tensor(np.array(batch["mask"]), device=device)
        b_act = torch.as_tensor(batch["action"], device=device)
        b_logp = torch.as_tensor(batch["logp"], device=device)
        b_val = torch.as_tensor(batch["value"], device=device)
        b_adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # --------------------------------------------------- PPO update
        clip_fracs, entropies, reference_kls = [], [], []
        idx = np.arange(n)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, args.minibatch):
                mb = idx[start:start + args.minibatch]
                logits, values = net(b_obs[mb], b_mask[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                pg1 = -b_adv[mb] * ratio
                pg2 = -b_adv[mb] * ratio.clamp(1 - CLIP, 1 + CLIP)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = 0.5 * (values - b_ret[mb]).pow(2).mean()
                entropy = dist.entropy().mean()
                with torch.no_grad():
                    reference_logits, _ = reference_net(b_obs[mb], b_mask[mb])
                    reference_log_probs = torch.log_softmax(reference_logits, dim=1)
                    reference_probs = reference_log_probs.exp()
                new_log_probs = torch.log_softmax(logits, dim=1)
                log_ratio = torch.nan_to_num(
                    reference_log_probs - new_log_probs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                reference_kl = (reference_probs * log_ratio).sum(dim=1).mean()
                loss = (
                    policy_loss
                    + VALUE_COEF * value_loss
                    - args.entropy_coef * entropy
                    + args.reference_kl_coef * reference_kl
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                clip_fracs.append(((ratio - 1).abs() > CLIP).float().mean().item())
                entropies.append(entropy.item())
                reference_kls.append(reference_kl.item())

        var_ret = float(b_ret.var())
        explained = 1.0 - float((b_ret - b_val).var()) / (var_ret + 1e-8)
        sps = args.rollout * args.num_envs / (time.time() - t0)
        metrics.emit({
            "t": "train", "step": global_step, "entropy": float(np.mean(entropies)),
            "explained_variance": explained, "clip_frac": float(np.mean(clip_fracs)),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()), "lr": args.lr, "sps": sps,
            "reference_kl": float(np.mean(reference_kls)),
            "alpha_root_k": current_budget.root_k,
            "alpha_samples": current_budget.samples,
            "alpha_depth": current_budget.depth,
        })
        print(f"update {update} | step {global_step:,} | {sps:,.0f} sps | "
              f"ent {np.mean(entropies):.2f} | ev {explained:.2f} | "
              f"clip {np.mean(clip_fracs):.2f} | batch {n:,} | "
              f"kl {np.mean(reference_kls):.3f} | "
              f"alpha {current_budget.root_k}x{current_budget.samples}d{current_budget.depth} | "
              f"seats {','.join(current_train_seats or ['self-play'])}")

        if update % args.eval_every == 0:
            for opponent in args.eval_opponents.split(","):
                opponent = opponent.strip()
                result = evaluate_vs(net, device, opponent, episodes=args.eval_episodes,
                                     seed=10_000 + update, args=args)
                metrics.emit({"t": "eval", "step": global_step, "vs": opponent, **result})
                print(f"  eval vs 3 {opponent}: {result['win_rate']*100:.1f}% "
                      f"({result['games']} games, baseline 25%)")
                if opponent == "alpha" and result["win_rate"] > best_alpha:
                    best_alpha = result["win_rate"]
                    path = save_checkpoint(run_dir, net, optimizer, global_step, config)
                    shutil.copy2(path, run_dir / "best_alpha.pt")
            save_checkpoint(run_dir, net, optimizer, global_step, config)

    save_checkpoint(run_dir, net, optimizer, global_step, config)
    for opponent in args.eval_opponents.split(","):
        opponent = opponent.strip()
        result = evaluate_vs(
            net,
            device,
            opponent,
            episodes=args.final_eval_episodes,
            seed=999,
            args=args,
        )
        metrics.emit({"t": "eval", "step": global_step, "vs": opponent, **result})
        print(f"final eval vs 3 {opponent}: {result['win_rate']*100:.1f}% | "
              f"avg VP {result['avg_vp']:.1f} | done at step {global_step:,}")


if __name__ == "__main__":
    main()
