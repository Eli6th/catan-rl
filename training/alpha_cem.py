"""Cross-entropy policy improvement against fixed engine opponents."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import catan_py
try:
    from .catanzero import (
        CATANZERO_VERSION,
        CatanZeroNet,
        evaluate_match,
        load_catanzero,
    )
except ImportError:
    from catanzero import (
        CATANZERO_VERSION,
        CatanZeroNet,
        evaluate_match,
        load_catanzero,
    )


def collect_trajectories(
    model,
    *,
    games: int,
    candidate_seat: int,
    num_envs: int,
    temperature: float,
    epsilon: float,
    seed: int,
    opponent: str,
    alpha_net: str,
):
    seats = [opponent] * 4
    seats[candidate_seat] = "policy"
    env_count = min(num_envs, games)
    env = catan_py.VecEnv(
        env_count,
        victory_target=7,
        visibility="realistic",
        zero_sum=True,
        seed=seed,
        seats=seats,
        alpha_net=alpha_net if opponent == "alpha" else None,
    )
    obs, masks, _ = env.observe()
    active = [[] for _ in range(env_count)]
    completed = []
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        while len(completed) < games:
            o = torch.as_tensor(obs, dtype=torch.float32)
            m = torch.as_tensor(masks, dtype=torch.bool)
            logits = model(o, m)["logits"] / temperature
            policy = torch.softmax(logits, dim=1).cpu().numpy()
            actions = np.zeros(env_count, dtype=np.uint32)
            for index in range(env_count):
                legal = masks[index].astype(np.float64)
                uniform = legal / legal.sum()
                mixed = (1.0 - epsilon) * policy[index] + epsilon * uniform
                mixed *= legal
                mixed /= mixed.sum()
                action = int(rng.choice(catan_py.NUM_ACTIONS, p=mixed))
                actions[index] = action
                active[index].append(
                    {
                        "obs": obs[index].copy(),
                        "mask": masks[index].copy(),
                        "action": action,
                        "opening": len(active[index]) < 4,
                    }
                )

            obs, masks, _, _, dones, terminals = env.step(actions)
            for index, done in enumerate(dones):
                if not done:
                    continue
                if len(completed) < games:
                    relative_outcome = np.asarray(
                        [
                            terminals[index, (candidate_seat + relative) % 4]
                            for relative in range(4)
                        ],
                        dtype=np.float32,
                    )
                    for sample in active[index]:
                        sample["outcome"] = relative_outcome
                    completed.append(
                        {
                            "samples": active[index],
                            "win": bool(terminals[index, candidate_seat] > 0),
                        }
                    )
                active[index] = []
    return completed


def fit_elites(
    model,
    trajectories,
    *,
    learning_rate: float,
    epochs: int,
    minibatch_size: int,
    elite_mix: float,
    opening_weight: float,
    type_head_only: bool,
):
    elites = [trajectory for trajectory in trajectories if trajectory["win"]]
    if not elites:
        return {"elite_games": 0, "samples": 0}
    samples = [sample for trajectory in elites for sample in trajectory["samples"]]
    reference = copy.deepcopy(model).eval()
    trainable_parameters = (
        list(model.policy_type.parameters())
        if type_head_only
        else list(model.parameters())
    )
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    totals = {"loss": 0.0, "elite_loss": 0.0, "kl_loss": 0.0}
    updates = 0

    for _ in range(epochs):
        order = torch.randperm(len(samples))
        for indexes in order.split(minibatch_size):
            batch = [samples[int(index)] for index in indexes]
            obs = torch.as_tensor(np.stack([sample["obs"] for sample in batch]))
            mask = torch.as_tensor(np.stack([sample["mask"] for sample in batch]))
            actions = torch.as_tensor(
                [sample["action"] for sample in batch], dtype=torch.int64
            )
            weights = torch.as_tensor(
                [
                    opening_weight if sample["opening"] else 1.0
                    for sample in batch
                ],
                dtype=torch.float32,
            )
            logits = model(obs, mask)["logits"]
            log_policy = torch.log_softmax(logits, dim=1)
            with torch.no_grad():
                reference_policy = torch.softmax(reference(obs, mask)["logits"], dim=1)
            elite_loss = -(
                log_policy.gather(1, actions.unsqueeze(1)).squeeze(1) * weights
            ).sum() / weights.sum()
            kl_terms = (
                reference_policy
                * (torch.log(reference_policy.clamp_min(1e-12)) - log_policy)
            )
            kl_loss = kl_terms.masked_fill(~mask, 0.0).sum(dim=1).mean()
            loss = elite_mix * elite_loss + (1.0 - elite_mix) * kl_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, 0.5)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["elite_loss"] += float(elite_loss.item())
            totals["kl_loss"] += float(kl_loss.item())
            updates += 1
    return {
        "elite_games": len(elites),
        "samples": len(samples),
        **{key: value / updates for key, value in totals.items()},
    }


def fit_outcome_head(
    model,
    trajectories,
    *,
    learning_rate: float,
    epochs: int,
    minibatch_size: int,
):
    """Calibrate only the outcome head from complete zero-sum games."""
    samples = [sample for trajectory in trajectories for sample in trajectory["samples"]]
    if not samples:
        return {"samples": 0}
    optimizer = torch.optim.Adam(model.outcome.parameters(), lr=learning_rate)
    totals = {"outcome_loss": 0.0}
    updates = 0

    for _ in range(epochs):
        order = torch.randperm(len(samples))
        for indexes in order.split(minibatch_size):
            batch = [samples[int(index)] for index in indexes]
            obs = torch.as_tensor(np.stack([sample["obs"] for sample in batch]))
            outcome = torch.as_tensor(
                np.stack([sample["outcome"] for sample in batch])
            )
            loss = (model(obs)["outcome"] - outcome).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.outcome.parameters(), 0.5)
            optimizer.step()
            totals["outcome_loss"] += float(loss.item())
            updates += 1
    return {
        "samples": len(samples),
        **{key: value / updates for key, value in totals.items()},
    }


def save_candidate(path: Path, source_payload, model, iteration: int, config: dict):
    payload = dict(source_payload)
    payload["catanzero_version"] = CATANZERO_VERSION
    payload["model_state"] = model.state_dict()
    payload["optimizer_state"] = {}
    payload["games"] = source_payload.get("games", 0)
    payload["cem_iteration"] = iteration
    payload["cem_config"] = config
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--elite-mix", type=float, default=0.20)
    parser.add_argument("--opening-weight", type=float, default=3.0)
    fit_mode = parser.add_mutually_exclusive_group()
    fit_mode.add_argument("--type-head-only", action="store_true")
    fit_mode.add_argument("--outcome-head-only", action="store_true")
    parser.add_argument("--eval-search-simulations", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=48)
    parser.add_argument("--seed", type=int, default=16_000_000)
    parser.add_argument(
        "--opponent",
        choices=("alpha", "heuristic", "heuristic_v2"),
        default="alpha",
    )
    parser.add_argument("--alpha-net", default="models/catan-512.ctnn")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    model = load_catanzero(args.checkpoint)
    source_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2))
    history = []
    best_rate = -1.0

    for iteration in range(1, args.iterations + 1):
        started = time.time()
        trajectories = []
        per_seat = args.games // 4
        for seat in range(4):
            trajectories.extend(
                collect_trajectories(
                    model,
                    games=per_seat,
                    candidate_seat=seat,
                    num_envs=args.num_envs,
                    temperature=args.temperature,
                    epsilon=args.epsilon,
                    seed=args.seed + iteration * 100_000 + seat * 10_000,
                    opponent=args.opponent,
                    alpha_net=args.alpha_net,
                )
            )
        if args.outcome_head_only:
            fit = fit_outcome_head(
                model,
                trajectories,
                learning_rate=args.lr,
                epochs=args.epochs,
                minibatch_size=args.minibatch_size,
            )
        else:
            fit = fit_elites(
                model,
                trajectories,
                learning_rate=args.lr,
                epochs=args.epochs,
                minibatch_size=args.minibatch_size,
                elite_mix=args.elite_mix,
                opening_weight=args.opening_weight,
                type_head_only=args.type_head_only,
            )
        checkpoint = args.run_dir / f"iteration_{iteration:02d}.pt"
        save_candidate(checkpoint, source_payload, model, iteration, config)
        evaluation = evaluate_match(
            model,
            None,
            args.opponent,
            args.eval_games,
            args.seed + iteration * 1_000_000,
            args.alpha_net if args.opponent == "alpha" else None,
            search_simulations=args.eval_search_simulations,
        )
        event = {
            "iteration": iteration,
            "collection_win_rate": sum(t["win"] for t in trajectories)
            / len(trajectories),
            "fit": fit,
            "evaluation": evaluation,
            "seconds": time.time() - started,
            "checkpoint": str(checkpoint),
        }
        history.append(event)
        (args.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(event, sort_keys=True), flush=True)
        if evaluation["win_rate"] > best_rate:
            best_rate = evaluation["win_rate"]
            save_candidate(
                args.run_dir / "best.pt",
                source_payload,
                model,
                iteration,
                config,
            )


if __name__ == "__main__":
    main()
