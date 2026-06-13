"""Laptop-scale hidden-information self-play with search distillation.

CatanZero keeps the existing fast Rust engine but changes the learning loop:

* official-information observations and re-determinized search states;
* low-budget multi-agent PUCT search as the policy-improvement operator;
* a checkpoint population plus scripted anchors;
* an EM-tuned mixture of learned policy prior and uniform exploration;
* auxiliary outcome, VP, progress-potential, and hidden-card belief heads.

The command is intentionally checkpoint-oriented:

    python training/catanzero.py train --minutes 10
    python training/catanzero.py evaluate <checkpoint.pt> --games 24
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import catan_py
try:
    from .ppo import PolicyValueNet
except ImportError:
    from ppo import PolicyValueNet


CATANZERO_VERSION = 2
SUPPORTED_CATANZERO_VERSIONS = {1, CATANZERO_VERSION}


class CatanZeroNet(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Linear(catan_py.OBS_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, catan_py.NUM_ACTIONS)
        self.policy_type = nn.Linear(hidden, len(catan_py.ACTION_TYPE_BOUNDARIES))
        nn.init.zeros_(self.policy_type.weight)
        nn.init.zeros_(self.policy_type.bias)
        boundaries = np.asarray(catan_py.ACTION_TYPE_BOUNDARIES)
        action_types = np.searchsorted(
            boundaries,
            np.arange(catan_py.NUM_ACTIONS),
            side="right",
        )
        self.register_buffer(
            "action_types",
            torch.as_tensor(action_types, dtype=torch.int64),
            persistent=False,
        )
        self.outcome = nn.Linear(hidden, 4)
        self.vp = nn.Linear(hidden, 4)
        self.progress = nn.Linear(hidden, 4)
        self.belief = nn.Linear(hidden, 30)

    def forward(self, obs, mask=None):
        h = self.trunk(obs)
        logits = self.policy(h) + self.policy_type(h)[:, self.action_types]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return {
            "logits": logits,
            "outcome": torch.tanh(self.outcome(h)),
            "vp": torch.sigmoid(self.vp(h)),
            "progress": torch.tanh(self.progress(h)),
            "belief": torch.sigmoid(self.belief(h)),
        }

    def effective_policy_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold the type head into flat action weights for Rust inference."""
        return (
            self.policy.weight + self.policy_type.weight[self.action_types],
            self.policy.bias + self.policy_type.bias[self.action_types],
        )


def load_catanzero_state(
    net: CatanZeroNet,
    model_state: dict[str, torch.Tensor],
) -> None:
    """Load flat v1 or factorized v2 weights without hiding incompatibilities."""
    incompatible = net.load_state_dict(model_state, strict=False)
    allowed_missing = {"policy_type.weight", "policy_type.bias"}
    invalid_missing = set(incompatible.missing_keys) - allowed_missing
    if invalid_missing or incompatible.unexpected_keys:
        raise ValueError(
            "incompatible CatanZero model state: "
            f"missing={sorted(invalid_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )


def warm_start_legacy(net: CatanZeroNet, path: Path) -> bool:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    legacy = PolicyValueNet(catan_py.OBS_DIM, catan_py.NUM_ACTIONS, ck["config"]["hidden"])
    legacy.load_state_dict(ck["model_state"])
    if ck["config"]["hidden"] != net.hidden:
        return False
    net.trunk.load_state_dict(legacy.trunk.state_dict())
    net.policy.load_state_dict(legacy.policy.state_dict())
    return True


def warm_start_catanzero(net: CatanZeroNet, path: Path) -> bool:
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("catanzero_version") not in SUPPORTED_CATANZERO_VERSIONS:
        return False
    if checkpoint["config"]["hidden"] != net.hidden:
        return False
    load_catanzero_state(net, checkpoint["model_state"])
    return True


def load_catanzero(path: Path) -> CatanZeroNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("catanzero_version") not in SUPPORTED_CATANZERO_VERSIONS:
        raise ValueError(
            "incompatible CatanZero checkpoint version: "
            f"{ck.get('catanzero_version')}"
        )
    expected = {
        "codec_version": catan_py.CODEC_VERSION,
        "obs_version": catan_py.OBS_VERSION,
        "obs_dim": catan_py.OBS_DIM,
        "num_actions": catan_py.NUM_ACTIONS,
    }
    mismatches = {
        key: (ck.get(key), value)
        for key, value in expected.items()
        if ck.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible CatanZero checkpoint: {mismatches}")
    net = CatanZeroNet(ck["config"]["hidden"])
    load_catanzero_state(net, ck["model_state"])
    net.eval()
    return net


def load_legacy(path: Path) -> PolicyValueNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(catan_py.OBS_DIM, catan_py.NUM_ACTIONS, ck["config"]["hidden"])
    net.load_state_dict(ck["model_state"])
    net.eval()
    return net


@torch.no_grad()
def policy_logits(model, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    m = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    result = model(o, m)
    logits = result["logits"] if isinstance(result, dict) else result[0]
    return logits[0].cpu().numpy()


@torch.no_grad()
def greedy_action(model, obs: np.ndarray, mask: np.ndarray) -> int:
    return int(np.argmax(policy_logits(model, obs, mask)))


class ExplorationEM:
    """EM for a policy-prior/uniform mixture.

    The MCTS visit target acts as weighted observations. The E-step computes
    which component better explains each searched action; the M-step updates
    the policy-component probability. Clamps retain deliberate exploration.
    """

    def __init__(self, policy_mix: float = 0.75, max_policy_mix: float = 0.9):
        self.policy_mix = policy_mix
        self.max_policy_mix = max_policy_mix

    def blend(self, policy: np.ndarray, legal: np.ndarray) -> np.ndarray:
        uniform = legal.astype(np.float64)
        uniform /= uniform.sum()
        mixed = self.policy_mix * policy + (1.0 - self.policy_mix) * uniform
        mixed *= legal
        return mixed / mixed.sum()

    def update(self, policy: np.ndarray, search: np.ndarray, legal: np.ndarray) -> float:
        uniform = legal.astype(np.float64)
        uniform /= uniform.sum()
        denom = self.policy_mix * policy + (1.0 - self.policy_mix) * uniform + 1e-12
        responsibility = self.policy_mix * policy / denom
        weight = float((search * responsibility).sum() / max(search.sum(), 1e-12))
        self.policy_mix = float(
            np.clip(
                0.98 * self.policy_mix + 0.02 * weight,
                0.1,
                self.max_policy_mix,
            )
        )
        return self.policy_mix


class Node:
    def __init__(self):
        self.prior = np.zeros(catan_py.NUM_ACTIONS, dtype=np.float64)
        self.visits = np.zeros(catan_py.NUM_ACTIONS, dtype=np.int32)
        self.value_sum = np.zeros((catan_py.NUM_ACTIONS, 4), dtype=np.float64)
        self.children: dict[int, Node] = {}
        self.expanded = False


def relative_to_absolute(values: np.ndarray, actor: int, players: int = 4) -> np.ndarray:
    out = np.zeros(4, dtype=np.float64)
    for rel in range(players):
        out[(actor + rel) % players] = float(values[rel])
    return out


@torch.no_grad()
def evaluate_leaf(model: CatanZeroNet, state, shaping_weight: float):
    obs = np.asarray(state.obs(), dtype=np.float32)
    mask = np.asarray(state.mask(), dtype=bool)
    actor = state.current_seat()
    o = torch.as_tensor(obs).unsqueeze(0)
    m = torch.as_tensor(mask).unsqueeze(0)
    result = model(o, m)
    logits = result["logits"][0].cpu().numpy()
    policy = np.zeros(catan_py.NUM_ACTIONS, dtype=np.float64)
    legal_logits = logits[mask]
    legal_logits = legal_logits - legal_logits.max()
    policy[mask] = np.exp(legal_logits)
    policy /= policy.sum()
    relative = (
        result["outcome"][0].cpu().numpy()
        + shaping_weight * result["progress"][0].cpu().numpy()
    )
    return obs, mask, policy, relative_to_absolute(relative, actor), actor


def run_mcts(
    root_state,
    model: CatanZeroNet,
    simulations: int,
    cpuct: float,
    explorer: ExplorationEM,
    shaping_weight: float,
    seed: int,
    gumbel_scale: float = 0.0,
    root_candidates: int = 8,
):
    if gumbel_scale > 0.0:
        return run_gumbel_mcts(
            root_state,
            model,
            simulations,
            cpuct,
            explorer,
            shaping_weight,
            seed,
            gumbel_scale,
            root_candidates,
        )
    root = Node()
    root_actor = root_state.current_seat()
    root_policy = None
    root_mask = None

    for simulation in range(max(2, simulations)):
        state = root_state.copy()
        state.redeterminize(root_actor, seed + simulation * 104729)
        node = root
        path = []

        while True:
            if state.is_done():
                value = np.asarray(state.outcome(), dtype=np.float64)
                break

            obs, mask, raw_policy, leaf_value, actor = evaluate_leaf(
                model, state, shaping_weight
            )
            if node is root and root_policy is None:
                root_policy = raw_policy.copy()
                root_mask = mask.copy()

            legal = np.flatnonzero(mask)
            if not node.expanded:
                node.prior = explorer.blend(raw_policy, mask)
                node.expanded = True
                value = leaf_value
                break

            total = max(1, int(node.visits.sum()))
            ranked_actions = []
            for action in legal:
                q = (
                    node.value_sum[action, actor] / node.visits[action]
                    if node.visits[action]
                    else 0.0
                )
                u = cpuct * node.prior[action] * math.sqrt(total) / (
                    1 + node.visits[action]
                )
                ranked_actions.append((q + u, int(action)))
            ranked_actions.sort(reverse=True)
            best_action = None
            for _, candidate_action in ranked_actions:
                try:
                    state.step(candidate_action)
                    best_action = candidate_action
                    break
                except ValueError:
                    # A child legal in one hidden-state determinization can be
                    # impossible in another. Do not let that stale child abort
                    # the information-set search.
                    node.prior[candidate_action] = 0.0
            if best_action is None:
                value = leaf_value
                break
            path.append((node, best_action))
            node = node.children.setdefault(best_action, Node())

        for visited, action in reversed(path):
            visited.visits[action] += 1
            visited.value_sum[action] += value

    assert root_policy is not None and root_mask is not None
    visits = root.visits.astype(np.float64)
    if visits.sum() == 0:
        visits = explorer.blend(root_policy, root_mask)
    else:
        visits /= visits.sum()
    return visits, root_policy, root_mask


def simulate_gumbel_candidate(
    root_state,
    root: Node,
    root_action: int,
    root_actor: int,
    model: CatanZeroNet,
    cpuct: float,
    explorer: ExplorationEM,
    shaping_weight: float,
    seed: int,
) -> bool:
    state = root_state.copy()
    state.redeterminize(root_actor, seed)
    try:
        state.step(root_action)
    except ValueError:
        return False
    node = root.children.setdefault(root_action, Node())
    path = [(root, root_action)]

    while True:
        if state.is_done():
            value = np.asarray(state.outcome(), dtype=np.float64)
            break
        _, mask, raw_policy, leaf_value, actor = evaluate_leaf(
            model, state, shaping_weight
        )
        legal = np.flatnonzero(mask)
        if not node.expanded:
            node.prior = explorer.blend(raw_policy, mask)
            node.expanded = True
            value = leaf_value
            break

        total = max(1, int(node.visits.sum()))
        ranked_actions = []
        for action in legal:
            q = (
                node.value_sum[action, actor] / node.visits[action]
                if node.visits[action]
                else 0.0
            )
            u = cpuct * node.prior[action] * math.sqrt(total) / (
                1 + node.visits[action]
            )
            ranked_actions.append((q + u, int(action)))
        ranked_actions.sort(reverse=True)
        selected = None
        for _, action in ranked_actions:
            try:
                state.step(action)
                selected = action
                break
            except ValueError:
                node.prior[action] = 0.0
        if selected is None:
            value = leaf_value
            break
        path.append((node, selected))
        node = node.children.setdefault(selected, Node())

    for visited, action in reversed(path):
        visited.visits[action] += 1
        visited.value_sum[action] += value
    return True


def action_type(action: int) -> int:
    """Return the semantic codec group containing an action id."""
    return int(np.searchsorted(catan_py.ACTION_TYPE_BOUNDARIES, action, side="right"))


def stratified_root_candidates(
    legal: np.ndarray,
    scores: np.ndarray,
    candidate_count: int,
) -> list[int]:
    """Select high-scoring legal roots while covering distinct move types."""
    ranked_indexes = sorted(
        range(len(legal)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )
    best_by_type: dict[int, int] = {}
    for index in ranked_indexes:
        best_by_type.setdefault(action_type(int(legal[index])), index)

    selected_indexes = sorted(
        best_by_type.values(),
        key=lambda index: float(scores[index]),
        reverse=True,
    )[:candidate_count]
    selected = set(selected_indexes)
    for index in ranked_indexes:
        if len(selected_indexes) >= candidate_count:
            break
        if index not in selected:
            selected_indexes.append(index)
            selected.add(index)
    return [int(legal[index]) for index in selected_indexes]


def run_gumbel_mcts(
    root_state,
    model: CatanZeroNet,
    simulations: int,
    cpuct: float,
    explorer: ExplorationEM,
    shaping_weight: float,
    seed: int,
    gumbel_scale: float,
    root_candidates: int,
):
    root = Node()
    root_actor = root_state.current_seat()
    _, root_mask, raw_policy, _, _ = evaluate_leaf(
        model, root_state, shaping_weight
    )
    root.prior = explorer.blend(raw_policy, root_mask)
    root.expanded = True

    legal = np.flatnonzero(root_mask)
    candidate_limit = min(len(legal), root_candidates)
    candidate_count = 1
    for count in range(candidate_limit, 1, -1):
        required = count * max(1, math.ceil(math.log2(count)))
        if required <= max(2, simulations):
            candidate_count = count
            break
    rng = np.random.default_rng(seed)
    gumbels = rng.gumbel(size=len(legal))
    prior_scores = np.log(root.prior[legal] + 1e-12) + gumbel_scale * gumbels
    candidates = stratified_root_candidates(legal, prior_scores, candidate_count)
    score_by_action = {
        int(action): float(score)
        for action, score in zip(legal, prior_scores)
    }
    root_scores = {action: score_by_action[action] for action in candidates}

    total_budget = max(simulations, candidate_count)
    spent = 0
    round_index = 0
    while len(candidates) > 1 and spent < total_budget:
        rounds_left = max(1, math.ceil(math.log2(len(candidates))))
        per_candidate = max(
            1,
            (total_budget - spent) // (len(candidates) * rounds_left),
        )
        for action in candidates:
            for repeat in range(per_candidate):
                if spent >= total_budget:
                    break
                simulation_seed = (
                    seed
                    + 104729 * (spent + 1)
                    + 8191 * (round_index + 1)
                    + 131 * (repeat + 1)
                )
                simulate_gumbel_candidate(
                    root_state,
                    root,
                    action,
                    root_actor,
                    model,
                    cpuct,
                    explorer,
                    shaping_weight,
                    simulation_seed,
                )
                spent += 1
        ranked = []
        for action in candidates:
            q = (
                root.value_sum[action, root_actor] / root.visits[action]
                if root.visits[action]
                else -1.0
            )
            ranked.append((root_scores[action] + q / max(cpuct, 0.1), action))
        ranked.sort(reverse=True)
        candidates = [action for _, action in ranked[: (len(ranked) + 1) // 2]]
        round_index += 1

    finalist = candidates[0]
    while spent < total_budget:
        simulate_gumbel_candidate(
            root_state,
            root,
            finalist,
            root_actor,
            model,
            cpuct,
            explorer,
            shaping_weight,
            seed + 104729 * (spent + 1),
        )
        spent += 1

    improved_policy = np.zeros(catan_py.NUM_ACTIONS, dtype=np.float64)
    improved_policy[finalist] = 1.0
    return improved_policy, raw_policy, root_mask


def sample_search_action(
    visits: np.ndarray, temperature: float, gumbel_scale: float, rng: np.random.Generator
) -> int:
    legal = visits > 0
    if temperature <= 0.05:
        return int(np.argmax(visits))
    logits = np.full_like(visits, -np.inf, dtype=np.float64)
    logits[legal] = np.log(visits[legal] + 1e-12) / temperature
    if gumbel_scale:
        logits[legal] += gumbel_scale * rng.gumbel(size=int(legal.sum()))
    logits[legal] -= logits[legal].max()
    probs = np.zeros_like(visits)
    probs[legal] = np.exp(logits[legal])
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


@dataclass
class Stage:
    name: str
    simulations: int
    shaping: float
    legacy_anchor: float
    auxiliary_scale: float
    cpuct: float
    temperature: float
    gumbel: float
    matchups: tuple[tuple[str, float], ...]
    root_candidates: int = 8


STAGES = (
    Stage(
        "bootstrap",
        4,
        0.25,
        0.50,
        0.10,
        1.5,
        1.0,
        0.35,
        (("heuristic", 0.65), ("self", 0.35)),
    ),
    Stage(
        "transition",
        8,
        0.10,
        0.25,
        0.50,
        1.25,
        0.65,
        0.20,
        (("heuristic_v2", 0.35), ("self", 0.35), ("historical", 0.30)),
    ),
    Stage(
        "league",
        16,
        0.0,
        0.10,
        1.0,
        1.0,
        0.30,
        0.05,
        (("heuristic_v2", 0.20), ("self", 0.35), ("historical", 0.45)),
    ),
)


def stages_for_profile(profile: str):
    if profile == "baseline":
        return STAGES
    if profile == "league_focus":
        return (
            STAGES[0],
            dataclass_replace(
                STAGES[1],
                matchups=(
                    ("heuristic_v2", 0.20),
                    ("self", 0.30),
                    ("historical", 0.50),
                ),
            ),
            dataclass_replace(
                STAGES[2],
                matchups=(
                    ("heuristic_v2", 0.10),
                    ("self", 0.25),
                    ("historical", 0.65),
                ),
            ),
        )
    if profile == "ppo_anchor":
        return (
            dataclass_replace(STAGES[0], legacy_anchor=0.65),
            dataclass_replace(
                STAGES[1],
                legacy_anchor=0.45,
                matchups=(
                    ("heuristic_v2", 0.20),
                    ("self", 0.30),
                    ("historical", 0.50),
                ),
            ),
            dataclass_replace(
                STAGES[2],
                legacy_anchor=0.30,
                matchups=(
                    ("heuristic_v2", 0.10),
                    ("self", 0.30),
                    ("historical", 0.60),
                ),
            ),
        )
    if profile == "league_anchor":
        return (
            dataclass_replace(STAGES[0], legacy_anchor=0.65),
            dataclass_replace(
                STAGES[1],
                legacy_anchor=0.45,
                matchups=(
                    ("heuristic_v2", 0.15),
                    ("self", 0.25),
                    ("historical", 0.60),
                ),
            ),
            dataclass_replace(
                STAGES[2],
                legacy_anchor=0.25,
                matchups=(
                    ("heuristic_v2", 0.05),
                    ("self", 0.20),
                    ("historical", 0.75),
                ),
            ),
        )
    if profile == "more_search":
        return (
            dataclass_replace(STAGES[0], simulations=8, gumbel=0.25),
            dataclass_replace(STAGES[1], simulations=12, gumbel=0.15),
            dataclass_replace(STAGES[2], simulations=24, gumbel=0.05),
        )
    if profile == "policy_heavy":
        return (
            dataclass_replace(STAGES[0], auxiliary_scale=0.0),
            dataclass_replace(STAGES[1], auxiliary_scale=0.25),
            dataclass_replace(STAGES[2], auxiliary_scale=0.75),
        )
    if profile == "exploration":
        return (
            dataclass_replace(STAGES[0], gumbel=0.50),
            dataclass_replace(STAGES[1], gumbel=0.30),
            dataclass_replace(STAGES[2], gumbel=0.10),
        )
    if profile == "alpha_focus":
        return (
            dataclass_replace(
                STAGES[0],
                simulations=8,
                legacy_anchor=0.10,
                auxiliary_scale=0.25,
                gumbel=0.50,
                matchups=(("alpha", 0.70), ("self", 0.20), ("historical", 0.10)),
            ),
            dataclass_replace(
                STAGES[1],
                simulations=12,
                legacy_anchor=0.05,
                auxiliary_scale=0.50,
                gumbel=0.30,
                matchups=(("alpha", 0.80), ("self", 0.10), ("historical", 0.10)),
            ),
            dataclass_replace(
                STAGES[2],
                simulations=16,
                legacy_anchor=0.0,
                auxiliary_scale=0.75,
                gumbel=0.15,
                matchups=(("alpha", 0.90), ("self", 0.05), ("historical", 0.05)),
            ),
        )
    if profile == "alpha_pure":
        return (
            dataclass_replace(
                STAGES[0],
                simulations=4,
                legacy_anchor=0.05,
                auxiliary_scale=0.25,
                gumbel=0.60,
                matchups=(("alpha", 1.0),),
            ),
            dataclass_replace(
                STAGES[1],
                simulations=4,
                legacy_anchor=0.0,
                auxiliary_scale=0.50,
                gumbel=0.40,
                matchups=(("alpha", 1.0),),
            ),
            dataclass_replace(
                STAGES[2],
                simulations=8,
                legacy_anchor=0.0,
                auxiliary_scale=0.75,
                gumbel=0.20,
                matchups=(("alpha", 1.0),),
            ),
        )
    if profile == "heuristic_focus":
        return (
            dataclass_replace(
                STAGES[0],
                simulations=4,
                legacy_anchor=0.20,
                auxiliary_scale=0.10,
                gumbel=0.35,
                matchups=(("heuristic_v2", 0.80), ("self", 0.15), ("historical", 0.05)),
            ),
            dataclass_replace(
                STAGES[1],
                simulations=8,
                legacy_anchor=0.10,
                auxiliary_scale=0.35,
                gumbel=0.20,
                matchups=(("heuristic_v2", 0.75), ("self", 0.15), ("historical", 0.10)),
            ),
            dataclass_replace(
                STAGES[2],
                simulations=12,
                legacy_anchor=0.05,
                auxiliary_scale=0.75,
                gumbel=0.05,
                matchups=(("heuristic_v2", 0.65), ("self", 0.20), ("historical", 0.15)),
            ),
        )
    if profile == "heuristic_mix":
        return (
            dataclass_replace(
                STAGES[0],
                simulations=4,
                legacy_anchor=0.50,
                auxiliary_scale=0.10,
                gumbel=0.35,
                matchups=(
                    ("heuristic", 0.35),
                    ("heuristic_v2", 0.35),
                    ("self", 0.20),
                    ("historical", 0.10),
                ),
            ),
            dataclass_replace(
                STAGES[1],
                simulations=8,
                legacy_anchor=0.35,
                auxiliary_scale=0.35,
                gumbel=0.20,
                matchups=(
                    ("heuristic", 0.30),
                    ("heuristic_v2", 0.40),
                    ("self", 0.15),
                    ("historical", 0.15),
                ),
            ),
            dataclass_replace(
                STAGES[2],
                simulations=12,
                legacy_anchor=0.25,
                auxiliary_scale=0.75,
                gumbel=0.05,
                matchups=(
                    ("heuristic", 0.25),
                    ("heuristic_v2", 0.40),
                    ("self", 0.20),
                    ("historical", 0.15),
                ),
            ),
        )
    raise ValueError(f"unknown training profile: {profile}")


def dataclass_replace(instance, **changes):
    values = {
        field: getattr(instance, field)
        for field in instance.__dataclass_fields__
    }
    values.update(changes)
    return type(instance)(**values)


def stage_for_progress(stages, progress: float, curriculum_start: float = 0.0):
    adjusted = curriculum_start + (1.0 - curriculum_start) * progress
    return stages[min(2, int(min(0.999, adjusted) * 3))]


def choose_weighted(items, rng: random.Random):
    x = rng.random()
    total = 0.0
    for value, weight in items:
        total += weight
        if x <= total:
            return value
    return items[-1][0]


def select_policy_anchor(champion, legacy):
    """Anchor fine-tuning to the incumbent policy when one is available."""
    return champion if champion is not None else legacy


def rotate_absolute(values, actor: int):
    return np.asarray([values[(actor + rel) % 4] for rel in range(4)], dtype=np.float32)


def normalized_policy(model, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = policy_logits(model, obs, mask)
    policy = np.zeros(catan_py.NUM_ACTIONS, dtype=np.float32)
    legal_logits = logits[mask]
    legal_logits = legal_logits - legal_logits.max()
    policy[mask] = np.exp(legal_logits)
    policy /= policy.sum()
    return policy


def teacher_policy(teacher, snapshot, seed: int, determinizations: int) -> np.ndarray:
    policy = np.zeros(catan_py.NUM_ACTIONS, dtype=np.float32)
    for index in range(determinizations):
        action = teacher.action(snapshot, seed + index * 104729)
        policy[action] += 1.0
    policy /= policy.sum()
    return policy


def _bootstrap_chunk(
    worker: int,
    games: int,
    seed: int,
    legacy_path: str,
    alpha_net: str,
    root_k: int,
    alpha_samples: int,
    alpha_depth: int,
    determinizations: int,
    champion_path: str | None,
    student_path: str | None,
    trajectory_weights: tuple[tuple[str, float], ...],
):
    legacy = load_legacy(Path(legacy_path))
    champion = load_catanzero(Path(champion_path)) if champion_path else None
    student = load_catanzero(Path(student_path)) if student_path else None
    teacher = catan_py.AlphaTeacher(
        alpha_net,
        root_k=root_k,
        samples=alpha_samples,
        depth=alpha_depth,
        visibility="realistic",
    )
    samples = []
    wins = [0, 0, 0, 0]
    decisions = 0
    for game_index in range(games):
        game_seed = seed + worker * 1_000_003 + game_index
        py_rng = random.Random(game_seed)
        np_rng = np.random.default_rng(game_seed)
        trajectory = choose_weighted(trajectory_weights, py_rng)
        learner_seat = py_rng.randrange(4)
        seats = ["policy"] * 4
        if trajectory == "heuristic":
            seats = ["heuristic_v2"] * 4
            seats[learner_seat] = "policy"
        env = catan_py.Env(
            victory_target=7,
            visibility="realistic",
            zero_sum=True,
            seed=game_seed,
            seats=seats,
        )
        game_samples = []
        winner = -1
        for decision in range(5000):
            actor = env.current_seat()
            obs = np.asarray(env.obs(), dtype=np.float32)
            mask = np.asarray(env.mask(), dtype=bool)
            snapshot = env.snapshot()
            policy = teacher_policy(
                teacher,
                snapshot,
                game_seed * 10000 + decision * determinizations,
                determinizations,
            )
            legacy_policy = normalized_policy(legacy, obs, mask)
            game_samples.append(
                {
                    "obs": obs,
                    "mask": mask,
                    "policy": policy.astype(np.float32),
                    "legacy_policy": legacy_policy,
                    "legacy_anchor": 0.0,
                    "raw_policy": legacy_policy,
                    "legal": mask,
                    "actor": actor,
                    "trajectory": trajectory,
                    "trajectory_id": f"{worker}:{game_index}",
                    "progress": np.asarray(
                        snapshot.potential_values(), dtype=np.float32
                    ),
                    "belief": np.asarray(env.private_target(), dtype=np.float32),
                }
            )
            if trajectory == "teacher":
                action = int(np_rng.choice(len(policy), p=policy))
            elif trajectory == "student" and student is not None:
                action = greedy_action(student, obs, mask)
            elif trajectory == "champion" and champion is not None:
                action = greedy_action(champion, obs, mask)
            else:
                action = greedy_action(legacy, obs, mask)
            _, _, done, winner, _ = env.step(action)
            decisions += 1
            if done:
                break

        final_vp = np.asarray(env.final_vp(), dtype=np.float32)
        outcome = np.full(4, -1.0 / 3.0, dtype=np.float32)
        if winner >= 0:
            outcome[winner] = 1.0
            wins[winner] += 1
        else:
            outcome.fill(0.0)
        for sample in game_samples:
            sample["outcome"] = rotate_absolute(outcome, sample["actor"])
            sample["vp"] = rotate_absolute(final_vp, sample["actor"])
        samples.extend(game_samples)
    return samples, {
        "games": games,
        "decisions": decisions,
        "wins": wins,
    }


def generate_bootstrap_samples(
    args,
    run_dir: Path,
    *,
    dagger: bool = False,
    student_path: Path | None = None,
):
    dataset_path = run_dir / ("dagger_dataset.pt" if dagger else "bootstrap_v2_dataset.pt")
    if args.bootstrap_dataset and not dagger:
        dataset_path = Path(args.bootstrap_dataset)
    if dataset_path.exists():
        payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        if payload.get("bootstrap_version") != 2:
            raise ValueError(
                f"{dataset_path} uses the obsolete bootstrap format; regenerate it"
            )
        samples = limit_bootstrap_samples(
            payload["samples"], args.bootstrap_sample_limit, args.seed
        )
        print(f"loaded {len(samples)} bootstrap samples from {dataset_path}")
        return samples
    requested_games = args.dagger_games if dagger else args.bootstrap_games
    if requested_games <= 0:
        return []

    workers = min(args.bootstrap_workers, requested_games)
    counts = [
        requested_games // workers + int(i < requested_games % workers)
        for i in range(workers)
    ]
    trajectory_weights = (
        (("student", 0.70), ("champion", 0.15), ("heuristic", 0.15))
        if dagger
        else (
            ("teacher", 0.35),
            ("champion", 0.35),
            ("heuristic", 0.15),
            ("legacy", 0.15),
        )
    )
    started = time.time()
    all_samples = []
    summaries = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _bootstrap_chunk,
                worker,
                games,
                args.seed + 500_000,
                args.legacy,
                args.alpha_net,
                args.teacher_root_k,
                args.teacher_samples,
                args.teacher_depth,
                args.teacher_determinizations,
                args.champion if Path(args.champion).exists() else None,
                str(student_path) if student_path else None,
                trajectory_weights,
            )
            for worker, games in enumerate(counts)
            if games
        ]
        for future in as_completed(futures):
            samples, summary = future.result()
            prefix = "dagger" if dagger else "bootstrap"
            for sample in samples:
                sample["trajectory_id"] = f"{prefix}:{sample['trajectory_id']}"
            all_samples.extend(samples)
            summaries.append(summary)
            print(
                f"bootstrap actor finished {summary['games']} games, "
                f"{summary['decisions']} decisions"
            )
    payload = {
        "bootstrap_version": 2,
        "samples": all_samples,
        "config": {
            "games": requested_games,
            "workers": workers,
            "root_k": args.teacher_root_k,
            "samples": args.teacher_samples,
            "depth": args.teacher_depth,
            "determinizations": args.teacher_determinizations,
            "trajectory_weights": trajectory_weights,
            "dagger": dagger,
        },
        "summaries": summaries,
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dataset_path)
    print(
        f"generated {len(all_samples)} bootstrap samples in "
        f"{time.time() - started:.1f}s -> {dataset_path}"
    )
    return all_samples


def limit_bootstrap_samples(samples, limit: int, seed: int):
    if limit <= 0 or len(samples) <= limit:
        return samples
    by_trajectory = {}
    for index, sample in enumerate(samples):
        trajectory_id = sample.get("trajectory_id", str(index))
        by_trajectory.setdefault(trajectory_id, []).append(sample)
    trajectory_ids = list(by_trajectory)
    random.Random(seed).shuffle(trajectory_ids)
    selected = []
    for trajectory_id in trajectory_ids:
        selected.extend(by_trajectory[trajectory_id])
        if len(selected) >= limit:
            break
    return selected


def play_training_game(
    model: CatanZeroNet,
    stage: Stage,
    explorer: ExplorationEM,
    population: list[nn.Module],
    legacy: nn.Module,
    champion: nn.Module | None,
    champion_sample_probability: float,
    alpha_net: str,
    vp_advantage_scale: float,
    seed: int,
):
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    matchup = choose_weighted(stage.matchups, py_rng)
    learner_seat = py_rng.randrange(4)

    if matchup in ("heuristic", "heuristic_v2", "alpha"):
        seats = [matchup] * 4
        seats[learner_seat] = "policy"
        controllers = {learner_seat: model}
    elif matchup == "historical" and population:
        seats = ["policy"] * 4
        controllers = {
            seat: (
                model
                if seat == learner_seat
                else (
                    champion
                    if champion is not None
                    and py_rng.random() < champion_sample_probability
                    else py_rng.choice(population)
                )
            )
            for seat in range(4)
        }
    else:
        matchup = "self"
        seats = ["policy"] * 4
        controllers = {seat: model for seat in range(4)}

    env = catan_py.Env(
        victory_target=7,
        visibility="realistic",
        zero_sum=True,
        potential_scale=0.2,
        seed=seed,
        seats=seats,
        alpha_net=alpha_net if matchup == "alpha" else None,
    )
    samples = []
    decisions = 0
    winner = -1

    while decisions < 5000:
        actor = env.current_seat()
        obs = np.asarray(env.obs(), dtype=np.float32)
        mask = np.asarray(env.mask(), dtype=bool)
        controller = controllers[actor]
        train_actor = controller is model

        if train_actor:
            snapshot = env.snapshot()
            progress = np.asarray(snapshot.potential_values(), dtype=np.float32)
            belief = np.asarray(env.private_target(), dtype=np.float32)
            visits, raw_policy, legal = run_mcts(
                snapshot,
                model,
                stage.simulations,
                stage.cpuct,
                explorer,
                stage.shaping,
                seed * 10000 + decisions,
                gumbel_scale=stage.gumbel,
                root_candidates=stage.root_candidates,
            )
            action = sample_search_action(
                visits, stage.temperature, 0.0, np_rng
            )
            samples.append(
                {
                    "obs": obs,
                    "mask": mask,
                    "policy": visits.astype(np.float32),
                    "legacy_policy": normalized_policy(legacy, obs, mask),
                    "legacy_anchor": stage.legacy_anchor,
                    "raw_policy": raw_policy.astype(np.float32),
                    "legal": legal,
                    "actor": actor,
                    "progress": progress,
                    "belief": belief,
                    "action": action,
                }
            )
        else:
            action = greedy_action(controller, obs, mask)

        _, _, done, winner, _ = env.step(action)
        decisions += 1
        if done:
            break

    final_vp = np.asarray(env.final_vp(), dtype=np.float32)
    outcome = np.zeros(4, dtype=np.float32)
    if winner >= 0:
        outcome.fill(-1.0 / 3.0)
        outcome[winner] = 1.0
    for sample in samples:
        sample["outcome"] = rotate_absolute(outcome, sample["actor"])
        sample["vp"] = rotate_absolute(final_vp, sample["actor"])
        relative_vp = sample["vp"]
        sample["policy_advantage"] = float(
            sample["outcome"][0]
            + vp_advantage_scale
            * (relative_vp[0] - float(relative_vp[1:].mean()))
        )
        fixed_opponent = matchup in ("alpha", "heuristic", "heuristic_v2")
        sample["reinforce_weight"] = float(fixed_opponent)
        sample["elite_weight"] = float(
            fixed_opponent and winner == learner_seat
        )
        explorer.update(sample["raw_policy"], sample["policy"], sample["legal"])
    return samples, {
        "matchup": matchup,
        "winner": int(winner),
        "learner_seat": learner_seat,
        "learner_win": int(winner == learner_seat),
        "decisions": decisions,
        "samples": len(samples),
        "exploration_mix": explorer.policy_mix,
    }


def play_training_batch(
    model: CatanZeroNet,
    stage: Stage,
    explorer: ExplorationEM,
    population: list[nn.Module],
    legacy: nn.Module,
    champion: nn.Module | None,
    champion_sample_probability: float,
    alpha_net: str,
    vp_advantage_scale: float,
    seed: int,
    games: int,
    workers: int,
):
    """Generate a round of games concurrently from one frozen learner."""
    frozen = copy.deepcopy(model).eval()
    worker_count = max(1, min(workers, games))
    jobs = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for offset in range(games):
            local_explorer = ExplorationEM(
                explorer.policy_mix,
                explorer.max_policy_mix,
            )
            jobs.append(
                executor.submit(
                    play_training_game,
                    frozen,
                    stage,
                    local_explorer,
                    population,
                    legacy,
                    champion,
                    champion_sample_probability,
                    alpha_net,
                    vp_advantage_scale,
                    seed + offset,
                )
            )
        results = [job.result() for job in jobs]

    total_samples = sum(stats["samples"] for _, stats in results)
    if total_samples:
        explorer.policy_mix = sum(
            stats["exploration_mix"] * stats["samples"] for _, stats in results
        ) / total_samples
    return results


def split_bootstrap_samples(samples, validation_fraction: float, seed: int):
    trajectories = sorted(
        {sample.get("trajectory_id", str(index)) for index, sample in enumerate(samples)}
    )
    rng = random.Random(seed)
    rng.shuffle(trajectories)
    if len(trajectories) < 2:
        raise ValueError("bootstrap validation requires at least two trajectories")
    validation_count = min(
        len(trajectories) - 1,
        max(1, int(len(trajectories) * validation_fraction)),
    )
    validation_ids = set(trajectories[:validation_count])
    train = [
        sample
        for index, sample in enumerate(samples)
        if sample.get("trajectory_id", str(index)) not in validation_ids
    ]
    validation = [
        sample
        for index, sample in enumerate(samples)
        if sample.get("trajectory_id", str(index)) in validation_ids
    ]
    return train, validation


def policy_metrics(model, samples, minibatch_size: int):
    model.eval()
    correct = 0
    cross_entropy = 0.0
    with torch.no_grad():
        for start in range(0, len(samples), minibatch_size):
            batch = samples[start : start + minibatch_size]
            obs = torch.as_tensor(np.stack([sample["obs"] for sample in batch]))
            mask = torch.as_tensor(np.stack([sample["mask"] for sample in batch]))
            target = torch.as_tensor(np.stack([sample["policy"] for sample in batch]))
            logits = model(obs)["logits"].masked_fill(~mask, -1e9)
            cross_entropy += float(
                (-(target * torch.log_softmax(logits, dim=1)).sum(dim=1)).sum().item()
            )
            correct += int((logits.argmax(dim=1) == target.argmax(dim=1)).sum().item())
    return {
        "agreement": correct / len(samples),
        "cross_entropy": cross_entropy / len(samples),
    }


def fit_policy_bootstrap(
    model,
    samples,
    *,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    validation_fraction: float,
    minibatch_size: int,
    seed: int,
):
    train_samples, validation_samples = split_bootstrap_samples(
        samples, validation_fraction, seed
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = policy_metrics(model, validation_samples, minibatch_size)
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_samples))
        for indexes in order.split(minibatch_size):
            batch = [train_samples[int(index)] for index in indexes]
            obs = torch.as_tensor(np.stack([sample["obs"] for sample in batch]))
            mask = torch.as_tensor(np.stack([sample["mask"] for sample in batch]))
            target = torch.as_tensor(np.stack([sample["policy"] for sample in batch]))
            logits = model(obs)["logits"].masked_fill(~mask, -1e9)
            loss = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        validation = policy_metrics(model, validation_samples, minibatch_size)
        history.append({"epoch": epoch, **validation})
        if validation["cross_entropy"] < best_validation["cross_entropy"] - 1e-4:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "best_epoch": best_epoch,
        "train": policy_metrics(model, train_samples, minibatch_size),
        "validation": best_validation,
        "history": history,
    }


def optimize(
    model,
    optimizer,
    samples,
    epochs: int = 3,
    minibatch_size: int = 2048,
    auxiliary_scale: float = 1.0,
    reinforce_scale: float = 0.0,
    entropy_scale: float = 0.0,
    search_policy_scale: float = 1.0,
    elite_scale: float = 0.0,
):
    if not samples:
        return {}
    obs = torch.as_tensor(np.stack([s["obs"] for s in samples]))
    mask = torch.as_tensor(np.stack([s["mask"] for s in samples]))
    policy = torch.as_tensor(np.stack([s["policy"] for s in samples]))
    legacy_policy = torch.as_tensor(np.stack([s["legacy_policy"] for s in samples]))
    legacy_anchor = torch.as_tensor(
        np.asarray([s["legacy_anchor"] for s in samples], dtype=np.float32)
    )
    outcome = torch.as_tensor(np.stack([s["outcome"] for s in samples]))
    vp = torch.as_tensor(np.stack([s["vp"] for s in samples]))
    progress = torch.as_tensor(np.stack([s["progress"] for s in samples]))
    belief = torch.as_tensor(np.stack([s["belief"] for s in samples]))
    actions = torch.as_tensor(
        np.asarray(
            [
                s.get("action", int(np.argmax(s["policy"])))
                for s in samples
            ],
            dtype=np.int64,
        )
    )
    advantages = torch.as_tensor(
        np.asarray(
            [s.get("policy_advantage", 0.0) for s in samples],
            dtype=np.float32,
        )
    )
    reinforce_weights = torch.as_tensor(
        np.asarray(
            [s.get("reinforce_weight", 0.0) for s in samples],
            dtype=np.float32,
        )
    )
    elite_weights = torch.as_tensor(
        np.asarray(
            [s.get("elite_weight", 0.0) for s in samples],
            dtype=np.float32,
        )
    )

    metric_totals = {}
    updates = 0
    for _ in range(epochs):
        for indexes in torch.randperm(len(samples)).split(minibatch_size):
            result = model(obs[indexes])
            masked_logits = result["logits"].masked_fill(~mask[indexes], -1e9)
            log_policy = torch.log_softmax(masked_logits, dim=1)
            probabilities = torch.softmax(masked_logits, dim=1)
            policy_loss = -(
                policy[indexes] * log_policy
            ).sum(dim=1).mean()
            legacy_loss = (
                -(legacy_policy[indexes] * log_policy).sum(dim=1)
                * legacy_anchor[indexes]
            ).mean()
            outcome_loss = (
                result["outcome"] - outcome[indexes]
            ).pow(2).mean()
            vp_loss = (result["vp"] - vp[indexes]).pow(2).mean()
            progress_loss = (
                result["progress"] - progress[indexes]
            ).pow(2).mean()
            belief_loss = (
                result["belief"] - belief[indexes]
            ).pow(2).mean()
            selected_log_policy = log_policy.gather(
                1, actions[indexes].unsqueeze(1)
            ).squeeze(1)
            policy_advantage = (
                advantages[indexes] - result["outcome"][:, 0].detach()
            )
            weights = reinforce_weights[indexes]
            if int((weights > 0).sum()) > 1:
                active = policy_advantage[weights > 0]
                policy_advantage = (
                    policy_advantage - active.mean()
                ) / (active.std(unbiased=False) + 1e-6)
            reinforce_loss = -(
                selected_log_policy * policy_advantage * weights
            ).sum() / weights.sum().clamp_min(1.0)
            elite = elite_weights[indexes]
            elite_loss = -(
                selected_log_policy * elite
            ).sum() / elite.sum().clamp_min(1.0)
            entropy = -(
                probabilities * log_policy
            ).sum(dim=1).mean()
            loss = (
                search_policy_scale * policy_loss
                + legacy_loss
                + reinforce_scale * reinforce_loss
                + elite_scale * elite_loss
                - entropy_scale * entropy
                + auxiliary_scale
                * (
                    outcome_loss
                    + 0.25 * vp_loss
                    + 0.25 * progress_loss
                    + 0.10 * belief_loss
                )
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metrics = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "legacy_loss": float(legacy_loss.item()),
                "outcome_loss": float(outcome_loss.item()),
                "vp_loss": float(vp_loss.item()),
                "progress_loss": float(progress_loss.item()),
                "belief_loss": float(belief_loss.item()),
                "reinforce_loss": float(reinforce_loss.item()),
                "elite_loss": float(elite_loss.item()),
                "entropy": float(entropy.item()),
            }
            for key, value in metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value
            updates += 1
    return {key: value / updates for key, value in metric_totals.items()}


def save_checkpoint(path: Path, model, optimizer, args, explorer, games, stage):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_args = {
        key: value
        for key, value in vars(args).items()
        if key not in {"func", "command"}
    }
    checkpoint = {
        "catanzero_version": CATANZERO_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {"hidden": args.hidden, **saved_args},
        "codec_version": catan_py.CODEC_VERSION,
        "obs_version": catan_py.OBS_VERSION,
        "obs_dim": catan_py.OBS_DIM,
        "num_actions": catan_py.NUM_ACTIONS,
        "games": games,
        "stage": stage.name,
        "exploration_mix": explorer.policy_mix,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def play_eval_game(
    candidate,
    opponent,
    seed: int,
    candidate_seat: int,
    opponent_kind: str,
    alpha_net: str | None = None,
    search_simulations: int = 0,
):
    return play_eval_lineup(
        candidate,
        opponent,
        seed,
        (candidate_seat,),
        opponent_kind,
        alpha_net,
        search_simulations,
    )


def play_eval_lineup(
    candidate,
    opponent,
    seed: int,
    candidate_seats,
    opponent_kind: str,
    alpha_net: str | None = None,
    search_simulations: int = 0,
):
    candidate_seats = set(candidate_seats)
    if opponent_kind in ("heuristic", "heuristic_v2", "alpha"):
        seats = [opponent_kind] * 4
        controllers = {seat: candidate for seat in candidate_seats}
        for seat in candidate_seats:
            seats[seat] = "policy"
    else:
        seats = ["policy"] * 4
        controllers = {
            seat: (candidate if seat in candidate_seats else opponent)
            for seat in range(4)
        }
    env = catan_py.Env(
        victory_target=7,
        visibility="realistic",
        seed=seed,
        seats=seats,
        alpha_net=alpha_net,
    )
    for decision in range(5000):
        seat = env.current_seat()
        obs = np.asarray(env.obs(), dtype=np.float32)
        mask = np.asarray(env.mask(), dtype=bool)
        if seat in candidate_seats and search_simulations:
            visits, _, _ = run_mcts(
                env.snapshot(),
                candidate,
                search_simulations,
                1.0,
                ExplorationEM(0.9),
                0.0,
                seed * 10000 + decision,
            )
            action = int(np.argmax(visits))
        else:
            action = greedy_action(controllers[seat], obs, mask)
        _, _, done, winner, _ = env.step(action)
        if done:
            return int(winner), np.asarray(env.final_vp())
    return -1, np.asarray(env.final_vp())


def evaluate_match(
    candidate,
    opponent,
    kind: str,
    games: int,
    seed: int,
    alpha_net=None,
    search_simulations: int = 0,
):
    wins = 0
    vp = 0.0
    for game in range(games):
        seat = game % 4
        paired_seed = seed + game // 4
        winner, final_vp = play_eval_game(
            candidate,
            opponent,
            paired_seed,
            seat,
            kind,
            alpha_net,
            search_simulations,
        )
        wins += int(winner == seat)
        vp += float(final_vp[seat] * 7.0)
    return {"games": games, "wins": wins, "win_rate": wins / games, "avg_vp": vp / games}


def evaluate_balanced_match(
    candidate,
    opponent,
    boards: int,
    seed: int,
    search_simulations: int = 0,
):
    wins = 0
    vp = 0.0
    games = 0
    for board in range(boards):
        for candidate_seats in itertools.combinations(range(4), 2):
            winner, final_vp = play_eval_lineup(
                candidate,
                opponent,
                seed + board,
                candidate_seats,
                "policy",
                search_simulations=search_simulations,
            )
            wins += int(winner in candidate_seats)
            vp += float(np.mean(final_vp[list(candidate_seats)]) * 7.0)
            games += 1
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "avg_vp": vp / games,
    }


def evaluate_suite(
    candidate,
    legacy,
    previous,
    champion,
    games: int,
    seed: int,
    alpha_net: str,
    alpha_games: int | None = None,
):
    alpha_count = alpha_games if alpha_games is not None else max(4, games // 2)
    results = {
        "heuristic_v1": evaluate_match(candidate, None, "heuristic", games, seed),
        "heuristic_v2": evaluate_match(candidate, None, "heuristic_v2", games, seed + 1000),
        "legacy_ppo": evaluate_match(candidate, legacy, "policy", games, seed + 2000),
        "alphabot": evaluate_match(
            candidate, None, "alpha", alpha_count, seed + 3000, alpha_net
        ),
        "heuristic_v2_search8": evaluate_match(
            candidate,
            None,
            "heuristic_v2",
            max(4, games // 2),
            seed + 3500,
            search_simulations=8,
        ),
    }
    if previous is not None:
        results["previous_catanzero"] = evaluate_match(
            candidate, previous, "policy", games, seed + 4000
        )
    if champion is not None:
        results["champion_catanzero"] = evaluate_match(
            candidate, champion, "policy", games, seed + 5000
        )
        results["champion_balanced"] = evaluate_balanced_match(
            candidate, champion, max(4, games // 2), seed + 5250
        )
        results["champion_search8"] = evaluate_match(
            candidate,
            champion,
            "policy",
            max(4, games // 2),
            seed + 5500,
            search_simulations=8,
        )
    return results


def benchmark_score(results) -> float:
    weights = {
        "heuristic_v1": 0.2,
        "heuristic_v2": 0.2,
        "legacy_ppo": 0.4,
        "alphabot": 0.2,
    }
    return sum(weights[name] * results[name]["win_rate"] for name in weights)


def selection_score(results, metric: str) -> float:
    if metric == "alphabot":
        return results["alphabot"]["win_rate"]
    return benchmark_score(results)


def champion_gate_passes(
    results, minimum_win_rate: float, balanced_minimum: float = 0.5
) -> bool:
    balanced = results.get("champion_balanced")
    if balanced is not None:
        return balanced["win_rate"] >= balanced_minimum
    champion = results.get("champion_catanzero")
    return champion is None or champion["win_rate"] >= minimum_win_rate


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    run_dir = Path(args.run_dir) if args.run_dir else Path("training/runs") / (
        time.strftime("%Y%m%d-%H%M") + "-catanzero"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = open(run_dir / "metrics.jsonl", "w", buffering=1)

    model = CatanZeroNet(args.hidden)
    champion_path = Path(args.champion)
    champion = load_catanzero(champion_path) if champion_path.exists() else None
    warm_started = warm_start_catanzero(model, champion_path)
    if not warm_started:
        warm_started = warm_start_legacy(model, Path(args.legacy))
    legacy = load_legacy(Path(args.legacy))
    policy_anchor = select_policy_anchor(champion, legacy)
    stages = stages_for_profile(args.training_profile)
    explorer = ExplorationEM(args.policy_mix, args.exploration_max)
    population: list[nn.Module] = [legacy]
    if champion is not None:
        population.append(champion)
    previous = None
    best_score = -float("inf")
    observed_score = -float("inf")
    games = 0
    pending_samples = []

    torch.set_num_threads(1)
    bootstrap_samples = generate_bootstrap_samples(args, run_dir)
    torch.set_num_threads(args.threads)
    if bootstrap_samples:
        print(
            f"policy-only pretraining {args.hidden}-wide student on "
            f"{len(bootstrap_samples)} teacher decisions"
        )
        bootstrap_metrics = fit_policy_bootstrap(
            model,
            bootstrap_samples,
            learning_rate=args.bootstrap_lr,
            max_epochs=args.bootstrap_max_epochs,
            patience=args.bootstrap_patience,
            validation_fraction=args.bootstrap_validation,
            minibatch_size=args.minibatch_size,
            seed=args.seed,
        )
        print(f"initial bootstrap: {json.dumps(bootstrap_metrics, sort_keys=True)}")

        initial_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        initial_checkpoint = run_dir / "bootstrap_initial.pt"
        save_checkpoint(
            initial_checkpoint,
            model,
            initial_optimizer,
            args,
            explorer,
            0,
            stages[0],
        )
        torch.set_num_threads(1)
        dagger_samples = generate_bootstrap_samples(
            args,
            run_dir,
            dagger=True,
            student_path=initial_checkpoint,
        )
        torch.set_num_threads(args.threads)
        if dagger_samples:
            combined_samples = bootstrap_samples + dagger_samples
            print(
                f"DAgger policy fitting on {len(combined_samples)} decisions "
                f"({len(dagger_samples)} student-state decisions)"
            )
            bootstrap_metrics["dagger"] = fit_policy_bootstrap(
                model,
                combined_samples,
                learning_rate=args.bootstrap_lr,
                max_epochs=args.bootstrap_max_epochs,
                patience=args.bootstrap_patience,
                validation_fraction=args.bootstrap_validation,
                minibatch_size=args.minibatch_size,
                seed=args.seed + 1,
            )
        (run_dir / "bootstrap_metrics.json").write_text(
            json.dumps(bootstrap_metrics, indent=2)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        bootstrap_checkpoint = run_dir / "bootstrap.pt"
        save_checkpoint(
            bootstrap_checkpoint,
            model,
            optimizer,
            args,
            explorer,
            0,
            stages[0],
        )
        frozen = copy.deepcopy(model).eval()
        bootstrap_results = evaluate_suite(
            frozen,
            legacy,
            None,
            champion,
            args.eval_games,
            args.seed + 900_000,
            args.alpha_net,
            args.alpha_eval_games,
        )
        bootstrap_checkpoint.with_suffix(".eval.json").write_text(
            json.dumps(bootstrap_results, indent=2)
        )
        score = selection_score(bootstrap_results, args.selection_metric)
        eligible = champion_gate_passes(
            bootstrap_results,
            args.champion_gate,
            args.champion_balanced_gate,
        ) and bootstrap_results["alphabot"]["win_rate"] >= args.alpha_gate
        observed_score = score
        shutil.copy2(bootstrap_checkpoint, run_dir / "best_observed.pt")
        shutil.copy2(
            bootstrap_checkpoint.with_suffix(".eval.json"),
            run_dir / "best_observed.eval.json",
        )
        (run_dir / "best_observed.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(bootstrap_checkpoint),
                    "benchmark_score": score,
                    "games": 0,
                    "champion_gate_passed": eligible,
                },
                indent=2,
            )
        )
        if eligible:
            best_score = score
            shutil.copy2(bootstrap_checkpoint, run_dir / "best.pt")
            shutil.copy2(
                bootstrap_checkpoint.with_suffix(".eval.json"),
                run_dir / "best.eval.json",
            )
        print(f"bootstrap checkpoint {bootstrap_checkpoint}")
        print(f"  fixed-opponent score: {score:.3f}; champion gate {eligible}")
        for name, result in bootstrap_results.items():
            print(
                f"  vs {name:<22}: {result['wins']}/{result['games']} "
                f"({result['win_rate']*100:.1f}%), avg VP {result['avg_vp']:.2f}"
            )
        previous = frozen
        population.append(frozen)
    elif not warm_started:
        raise ValueError("training requires bootstrap samples or a compatible warm start")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    deadline = start + args.minutes * 60
    next_checkpoint = start + args.checkpoint_minutes * 60

    while time.time() < deadline:
        progress = min(0.999, (time.time() - start) / max(1.0, args.minutes * 60))
        stage = stage_for_progress(stages, progress, args.curriculum_start)
        torch.set_num_threads(1)
        batch_results = play_training_batch(
            model,
            stage,
            explorer,
            population,
            policy_anchor,
            champion,
            args.champion_sample_probability,
            args.alpha_net,
            args.vp_advantage_scale,
            args.seed + games,
            args.selfplay_workers,
            args.selfplay_workers,
        )
        torch.set_num_threads(args.threads)
        for samples, game_stats in batch_results:
            pending_samples.extend(samples)
            games += 1
            event = {
                "t": "game",
                "games": games,
                "stage": stage.name,
                "simulations": stage.simulations,
                "exploration_mix": explorer.policy_mix,
                **game_stats,
            }
            metrics_file.write(json.dumps(event) + "\n")
            print(
                f"game {games:>3} | {stage.name:<10} | {game_stats['matchup']:<12} | "
                f"win {game_stats['learner_win']} | samples {game_stats['samples']:>3} | "
                f"EM policy {explorer.policy_mix:.2f}"
            )

        train_metrics = {}
        if len(pending_samples) >= args.batch_samples:
            train_metrics = optimize(
                model,
                optimizer,
                pending_samples,
                minibatch_size=args.minibatch_size,
                auxiliary_scale=stage.auxiliary_scale,
                reinforce_scale=args.reinforce_scale,
                entropy_scale=args.entropy_scale,
                search_policy_scale=args.search_policy_scale,
                elite_scale=args.elite_scale,
            )
            pending_samples.clear()
            metrics_file.write(
                json.dumps(
                    {
                        "t": "update",
                        "games": games,
                        "stage": stage.name,
                        **train_metrics,
                    }
                )
                + "\n"
            )

        if time.time() >= next_checkpoint or time.time() >= deadline:
            if pending_samples:
                train_metrics = optimize(
                    model,
                    optimizer,
                    pending_samples,
                    minibatch_size=args.minibatch_size,
                    auxiliary_scale=stage.auxiliary_scale,
                    reinforce_scale=args.reinforce_scale,
                    entropy_scale=args.entropy_scale,
                    search_policy_scale=args.search_policy_scale,
                    elite_scale=args.elite_scale,
                )
                pending_samples.clear()
            checkpoint = run_dir / "checkpoints" / f"game_{games:04d}.pt"
            save_checkpoint(checkpoint, model, optimizer, args, explorer, games, stage)
            frozen = copy.deepcopy(model).eval()
            results = evaluate_suite(
                frozen,
                legacy,
                previous,
                champion,
                args.eval_games,
                args.seed + games * 100,
                args.alpha_net,
                args.alpha_eval_games,
            )
            (checkpoint.with_suffix(".eval.json")).write_text(json.dumps(results, indent=2))
            score = selection_score(results, args.selection_metric)
            eligible = champion_gate_passes(
                results,
                args.champion_gate,
                args.champion_balanced_gate,
            ) and results["alphabot"]["win_rate"] >= args.alpha_gate
            if score > observed_score:
                observed_score = score
                shutil.copy2(checkpoint, run_dir / "best_observed.pt")
                shutil.copy2(
                    checkpoint.with_suffix(".eval.json"),
                    run_dir / "best_observed.eval.json",
                )
                (run_dir / "best_observed.json").write_text(
                    json.dumps(
                        {
                            "checkpoint": str(checkpoint),
                            "benchmark_score": score,
                            "games": games,
                            "champion_gate_passed": eligible,
                        },
                        indent=2,
                    )
                )
            if eligible and score > best_score:
                best_score = score
                shutil.copy2(checkpoint, run_dir / "best.pt")
                shutil.copy2(checkpoint.with_suffix(".eval.json"), run_dir / "best.eval.json")
                (run_dir / "best.json").write_text(
                    json.dumps(
                        {
                            "checkpoint": str(checkpoint),
                            "benchmark_score": score,
                            "games": games,
                            "champion_gate_passed": True,
                        },
                        indent=2,
                    )
                )
            print(f"checkpoint {checkpoint}")
            print(
                f"  fixed-opponent score: {score:.3f} "
                f"(eligible best {best_score:.3f}); champion gate {eligible}"
            )
            for name, result in results.items():
                print(
                    f"  vs {name:<22}: {result['wins']}/{result['games']} "
                    f"({result['win_rate']*100:.1f}%), avg VP {result['avg_vp']:.2f}"
                )
            previous = frozen
            population.append(frozen)
            population = population[-args.population_size :]
            if champion is not None and all(item is not champion for item in population):
                population = [champion, *population[-(args.population_size - 1) :]]
            next_checkpoint += args.checkpoint_minutes * 60

    print(f"finished {games} games; artifacts: {run_dir}")


def evaluate_command(args):
    candidate = load_catanzero(Path(args.checkpoint))
    legacy = load_legacy(Path(args.legacy))
    champion_path = Path(args.champion)
    champion = load_catanzero(champion_path) if champion_path.exists() else None
    results = evaluate_suite(
        candidate,
        legacy,
        None,
        champion,
        args.games,
        args.seed,
        args.alpha_net,
        args.alpha_games,
    )
    print(json.dumps(results, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--minutes", type=float, default=10)
    train_parser.add_argument("--checkpoint-minutes", type=float, default=3)
    train_parser.add_argument("--eval-games", type=int, default=8)
    train_parser.add_argument("--batch-samples", type=int, default=2048)
    train_parser.add_argument("--minibatch-size", type=int, default=1024)
    train_parser.add_argument("--population-size", type=int, default=8)
    train_parser.add_argument("--hidden", type=int, default=512)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--policy-mix", type=float, default=0.75)
    train_parser.add_argument("--exploration-max", type=float, default=0.9)
    train_parser.add_argument(
        "--training-profile",
        choices=(
            "baseline",
            "league_focus",
            "ppo_anchor",
            "league_anchor",
            "more_search",
            "policy_heavy",
            "exploration",
            "alpha_focus",
            "alpha_pure",
            "heuristic_focus",
            "heuristic_mix",
        ),
        default="baseline",
    )
    train_parser.add_argument(
        "--curriculum-start",
        type=float,
        default=0.0,
        help="Start fraction in [0, 1); use 0.34 or 0.67 for warm-start continuation.",
    )
    train_parser.add_argument(
        "--champion-sample-probability",
        type=float,
        default=0.35,
        help="Probability that each historical opponent is the incumbent champion.",
    )
    train_parser.add_argument(
        "--reinforce-scale",
        type=float,
        default=0.0,
        help="Chosen-action policy-gradient weight for terminal/VP advantage.",
    )
    train_parser.add_argument(
        "--entropy-scale",
        type=float,
        default=0.0,
        help="Entropy bonus applied during self-play optimization.",
    )
    train_parser.add_argument(
        "--vp-advantage-scale",
        type=float,
        default=0.25,
        help="Relative final-VP contribution to policy-gradient advantage.",
    )
    train_parser.add_argument(
        "--search-policy-scale",
        type=float,
        default=1.0,
        help="Weight on MCTS visit-count distillation.",
    )
    train_parser.add_argument(
        "--elite-scale",
        type=float,
        default=0.0,
        help="Behavior-cloning weight for actions from winning AlphaBot games.",
    )
    train_parser.add_argument(
        "--selection-metric",
        choices=("benchmark", "alphabot"),
        default="benchmark",
    )
    train_parser.add_argument(
        "--alpha-eval-games",
        type=int,
        default=None,
        help="Paired-seat AlphaBot games per checkpoint.",
    )
    train_parser.add_argument("--alpha-gate", type=float, default=0.0)
    train_parser.add_argument("--threads", type=int, default=10)
    train_parser.add_argument("--selfplay-workers", type=int, default=4)
    train_parser.add_argument("--bootstrap-games", type=int, default=96)
    train_parser.add_argument("--dagger-games", type=int, default=48)
    train_parser.add_argument("--bootstrap-workers", type=int, default=8)
    train_parser.add_argument("--bootstrap-dataset", default=None)
    train_parser.add_argument(
        "--bootstrap-sample-limit",
        type=int,
        default=0,
        help="Limit a loaded bootstrap dataset while retaining whole trajectories.",
    )
    train_parser.add_argument("--bootstrap-lr", type=float, default=3e-4)
    train_parser.add_argument("--bootstrap-max-epochs", type=int, default=30)
    train_parser.add_argument("--bootstrap-patience", type=int, default=5)
    train_parser.add_argument("--bootstrap-validation", type=float, default=0.2)
    train_parser.add_argument("--teacher-root-k", type=int, default=8)
    train_parser.add_argument("--teacher-samples", type=int, default=32)
    train_parser.add_argument("--teacher-depth", type=int, default=120)
    train_parser.add_argument("--teacher-determinizations", type=int, default=4)
    train_parser.add_argument("--champion-gate", type=float, default=0.25)
    train_parser.add_argument("--champion-balanced-gate", type=float, default=0.5)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--legacy", default="models/catan-512-best.pt")
    train_parser.add_argument(
        "--champion",
        default="training/runs/20260611-catanzero-v2/best.pt",
    )
    train_parser.add_argument("--alpha-net", default="models/catan-512.ctnn")
    train_parser.add_argument("--run-dir", default=None)
    train_parser.set_defaults(func=train)

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("checkpoint")
    eval_parser.add_argument("--games", type=int, default=24)
    eval_parser.add_argument("--seed", type=int, default=9000)
    eval_parser.add_argument("--legacy", default="models/catan-512-best.pt")
    eval_parser.add_argument(
        "--champion",
        default="training/runs/20260611-catanzero-v2/best.pt",
    )
    eval_parser.add_argument("--alpha-net", default="models/catan-512.ctnn")
    eval_parser.add_argument("--alpha-games", type=int, default=None)
    eval_parser.set_defaults(func=evaluate_command)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
