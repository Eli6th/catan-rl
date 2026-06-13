import numpy as np
import torch
from argparse import Namespace
from pathlib import Path

import catan_py
from training.alpha_cem import fit_outcome_head
from training.catanzero import (
    CATANZERO_VERSION,
    CatanZeroNet,
    ExplorationEM,
    champion_gate_passes,
    evaluate_balanced_match,
    evaluate_match,
    limit_bootstrap_samples,
    load_catanzero,
    optimize,
    relative_to_absolute,
    run_mcts,
    save_checkpoint,
    select_policy_anchor,
    selection_score,
    split_bootstrap_samples,
    stage_for_progress,
    stages_for_profile,
    stratified_root_candidates,
    STAGES,
    warm_start_catanzero,
    warm_start_legacy,
)


def test_relative_to_absolute_rotates_value_vector():
    values = np.array([1.0, 0.2, -0.3, -0.9])
    assert np.allclose(relative_to_absolute(values, 2), [-0.3, -0.9, 1.0, 0.2])


def test_em_mixture_moves_toward_component_explaining_search():
    legal = np.array([True, True, True, False])
    policy = np.array([0.8, 0.1, 0.1, 0.0])
    search = np.array([0.9, 0.05, 0.05, 0.0])
    explorer = ExplorationEM(0.5)
    before = explorer.policy_mix
    explorer.update(policy, search, legal)
    assert explorer.policy_mix > before


def test_em_mixture_retains_exploration_floor():
    legal = np.array([True, True, True, False])
    policy = np.array([0.99, 0.005, 0.005, 0.0])
    search = np.array([0.0, 0.5, 0.5, 0.0])
    explorer = ExplorationEM(0.11)
    for _ in range(100):
        explorer.update(policy, search, legal)
    assert explorer.policy_mix >= 0.1


def test_em_mixture_retains_exploration_reserve():
    legal = np.array([True, True, True, False])
    policy = np.array([0.99, 0.005, 0.005, 0.0])
    search = policy.copy()
    explorer = ExplorationEM(0.89)
    for _ in range(100):
        explorer.update(policy, search, legal)
    assert explorer.policy_mix <= 0.9


def test_warm_start_can_skip_to_late_curriculum():
    stages = stages_for_profile("league_focus")
    assert stage_for_progress(stages, 0.0, 0.34).name == "transition"
    assert stage_for_progress(stages, 0.0, 0.67).name == "league"


def test_bootstrap_limit_keeps_trajectories_whole():
    samples = [
        {"trajectory_id": f"game-{game}", "index": index}
        for game in range(5)
        for index in range(3)
    ]
    limited = limit_bootstrap_samples(samples, 7, 9)
    counts = {}
    for sample in limited:
        counts[sample["trajectory_id"]] = counts.get(sample["trajectory_id"], 0) + 1
    assert len(limited) >= 7
    assert set(counts.values()) == {3}


def test_policy_objective_is_finite_with_illegal_actions():
    model = CatanZeroNet(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mask = np.zeros(catan_py.NUM_ACTIONS, dtype=bool)
    mask[[0, 3]] = True
    policy = np.zeros(catan_py.NUM_ACTIONS, dtype=np.float32)
    policy[0] = 1.0
    sample = {
        "obs": np.zeros(catan_py.OBS_DIM, dtype=np.float32),
        "mask": mask,
        "policy": policy,
        "legacy_policy": policy,
        "legacy_anchor": 0.5,
        "outcome": np.zeros(4, dtype=np.float32),
        "vp": np.zeros(4, dtype=np.float32),
        "progress": np.zeros(4, dtype=np.float32),
        "belief": np.zeros(30, dtype=np.float32),
    }
    sample["action"] = 0
    sample["policy_advantage"] = 1.0
    metrics = optimize(
        model,
        optimizer,
        [sample],
        epochs=1,
        reinforce_scale=0.5,
        entropy_scale=0.01,
        search_policy_scale=0.5,
        elite_scale=0.25,
    )
    assert all(np.isfinite(value) for value in metrics.values())


def test_alpha_profile_trains_mostly_against_alphabot():
    stages = stages_for_profile("alpha_focus")
    assert dict(stages[1].matchups)["alpha"] == 0.8
    assert dict(stages[2].matchups)["alpha"] == 0.9
    pure = stages_for_profile("alpha_pure")
    assert pure[1].matchups == (("alpha", 1.0),)


def test_heuristic_mix_profile_preserves_both_scripted_opponents():
    stages = stages_for_profile("heuristic_mix")
    transition = dict(stages[1].matchups)
    assert transition["heuristic"] == 0.3
    assert transition["heuristic_v2"] == 0.4
    assert stages[1].legacy_anchor == 0.35


def test_policy_anchor_prefers_incumbent_champion():
    legacy = object()
    champion = object()
    assert select_policy_anchor(champion, legacy) is champion
    assert select_policy_anchor(None, legacy) is legacy


def test_batched_hybrid_policy_returns_legal_actions():
    env = catan_py.VecEnv(
        4,
        victory_target=7,
        visibility="realistic",
        seed=17,
        seats=["policy", "heuristic_v2", "heuristic_v2", "heuristic_v2"],
    )
    policy = catan_py.OpeningHybridPolicy(
        "models/catan-512.ctnn",
        specialist_net_path="models/catan-512.ctnn",
        specialist_net_min_vp=0,
        specialist_net_seat_mask=1,
        heuristic="v2",
        seed=17,
        strategy_settlement_weight=5.0,
        opening_production_weight=1.0,
        heuristic_refinement=True,
        endgame_conversion=True,
        road_refinement=True,
    )
    _, masks, _ = env.observe()
    masks = np.asarray(masks)
    actions = np.asarray(policy.actions(env), dtype=np.int64)
    assert actions.shape == (4,)
    assert masks[np.arange(4), actions].all()


def test_alphabot_selection_score_uses_requested_target():
    results = {
        "alphabot": {"win_rate": 0.7},
        "heuristic_v1": {"win_rate": 0.0},
        "heuristic_v2": {"win_rate": 0.0},
        "legacy_ppo": {"win_rate": 0.0},
    }
    assert selection_score(results, "alphabot") == 0.7


def test_checkpoint_config_excludes_cli_callback(tmp_path):
    model = CatanZeroNet(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    args = Namespace(hidden=16, command="train", func=lambda _: None, seed=7)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, args, ExplorationEM(), 1, STAGES[0])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["config"] == {"hidden": 16, "seed": 7}
    assert checkpoint["catanzero_version"] == CATANZERO_VERSION
    assert checkpoint["obs_dim"] == catan_py.OBS_DIM
    assert checkpoint["num_actions"] == catan_py.NUM_ACTIONS


def test_compact_student_skips_incompatible_legacy_copy():
    model = CatanZeroNet(hidden=16)
    assert not warm_start_legacy(model, "models/catan-512-best.pt")


def test_512_student_warm_starts_from_champion():
    model = CatanZeroNet(hidden=512)
    assert warm_start_catanzero(
        model, Path("training/runs/20260611-catanzero-v2/best.pt")
    )


def test_factorized_policy_head_preserves_flat_logits_at_initialization():
    model = CatanZeroNet(hidden=16)
    obs = torch.randn(3, catan_py.OBS_DIM)
    hidden = model.trunk(obs)

    assert torch.allclose(model(obs)["logits"], model.policy(hidden))


def test_factorized_policy_head_shifts_one_action_type_together():
    model = CatanZeroNet(hidden=16)
    obs = torch.zeros(1, catan_py.OBS_DIM)
    with torch.no_grad():
        model.policy_type.bias[0] = 2.0
    hidden = model.trunk(obs)
    shift = model(obs)["logits"] - model.policy(hidden)

    assert torch.allclose(shift[0, :54], torch.full((54,), 2.0))
    assert torch.allclose(
        shift[0, 54:],
        torch.zeros(catan_py.NUM_ACTIONS - 54),
    )


def test_outcome_calibration_changes_only_outcome_head():
    model = CatanZeroNet(hidden=16)
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    trajectories = [
        {
            "samples": [
                {
                    "obs": np.zeros(catan_py.OBS_DIM, dtype=np.float32),
                    "outcome": np.asarray([1.0, -1 / 3, -1 / 3, -1 / 3]),
                }
            ]
        }
    ]

    fit_outcome_head(
        model,
        trajectories,
        learning_rate=1e-2,
        epochs=2,
        minibatch_size=1,
    )

    changed = {
        key
        for key, value in model.state_dict().items()
        if not torch.equal(value, before[key])
    }
    assert changed == {"outcome.weight", "outcome.bias"}


def test_bootstrap_split_keeps_trajectories_disjoint():
    samples = [
        {"trajectory_id": f"game-{game}", "index": index}
        for game in range(5)
        for index in range(3)
    ]
    train, validation = split_bootstrap_samples(samples, 0.2, 7)
    train_ids = {sample["trajectory_id"] for sample in train}
    validation_ids = {sample["trajectory_id"] for sample in validation}
    assert train_ids.isdisjoint(validation_ids)


def test_champion_gate_requires_fair_share():
    assert champion_gate_passes(
        {"champion_catanzero": {"win_rate": 0.25}}, 0.25
    )
    assert not champion_gate_passes(
        {"champion_catanzero": {"win_rate": 0.125}}, 0.25
    )
    assert champion_gate_passes(
        {"champion_balanced": {"win_rate": 0.5}}, 0.25
    )
    assert not champion_gate_passes(
        {"champion_balanced": {"win_rate": 0.49}}, 0.25
    )


def test_identical_policy_evaluation_balances_all_seats():
    model = load_catanzero(Path("training/runs/20260611-catanzero-v2/best.pt"))
    result = evaluate_match(model, model, "policy", 4, 123)
    assert result["wins"] == 1


def test_identical_policy_balanced_match_splits_wins_evenly():
    model = load_catanzero(Path("training/runs/20260611-catanzero-v2/best.pt"))
    result = evaluate_balanced_match(model, model, 1, 123)
    assert result["wins"] == result["games"] // 2


def test_alpha_teacher_returns_legal_realistic_action():
    env = catan_py.Env(victory_target=7, visibility="realistic", seed=17)
    teacher = catan_py.AlphaTeacher(
        "models/catan-512.ctnn",
        root_k=2,
        samples=1,
        depth=2,
        visibility="realistic",
    )
    action = teacher.action(env.snapshot(), 99)
    assert np.asarray(env.mask(), dtype=bool)[action]


def test_opponent_aware_planner_returns_legal_action():
    env = catan_py.Env(
        victory_target=7,
        visibility="realistic",
        seed=17,
        seats=["policy", "alpha", "alpha", "alpha"],
        alpha_net="models/catan-512.ctnn",
    )
    planner = catan_py.OpponentAwarePlanner(
        "models/catan-512.ctnn",
        root_k=1,
        samples=1,
        continuation_decisions=0,
    )
    action = planner.action(env, 99)
    assert np.asarray(env.mask(), dtype=bool)[action]


def test_information_set_mcts_survives_many_redeterminizations():
    env = catan_py.Env(victory_target=7, visibility="realistic", seed=91)
    model = CatanZeroNet(hidden=16)
    visits, _, legal = run_mcts(
        env.snapshot(),
        model,
        simulations=32,
        cpuct=1.0,
        explorer=ExplorationEM(0.8),
        shaping_weight=0.0,
        seed=1234,
    )
    assert np.isclose(visits.sum(), 1.0)
    assert np.all(visits[~legal] == 0)


def test_gumbel_sequential_halving_is_legal_and_deterministic():
    env = catan_py.Env(victory_target=7, visibility="realistic", seed=93)
    model = CatanZeroNet(hidden=16)
    kwargs = {
        "simulations": 8,
        "cpuct": 1.0,
        "explorer": ExplorationEM(0.8),
        "shaping_weight": 0.0,
        "seed": 4321,
        "gumbel_scale": 0.5,
        "root_candidates": 4,
    }
    first, _, legal = run_mcts(env.snapshot(), model, **kwargs)
    second, _, _ = run_mcts(env.snapshot(), model, **kwargs)
    assert np.isclose(first.sum(), 1.0)
    assert np.all(first[~legal] == 0)
    assert np.count_nonzero(first) <= 4
    assert np.allclose(first, second)


def test_stratified_root_candidates_cover_distinct_action_types():
    legal = np.asarray([0, 1, 54, 55, 108, 109, 298])
    scores = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])

    assert stratified_root_candidates(legal, scores, 4) == [0, 54, 108, 298]
