import copy

import numpy as np
import pytest

import catan_py
from training.catanzero import CatanZeroNet
from training.counterfactual_policy_improvement import (
    fit_counterfactuals,
    observed_cities,
    observed_vp,
    relative_road_target,
    score_target,
    should_retain_episode,
)


def test_observed_vp_includes_hidden_victory_point_cards():
    obs = np.zeros(1350, dtype=np.float32)
    obs[1196 + 3] = 4.0 / 7.0
    obs[1264 + 5 + 1] = 2.0 / 5.0
    assert observed_vp(obs) == pytest.approx(6.0)


def test_observed_cities_decodes_candidate_inventory():
    obs = np.zeros(1350, dtype=np.float32)
    obs[1196 + 5] = 2.0 / 4.0
    assert observed_cities(obs) == 2


def test_score_target_prefers_higher_counterfactual_value():
    target, spread = score_target([(10, 0.1), (20, 0.4)], temperature=0.1)
    assert target[20] > target[10]
    assert np.isclose(target.sum(), 1.0)
    assert np.isclose(spread, 0.3)


def test_losses_only_filter_rejects_wins_and_keeps_non_wins():
    win = np.asarray([-1 / 3, 1.0, -1 / 3, -1 / 3], dtype=np.float32)
    draw = np.zeros(4, dtype=np.float32)

    assert not should_retain_episode(win, 1, losses_only=True)
    assert should_retain_episode(win, 0, losses_only=True)
    assert should_retain_episode(draw, 1, losses_only=True)
    assert should_retain_episode(win, 1, losses_only=False)


def test_relative_road_target_rotates_absolute_holder():
    absolute = np.asarray([0, 0, 0, 1, 0], dtype=np.float32)
    assert relative_road_target(absolute, 1).tolist() == [0, 0, 1, 0, 0]

    no_holder = np.asarray([0, 0, 0, 0, 1], dtype=np.float32)
    assert relative_road_target(no_holder, 3).tolist() == [0, 0, 0, 0, 1]


def test_relative_road_target_rejects_missing_terminal_label():
    with pytest.raises(ValueError, match="one-hot"):
        relative_road_target(np.zeros(5, dtype=np.float32), 0)


def test_road_auxiliary_can_train_on_non_policy_samples():
    model = CatanZeroNet(hidden=8)
    reference = copy.deepcopy(model).eval()
    mask = np.zeros(catan_py.NUM_ACTIONS, dtype=bool)
    mask[:2] = True
    target = np.zeros(catan_py.NUM_ACTIONS, dtype=np.float32)
    target[0] = 1.0
    sample = {
        "obs": np.zeros(catan_py.OBS_DIM, dtype=np.float32),
        "mask": mask,
        "target": target,
        "weight": 1.0,
        "policy_weight": 0.0,
        "road_target": np.asarray([1, 0, 0, 0, 0], dtype=np.float32),
        "vp": 4.0,
        "spread": 0.1,
        "deployed": 0,
        "teacher_best": 1,
    }

    result = fit_counterfactuals(
        model,
        reference,
        [sample],
        learning_rate=1e-6,
        epochs=1,
        minibatch_size=1,
        kl_scale=1.0,
        road_aux_scale=0.1,
    )

    assert result["policy_samples"] == 0
    assert result["disagreements"] == 0
    assert np.isfinite(result["loss"])
