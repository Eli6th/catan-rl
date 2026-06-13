import numpy as np
import pytest

import catan_py
from training.ppo import (
    PolicyValueNet,
    configure_trainable_parameters,
    rotate_single_policy_seat,
)


def test_rotate_single_policy_seat_covers_all_positions():
    seats = ["policy", "alpha", "alpha", "alpha"]

    assert [rotate_single_policy_seat(seats, index).index("policy") for index in range(4)] == [
        0,
        1,
        2,
        3,
    ]


def test_rotate_single_policy_seat_rejects_ambiguous_lineup():
    with pytest.raises(ValueError, match="exactly one policy"):
        rotate_single_policy_seat(["policy", "alpha", "policy", "alpha"], 1)


def test_opening_heuristic_vecenv_skips_settlements_but_not_setup_roads():
    env = catan_py.VecEnv(
        4,
        victory_target=7,
        visibility="realistic",
        seed=123,
        seats=["policy", "heuristic_v2", "heuristic_v2", "heuristic_v2"],
        policy_opening_heuristic=True,
    )
    _, masks, _ = env.observe()
    masks = np.asarray(masks)
    settlement_end = catan_py.ACTION_TYPE_BOUNDARIES[0]
    road_start = catan_py.ACTION_TYPE_BOUNDARIES[1]
    road_end = catan_py.ACTION_TYPE_BOUNDARIES[2]
    assert not masks[:, :settlement_end].any()
    assert masks[:, road_start:road_end].any(axis=1).all()


def test_policy_head_only_freezes_trunk_and_value():
    net = PolicyValueNet(catan_py.OBS_DIM, catan_py.NUM_ACTIONS, 32)
    trainable = configure_trainable_parameters(net, policy_head_only=True)
    assert {id(parameter) for parameter in trainable} == {
        id(parameter) for parameter in net.policy.parameters()
    }
    assert not any(parameter.requires_grad for parameter in net.trunk.parameters())
    assert not any(parameter.requires_grad for parameter in net.value.parameters())
