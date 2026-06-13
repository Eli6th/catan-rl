"""Export a PolicyValueNet checkpoint to the CTNN binary format consumed by
the Rust AlphaBot (catan-env/src/net.rs).

Layout (little-endian):
  magic b"CTNN" | u32 version=1 | u32 obs_dim | u32 num_actions | u32 hidden
  f32 tensors row-major in order:
    w1[h][obs] b1[h] w2[h][h] b2[h] wp[acts][h] bp[acts] wv[h] bv[1]
  self-check vector:
    f32 test_obs[obs_dim] | f32 expected_value | f32 expected_logits[8]
The Rust loader recomputes the check vector and refuses mismatched weights —
the file proves its own integrity.

    python training/export_net.py <checkpoint.pt> <out.ctnn>
"""

import struct
import sys

import numpy as np
import torch

import catan_py
from ppo import PolicyValueNet
from catanzero import (
    CatanZeroNet,
    SUPPORTED_CATANZERO_VERSIONS,
    load_catanzero_state,
)


def main():
    ck_path, out_path = sys.argv[1], sys.argv[2]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert ck["codec_version"] == catan_py.CODEC_VERSION
    assert ck["obs_version"] == catan_py.OBS_VERSION
    hidden = ck["config"]["hidden"]
    is_catanzero = ck.get("catanzero_version") in SUPPORTED_CATANZERO_VERSIONS
    if is_catanzero:
        net = CatanZeroNet(hidden)
        load_catanzero_state(net, ck["model_state"])
    else:
        net = PolicyValueNet(catan_py.OBS_DIM, catan_py.NUM_ACTIONS, hidden)
        net.load_state_dict(ck["model_state"])
    net.eval()
    s = ck["model_state"]

    # Deterministic pseudo-random test observation in [0, 1].
    rng = np.random.default_rng(1234)
    test_obs = rng.random(catan_py.OBS_DIM, dtype=np.float32)
    with torch.no_grad():
        o = torch.as_tensor(test_obs).unsqueeze(0)
        mask = torch.ones(1, catan_py.NUM_ACTIONS, dtype=torch.bool)
        result = net(o, mask)
        if is_catanzero:
            logits = result["logits"]
            h = net.trunk(o)
            value = torch.nn.functional.linear(
                h,
                net.outcome.weight[0:1],
                net.outcome.bias[0:1],
            ).squeeze(1)
            value_weight = s["outcome.weight"][0:1]
            value_bias = s["outcome.bias"][0:1]
        else:
            logits, value = result
            value_weight = s["value.weight"]
            value_bias = s["value.bias"]
        policy_weight, policy_bias = (
            net.effective_policy_parameters()
            if is_catanzero
            else (net.policy.weight, net.policy.bias)
        )

    with open(out_path, "wb") as f:
        f.write(b"CTNN")
        f.write(struct.pack("<III", 1, catan_py.OBS_DIM, catan_py.NUM_ACTIONS))
        f.write(struct.pack("<I", hidden))
        for key in (
            "trunk.0.weight",
            "trunk.0.bias",
            "trunk.2.weight",
            "trunk.2.bias",
        ):
            f.write(s[key].numpy().astype("<f4").tobytes())
        f.write(policy_weight.detach().numpy().astype("<f4").tobytes())
        f.write(policy_bias.detach().numpy().astype("<f4").tobytes())
        f.write(value_weight.numpy().astype("<f4").tobytes())
        f.write(value_bias.numpy().astype("<f4").tobytes())
        f.write(test_obs.astype("<f4").tobytes())
        f.write(struct.pack("<f", float(value[0])))
        f.write(logits[0, :8].numpy().astype("<f4").tobytes())
    step = ck.get("global_step", ck.get("games", 0))
    print(f"exported {ck_path} (hidden {hidden}, step {step:,}) "
          f"-> {out_path} | check value {float(value[0]):.4f}")


if __name__ == "__main__":
    main()
