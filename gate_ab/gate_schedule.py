"""Replay the residual window's two-threshold recursion over a demo, offline.

Answers whether the gate actually opens and closes during an episode, or whether it is on for the
whole thing (which would make an A/B against residualGateDistance=-1 vacuous). Imports isaacgym
before torch, the way main/rl/eval_score.py does, because the dataset package pulls it in.
"""
from isaacgym import gymapi, gymtorch  # noqa: F401  (must precede torch)
import numpy as np
import torch
from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_rotmat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

DEMO, ENGAGE, RELEASE, FADE, MAXLEN = "m_131154", 0.03, 0.045, 12, 1200


def env_offset():
    """Table-frame transform the loaders expect; copied from eval_score.Eval.get_env_offset."""
    t = np.eye(4)
    t[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    t[:3, 3] = np.array([0, 0, 0.4 + 0.4])
    return torch.tensor(t, device="cuda:0", dtype=torch.float32)


def window_trace(pinch):
    """Run the env's hysteresis + fade over a per-frame pinch distance, returning the weight trace.

    Args:
        pinch: (T,) min(thumb, index) human-fingertip-to-object distance in metres.

    Returns:
        (T,) float32 residual weight in [0, 1], matching residual_gate_weights step for step.
    """
    gate, fade, out = False, 0.0, []
    for d in pinch:
        gate = gate and (d < RELEASE)
        gate = gate or (d <= ENGAGE)
        fade = min(max(fade + (1.0 if gate else -1.0), 0.0), float(FADE))
        out.append(fade / FADE)
    return np.array(out, dtype=np.float32)


for side, hand in (("rh", "right"), ("lh", "left")):
    ds = ManipDataFactory.create_data(
        manipdata_type=ManipDataFactory.dataset_type(DEMO), side=hand, device="cuda:0",
        mujoco2gym_transf=env_offset(), max_seq_len=MAXLEN,
        dexhand=DexHandFactory.create_hand("inspire", hand),
    )
    d = ds[DEMO]
    pinch = d["tips_distance"][:, :2].min(dim=-1).values.cpu().numpy()
    w = window_trace(pinch)
    on = w > 0
    edges = np.where(np.diff(on.astype(int)) != 0)[0] + 1
    print(f"\n=== {side.upper()} ({DEMO}) — {len(pinch)} frames ===")
    print(f"  pinch dist: min {pinch.min()*100:5.2f} cm   max {pinch.max()*100:6.2f} cm")
    print(f"  residual ON for {on.sum()}/{len(on)} frames ({100.0*on.mean():.1f}% of the episode)")
    print(f"  transitions at frames: {edges.tolist()}")
    print(f"  weight at frames 0/60/120/180/251: "
          f"{w[0]:.2f} {w[60]:.2f} {w[120]:.2f} {w[180]:.2f} {w[-1]:.2f}")
    print(f"  last 40 frames: " + " ".join(f"{v:.1f}" for v in w[-40::4]))
