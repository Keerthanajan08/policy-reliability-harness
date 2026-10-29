"""
Render one rollout of a trained imitation policy to a GIF.

    python -m policy_reliability.imitation.clip --checkpoint models/imitation/chunk8_seed0.pt \\
        --scene 1003 --out results/imitation/clips/chunk8_scene1003.gif
"""

import argparse
from pathlib import Path

import gym_pusht  # noqa: F401  (registers the environment)
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from policy_reliability.imitation.evaluate import MAX_STEPS, ROOT, load_policy, observation
from policy_reliability.imitation.model import Controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    controller = Controller(load_policy(ROOT / args.checkpoint))
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="environment_state_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=MAX_STEPS,
    )
    obs, _ = env.reset(seed=args.scene)
    torch.manual_seed(args.scene)  # same sampling noise as evaluate.py
    frames, success = [], False
    for step in range(MAX_STEPS):
        if step % 2 == 0:
            frame = Image.fromarray(env.render()).resize((256, 256), Image.Resampling.BILINEAR)
            frames.append(np.asarray(frame))
        action = np.clip(controller.act(observation(obs)), 0.0, 512.0)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            break
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=0.2, loop=0)
    print(f"{args.out.name}: success={success}, {step + 1} steps")


if __name__ == "__main__":
    main()
