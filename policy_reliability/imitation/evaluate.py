"""
Roll out trained imitation policies in the PushT simulator (state observations).

    python -m policy_reliability.imitation.evaluate --checkpoint models/imitation/chunk8_seed0.pt \\
        --scenes 1000-1099

Appends one JSON line per episode to results/imitation/episodes.jsonl; re-running
skips episodes already recorded. Evaluation scenes start at seed 1000 so they
never coincide with the diffusion-policy scenes (0-14).
"""

import argparse
import json
import time
from pathlib import Path

import gym_pusht  # noqa: F401  (registers the environment)
import gymnasium as gym
import numpy as np
import torch

from policy_reliability.imitation.model import Controller, Variant, build_policy
from policy_reliability.metrics import parse_seeds

ROOT = Path(__file__).resolve().parents[2]
MAX_STEPS = 300


def load_policy(path: Path) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = build_policy(Variant(**ckpt["variant"]))
    policy.load_state_dict(ckpt["state_dict"])
    return policy


def observation(obs: dict) -> np.ndarray:
    return np.concatenate([obs["environment_state"], obs["agent_pos"]]).astype(np.float32)


def rollout(controller: Controller, scene: int) -> dict:
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="environment_state_agent_pos", max_episode_steps=MAX_STEPS
    )
    obs, _ = env.reset(seed=scene)
    torch.manual_seed(scene)  # diffusion sampling noise: reproducible per scene
    controller.reset()
    progress: list[float] = []
    success = False
    for _ in range(MAX_STEPS):
        action = np.clip(controller.act(observation(obs)), 0.0, 512.0)
        obs, reward, terminated, truncated, info = env.step(action)
        progress.append(float(reward))
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            break
    env.close()
    return {
        "success": success,
        "steps": len(progress),
        "final_progress": round(progress[-1], 4),
        "max_progress": round(max(progress), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", default="1000-1099")
    parser.add_argument("--out", type=Path, default=ROOT / "results/imitation/episodes.jsonl")
    args = parser.parse_args()
    torch.set_num_threads(1)

    checkpoint = str(args.checkpoint if args.checkpoint.is_absolute() else args.checkpoint)
    done: set[tuple[str, int]] = set()
    if args.out.exists():
        done = {(r["checkpoint"], r["scene"]) for r in map(json.loads, args.out.open()) if r}

    policy = load_policy(
        ROOT / args.checkpoint if not args.checkpoint.is_absolute() else args.checkpoint
    )
    controller = Controller(policy)
    name = Path(checkpoint).stem  # e.g. chunk8_seed0
    variant, _, seed = name.rpartition("_seed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started, wins, n = time.time(), 0, 0
    with args.out.open("a") as f:
        for scene in parse_seeds(args.scenes):
            if (name, scene) in done:
                continue
            result = rollout(controller, scene)
            f.write(
                json.dumps(
                    {
                        "checkpoint": name,
                        "variant": variant,
                        "train_seed": int(seed),
                        "scene": scene,
                        **result,
                    }
                )
                + "\n"
            )
            wins += result["success"]
            n += 1
    print(f"{name}: {wins}/{n} new episodes succeeded ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
