"""
Run evaluation episodes for one condition and append one JSON line per episode.

    python -m policy_reliability.run --condition nominal --seeds 0-14 \\
        --out results/episodes.jsonl --clips results/clips

Re-running skips (condition, seed) pairs already in the output file, so a long
evaluation can be split into chunks or resumed after an interruption.
"""

import argparse
import json
import time
from pathlib import Path

import gym_pusht  # noqa: F401  (registers the environment)
import gymnasium as gym
import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation

from policy_reliability.conditions import get_condition
from policy_reliability.metrics import parse_seeds
from policy_reliability.policy import SAMPLERS, InstrumentedPolicy

MAX_STEPS = 300  # PushT's standard episode length
ROOT = Path(__file__).resolve().parent.parent


def done_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {
            (r["condition"], r.get("sampler", "ddim10"), r["seed"]) for r in map(json.loads, f) if r
        }


def run_episode(
    policy: InstrumentedPolicy, condition_name: str, sampler: str, seed: int, keep_frames: bool
):
    condition = get_condition(condition_name)
    rng = np.random.default_rng(10_000 + seed)  # perturbation noise, separate from the scene
    torch.manual_seed(seed)  # diffusion sampling noise
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=MAX_STEPS)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    coverage: list[float] = []
    frames: list[np.ndarray] = []
    success, started = False, time.time()

    for _ in range(MAX_STEPS):
        if keep_frames:
            frames.append(obs["pixels"])
        seen = condition.perturb_obs(obs, rng)
        batch = preprocess_observation({k: v[None] for k, v in seen.items()})
        batch["task"] = [""]
        action = policy.act(batch).numpy()[0]
        obs, reward, terminated, truncated, info = env.step(condition.perturb_action(action, rng))
        # PushT's reward is progress: coverage / 0.95 capped at 1 (1.0 = success region).
        coverage.append(round(float(reward), 4))
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            break
    env.close()

    record = {
        "condition": condition_name,
        "sampler": sampler,
        "seed": seed,
        "success": success,
        "steps": len(coverage),
        "final_coverage": coverage[-1],
        "max_coverage": max(coverage),
        "coverage": coverage,
        "decision_steps": list(policy.log.steps),
        "dispersion": [round(d, 6) for d in policy.log.dispersion],
        "wall_seconds": round(time.time() - started, 1),
    }
    return record, frames


def save_clip(frames: list[np.ndarray], path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    big = [np.kron(f, np.ones((3, 3, 1), dtype=np.uint8)) for f in frames[::2]]  # 288 px, 5 fps
    imageio.mimsave(path, big, duration=0.2, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seeds", required=True, help="e.g. 0-14 or 0,3,7")
    parser.add_argument("--out", type=Path, default=ROOT / "results/episodes.jsonl")
    parser.add_argument("--clips", type=Path, default=None, help="save a GIF per episode here")
    parser.add_argument(
        "--samples", type=int, default=3, help="1 = success-rate only, no disagreement signal"
    )
    parser.add_argument(
        "--sampler", choices=SAMPLERS, default="ddim10", help="ddpm100 is ~10x slower"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--processors", type=Path, default=ROOT / "models/diffusion_pusht")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    get_condition(args.condition)  # fail fast on a typo
    skip = done_keys(args.out)
    todo = [s for s in parse_seeds(args.seeds) if (args.condition, args.sampler, s) not in skip]
    if not todo:
        print("nothing to do")
        return

    policy = InstrumentedPolicy(args.processors, samples=args.samples, sampler=args.sampler)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in todo:
        record, frames = run_episode(
            policy, args.condition, args.sampler, seed, keep_frames=args.clips is not None
        )
        with args.out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if args.clips is not None:
            tag = "success" if record["success"] else "failure"
            suffix = "" if args.sampler == "ddim10" else f"_{args.sampler}"
            save_clip(frames, args.clips / f"{args.condition}{suffix}_seed{seed}_{tag}.gif")
        print(
            f"{args.condition}/{args.sampler} seed={seed} success={record['success']} "
            f"steps={record['steps']} coverage={record['final_coverage']:.2f} "
            f"({record['wall_seconds']:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
