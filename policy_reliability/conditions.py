"""
Test conditions: controlled perturbations applied between the simulator and
the policy, so the same pretrained policy can be measured under the kinds of
drift a real deployment sees.

Each condition perturbs observations on the way in and/or actions on the way
out. Perturbations use their own seeded RNG, so a run is reproducible and two
conditions with the same episode seed start from the same scene.
"""

from dataclasses import dataclass

import numpy as np

Obs = dict[str, np.ndarray]

# PushT's workspace is 512 x 512 px; actions are target positions in that space.
WORKSPACE = 512.0


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    pixel_noise_std: float = 0.0  # on the 0-255 image scale
    state_noise_px: float = 0.0  # on the agent position the policy sees
    action_noise_px: float = 0.0  # on the target position actually executed

    def perturb_obs(self, obs: Obs, rng: np.random.Generator) -> Obs:
        out = dict(obs)
        if self.pixel_noise_std:
            noisy = obs["pixels"].astype(np.float32)
            noisy += rng.normal(0.0, self.pixel_noise_std, size=noisy.shape)
            out["pixels"] = np.clip(noisy, 0, 255).astype(np.uint8)
        if self.state_noise_px:
            out["agent_pos"] = obs["agent_pos"] + rng.normal(0.0, self.state_noise_px, size=2)
        return out

    def perturb_action(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.action_noise_px:
            return action
        noisy = action + rng.normal(0.0, self.action_noise_px, size=action.shape)
        return np.clip(noisy, 0.0, WORKSPACE).astype(action.dtype)


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in (
        Condition("nominal", "No perturbation: the policy as trained."),
        Condition(
            "sensor_noise",
            "Camera noise (σ = 20/255 per pixel) and position-sensor noise (σ = 4 px).",
            pixel_noise_std=20.0,
            state_noise_px=4.0,
        ),
        Condition(
            "actuation_noise",
            "Imprecise actuation: executed target position off by σ = 12 px.",
            action_noise_px=12.0,
        ),
    )
}


def get_condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise ValueError(f"unknown condition {name!r}; choose from {sorted(CONDITIONS)}") from None
