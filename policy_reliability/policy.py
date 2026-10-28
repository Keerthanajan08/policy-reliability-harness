"""
The policy under test: LeRobot's pretrained Diffusion Policy for PushT,
instrumented to report its own uncertainty.

Uncertainty signal: at every decision the policy draws K candidate action
chunks from its diffusion model (same observation, different noise) and
executes the first one, exactly as it normally would. The spread between
the K candidates (mean std across samples, in the policy's normalised action
space) measures how much the policy disagrees with itself. The hypothesis
tested in this repo: high self-disagreement precedes failure.

Two samplers are supported: "ddim10" (DDIM, 10 steps: the fast default) and
"ddpm100" (the checkpoint's original 100-step DDPM), so the effect of fast
sampling on reliability can be measured directly.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

HUB_ID = "lerobot/diffusion_pusht"
SAMPLERS = ("ddim10", "ddpm100")


@dataclass
class UncertaintyLog:
    """One entry per policy decision (a new action chunk)."""

    steps: list[int] = field(default_factory=list)
    dispersion: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.steps.clear()
        self.dispersion.clear()


class InstrumentedPolicy:
    def __init__(
        self,
        processors_dir: Path,
        samples: int = 3,
        sampler: str = "ddim10",
        device: str = "cpu",
    ) -> None:
        if samples < 1:
            raise ValueError("samples must be >= 1 (2+ to measure disagreement)")
        if sampler not in SAMPLERS:
            raise ValueError(f"unknown sampler {sampler!r}; choose from {SAMPLERS}")
        self.samples = samples
        self.policy = DiffusionPolicy.from_pretrained(HUB_ID)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(processors_dir),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        model = self.policy.diffusion
        if sampler == "ddim10":
            model.noise_scheduler = DDIMScheduler.from_config(model.noise_scheduler.config)
            model.num_inference_steps = 10
        else:  # the checkpoint's own DDPM scheduler, all training timesteps
            model.num_inference_steps = model.noise_scheduler.config.num_train_timesteps
        self.log = UncertaintyLog()
        self._step = 0
        model.generate_actions = self._generate_with_uncertainty  # type: ignore[method-assign]

    def _generate_with_uncertainty(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        model = self.policy.diffusion
        size, n_obs = batch["observation.state"].shape[:2]
        cond = model._prepare_global_conditioning(batch)  # encode the images once
        chunks = model.conditional_sample(
            size * self.samples, global_cond=cond.repeat_interleave(self.samples, 0)
        )
        start = n_obs - 1
        chunks = chunks[:, start : start + model.config.n_action_steps]
        chunks = chunks.reshape(size, self.samples, *chunks.shape[1:])
        if self.samples > 1:  # with one sample there's nothing to disagree with
            self.log.steps.append(self._step)
            self.log.dispersion.append(float(chunks.std(dim=1).mean()))
        return chunks[:, 0]  # execute one sample, as the unmodified policy would

    def reset(self) -> None:
        self.policy.reset()
        self.log.clear()
        self._step = 0

    @torch.inference_mode()
    def act(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        action = self.post(self.policy.select_action(self.pre(observation)))
        self._step += 1
        return action
