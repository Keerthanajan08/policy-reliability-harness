"""
A diffusion policy trained from scratch on the same demonstrations.

Regression (the MLP policies in model.py) learns the *average* action chunk for
a state. When demonstrators sometimes go left and sometimes right around the
block, that average is a chunk nobody demonstrated. A diffusion policy instead
learns to turn random noise into a chunk *sampled* from the demonstrated
distribution, so it commits to one of the modes.

The denoiser is Diffusion Policy's conditional 1D U-Net (temporal convolutions
over the action chunk, FiLM-conditioned on the observation history and the
diffusion timestep), scaled down to train on a CPU.
Training follows Diffusion Policy: 100-step DDPM, squared-cosine noise
schedule, predict the added noise, EMA weights for evaluation. At run time it
samples with 10-step DDIM.
"""

import copy
import math

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch import nn

from policy_reliability.imitation.data import ACT_DIM, OBS_DIM

TRAIN_TIMESTEPS = 100
INFERENCE_STEPS = 10


def timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 5) -> None:
        super().__init__(
            nn.Conv1d(cin, cout, kernel, padding=kernel // 2), nn.GroupNorm(8, cout), nn.Mish()
        )


class ResBlock(nn.Module):
    """Two temporal convolutions; the conditioning scales and shifts (FiLM) in between."""

    def __init__(self, cin: int, cout: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1, self.conv2 = ConvBlock(cin, cout), ConvBlock(cout, cout)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond)[:, :, None].chunk(2, dim=1)
        h = self.conv1(x) * (1 + scale) + shift
        return self.conv2(h) + self.skip(x)


class Denoiser(nn.Module):
    """
    Diffusion Policy's conditional 1D U-Net, scaled down to train on a CPU.

    Convolving over the time axis of the action chunk (rather than flattening it,
    as an MLP would) builds in that neighbouring actions should be smooth and
    consistent, which is where the precision for the last few pixels comes from.
    """

    def __init__(self, history: int, chunk: int, dims: tuple[int, ...] = (64, 128)) -> None:
        super().__init__()
        self.chunk = chunk
        cond_dim = 128 + history * OBS_DIM
        self.t_enc = nn.Sequential(nn.Linear(128, 512), nn.Mish(), nn.Linear(512, 128))
        widths = [ACT_DIM, *dims]
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            last = i == len(dims) - 1
            self.down.append(
                nn.ModuleList(
                    [
                        ResBlock(a, b, cond_dim),
                        ResBlock(b, b, cond_dim),
                        nn.Identity() if last else nn.Conv1d(b, b, 3, stride=2, padding=1),
                    ]
                )
            )
        self.mid = nn.ModuleList([ResBlock(dims[-1], dims[-1], cond_dim) for _ in range(2)])
        self.up = nn.ModuleList()
        for a, b in zip(dims[::-1][:-1], dims[::-1][1:], strict=True):
            self.up.append(
                nn.ModuleList(
                    [
                        ResBlock(2 * a, b, cond_dim),
                        ResBlock(b, b, cond_dim),
                        nn.ConvTranspose1d(b, b, 4, stride=2, padding=1),
                    ]
                )
            )
        self.out = nn.Sequential(ConvBlock(dims[0], dims[0]), nn.Conv1d(dims[0], ACT_DIM, 1))

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([self.t_enc(timestep_embedding(t)), obs], dim=-1)
        x = noisy.reshape(len(noisy), self.chunk, ACT_DIM).transpose(1, 2)  # (B, ACT_DIM, T)
        skips = []
        for res1, res2, down in self.down:
            x = res2(res1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for res in self.mid:
            x = res(x, cond)
        for res1, res2, up in self.up:
            x = res2(res1(torch.cat([x, skips.pop()], dim=1), cond), cond)
            x = up(x)
        return self.out(x).transpose(1, 2).reshape(len(noisy), -1)


def noise_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )


class DiffusionPolicy(nn.Module):
    """Same interface as MLPPolicy: predict(obs_history) -> (1, chunk * ACT_DIM)."""

    def __init__(self, variant) -> None:
        super().__init__()
        self.variant = variant
        self.denoiser = Denoiser(variant.history, variant.chunk)
        self.ema = copy.deepcopy(self.denoiser).requires_grad_(False)
        self.train_sched = noise_scheduler()
        self.sample_sched = DDIMScheduler.from_config(self.train_sched.config)
        self.sample_sched.set_timesteps(INFERENCE_STEPS)
        self.generator: torch.Generator | None = None

    def loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(actions)
        t = torch.randint(0, TRAIN_TIMESTEPS, (len(actions),))
        noisy = self.train_sched.add_noise(actions, noise, t)
        return nn.functional.mse_loss(self.denoiser(noisy, t, obs), noise)

    @torch.no_grad()
    def update_ema(self, decay: float) -> None:
        for e, p in zip(self.ema.parameters(), self.denoiser.parameters(), strict=True):
            e.mul_(decay).add_(p.detach(), alpha=1 - decay)

    @torch.inference_mode()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.randn(
            len(obs), self.variant.chunk * ACT_DIM, generator=self.generator, dtype=obs.dtype
        )
        for t in self.sample_sched.timesteps:
            eps = self.ema(x, t.expand(len(obs)), obs)
            x = self.sample_sched.step(eps, t, x).prev_sample
        return x.clamp(-1, 1)

    def config(self) -> dict:
        from dataclasses import asdict

        return asdict(self.variant)
