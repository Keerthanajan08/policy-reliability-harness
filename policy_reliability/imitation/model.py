"""
The policy network and the two ways of executing its predictions.

The network is deliberately simple (an MLP), so differences between variants
come from *what it predicts and how predictions are executed*, not capacity.
"""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from policy_reliability.imitation.data import ACT_DIM, OBS_DIM, denormalize, normalize


@dataclass(frozen=True)
class Variant:
    name: str
    history: int  # observation frames fed to the network
    chunk: int  # future actions predicted per query
    execute: str  # "open_loop": run the whole chunk; "ensemble": re-plan every step


VARIANTS = {
    v.name: v
    for v in (
        Variant("bc_single_step", history=2, chunk=1, execute="open_loop"),
        Variant("chunk4", history=2, chunk=4, execute="open_loop"),
        Variant("chunk8", history=2, chunk=8, execute="open_loop"),
        Variant("chunk16", history=2, chunk=16, execute="open_loop"),
        Variant("chunk8_ensemble", history=2, chunk=8, execute="ensemble"),
        Variant("chunk16_ensemble", history=2, chunk=16, execute="ensemble"),
    )
}


class MLPPolicy(nn.Module):
    def __init__(self, variant: Variant, hidden: int = 512, layers: int = 3) -> None:
        super().__init__()
        self.variant = variant
        dims = [variant.history * OBS_DIM] + [hidden] * layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            blocks += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        blocks.append(nn.Linear(dims[-1], variant.chunk * ACT_DIM))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config(self) -> dict:
        return asdict(self.variant)


class Controller:
    """
    Turns a trained policy into one action per environment step.

    open_loop: query once per chunk and execute all of it (ACT's "action chunking").
    ensemble:  query every step; the action for step t is a weighted average of every
               chunk that predicted step t, with weight exp(-m * age), so the oldest
               prediction counts most (ACT's temporal ensembling, m = 0.01).
    """

    def __init__(self, policy: MLPPolicy, ensemble_m: float = 0.01) -> None:
        self.policy = policy.eval()
        self.v = policy.variant
        self.m = ensemble_m
        self.reset()

    def reset(self) -> None:
        self.history: deque[np.ndarray] = deque(maxlen=self.v.history)
        self.queue: deque[np.ndarray] = deque()
        self.t = 0
        self.predictions: dict[int, list[np.ndarray]] = {}

    @torch.inference_mode()
    def _predict(self) -> np.ndarray:
        hist = list(self.history)
        while len(hist) < self.v.history:  # pad with the first frame, as in training
            hist.insert(0, hist[0])
        x = torch.from_numpy(np.concatenate(hist)[None]).float()
        return self.policy(x).numpy().reshape(self.v.chunk, ACT_DIM)

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.history.append(normalize(obs.astype(np.float32)))
        if self.v.execute == "open_loop":
            if not self.queue:
                self.queue.extend(self._predict())
            action = self.queue.popleft()
        else:
            chunk = self._predict()
            for k, a in enumerate(chunk):
                self.predictions.setdefault(self.t + k, []).append(a)
            preds = np.stack(self.predictions.pop(self.t))
            weights = np.exp(-self.m * np.arange(len(preds)))  # oldest first
            action = (weights[:, None] * preds).sum(0) / weights.sum()
        self.t += 1
        return denormalize(action)
