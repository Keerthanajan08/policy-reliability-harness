"""
Demonstrations -> (observation history, future action chunk) training pairs.

Observation per step: the T-block's 16 keypoint coordinates + the pusher's (x, y),
all in PushT's 0-512 px workspace. Both observations and actions are scaled to
[-1, 1] with fixed workspace bounds, so no statistics leak from the validation set.
"""

from dataclasses import dataclass

import numpy as np

WORKSPACE = 512.0
OBS_DIM = 18  # 16 keypoint coordinates + pusher (x, y)
ACT_DIM = 2


def normalize(x: np.ndarray) -> np.ndarray:
    return (x / WORKSPACE) * 2.0 - 1.0


def denormalize(x: np.ndarray) -> np.ndarray:
    return (x + 1.0) / 2.0 * WORKSPACE


@dataclass
class Demos:
    """Concatenated demonstrations, with episode boundaries."""

    obs: np.ndarray  # (N, OBS_DIM), raw pixels
    actions: np.ndarray  # (N, ACT_DIM), raw pixels
    episode: np.ndarray  # (N,), episode index per frame

    def episodes(self) -> list[int]:
        return sorted(set(self.episode.tolist()))

    def subset(self, episodes: set[int]) -> "Demos":
        mask = np.isin(self.episode, list(episodes))
        return Demos(self.obs[mask], self.actions[mask], self.episode[mask])


def load_pusht_demos() -> Demos:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot/pusht_keypoints", download_videos=False)
    cols = ds.hf_dataset.with_format("numpy")[:]
    obs = np.concatenate(
        [np.asarray(cols["observation.environment_state"]), np.asarray(cols["observation.state"])],
        axis=1,
    ).astype(np.float32)
    return Demos(
        obs=obs,
        actions=np.asarray(cols["action"], dtype=np.float32),
        episode=np.asarray(cols["episode_index"]),
    )


def split(demos: Demos, val_fraction: float, seed: int = 0) -> tuple[Demos, Demos]:
    """Split by whole episodes, so validation frames never come from a training demo."""
    eps = np.array(demos.episodes())
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_val = max(1, round(len(eps) * val_fraction))
    val = set(eps[:n_val].tolist())
    train = set(eps[n_val:].tolist())
    return demos.subset(train), demos.subset(val)


def make_windows(demos: Demos, history: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For every frame t: inputs are observations t-history+1..t (earlier frames
    repeat the episode's first observation), targets are actions t..t+chunk-1
    (later steps repeat the episode's last action). Returns normalised arrays
    of shape (N, history*OBS_DIM) and (N, chunk*ACT_DIM).
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for ep in demos.episodes():
        idx = np.flatnonzero(demos.episode == ep)
        obs, act = normalize(demos.obs[idx]), normalize(demos.actions[idx])
        n = len(idx)
        for t in range(n):
            hist = [obs[max(0, t - k)] for k in range(history - 1, -1, -1)]
            fut = [act[min(n - 1, t + k)] for k in range(chunk)]
            xs.append(np.concatenate(hist))
            ys.append(np.concatenate(fut))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)
