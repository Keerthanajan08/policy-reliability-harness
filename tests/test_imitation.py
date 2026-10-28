import numpy as np
import pytest
import torch

from policy_reliability.imitation.data import (
    ACT_DIM,
    OBS_DIM,
    Demos,
    denormalize,
    make_windows,
    normalize,
    split,
)
from policy_reliability.imitation.model import Controller, MLPPolicy, Variant


def demos(lengths: list[int]) -> Demos:
    """Frame f of episode e has every observation value = 10*e + f and action = (f, f)."""
    obs, act, ep = [], [], []
    for e, n in enumerate(lengths):
        for f in range(n):
            obs.append(np.full(OBS_DIM, 10 * e + f, np.float32))
            act.append(np.array([f, f], np.float32))
            ep.append(e)
    return Demos(np.stack(obs), np.stack(act), np.array(ep))


def test_normalisation_round_trip():
    x = np.array([0.0, 256.0, 512.0])
    assert normalize(x).tolist() == [-1.0, 0.0, 1.0]
    assert np.allclose(denormalize(normalize(x)), x)


def test_windows_pad_at_episode_edges_and_never_cross_them():
    x, y = make_windows(demos([3, 2]), history=2, chunk=3)
    assert x.shape == (5, 2 * OBS_DIM) and y.shape == (5, 3 * ACT_DIM)
    first = denormalize(x[0].reshape(2, OBS_DIM))[:, 0]
    assert first.tolist() == [0, 0]  # history before frame 0 repeats frame 0
    last_of_ep0 = denormalize(y[2].reshape(3, ACT_DIM))[:, 0]
    assert last_of_ep0.tolist() == [2, 2, 2]  # future past the end repeats the last action
    ep1_start = denormalize(x[3].reshape(2, OBS_DIM))[:, 0]
    assert ep1_start.tolist() == [10, 10]  # episode 1 never sees episode 0's frames


def test_split_is_by_whole_episodes():
    train, val = split(demos([2] * 10), val_fraction=0.2)
    assert len(val.episodes()) == 2 and len(train.episodes()) == 8
    assert not set(train.episodes()) & set(val.episodes())


class FixedChunk(MLPPolicy):
    """Returns a chunk whose k-th action is (query_index + k) / 100, to trace execution."""

    def __init__(self, variant: Variant) -> None:
        super().__init__(variant, hidden=8, layers=1)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk = torch.tensor(
            [[(self.calls + k) / 100.0] * ACT_DIM for k in range(self.variant.chunk)]
        )
        self.calls += 1
        return chunk.reshape(1, -1)


def test_open_loop_queries_once_per_chunk():
    policy = FixedChunk(Variant("t", history=1, chunk=4, execute="open_loop"))
    ctrl = Controller(policy)
    obs = np.zeros(OBS_DIM, np.float32)
    actions = [normalize(ctrl.act(obs))[0] for _ in range(6)]
    assert policy.calls == 2
    assert actions[:4] == pytest.approx([0.0, 0.01, 0.02, 0.03])
    assert actions[4] == pytest.approx(0.01)  # second chunk starts at query index 1


def test_ensemble_averages_every_prediction_for_the_current_step():
    policy = FixedChunk(Variant("t", history=1, chunk=3, execute="ensemble"))
    ctrl = Controller(policy, ensemble_m=0.0)  # equal weights: a plain average
    obs = np.zeros(OBS_DIM, np.float32)
    a0 = normalize(ctrl.act(obs))[0]  # only query 0 predicts step 0: 0.00
    a1 = normalize(ctrl.act(obs))[0]  # step 1: query 0 says 0.01, query 1 says 0.01
    a2 = normalize(ctrl.act(obs))[0]  # step 2: 0.02, 0.02, 0.02
    assert (a0, a1, a2) == pytest.approx((0.0, 0.01, 0.02))
    assert policy.calls == 3


def test_ensemble_weights_favour_older_predictions():
    class Drift(FixedChunk):
        def forward(self, x):
            chunk = torch.full((self.variant.chunk, ACT_DIM), float(self.calls))
            self.calls += 1
            return chunk.reshape(1, -1)

    ctrl = Controller(Drift(Variant("t", history=1, chunk=2, execute="ensemble")), ensemble_m=1.0)
    obs = np.zeros(OBS_DIM, np.float32)
    ctrl.act(obs)
    a1 = normalize(ctrl.act(obs))[0]  # predictions 0 (old, weight 1) and 1 (new, weight e^-1)
    assert a1 == pytest.approx((0 * 1 + 1 * np.exp(-1)) / (1 + np.exp(-1)))
