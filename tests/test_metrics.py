import math

import numpy as np
import pytest

from policy_reliability.conditions import CONDITIONS, WORKSPACE, get_condition
from policy_reliability.metrics import (
    alarm_report,
    auroc,
    early_score,
    first_alarm_step,
    mcnemar_exact,
    parse_seeds,
    peak_after_warmup,
    quantile,
    rolling,
    summarize,
    wilson_interval,
)


def episode(success: bool, dispersion: list[float], steps: int = 120, coverage: float = 0.97):
    return {
        "condition": "nominal",
        "seed": 0,
        "success": success,
        "steps": steps,
        "final_coverage": coverage,
        "decision_steps": [8 * i for i in range(len(dispersion))],
        "dispersion": dispersion,
    }


class TestWilson:
    def test_known_value(self):
        lo, hi = wilson_interval(8, 10)
        assert lo == pytest.approx(0.490, abs=1e-3)
        assert hi == pytest.approx(0.943, abs=1e-3)

    def test_extremes_stay_in_range(self):
        assert wilson_interval(0, 5)[0] == 0.0
        assert wilson_interval(5, 5)[1] == pytest.approx(1.0)
        assert wilson_interval(0, 0) == (0.0, 0.0)


class TestAuroc:
    def test_perfect_and_inverted(self):
        assert auroc([3, 4], [1, 2]) == 1.0
        assert auroc([1, 2], [3, 4]) == 0.0

    def test_ties_count_half(self):
        assert auroc([1], [1]) == 0.5

    def test_empty_side(self):
        assert auroc([], [1]) is None


def test_rolling_mean():
    assert rolling([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]


def test_early_score_uses_only_the_first_decisions():
    assert early_score(episode(False, [1, 1, 9, 9]), decisions=2) == 1


def test_quantile_interpolates():
    assert quantile([0, 10], 0.5) == 5
    with pytest.raises(ValueError):
        quantile([], 0.5)


def test_first_alarm_step():
    rec = episode(False, [0.1, 0.1, 0.5, 0.9])
    assert first_alarm_step(rec, threshold=0.4, window=1) == 16
    assert first_alarm_step(rec, threshold=0.4, window=2) == 24  # smoothing delays the alarm
    assert first_alarm_step(rec, threshold=0.95, window=1) is None


def test_warmup_ignores_an_uncertain_start():
    rec = episode(True, [0.9, 0.9, 0.1, 0.1])
    assert first_alarm_step(rec, threshold=0.5, window=1) == 0
    assert first_alarm_step(rec, threshold=0.5, window=1, warmup=2) is None
    assert peak_after_warmup(rec, window=1, warmup=2) == 0.1
    assert peak_after_warmup(episode(True, [0.3]), window=1, warmup=5) == 0.0


def test_alarm_report_counts_lead_time_and_false_alarms():
    records = [
        episode(False, [0.1, 0.5, 0.5], steps=300),  # alarm at step 8 -> 29.2 s before the end
        episode(False, [0.1, 0.1, 0.1], steps=300),  # never flagged
        episode(True, [0.6, 0.1], steps=100),  # false alarm
        episode(True, [0.1, 0.1], steps=100),
    ]
    report = alarm_report(records, threshold=0.3, window=1)
    assert (report.failures, report.failures_flagged) == (2, 1)
    assert report.median_lead_seconds == pytest.approx(29.2)
    assert (report.successes, report.successes_flagged) == (2, 1)


def test_summarize():
    records = [
        episode(True, [0.1], steps=100),
        episode(True, [0.1], steps=140),
        episode(False, [0.2]),
    ]
    s = summarize("nominal", records)
    assert (s.episodes, s.successes) == (3, 2)
    assert s.success_rate == pytest.approx(2 / 3)
    assert s.median_seconds_to_success == pytest.approx(12.0)


def test_parse_seeds():
    assert parse_seeds("0-2,7") == [0, 1, 2, 7]


class TestConditions:
    def test_nominal_changes_nothing(self):
        rng = np.random.default_rng(0)
        obs = {"pixels": np.full((4, 4, 3), 100, np.uint8), "agent_pos": np.array([10.0, 20.0])}
        out = CONDITIONS["nominal"].perturb_obs(obs, rng)
        assert np.array_equal(out["pixels"], obs["pixels"])
        action = np.array([5.0, 6.0], np.float32)
        assert np.array_equal(CONDITIONS["nominal"].perturb_action(action, rng), action)

    def test_sensor_noise_keeps_valid_pixels(self):
        rng = np.random.default_rng(0)
        obs = {"pixels": np.zeros((32, 32, 3), np.uint8), "agent_pos": np.array([10.0, 20.0])}
        out = get_condition("sensor_noise").perturb_obs(obs, rng)
        assert out["pixels"].dtype == np.uint8
        assert out["pixels"].max() > 0  # noise was added, clipped at 0 from below
        assert not np.array_equal(out["agent_pos"], obs["agent_pos"])
        assert obs["pixels"].max() == 0  # the original is untouched

    def test_actuation_noise_stays_in_the_workspace(self):
        rng = np.random.default_rng(0)
        cond = get_condition("actuation_noise")
        for _ in range(200):
            a = cond.perturb_action(np.array([0.0, WORKSPACE], np.float32), rng)
            assert a.min() >= 0.0 and a.max() <= WORKSPACE
        noisy = [cond.perturb_action(np.array([256.0, 256.0]), rng)[0] for _ in range(2000)]
        assert np.std(noisy) == pytest.approx(12.0, rel=0.1)

    def test_unknown_condition(self):
        with pytest.raises(ValueError, match="unknown condition"):
            get_condition("typo")


def test_math_sanity():
    assert math.isclose(sum(rolling([2, 2, 2], 3)), 6)


def test_mcnemar_exact():
    assert mcnemar_exact(4, 0) == 0.125  # 2 * (1/2)^4
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(10, 0) < 0.01
