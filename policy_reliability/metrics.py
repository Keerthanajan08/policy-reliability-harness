"""
Pure metric functions over episode records (no simulator, no torch), so they
are fast to test.

An episode record (see run.py) carries per-step coverage and, per policy
decision, the policy's self-disagreement ("dispersion").
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

Record = dict[str, Any]
CONTROL_HZ = 10  # PushT steps per second


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a success rate; honest at small n, unlike p ± 2σ."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """
    Probability that a random positive scores above a random negative (ties count
    half). 0.5 = no signal, 1.0 = perfect separation. None if either side is empty.
    """
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def early_score(record: Record, decisions: int) -> float:
    """Mean dispersion over the first `decisions` policy decisions: a score available early."""
    values = record["dispersion"][:decisions]
    return sum(values) / len(values) if values else 0.0


def rolling(values: Sequence[float], window: int) -> list[float]:
    """Trailing mean; the first entries average over what's available."""
    out: list[float] = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def first_alarm_step(record: Record, threshold: float, window: int, warmup: int = 0) -> int | None:
    """
    Env step at which the rolling dispersion first exceeds `threshold`, or None.
    The first `warmup` decisions are ignored: policies are naturally unsure at the
    very start of an episode, before they have seen the scene settle.
    """
    smoothed = rolling(record["dispersion"], window)
    for i, (step, value) in enumerate(zip(record["decision_steps"], smoothed, strict=True)):
        if i >= warmup and value > threshold:
            return int(step)
    return None


def peak_after_warmup(record: Record, window: int, warmup: int) -> float:
    """Highest rolling dispersion an episode reaches once the warm-up is over."""
    smoothed = rolling(record["dispersion"], window)[warmup:]
    return max(smoothed) if smoothed else 0.0


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass
class ConditionSummary:
    condition: str
    episodes: int
    successes: int
    success_ci: tuple[float, float]
    median_seconds_to_success: float | None
    mean_final_coverage: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


def summarize(condition: str, records: Sequence[Record]) -> ConditionSummary:
    wins = [r for r in records if r["success"]]
    return ConditionSummary(
        condition=condition,
        episodes=len(records),
        successes=len(wins),
        success_ci=wilson_interval(len(wins), len(records)),
        median_seconds_to_success=(median(r["steps"] for r in wins) / CONTROL_HZ if wins else None),
        mean_final_coverage=(
            sum(r["final_coverage"] for r in records) / len(records) if records else 0.0
        ),
    )


@dataclass
class AlarmReport:
    threshold: float
    failures: int
    failures_flagged: int
    median_lead_seconds: float | None  # how long before the episode ended the alarm fired
    successes: int
    successes_flagged: int  # false alarms: would have called a human unnecessarily


def alarm_report(
    records: Sequence[Record], threshold: float, window: int, warmup: int = 0
) -> AlarmReport:
    fails = [r for r in records if not r["success"]]
    wins = [r for r in records if r["success"]]
    leads: list[float] = []
    flagged_fails = 0
    for r in fails:
        step = first_alarm_step(r, threshold, window, warmup)
        if step is not None:
            flagged_fails += 1
            leads.append((r["steps"] - step) / CONTROL_HZ)
    flagged_wins = sum(first_alarm_step(r, threshold, window, warmup) is not None for r in wins)
    return AlarmReport(
        threshold=threshold,
        failures=len(fails),
        failures_flagged=flagged_fails,
        median_lead_seconds=median(leads) if leads else None,
        successes=len(wins),
        successes_flagged=flagged_wins,
    )


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def mcnemar_exact(improved: int, worsened: int) -> float:
    """
    Two-sided exact McNemar test for paired success/failure outcomes (same seeds,
    two variants). Only the discordant pairs carry information: under "no
    difference", each one is equally likely to go either way.
    """
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
