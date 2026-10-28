import json
import random

import policy_reliability.report as report


def fake_episode(condition, seed, success):
    rng = random.Random(seed)
    base = 0.01 if success else 0.03
    n = 15 if success else 38
    return {
        "condition": condition,
        "seed": seed,
        "success": success,
        "steps": n * 8,
        "final_coverage": 0.97 if success else 0.6,
        "max_coverage": 0.97 if success else 0.7,
        "coverage": [0.5] * (n * 8),
        "decision_steps": [8 * i for i in range(n)],
        "dispersion": [base + rng.random() * 0.005 for _ in range(n)],
        "wall_seconds": 1.0,
    }


def test_report_end_to_end(tmp_path, monkeypatch):
    rows = [fake_episode("nominal", s, s % 4 != 0) for s in range(8)]
    rows += [fake_episode("actuation_noise", s, s % 2 == 0) for s in range(8)]
    ablation = [fake_episode("nominal", s, True) for s in range(4)]
    for r in ablation:
        r["sampler"], r["dispersion"], r["decision_steps"] = "ddpm100", [], []
    rows += ablation
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(report, "RESULTS", tmp_path)

    report.main()

    md = (tmp_path / "report.md").read_text()
    assert "| nominal | 8 | 75% " in md
    assert "| actuation_noise | 8 | 50% " in md
    assert "| all | 1.00 |" in md  # synthetic failures disagree more from the start
    assert "| nominal (ddpm100) | 4 | 100% " in md  # sampler ablation gets its own row
    assert "| nominal (ddpm100) | 0 | 0 | 0 | – / – / – |" in md  # ...but no signal stats
    # nominal seeds 0 and 4 failed by default (s % 4 == 0) and succeed in the ablation
    assert "| nominal (ddpm100) | 4 | 3 | 1 | 0 | 0 | 1.000 |" in md
    assert (tmp_path / "success_by_condition.png").stat().st_size > 1000
    assert (tmp_path / "disagreement_over_time.png").stat().st_size > 1000
