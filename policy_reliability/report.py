"""
Turn results/episodes.jsonl into results/report.md and two charts.

    python -m policy_reliability.report

Failure prediction is evaluated two ways:
  * Early warning (AUROC): does the policy's self-disagreement over its first
    few decisions separate episodes that will fail from ones that will succeed?
  * Online alarm: a monitor that fires when rolling disagreement crosses a
    threshold after a short warm-up. The threshold is set from a false-alarm
    budget on NOMINAL successful episodes only (at most ~10% of them may ever
    trigger it), then applied unchanged everywhere, so the perturbed
    conditions are a fair, out-of-sample test.
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from policy_reliability.conditions import CONDITIONS  # noqa: E402
from policy_reliability.metrics import (  # noqa: E402
    CONTROL_HZ,
    alarm_report,
    auroc,
    early_score,
    mcnemar_exact,
    peak_after_warmup,
    quantile,
    rolling,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EARLY_DECISIONS = 6  # ~4.8 s of a 30 s episode (8 steps per decision at 10 Hz)
WINDOW = 3  # decisions in the alarm's rolling mean
WARMUP = 2  # decisions ignored at the start of each episode (~1.6 s)
FALSE_ALARM_BUDGET = 0.10  # share of nominal successes allowed to trigger the alarm
SAMPLER_LABELS = {"ddim10": "10-step DDIM", "ddpm100": "100-step DDPM"}
NEAR_MISS_PROGRESS = 0.90  # a failure ending at >= 90% progress (~86% covered) is a near miss

# Validated categorical slots 1-2 (see the dataviz palette); text stays in ink colours.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"


def group_name(record: dict) -> str:
    """Default-sampler runs keep the plain condition name; ablations get a suffix."""
    sampler = record.get("sampler", "ddim10")
    return record["condition"] if sampler == "ddim10" else f"{record['condition']} ({sampler})"


def load(path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                groups[group_name(record)].append(record)
    # Canonical condition order, each followed by its sampler ablations.
    order = [name for c in CONDITIONS for name in sorted(groups) if name.split(" ")[0] == c]
    return {name: groups[name] for name in order}


def has_signal(records: list[dict]) -> bool:
    return any(r["dispersion"] for r in records)


def calibrate_threshold(nominal: list[dict]) -> float:
    """Lowest threshold that at most FALSE_ALARM_BUDGET of nominal successes ever exceed."""
    peaks = [peak_after_warmup(r, WINDOW, WARMUP) for r in nominal if r["success"]]
    if not peaks:
        raise SystemExit("need at least one successful nominal episode to calibrate the alarm")
    return quantile(peaks, 1 - FALSE_ALARM_BUDGET)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


def chart_success(summaries, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 0.9 + 0.75 * len(summaries)), facecolor=SURFACE)
    _style(ax)
    has_ablation = any("(" in s.condition for s in summaries)

    def label(name: str) -> str:
        base, _, variant = name.partition(" (")
        base = base.replace("_", " ")
        if variant:
            return f"{base}: {SAMPLER_LABELS.get(variant.rstrip(')'), variant.rstrip(')'))}"
        return f"{base}: {SAMPLER_LABELS['ddim10']}" if has_ablation else base

    names = [label(s.condition) for s in summaries][::-1]
    rates = [s.success_rate * 100 for s in summaries][::-1]
    lows = [s.success_rate * 100 - s.success_ci[0] * 100 for s in summaries][::-1]
    highs = [s.success_ci[1] * 100 - s.success_rate * 100 for s in summaries][::-1]
    y = range(len(names))
    ax.barh(y, rates, height=0.5, color=BLUE)
    ax.errorbar(rates, y, xerr=[lows, highs], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
    for i, s in enumerate(summaries[::-1]):
        ax.text(
            min(s.success_ci[1] * 100 + 2, 101),
            i,
            f"{s.successes}/{s.episodes}",
            va="center",
            color=INK,
            fontsize=10,
        )
    ax.set_yticks(list(y), names, color=INK, fontsize=11)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_title("Success rate by condition (95% CI)", loc="left", color=INK, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_traces(data: dict[str, list[dict]], threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    _style(ax)
    groups = {"Succeeded": [], "Failed": []}
    for records in data.values():
        for r in records:
            groups["Succeeded" if r["success"] else "Failed"].append(r)

    for label, color in (("Succeeded", BLUE), ("Failed", ORANGE)):
        series = [rolling(r["dispersion"], WINDOW) for r in groups[label]]
        if not series:
            continue
        length = max(len(s) for s in series)
        xs, med, lo, hi = [], [], [], []
        for i in range(length):
            column = [s[i] for s in series if i < len(s)]
            if len(column) < 2:  # too few episodes still running to summarise
                break
            xs.append(i * 8 / CONTROL_HZ)
            med.append(median(column))
            lo.append(quantile(column, 0.25))
            hi.append(quantile(column, 0.75))
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(xs, med, color=color, lw=2, label=f"{label} (n={len(series)})")
        if xs:  # labels above/below the line end so the two never collide
            ax.text(
                xs[-1],
                med[-1],
                f" {label}",
                color=INK,
                fontsize=10,
                va="bottom" if label == "Succeeded" else "top",
            )

    warmup_s = WARMUP * 8 / CONTROL_HZ
    ax.axvspan(0, warmup_s, color=GRID, alpha=0.6, lw=0)
    ax.text(
        warmup_s / 2,
        ax.get_ylim()[1],
        "warm-up\n(ignored)",
        color=INK2,
        fontsize=8,
        ha="center",
        va="top",
    )
    ax.axhline(threshold, color=MUTED, lw=1.2, ls=(0, (4, 4)))
    ax.text(warmup_s + 0.2, threshold, " alarm threshold", color=INK2, fontsize=9, va="bottom")
    ax.set_xlabel("Seconds into the episode", color=INK2)
    ax.set_ylabel("Policy self-disagreement", color=INK2)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK, loc="upper right")
    ax.set_title(
        "Median rolling self-disagreement (band: middle 50% of episodes)",
        loc="left",
        color=INK,
        fontsize=12,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def main() -> None:
    data = load(RESULTS / "episodes.jsonl")
    if "nominal" not in data:
        raise SystemExit("need nominal episodes to calibrate the alarm")
    summaries = [summarize(name, records) for name, records in data.items()]
    threshold = calibrate_threshold(data["nominal"])
    # The disagreement signal is compared only within the default sampler: other samplers
    # (and single-sample runs) have a different or no disagreement scale.
    signal = {
        name: records
        for name, records in data.items()
        if has_signal(records) and name == records[0]["condition"]
    }
    all_records = [r for records in signal.values() for r in records]

    lines = ["# Results", ""]
    lines += [
        "| Condition | Episodes | Success rate (95% CI) | Median time to success "
        "| Mean final progress |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        t = f"{s.median_seconds_to_success:.1f} s" if s.median_seconds_to_success else "–"
        lines.append(
            f"| {s.condition} | {s.episodes} | {pct(s.success_rate)} "
            f"({pct(s.success_ci[0])}–{pct(s.success_ci[1])}) | {t} | "
            f"{s.mean_final_coverage:.2f} |"
        )

    ablations = [name for name in data if name != data[name][0]["condition"]]
    if ablations:
        lines += ["", "## Sampler comparison (same seeds)", ""]
        lines += [
            "Paired by seed against the default sampler (DDIM, 10 steps). *Fixed*: failed with "
            "the default, succeeded with the variant; *broke*: the reverse. p is a two-sided "
            "exact McNemar test.",
            "",
            "| Variant | Paired seeds | Both succeed | Fixed | Broke | Both fail | p |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in ablations:
            base = {r["seed"]: r["success"] for r in data.get(data[name][0]["condition"], [])}
            pairs = [(base[r["seed"]], r["success"]) for r in data[name] if r["seed"] in base]
            fixed = sum(not b and v for b, v in pairs)
            broke = sum(b and not v for b, v in pairs)
            both = sum(b and v for b, v in pairs)
            neither = sum(not b and not v for b, v in pairs)
            lines.append(
                f"| {name} | {len(pairs)} | {both} | {fixed} | {broke} | {neither} | "
                f"{mcnemar_exact(fixed, broke):.3f} |"
            )

    lines += ["", "## How episodes failed", ""]
    lines += [
        f"A failure ending at progress >= {NEAR_MISS_PROGRESS} is a *near miss* (the T is almost "
        "placed but never crosses the 95% bar); anything lower means the policy *lost the "
        "block*. Peak disagreement is the highest rolling value after warm-up.",
        "",
        "| Condition | Failures | Near misses | Lost the block "
        "| Median peak disagreement: successes / near misses / lost |",
        "|---|---|---|---|---|",
    ]

    def med_peak(rs: list[dict]) -> str:
        if not rs or not has_signal(rs):
            return "–"
        return f"{median(peak_after_warmup(r, WINDOW, WARMUP) for r in rs):.4f}"

    for name, records in data.items():
        fails = [r for r in records if not r["success"]]
        near = [r for r in fails if r["final_coverage"] >= NEAR_MISS_PROGRESS]
        lost = [r for r in fails if r["final_coverage"] < NEAR_MISS_PROGRESS]
        wins = [r for r in records if r["success"]]
        lines.append(
            f"| {name} | {len(fails)} | {len(near)} | {len(lost)} | "
            f"{med_peak(wins)} / {med_peak(near)} / {med_peak(lost)} |"
        )

    lines += ["", "## Failure prediction", ""]
    lines += [
        f"Early warning uses mean self-disagreement over the first {EARLY_DECISIONS} decisions "
        f"(~{EARLY_DECISIONS * 8 / CONTROL_HZ:.1f} s). The alarm fires when the {WINDOW}-decision "
        f"rolling mean exceeds {threshold:.4f} after a {WARMUP}-decision warm-up; the threshold "
        f"allows at most {FALSE_ALARM_BUDGET:.0%} of nominal successes to trigger it.",
        "",
        "| Condition | Early-warning AUROC | Failures flagged | Median warning before end "
        "| False alarms on successes |",
        "|---|---|---|---|---|",
    ]
    for name, records in [*signal.items(), ("all", all_records)]:
        fails = [early_score(r, EARLY_DECISIONS) for r in records if not r["success"]]
        wins = [early_score(r, EARLY_DECISIONS) for r in records if r["success"]]
        score = auroc(fails, wins)
        rep = alarm_report(records, threshold, WINDOW, WARMUP)
        lead = f"{rep.median_lead_seconds:.1f} s" if rep.median_lead_seconds is not None else "–"
        lines.append(
            f"| {name} | {'–' if score is None else f'{score:.2f}'} | "
            f"{rep.failures_flagged}/{rep.failures} | {lead} | "
            f"{rep.successes_flagged}/{rep.successes} |"
        )
    lines += [
        "",
        "Nominal false alarms are in-sample (the threshold was calibrated on those episodes); "
        "the perturbed conditions are out-of-sample.",
        "",
    ]

    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    chart_success(summaries, RESULTS / "success_by_condition.png")
    chart_traces(signal, threshold, RESULTS / "disagreement_over_time.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
