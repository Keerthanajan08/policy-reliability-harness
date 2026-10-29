"""
results/imitation/episodes.jsonl -> results/imitation/report.md + a chart.

    python -m policy_reliability.imitation.report

Each variant is trained with several seeds and every model is evaluated on
the same scenes. Success rates pool all (seed, scene) episodes, with the
per-seed spread shown separately, so one lucky training run can't carry a
variant. Variants are compared pairwise on matching (seed index, scene) pairs.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from policy_reliability.imitation.model import VARIANTS  # noqa: E402
from policy_reliability.metrics import mcnemar_exact, wilson_interval  # noqa: E402
from policy_reliability.report import BLUE, GRID, INK, INK2, ORANGE, SURFACE, _style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "imitation"

LABELS = {
    "bc_single_step": "Single-step BC",
    "chunk4": "Chunking (4)",
    "chunk8": "Chunking (8)",
    "chunk16": "Chunking (16)",
    "chunk8_ensemble": "Chunking (8) + ensembling",
    "chunk16_ensemble": "Chunking (16) + ensembling",
    "chunk16_exec8": "Regression, plan 16 / run 8",
    "diffusion": "Diffusion, plan 16 / run 8",
}


def load(path: Path) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_variant[r["variant"]].append(r)
    return {v: by_variant[v] for v in VARIANTS if by_variant.get(v)}


def per_seed_rates(records: list[dict]) -> dict[int, float]:
    seeds: dict[int, list[bool]] = defaultdict(list)
    for r in records:
        seeds[r["train_seed"]].append(r["success"])
    return {s: sum(v) / len(v) for s, v in sorted(seeds.items())}


def paired(a: list[dict], b: list[dict]) -> tuple[int, int, int]:
    """(pairs, b fixes a failure of a, b breaks a success of a) on matching seed and scene."""
    index = {(r["train_seed"], r["scene"]): r["success"] for r in a}
    fixed = broke = n = 0
    for r in b:
        key = (r["train_seed"], r["scene"])
        if key in index:
            n += 1
            fixed += (not index[key]) and r["success"]
            broke += index[key] and not r["success"]
    return n, fixed, broke


def chart(data: dict[str, list[dict]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 1.4 + 0.55 * len(data)), facecolor=SURFACE)
    _style(ax)
    names = list(data)[::-1]
    stats = []
    for name in names:
        recs = data[name]
        wins = sum(r["success"] for r in recs)
        lo, hi = (x * 100 for x in wilson_interval(wins, len(recs)))
        stats.append((wins, len(recs), wins / len(recs) * 100, lo, hi, per_seed_rates(recs)))
    top = max(max(hi, *(v * 100 for v in seeds.values())) for *_, hi, seeds in stats)
    upper = 10 * (int(top // 10) + 1)
    for i, (wins, n, rate, lo, hi, seeds) in enumerate(stats):
        generative = VARIANTS[names[i]].kind == "diffusion"
        ax.barh(i, rate, height=0.55, color=ORANGE if generative else BLUE)
        ax.errorbar(
            rate,
            i,
            xerr=[[rate - lo], [hi - rate]],
            fmt="none",
            ecolor=INK2,
            elinewidth=1.2,
            capsize=3,
        )
        for seed_rate in seeds.values():  # each training seed as a hollow dot
            ax.plot(seed_rate * 100, i, "o", ms=5, mfc=SURFACE, mec=INK, mew=1.2)
        # Counts in their own column, so they never collide with dots or whiskers.
        ax.text(upper * 1.13, i, f"{wins}/{n}", va="center", ha="right", color=INK, fontsize=10)
    ax.set_yticks(range(len(names)), [LABELS.get(n, n) for n in names], color=INK, fontsize=10)
    ax.set_xlim(0, upper * 1.14)
    ticks = list(range(0, upper + 1, 10))
    ax.set_xticks(ticks, [f"{t}%" for t in ticks])
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_title(
        "Imitation learning from scratch: task success\n"
        "bar = all seeds pooled (95% CI) · dot = one training seed\n"
        "blue = regression (MLP) · orange = diffusion (1D U-Net)",
        loc="left",
        color=INK,
        fontsize=11,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def main() -> None:
    data = load(RESULTS / "episodes.jsonl")
    training = [json.loads(line) for line in (RESULTS / "training.jsonl").open() if line.strip()]
    lines = ["# Imitation learning from scratch (PushT, state observations)", ""]
    any_rec = next((t for t in training if "train_demos" in t), None)
    if any_rec:
        lines += [
            f"Trained on {any_rec['train_demos']} human demonstrations "
            f"({any_rec['train_pairs']} training pairs); {any_rec['val_demos']} held-out "
            "demonstrations select the best checkpoint by validation L1.",
            "",
        ]
    lines += [
        "| Variant | Training seeds | Episodes | Success (95% CI) | Per-seed success "
        "| Mean final progress |",
        "|---|---|---|---|---|---|",
    ]
    for name, recs in data.items():
        wins = sum(r["success"] for r in recs)
        lo, hi = wilson_interval(wins, len(recs))
        seeds = per_seed_rates(recs)
        progress = sum(r["final_progress"] for r in recs) / len(recs)
        lines.append(
            f"| {LABELS.get(name, name)} | {len(seeds)} | {len(recs)} | {pct(wins / len(recs))} "
            f"({pct(lo)}–{pct(hi)}) | {', '.join(pct(v) for v in seeds.values())} | "
            f"{progress:.2f} |"
        )

    lines += [
        "",
        "## Paired comparisons (same training-seed index and scene)",
        "",
        "| Baseline → variant | Pairs | Fixed | Broke | p (exact McNemar) |",
        "|---|---|---|---|---|",
    ]
    # Only the comparisons that answer a question: does chunking help (vs single-step BC),
    # does chunk length matter (vs chunk 8), and does ensembling help (same chunk length).
    questions = [("bc_single_step", v) for v in data if v != "bc_single_step"]
    questions += [("chunk8", v) for v in ("chunk4", "chunk16") if v in data]
    questions += [(f"chunk{k}", f"chunk{k}_ensemble") for k in (8, 16)]
    # Does a generative model beat regression? Matched (same horizon and execution),
    # and against the best regression policy.
    questions += [("chunk16_exec8", "diffusion"), ("chunk8_ensemble", "diffusion")]
    for a, b in questions:
        if a not in data or b not in data:
            continue
        n, fixed, broke = paired(data[a], data[b])
        lines.append(
            f"| {LABELS.get(a, a)} → {LABELS.get(b, b)} | {n} | {fixed} | {broke} | "
            f"{mcnemar_exact(fixed, broke):.2g} |"
        )

    lines += [
        "",
        "## Training",
        "",
        "| Variant | Seed | Validation loss | Seconds |",
        "|---|---|---|---|",
    ]
    for t in training:
        val = (
            f"{t['best_val_l1']:.4f} (best L1)"
            if "best_val_l1" in t
            else f"{t['final_val_denoising_mse']:.4f} (denoising MSE, final)"
        )
        lines.append(
            f"| {LABELS.get(t['variant'], t['variant'])} | {t['seed']} | {val} | "
            f"{t['seconds']:.0f} |"
        )
    lines += [
        "",
        "Validation losses are not comparable across variants: predicting 16 future actions "
        "is a harder target than predicting 1, and a denoising loss measures something else "
        "entirely. Task success is the comparison that counts.",
        "",
    ]

    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    chart(data, RESULTS / "success_by_variant.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
