"""
Train an imitation-learning policy on PushT demonstrations.

    python -m policy_reliability.imitation.train --variant chunk8 --seed 0

Saves models/imitation/<variant>_seed<seed>.pt (the checkpoint with the best
validation loss) and appends training curves to results/imitation/training.jsonl.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from policy_reliability.imitation.data import load_pusht_demos, make_windows, split
from policy_reliability.imitation.model import VARIANTS, MLPPolicy, build_policy

ROOT = Path(__file__).resolve().parents[2]


def train(
    variant_name: str,
    seed: int,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
) -> dict:
    variant = VARIANTS[variant_name]
    torch.manual_seed(seed)
    np.random.seed(seed)

    demos = load_pusht_demos()
    # The demo split is fixed (seed 0) for every variant and training seed, so all
    # models see exactly the same training and validation demonstrations.
    train_demos, val_demos = split(demos, val_fraction, seed=0)
    x_tr, y_tr = make_windows(train_demos, variant.history, variant.chunk)
    x_va, y_va = make_windows(val_demos, variant.history, variant.chunk)
    x_tr_t, y_tr_t = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    x_va_t, y_va_t = torch.from_numpy(x_va), torch.from_numpy(y_va)

    if variant.kind == "diffusion":
        return train_diffusion(variant, seed, x_tr_t, y_tr_t, x_va_t, y_va_t, epochs, batch_size)

    model = MLPPolicy(variant)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(x_tr) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    loss_fn = nn.L1Loss()  # L1, as in ACT: less pulled towards the mean of two modes than MSE

    best_val, best_state, curve = math.inf, None, []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(steps_per_epoch):
            batch = perm[i * batch_size : (i + 1) * batch_size]
            loss = loss_fn(model(x_tr_t[batch]), y_tr_t[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_va_t), y_va_t).item()
        curve.append({"epoch": epoch + 1, "train_l1": total / len(x_tr), "val_l1": val})
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    path = save(model, best_state, variant_name, seed)
    return {
        "variant": variant_name,
        "seed": seed,
        "train_demos": len(train_demos.episodes()),
        "val_demos": len(val_demos.episodes()),
        "train_pairs": len(x_tr),
        "best_val_l1": best_val,
        "seconds": round(time.time() - started, 1),
        "curve": curve,
        "checkpoint": str(path.relative_to(ROOT)),
    }


def save(model: nn.Module, state: dict, variant_name: str, seed: int) -> Path:
    out_dir = ROOT / "models" / "imitation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant_name}_seed{seed}.pt"
    torch.save({"variant": model.config(), "state_dict": state}, path)
    return path


def train_diffusion(variant, seed, x_tr, y_tr, x_va, y_va, epochs, batch_size) -> dict:
    """
    Diffusion Policy recipe: AdamW, cosine schedule with warm-up, EMA weights (learning
    rate 3e-4 rather than 1e-4, to fit a short CPU training budget).
    There is no "best checkpoint": a lower denoising loss doesn't mean better actions,
    so (as in Diffusion Policy) the final EMA weights are evaluated.
    """
    model = build_policy(variant)
    opt = torch.optim.AdamW(model.denoiser.parameters(), lr=3e-4, weight_decay=1e-6)
    steps_per_epoch = math.ceil(len(x_tr) / batch_size)
    total, warmup = epochs * steps_per_epoch, 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))),
    )
    curve, started, step = [], time.time(), 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        running = 0.0
        for i in range(steps_per_epoch):
            batch = perm[i * batch_size : (i + 1) * batch_size]
            loss = model.loss(x_tr[batch], y_tr[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            # EMA warm-up (as in Diffusion Policy): average less while weights move fast.
            model.update_ema(min(0.9999, 1 - (1 + step) ** -0.75))
            running += loss.item() * len(batch)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                torch.manual_seed(0)  # same noise every time, so the curve is comparable
                val = model.loss(x_va, y_va).item()
            curve.append({"epoch": epoch + 1, "train_mse": running / len(x_tr), "val_mse": val})
            print(f"  epoch {epoch + 1}: train {running / len(x_tr):.4f} val {val:.4f}", flush=True)
    path = save(model, model.state_dict(), variant.name, seed)
    return {
        "variant": variant.name,
        "seed": seed,
        "train_pairs": len(x_tr),
        "final_val_denoising_mse": curve[-1]["val_mse"],
        "seconds": round(time.time() - started, 1),
        "curve": curve,
        "checkpoint": str(path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    record = train(args.variant, args.seed, epochs=args.epochs)
    log = ROOT / "results" / "imitation" / "training.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(record) + "\n")
    loss = record.get("best_val_l1", record.get("final_val_denoising_mse"))
    print(
        f"{args.variant} seed={args.seed}: val loss {loss:.4f} "
        f"({record['seconds']:.0f}s, {record['train_pairs']} pairs)",
        flush=True,
    )


if __name__ == "__main__":
    main()
