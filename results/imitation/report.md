# Imitation learning from scratch (PushT, state observations)

Trained on 185 human demonstrations (23214 training pairs); 21 held-out demonstrations select the best checkpoint by validation L1.

| Variant | Training seeds | Episodes | Success (95% CI) | Per-seed success | Mean final progress |
|---|---|---|---|---|---|
| Single-step BC | 3 | 300 | 1% (0%–3%) | 2%, 1%, 0% | 0.24 |
| Chunking (4) | 3 | 300 | 1% (0%–3%) | 1%, 2%, 0% | 0.42 |
| Chunking (8) | 3 | 300 | 21% (16%–26%) | 19%, 18%, 25% | 0.54 |
| Chunking (16) | 3 | 300 | 5% (3%–8%) | 4%, 2%, 8% | 0.23 |
| Chunking (8) + ensembling | 3 | 300 | 25% (21%–31%) | 38%, 13%, 25% | 0.62 |
| Chunking (16) + ensembling | 3 | 300 | 5% (3%–8%) | 2%, 2%, 11% | 0.31 |
| Regression, plan 16 / run 8 | 3 | 300 | 13% (10%–17%) | 6%, 5%, 28% | 0.53 |
| Diffusion, plan 16 / run 8 | 3 | 300 | 60% (54%–65%) | 65%, 61%, 53% | 0.89 |

## Paired comparisons (same training-seed index and scene)

| Baseline → variant | Pairs | Fixed | Broke | p (exact McNemar) |
|---|---|---|---|---|
| Single-step BC → Chunking (4) | 300 | 3 | 3 | 1 |
| Single-step BC → Chunking (8) | 300 | 61 | 2 | 4.4e-16 |
| Single-step BC → Chunking (16) | 300 | 14 | 3 | 0.013 |
| Single-step BC → Chunking (8) + ensembling | 300 | 75 | 2 | 4e-20 |
| Single-step BC → Chunking (16) + ensembling | 300 | 15 | 3 | 0.0075 |
| Single-step BC → Regression, plan 16 / run 8 | 300 | 39 | 3 | 5.6e-09 |
| Single-step BC → Diffusion, plan 16 / run 8 | 300 | 177 | 1 | 9.3e-52 |
| Chunking (8) → Chunking (4) | 300 | 2 | 61 | 4.4e-16 |
| Chunking (8) → Chunking (16) | 300 | 11 | 59 | 4.5e-09 |
| Chunking (8) → Chunking (8) + ensembling | 300 | 52 | 38 | 0.17 |
| Chunking (16) → Chunking (16) + ensembling | 300 | 14 | 13 | 1 |
| Regression, plan 16 / run 8 → Diffusion, plan 16 / run 8 | 300 | 153 | 13 | 1.7e-31 |
| Chunking (8) + ensembling → Diffusion, plan 16 / run 8 | 300 | 133 | 30 | 1.1e-16 |

## Training

| Variant | Seed | Validation loss | Seconds |
|---|---|---|---|
| Single-step BC | 0 | 0.0174 (best L1) | 90 |
| Chunking (8) | 0 | 0.0536 (best L1) | 79 |
| Chunking (16) + ensembling | 0 | 0.0772 (best L1) | 82 |
| Single-step BC | 1 | 0.0175 (best L1) | 80 |
| Chunking (8) | 1 | 0.0525 (best L1) | 85 |
| Chunking (16) + ensembling | 1 | 0.0767 (best L1) | 86 |
| Single-step BC | 2 | 0.0175 (best L1) | 88 |
| Chunking (8) | 2 | 0.0529 (best L1) | 89 |
| Chunking (16) + ensembling | 2 | 0.0767 (best L1) | 92 |
| Chunking (4) | 0 | 0.0344 (best L1) | 75 |
| Chunking (16) | 0 | 0.0772 (best L1) | 74 |
| Chunking (8) + ensembling | 0 | 0.0536 (best L1) | 74 |
| Chunking (4) | 1 | 0.0346 (best L1) | 73 |
| Chunking (16) | 1 | 0.0767 (best L1) | 76 |
| Chunking (8) + ensembling | 1 | 0.0525 (best L1) | 76 |
| Chunking (4) | 2 | 0.0346 (best L1) | 81 |
| Chunking (16) | 2 | 0.0767 (best L1) | 81 |
| Chunking (8) + ensembling | 2 | 0.0529 (best L1) | 76 |
| Regression, plan 16 / run 8 | 0 | 0.0772 (best L1) | 49 |
| Diffusion, plan 16 / run 8 | 0 | 0.0416 (denoising MSE, final) | 3487 |
| Regression, plan 16 / run 8 | 1 | 0.0767 (best L1) | 49 |
| Diffusion, plan 16 / run 8 | 1 | 0.0397 (denoising MSE, final) | 3535 |
| Regression, plan 16 / run 8 | 2 | 0.0767 (best L1) | 52 |
| Diffusion, plan 16 / run 8 | 2 | 0.0409 (denoising MSE, final) | 3683 |

Validation losses are not comparable across variants: predicting 16 future actions is a harder target than predicting 1, and a denoising loss measures something else entirely. Task success is the comparison that counts.
