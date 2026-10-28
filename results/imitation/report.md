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

## Paired comparisons (same training-seed index and scene)

| Baseline → variant | Pairs | Fixed | Broke | p (exact McNemar) |
|---|---|---|---|---|
| Single-step BC → Chunking (4) | 300 | 3 | 3 | 1 |
| Single-step BC → Chunking (8) | 300 | 61 | 2 | 4.4e-16 |
| Single-step BC → Chunking (16) | 300 | 14 | 3 | 0.013 |
| Single-step BC → Chunking (8) + ensembling | 300 | 75 | 2 | 4e-20 |
| Single-step BC → Chunking (16) + ensembling | 300 | 15 | 3 | 0.0075 |
| Chunking (8) → Chunking (4) | 300 | 2 | 61 | 4.4e-16 |
| Chunking (8) → Chunking (16) | 300 | 11 | 59 | 4.5e-09 |
| Chunking (8) → Chunking (8) + ensembling | 300 | 52 | 38 | 0.17 |
| Chunking (16) → Chunking (16) + ensembling | 300 | 14 | 13 | 1 |

## Training

| Variant | Seed | Best validation L1 | Seconds |
|---|---|---|---|
| Single-step BC | 0 | 0.0174 | 90 |
| Chunking (8) | 0 | 0.0536 | 79 |
| Chunking (16) + ensembling | 0 | 0.0772 | 82 |
| Single-step BC | 1 | 0.0175 | 80 |
| Chunking (8) | 1 | 0.0525 | 85 |
| Chunking (16) + ensembling | 1 | 0.0767 | 86 |
| Single-step BC | 2 | 0.0175 | 88 |
| Chunking (8) | 2 | 0.0529 | 89 |
| Chunking (16) + ensembling | 2 | 0.0767 | 92 |
| Chunking (4) | 0 | 0.0344 | 75 |
| Chunking (16) | 0 | 0.0772 | 74 |
| Chunking (8) + ensembling | 0 | 0.0536 | 74 |
| Chunking (4) | 1 | 0.0346 | 73 |
| Chunking (16) | 1 | 0.0767 | 76 |
| Chunking (8) + ensembling | 1 | 0.0525 | 76 |
| Chunking (4) | 2 | 0.0346 | 81 |
| Chunking (16) | 2 | 0.0767 | 81 |
| Chunking (8) + ensembling | 2 | 0.0529 | 76 |

Validation L1 is not comparable across variants: predicting 16 future actions is a harder target than predicting 1. Task success is the comparison that counts.
