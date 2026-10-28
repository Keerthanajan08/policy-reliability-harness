# Results

| Condition | Episodes | Success rate (95% CI) | Median time to success | Mean final progress |
|---|---|---|---|---|
| nominal | 15 | 53% (30%–75%) | 17.2 s | 0.93 |
| nominal (ddpm100) | 15 | 80% (55%–93%) | 18.1 s | 0.94 |

## Sampler comparison (same seeds)

Paired by seed against the default sampler (DDIM, 10 steps). *Fixed*: failed with the default, succeeded with the variant; *broke*: the reverse. p is a two-sided exact McNemar test.

| Variant | Paired seeds | Both succeed | Fixed | Broke | Both fail | p |
|---|---|---|---|---|---|---|
| nominal (ddpm100) | 15 | 8 | 4 | 0 | 3 | 0.125 |

## How episodes failed

A failure ending at progress >= 0.9 is a *near miss* (the T is almost placed but never crosses the 95% bar); anything lower means the policy *lost the block*. Peak disagreement is the highest rolling value after warm-up.

| Condition | Failures | Near misses | Lost the block | Median peak disagreement: successes / near misses / lost |
|---|---|---|---|---|
| nominal | 7 | 5 | 2 | 0.0167 / 0.0152 / 0.0309 |
| nominal (ddpm100) | 3 | 1 | 2 | – / – / – |

## Failure prediction

Early warning uses mean self-disagreement over the first 6 decisions (~4.8 s). The alarm fires when the 3-decision rolling mean exceeds 0.0373 after a 2-decision warm-up; the threshold allows at most 10% of nominal successes to trigger it.

| Condition | Early-warning AUROC | Failures flagged | Median warning before end | False alarms on successes |
|---|---|---|---|---|
| nominal | 0.34 | 2/7 | 20.0 s | 1/8 |
| all | 0.34 | 2/7 | 20.0 s | 1/8 |

Nominal false alarms are in-sample (the threshold was calibrated on those episodes); the perturbed conditions are out-of-sample.
