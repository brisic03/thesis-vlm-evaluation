# Step 5 — Are our numbers real? (statistics)

> Plain-language first, then the verdicts. CSV-only, no GPU. This step does not
> find anything new — it stamps each earlier finding as "real" or "could be luck".
> Script: `scripts/step5/step5_statistics.py`. Last updated: 2026-06-19.

## What this step checks (no jargon)

```
  With only 177 counting questions, getting ~4 more right/wrong by chance
  shifts the score ~2 points. So small gaps may mean NOTHING.

  Three tools:
   1. error bars  : a range each % really lives in   (95% confidence interval)
   2. flip-test   : on the SAME questions, did B truly beat A, or coin-toss?
                    (McNemar's test — looks only at questions that flipped)
   3. bootstrap   : a range for the VisDrone average-miss (MAE/RMSE)

  Golden rule: anything NOT real gets labelled "not significant".
```

---

## Result 1 — accuracy with error bars (NExT-QA)

```
  Counting (177 Qs)            Accuracy with 95% error bar
  vote          ├────────●────────┤   57.1%  (49.8 - 64.4)
  composite@8   ├────────●────────┤   53.1%  (45.8 - 60.5)
  repair-max    ├────────●────────┤   59.3%  (52.1 - 66.6)
  oracle        ├───────●───────┤     69.5%  (62.7 - 76.3)   <- TinyLLaVA
                45%      55%      65%      75%

  Every counting bar overlaps the next heavily, EXCEPT the oracle sits clearly
  to the right. That overlap is the whole story: the method differences are
  inside the noise; only the oracle (the ceiling) stands apart.
```

Full table in `results/step5/step5_accuracy_ci.csv`.

## Result 2 — the flip-tests (which differences are REAL)

```
  comparison (same questions)              diff     verdict
  ─────────────────────────────────────────────────────────────────────
  COUNTING (177)
  composite vs vote   (TinyLLaVA)          -4.0%    not significant (p=.21)
  composite vs vote   (MobileVLM)          -3.4%    not significant (p=.38)
  repair-max vs vote  (TinyLLaVA)          +2.3%    not significant (p=.66)
  repair-max vs vote  (MobileVLM)          +1.1%    not significant (p=.89)
  repair-mode/median  (both models)        ~0       not significant (p=1.0)
  TinyLLaVA vs MobileVLM (vote)             0.0%    not significant (p=1.0)

  ALL-777
  composite vs vote   (MobileVLM)          -0.1%    not significant (p=1.0)
  composite vs vote   (TinyLLaVA)          -4.6%    *** SIGNIFICANT (p=.0001) ***
  TinyLLaVA vs MobileVLM (vote)            -0.9%    not significant (p=.57)

  THE HEADROOM (oracle = best single frame)
  oracle vs vote      (TinyLLaVA)         +12.4%    *** SIGNIFICANT (p=.0003) ***
  oracle vs vote      (MobileVLM)         +18.1%    *** SIGNIFICANT (p<.0001) ***
```

**Reading it:**
- **Almost every counting comparison is noise.** The composite penalty and the
  little repair gains (+1–2 pts) are NOT statistically real on 177 questions.
- **The two models are statistically tied** on NExT-QA — we cannot claim either
  is better.
- **One real NExT-QA negative:** for TinyLLaVA, composite is *significantly worse*
  than the vote on all 777 questions (−4.6%, p=0.0001). So "all frames at once"
  genuinely hurts TinyLLaVA's general QA — not just noise.
- **The oracle gap is rock-solid** (p≤0.0003 both models). The correct answer is
  in the model's own per-frame words **significantly** more often than voting (or
  any simple rule) recovers. This is the defensible Step 4 finding: the
  information is there; the bottleneck is reliably *picking the right frame*.

```
  vote 57% ─────────────────●  what voting gets
  repair  59% ─●  (+2, NOT significant — noise)
  oracle 70-75% ───────────────────────●  *** significantly higher ***
                └── this gap is REAL; the small repair bump is not ──┘
```

Full table in `results/step5/step5_mcnemar.csv`.

## Result 3 — VisDrone average-miss with bootstrap error bars

```
  MAE (lower = better)         95% bootstrap range
  TinyLLaVA  ├──●──┤            2.69   (2.14 - 3.42)
  MobileVLM        ├───●───┤    7.80   (6.27 - 9.45)
             2     4     6     8    10
  The two MAE ranges DO NOT overlap -> the gap is statistically real.

  RMSE (punishes big misses)   95% bootstrap range
  TinyLLaVA  ├────────●────────┤      7.14  (3.21 - 10.76)  <- very wide
  MobileVLM            ├─────●─────┤  20.69 (16.81 - 24.24)
```

**Important caveat (stated honestly):** the non-overlapping MAE does *not* mean
"TinyLLaVA counts better." TinyLLaVA only answered 399/545 questions (it refused
the crowded ones) and guessed "0" most of the time, so it is scored on an easier
subset with small errors. The statistic confirms the two numbers genuinely
differ; the Step 2B interpretation (refusal + under-count bias) is what explains
*why*. TinyLLaVA's RMSE range is very wide (3.2–10.8) because a handful of big
misses dominate. Full table in `results/step5/step5_visdrone_bootstrap.csv`.

---

## Bottom line — what we are allowed to claim

```
  CAN claim (statistically real):
   + Composite does NOT beat the vote (it's worse for TinyLLaVA on all-777, p=.0001).
   + The correct count is in the model's own words far more often than voting
     captures (oracle gap +12-18 pts, p<=.0003)  <- the strongest finding.
   + VisDrone MAE differs between models (CIs don't overlap), and the models fail
     in opposite directions (under- vs over-count) — robust qualitative result.

  CANNOT claim (within noise):
   - Symbolic repair beats the vote (the +1-2 pt gains are not significant).
   - One model counts better than the other on video (statistically tied).
   - Composite changes counting accuracy on the 177-question subset.
```
