# Steps 2 & 3 — Running Results Log

> Plain-language log of what we did, why, how, and what we got.
> Models: **TinyLLaVA-3.1B** and **MobileVLM-V2-3B**. Machine: danavis3 (4 GPUs).
> Last updated: 2026-06-19. Status: **Steps 2A, 2B, 3A, 3B all done.** (Step 5
> adds confidence intervals / significance tests on top of these.)

## TL;DR (the headlines so far)

```
  1. Composite NEVER beats the old per-frame vote. It is a touch WORSE on
     counting for both models, worse across the board for TinyLLaVA, and only
     TIES the vote for MobileVLM on general questions.
        counting:        vote 57.1%   vs   composite@8 ~53%   (both models)
        all descriptive: vote 77.3/76.4 vs composite 72.7/76.3
     -> "show all frames at once" is not the fix. Step 4 (symbolic repair)
        is still worth doing.

  2. More frames = flat. 1, 2, 4, 8, 16 frames all give ~53-54%.
     MobileVLM literally BREAKS at 16 frames (runs out of context -> gibberish).

  3. VisDrone numeric counting: both models are poor, but fail in OPPOSITE ways.
        TinyLLaVA  -> barely counts: 66% of its answers are "0" (always wrong, the
                      object is present) and it REFUSES 27% (esp. crowded scenes).
                      Its low MAE 2.69 is an ARTIFACT of guessing ~0 on small counts.
        MobileVLM  -> always answers; exactly right 2x more often, but OVER-counts
                      with round-number defaults ("10" 35%, "100" 5%) -> MAE 7.80.
```

---

## The big picture — what these two steps ask

```
  Step 1 already showed: the models often SEE the right count in a single
  frame, but the per-frame VOTE throws that information away.

  Step 2 + 3 ask two separate questions:

  ┌─ NExT-QA (video counting) ──────────────────────────────────────┐
  │  Does showing ALL frames in ONE prompt ("composite") beat the    │
  │  old one-frame-at-a-time vote?   And how many frames is enough?  │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ VisDrone (aerial counting) ────────────────────────────────────┐
  │  Stop using A/B/C/D buckets. Just ask "how many Xs?" and read    │
  │  the number. How far off is it (MAE / RMSE)?                     │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Two ways to ask a counting question (the core idea of Step 2A/3A)

```
  OLD WAY — per-frame majority vote (the existing baseline)
  ┌──────┐ ┌──────┐ ┌──────┐        each frame answered ALONE,
  │frame1│ │frame2│ │ ...8 │        then we count the votes
  └──┬───┘ └──┬───┘ └──┬───┘
     │ "C"    │ "E"    │ "E"   ──►  majority = "E"
     └────────┴────────┘            (information from any single
                                     good frame can be out-voted)

  NEW WAY — composite (all frames, one answer)
  ┌──────┬──────┬──────┬─────┐
  │frame1│frame2│ .... │frame8│  all shown at once, model gives
  └──────┴──────┴──────┴─────┘   ONE answer using everything
            │
            └──►  "E"            (this is how the model was designed
                                  to be used)
```

We feed N frames into a single prompt by putting N image placeholders before
the question, then ask for one letter. Both models are LLaVA-style and accept
several images in one prompt natively (verified before running).

---

## Step 3A — does adding more frames help? (frame sweep)

We run the composite method at **1, 2, 4, 8, 16 frames** on the 177 "how many"
questions. Same questions each time, only the number of frames changes. This
gives both:
- the **8-frame** column → that *is* the Step 2A counting number, and
- the **whole curve** → Step 3A (does accuracy peak, climb, or stay flat?).

```
   accuracy
     ^
     |              ?   ?            <- we do NOT assume a peak exists;
     |        ?                         we report the real shape
     |   ?
     +---+---+---+---+---+--> frames
         1   2   4   8   16
```

**Result — accuracy (%) on the 177 counting questions, by frame count:**

| frames | TinyLLaVA | MobileVLM |
|---:|---:|---:|
| 1  | 54.2 | 53.7 |
| 2  | 52.0 | 54.2 |
| 4  | 53.1 | 54.2 |
| 8  | 53.1 | 53.7 |
| 16 | 54.2 | **7.3 (broken)** |

```
  accuracy
   57 |········ vote baseline (57.1%) ················
   55 |
   54 |  T        T                        T          <- TinyLLaVA: dead flat
   53 |  M  T  M  M  T M  T,M
   52 |        T
      |                                    M = 7.3 (16f, MobileVLM BROKE)
      +----+----+----+----+----+
          1    2    4    8   16  frames
```

**Shape: FLAT.** Adding frames does not help either model count. There is no
peak. The honest reading: for these small models, the bottleneck is not how
much video they see — it is their counting ability per se.

**MobileVLM @ 16 frames is not a real data point — it is a failure mode.**
16 images' worth of vision tokens overflow MobileVLM's context window, so it
emits gibberish ("The beach and The beach and a a a 26 …") and only 94/177
replies even contained a letter. We report it as "breaks at 16," not as 7.3%
accuracy. TinyLLaVA handles 16 frames fine.

```
  Why MobileVLM breaks but TinyLLaVA does not:
     each frame  ->  a fixed block of vision tokens
     16 frames   ->  block x 16  >  MobileVLM's context limit  ->  overflow
     (at 8 frames it still fits, which is why <=8 works for both)
```

---

## Step 2A — composite vs the old vote (full picture)

After the sweep we also run composite @ 8 frames on **all 777** descriptive
questions, and compare to the old per-frame-vote baseline on the same questions.

**Counting subset (177 questions) — the gate question:**

| method | TinyLLaVA | MobileVLM |
|---|---:|---:|
| per-frame vote (old baseline) | **57.1** | **57.1** |
| composite @ 8 frames | 53.1 | 53.7 |
| difference | −4.0 | −3.4 |

**Gate answer: composite does NOT close the counting gap.** It is slightly
*worse* than the vote. (Whether −4 points is statistically meaningful on 177
questions is a Step 5 question — it is small and may well be noise; either way
composite clearly does not *fix* counting.)

**All 777 descriptive questions:**

| method | TinyLLaVA | MobileVLM |
|---|---:|---:|
| per-frame vote (old baseline) | **77.3** | **76.4** |
| composite @ 8 frames | 72.7 | 76.3 |
| difference | −4.6 | −0.1 |

**Full Step 2A picture (composite − vote):**

```
              counting        all descriptive
  TinyLLaVA    -4.0              -4.6        <- composite worse on both
  MobileVLM    -3.4              -0.1        <- worse on counting, tied overall
```

**Reading:** composite never wins. For MobileVLM it only *ties* the vote on
general questions and loses on counting; for TinyLLaVA it loses across the board.
The simple per-frame majority vote — which effectively ensembles an answer from
each frame — is as good as or better than feeding all frames at once. So the
"use the model the way it was designed" hypothesis does not pay off here, and
counting in particular is the worst case for composite. (Step 5 will attach CIs /
McNemar significance; the −0.1 and −3 to −5 point gaps are small, but the
direction is consistent and composite clearly does not *improve* anything.)

---

## Step 2B — VisDrone numeric counting

The old VisDrone counting used buckets (0,1,2,3,"4 or more"). Step 0 showed that
is weak (a model can score ~38% just by always saying "1"). So we switch to a
real counting task:

```
   prompt per image:
     "How many cars are in this image? Answer with a number only."

   read the number  ──►  compare to the TRUE count from the labels
                          |
                          ├─ MAE  = average |guess - truth|
                          ├─ RMSE = punishes big misses harder
                          └─ parse-failure rate = how often the reply
                             had no usable number (NEVER counted as 0)
```

### Where the true counts come from (and how we know they're right)

The questions file only stored the bucket, not the exact number — useless for
MAE on the "4 or more" cases. So we recomputed exact counts from the VisDrone
label files (downloaded the DET-val set, 548 images + labels).

```
  YOLO label line:  "3  0.46 0.57 0.028 0.085"
                      │   │    │     └─w──┘ └h┘  (fractions of the image)
                      │   center x,y
                      └─ class id (0..9) = VisDrone class 1..10

  area of a box = w * h   (already a fraction of the image)
  keep boxes with area >= 0.001  (same filter the questions used)
  count the boxes of the asked class  ->  TRUE count
```

**Sanity check (passed):** for every question where the true count is 0–3 (so the
bucket is exact), our recomputed number matched the stored bucket — **362/362**.
And all 183 "4 or more" questions recomputed to ≥4. So the counts are correct.
True counts range **1–39**, average **4.59** per image.

### Results — MAE / RMSE / parse-failure

| model | MAE | RMSE | parse-failure | n scored |
|---|---:|---:|---:|---:|
| TinyLLaVA | 2.69 | 7.14 | **26.8%** | 399 / 545 |
| MobileVLM | 7.80 | 20.69 | 0.0% | 545 / 545 |

**Do NOT read this as "TinyLLaVA wins."** The two numbers are not comparable yet,
because TinyLLaVA only answered 73% of the questions. Here is what is really
going on:

```
  TinyLLaVA: when the scene is crowded it says "Many" / "I cannot give an exact
  number" and gives NO number at all.
     avg true count where it ANSWERED  = 2.81   (easy, sparse scenes)
     avg true count where it REFUSED   = 9.47   (hard, crowded scenes)
  -> its low MAE is measured only on the EASY images it chose to answer.

  MobileVLM: always commits to a number (0% refusal), so it is scored on
  everything, including the hard crowded scenes -> bigger average error.
```

**Fair comparison — same 399 images where BOTH gave a number:**

| model | MAE | RMSE |
|---|---:|---:|
| TinyLLaVA | 2.69 | 7.14 |
| MobileVLM | 4.09 | 12.57 |

On equal footing TinyLLaVA is still better *when it commits*, but it dodges the
hard cases. (This is exactly why the plan says never treat a refusal as 0 — doing
so would have falsely given TinyLLaVA an MAE of ~4.5 and hidden the refusal
behaviour.)

### How they fail (signed error = guess − truth)

```
  TinyLLaVA   bias = -1.04   guesses LOW.   262 of 399 answers were "0".
  MobileVLM   bias = +5.30   guesses HIGH.  defaults to round numbers:
                                            "10" x193,  "100" x26,  "20" x16
```

Both small VLMs are weak at precise aerial counting, but in opposite directions:
TinyLLaVA under-counts / refuses, MobileVLM over-counts with round-number guesses.

### How far off, and how often (error distributions)

Script: `scripts/visdrone/step2b_error_distribution.py`. "Answered" = questions
where the model gave a parseable number (TinyLLaVA 399/545, MobileVLM 545/545).

**Absolute error |guess − truth|, share of answered questions:**

```
                 TinyLLaVA                       MobileVLM
  exact (0)    9.8% ████▏                   19.1% ████████▌
  off by 1    44.9% ██████████████████████▍ 22.6% ██████████
  off by 2    18.5% █████████▎               9.9% ████▍
  off by 3-5  18.3% █████████▏              17.2% ███████▋
  off by 6-10  6.3% ███▏                    23.1% ██████████▎   <- big tail
  off by >10   2.3% █▏                       8.1% ███▌          <- big tail
```

**Cumulative — "how often close enough":**

| within | TinyLLaVA | MobileVLM |
|---|---:|---:|
| exact     | 9.8%  | **19.1%** |
| ±1        | **54.6%** | 41.7% |
| ±2        | **73.2%** | 51.6% |
| ±5        | **91.5%** | 68.8% |
| off by >5 | 8.6%  | **31.2%** |

**Signed error (− = under-count, + = over-count):**

```
  TinyLLaVA  bias -1.04 (UNDER)         MobileVLM  bias +5.30 (OVER)
   -5..-3   █████████████████ 17.0%      -5..-3  ████████████████ 8.4%
       -2   █████████████████ 16.8%          -2  ████████████████ 8.3%
       -1   ████████████████████████ 40.6%   -1  █████████████████████████ 14.7%
  0 exact   ██████ 9.8%                  0 exact  ███████████████████████████████ 19.1%
       +1   ████  4.3%                        +1  ████████████████ 7.9%
   +3..+5   ▏ 1.3%                        +3..+5  █████████████████ 8.8%
  +6..+10   ▏ 1.8%                       +6..+10  ████████████████████████████ 20.2%  <- overshoot
 way over   ▏ 1.3%                      way over  ███████████ 5.3%   <- overshoot
```

**What numbers they actually output (top guesses):**

```
  TinyLLaVA (true mean 2.8)        MobileVLM (true mean 4.6)
    "0"  65.7%  <- always WRONG      "10"  35.4%  <- default "lots"
    "1"  11.8%     (objects are        "1"  20.4%
    "2"   7.8%      always present,     "0"  15.6%
    "10"  4.3%      so true >= 1)       "2"  14.7%
    "3"   4.0%                        "100"  4.8%  <- default "tons"
```

**Refusal is count-dependent (TinyLLaVA):**

```
  true count | refusal rate
     1-2      | 13%  ░░░░░
     3-5      | 27%  ░░░░░░░░░░
     6-10     | 40%  ░░░░░░░░░░░░░░░
     11+      | 81%  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   <- bails out when it's hard
```

**What this means (do not mis-read the MAE):**
- **TinyLLaVA barely counts.** 66% of its committed answers are "0", and since
  every asked object is present (true ≥ 1) *every one of those is wrong*. With
  the 27% refusals on top, it almost never gives a useful count. Its low MAE
  (2.69) is an artifact — true counts are small (mean 2.8), so guessing ≈0 is
  only "off by 1–2."
- **MobileVLM actually tries** and is bimodal: exactly right twice as often
  (19.1% vs 9.8%) on sparse scenes, but dumps round-number defaults ("10"/"100")
  on crowded scenes — that fat +6-and-beyond tail is what drives MAE 7.80 /
  RMSE 20.69.

---

## Step 3B — VisDrone object-size sweep

**Idea:** is the error caused by tiny objects the models cannot see? We keep only
true objects whose box is bigger than a size threshold, recompute the true count,
and compare to the SAME model guess. The guess is fixed; only the truth shrinks
as we raise the threshold. Thresholds: 0.05%, 0.1%, 0.2%, 0.5%, 1% of image area.

| threshold | TinyLLaVA MAE | MobileVLM MAE |
|---|---:|---:|
| 0.05% | 3.57 | 7.74 |
| 0.1%  | 2.69 | 7.80 |
| 0.2%  | 1.95 | 8.43 |
| 0.5%  | 1.71 | 9.20 |
| 1.0%  | 1.65 | 9.60 |

```
   MAE                                  the two models move OPPOSITE ways:
   10 |                         M  M
    8 | M  M  M                            MobileVLM: error RISES as we drop
    6 |                                    small objects -> it OVER-counts, so
    4 | T                                  smaller truth = wider gap.
    2 |    T  T  T  T
    0 +----+----+----+----+----+           TinyLLaVA: error FALLS -> it
       .05  .1  .2  .5   1   threshold %   UNDER-counts (misses small objects),
                                           so dropping them helps it match.
```

**Shape: opposite slopes, no cliff.** Excluding small objects helps TinyLLaVA
(confirms it misses small things) and hurts MobileVLM (confirms it over-guesses).
There is no single threshold where either error suddenly collapses. RMSE stays
high throughout for both — the occasional huge miss is not a small-object effect.

> Caveat on this sweep: it changes what counts as GROUND TRUTH (drop GT objects
> below the cutoff) while leaving the model's guess fixed. So it measures "does
> the fixed guess match a smaller truth," not "are small objects hard to see."
> The cut below answers the latter directly.

### Step 3B (direct cut) — error vs the ACTUAL size of the counted objects

Instead of moving a GT cutoff, here we bin each question by the median size of
the objects being counted, and measure error per band. Script:
`scripts/visdrone/step3b_error_by_objsize.py`.

| object size (% img) | TinyLLaVA MAE | MobileVLM MAE | mean true count |
|---|---:|---:|---:|
| 0.1-0.2% | 3.46 | 4.52 | ~6.0 |
| 0.2-0.5% | 2.32 | 2.70 | ~5.2 |
| 0.5-1%   | 2.38 | 1.59 | ~3.5 |
| 1-2%     | 0.94 | 1.15 | ~3.4 |
| >2%      | 0.80 | 0.60 | ~1.9 |

```
   MAE
   4.5 | M                          BOTH models improve sharply as the
   3.5 | T                          objects get bigger:
   2.5 |    M  T                       MobileVLM 4.52 -> 0.60
   1.5 |       T  M                     TinyLLaVA 3.46 -> 0.80
   0.5 |             T  M  TM
   0.0 +----+----+----+----+----
       0.1- 0.2- 0.5- 1-2% >2%   object size (% of image)
       0.2% 0.5% 1%
```

**Small objects are the hard cases — clearly, for both models** (error falls
~4-7x from smallest to largest band). On big (and sparse) objects both are nearly
exact. **Confound:** small-object scenes are also crowded (mean true count drops
6.0 -> 1.9 across the bands), so "small" and "many" are entangled and this cut
cannot separate them. No contradiction with the threshold sweep above: that one
shrank the GT under a fixed inflated guess; this one groups by the real scene.

---

## Run status

| Run | What | GPU | Status |
|---|---|---|---|
| A | NExT-QA counting frame sweep {1,2,4,8,16}, both models | 0 (Tiny), 1 (Mobile) | ✅ done |
| B | NExT-QA all 777 descriptive, composite @8, both models | 0+1 (after A) | ✅ done |
| C | VisDrone numeric counting, both models | 2 (Tiny), 3 (Mobile) | ✅ done |
| 3B | VisDrone size sweep (CSV analysis) | — | ✅ done |

## Output files

| File | What |
|---|---|
| `results/step3/step3a_{model}_framesweep.csv` | per-question composite results at 1/2/4/8/16 frames |
| `results/step2/step2a_{model}_composite8.csv` | composite@8 on all 777 descriptive |
| `results/step2/step2a_composite_vs_vote.csv` | the Step 2A comparison table |
| `results/step2/step2b_{model}_numeric.csv` | VisDrone numeric per-image guesses + parse flag |
| `results/step2/visdrone_counting_gt.csv` | re-derived true counts (+ per-object areas for 3B) |
| `results/step3/step3b_size_sweep.csv` | MAE/RMSE vs object-size threshold |
