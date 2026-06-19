# Step 4 — Symbolic Repair

> Plain-language explanation first (exactly as walked through), then results.
> CSV-only analysis, no GPU — reuses the per-frame text saved in Step 1.
> Models: TinyLLaVA-3.1B and MobileVLM-V2-3B. Last updated: 2026-06-19.

---

## The problem we're trying to fix

Step 1 already taught us something important. When we ask a counting question about a video, the model looks at each frame on its own. Often, **at least one frame the model looks at, it actually says the right number** — but the old method throws that away.

```
  THE OLD METHOD ("vote on the letters")

  Question: "How many people?"   Options: A=1  B=2  C=3  D=4  E=5

  frame 1 -> model picks "B"  (2 people)
  frame 2 -> model picks "E"  (5 people)   <- this one is RIGHT
  frame 3 -> model picks "B"  (2 people)
  frame 4 -> model picks "B"  (2 people)
  frame 5 -> model picks "B"  (2 people)
                     |
                     v
        count the letters:  B wins (4 votes)
        FINAL ANSWER = B (2)   ... WRONG. Truth was 5.

  The one correct frame got out-voted and lost.
```

Step 1 measured exactly how often the right number is hiding in the model's own words: **TinyLLaVA 37%, MobileVLM 43%** of the wrong answers were actually recoverable. That's a lot of free accuracy sitting on the table.

## What Step 4 does

Instead of voting on letters, we **read the actual numbers the model said in each frame**, and combine them with a simple rule. No new model, no extra detector — just reusing the words the model already produced in Step 1.

```
  THE NEW METHOD ("read the numbers, then combine")

  Question: "How many people?"

  frame 1 -> model says "...there are 2 people..."   -> 2
  frame 2 -> model says "...I see 5 people..."        -> 5
  frame 3 -> model says "...2 people here..."         -> 2
  frame 4 -> model says "...about 2..."               -> 2
  frame 5 -> model says "...5 people..."              -> 5
                     |
                     v
            list of numbers:  [2, 5, 2, 2, 5]
                     |
                     v
            combine with a simple rule
                     |
                     v
            map the result to the nearest option (A..E)
```

## The three simple "combine" rules we test

```
   numbers seen across frames:  [2, 5, 2, 2, 5]

   ┌─────────┬──────────────────────────────┬────────┐
   │  RULE   │ what it means                 │ result │
   ├─────────┼──────────────────────────────┼────────┤
   │ MODE    │ the most common number        │   2    │
   │ MEDIAN  │ the middle number             │   2    │
   │ MAX     │ the biggest number seen        │   5    │ <- right!
   └─────────┴──────────────────────────────┴────────┘
```

Why try `MAX`? Counting models usually **miss** things (objects hidden, blurry, off-screen in some frames), they rarely **invent** extra ones. So the frame that saw the *most* is often the one that saw *all* of them. That's a hypothesis Step 4 tests, not a given.

## What we compare it against

```
   On the SAME 177 counting questions, line up four numbers:

   ┌────────────────────────┬──────────────────────────────────┐
   │ old letter vote        │  57.1%   (the baseline)           │
   │ composite (all frames) │  ~53%    (Step 2 — didn't help)   │
   │ symbolic repair: mode  │   ?                               │
   │ symbolic repair: median│   ?      <- Step 4 fills these in │
   │ symbolic repair: max   │   ?                               │
   └────────────────────────┴──────────────────────────────────┘
```

## What the outcomes would mean

```
  if repair  >>  vote   ->  "just reading the model's own numbers
                            recovers accuracy that voting threw away"
                            (a cheap, detector-free fix — a nice finding)

  if repair  ~=  vote   ->  the stated numbers are as confused as the
                            letters; the model is genuinely blind, not
                            mis-counted

  if repair  ~=  composite -> repair matches the 'proper' method for free
```

## Good news on cost

**We already have everything we need.** The per-frame text with the stated numbers was saved in Step 1 (`results/step1/`). So Step 4 is **pure number-crunching on existing files — no GPU, no re-running the models.** It's fast.

```
   Step 1 output  ──►  parse numbers per frame  ──►  apply 3 rules
   (already on disk)        (already done in           ──►  map to option
                             Step 1's analysis)          ──►  accuracy table
```

---

# Results

Script: `scripts/step4/step4_symbolic_repair.py`. Same 177 "how many" questions,
both models. "Oracle" = if *any* single frame's number maps to the correct option
(the best you could do with perfect frame-picking — a ceiling, not a usable
method).

**Getting the number out of each frame's text — multiple regex strategies.**
Rather than grab the first digit we see, the parser tries several patterns in
order of reliability and stops at the first hit:

```
  1. number glued to the asked object   "5 people" / "three cars"   (most reliable)
  2. explicit count phrase              "there are 5", "I see 5", "count is 5"
  3. the whole reply is a bare number   "5", "zero"
  4. fallback: the LAST number          (the prompt asks to state the count last)
  5. last resort: the FIRST number anywhere
```

We checked the robust parser against a naive "first number" parser across **all
2832 frames** (177 questions x 8 frames x 2 models): they agreed on **100%** of
frames. So on this data the model's per-frame replies are simple enough (usually
a bare number or a single "there are N ..." sentence) that number-extraction is
**not** a source of error — the parser is robust, and the results below do not
depend on the parsing choice.

| method | TinyLLaVA | MobileVLM |
|---|---:|---:|
| per-frame vote (baseline) | 57.1 | 57.1 |
| composite @ 8 frames | 53.1 | 53.7 |
| symbolic repair: mode | 53.1 | 57.1 |
| symbolic repair: median | 53.1 | 56.5 |
| **symbolic repair: max** | **59.3** | **58.2** |
| oracle: best single frame (ceiling) | 69.5 | 75.1 |

```
  Accuracy on the 177 counting questions (%)

  TinyLLaVA                              MobileVLM
  vote      ███████████████████ 57.1     vote      ███████████████████ 57.1
  composite █████████████████ 53.1       composite █████████████████ 53.7
  rep:mode  █████████████████ 53.1       rep:mode  ███████████████████ 57.1
  rep:median█████████████████ 53.1       rep:median██████████████████ 56.5
  rep:MAX   ████████████████████ 59.3 *  rep:MAX   ███████████████████ 58.2 *
  oracle    ███████████████████████ 69.5 oracle    ████████████████████████ 75.1
            └─ * MAX beats both vote and composite ─┘   └─ ceiling is far higher ─┘
```

## What we found

1. **The MAX rule wins — but only by a little.** Reading the model's own numbers
   and taking the biggest beats both the letter vote (+2.2 pts TinyLLaVA, +1.1
   MobileVLM) and the composite method. Mode and median do *not* help (they sit
   at or below the vote). This matches the hypothesis: the models **miss** objects
   rather than hallucinate extra ones, so the frame that saw the most is usually
   the most complete.

2. **The ceiling is much higher than any simple rule reaches.** If we could always
   pick the single best frame, accuracy would be **69.5% (TinyLLaVA) / 75.1%
   (MobileVLM)** — roughly +12 to +18 points over the vote. So the right answer
   really is present in the model's own words far more often than any of mode/
   median/max manages to extract. The simple rules leave most of that on the table.

```
   How much accuracy is "in the words" vs how much we actually grab:

   vote 57% ───────────────────────────────► what voting gets
   MAX  59% ─►  (+2)   simple rule grabs a sliver
   oracle 70-75% ─────────────────────────────────────► what is actually there
                  └──────────── big gap = headroom a smarter
                                aggregator could chase ────────┘
```

## What this means for the thesis

- **Detector-free repair gives a small, real gain** (MAX > vote > composite). It is
  essentially free (no GPU, no extra model), so it is worth reporting as the best
  simple aggregation.
- **But it is not a full fix.** The gap to the oracle (~12–18 pts) says the
  bottleneck is now *picking/trusting the right frame*, not the model's raw
  ability to see the count. That gap is the interesting story: the information is
  there, simple statistics can't reliably find it.
- Significance of the small +1 to +2 point gains is a **Step 5** question (these
  are ~2–4 questions out of 177 — likely within noise; the oracle gap is the
  robust finding).
