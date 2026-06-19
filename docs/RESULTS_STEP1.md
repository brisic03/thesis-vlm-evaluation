# Step 1 Results — Can the model count, or does it just vote badly?

*Plain-language write-up of the Step 1 run on danavis3, 2026-06-19.*
*Models: TinyLLaVA-3.1B and MobileVLM V2-3B. Dataset: NExT-QA counting questions.*

---

## 1. The big question (the "why")

We are testing two small vision-language models on counting questions like
*"how many goats can be spotted?"*. They get a lot of these **wrong**.

But "wrong" can mean two very different things:

```
   REASON A: the model is BLIND              REASON B: the model can SEE,
   it never sees the right number            but the way we ADD UP its
   anywhere.                                  answers throws the right
                                              number away.
   ┌────────────────────────┐                ┌────────────────────────┐
   │  "how many goats?"      │                │  "how many goats?"      │
   │  model: 1, 2, 3, 1, 2   │                │  model: 8, 2, 8, 2, 8   │
   │  truth: 8               │                │  truth: 8               │
   │                         │                │                         │
   │  8 never appears.       │                │  8 IS there! But our    │
   │  Nothing to fix.        │                │  vote picked "2".       │
   └────────────────────────┘                └────────────────────────┘
        no hope of repair                          fixable!
```

If most mistakes are **Reason B**, then a cheap trick — read the numbers the
model actually said, and add them up more carefully — could rescue a lot of
wrong answers. That cheap trick is "Step 4" in our plan.

**Step 1 measures how many of the mistakes are Reason B.** We call that the
**recoverable fraction**.

---

## 2. How the model is used today (the baseline "how")

Each video is turned into **8 still frames**. The model looks at **each frame
on its own**, picks a letter (A–E), and we take the **majority vote**.

```
  VIDEO: "how many goats can be spotted?"   choices: A.eight B.two C.one D.three E.four
  ──────────────────────────────────────────────────────────────────────────

   frame1  frame2  frame3  frame4  frame5  frame6  frame7  frame8
   ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐
   │ 🐐 │  │🐐🐐│  │🐐..│  │🐐🐐│  │🐐🐐│  │🐐🐐│  │🐐🐐│  │🐐..│
   └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘
     C       D       B       B       B       B       B       D
    one    three    two     two     two     two     two    three
                                                              │
                                          majority vote ──────┘
                                          "two" wins (5 of 8)

   TRUE ANSWER = eight (A)        ✗  WRONG
```

On the counting subset this baseline scores:

```
   ┌──────────────┬───────────────┬─────────────────┬──────────────┐
   │ Model        │ how-many acc  │ how-many wrong  │ overall acc  │
   ├──────────────┼───────────────┼─────────────────┼──────────────┤
   │ TinyLLaVA-3B │    57.1%      │   76 of 177     │    77.3%     │
   │ MobileVLM-3B │    57.1%      │   76 of 177     │    76.4%     │
   └──────────────┴───────────────┴─────────────────┴──────────────┘
```

Both get exactly **76 counting questions wrong**. Those 76 are what Step 1
investigates.

---

## 3. What we changed for Step 1 (the new "how")

Instead of forcing a letter, we ask the model to **talk first, then count**.
Same 8 frames, but a different question:

```
  OLD prompt (pick a letter):
  ┌────────────────────────────────────────────────┐
  │ <image>                                          │
  │ Question: how many goats can be spotted?         │
  │ A.eight B.two C.one D.three E.four               │
  │ Answer with only the letter.                     │
  └────────────────────────────────────────────────┘
            ↓
          "B"          ← just a letter, no insight

  NEW prompt (describe, then count):
  ┌────────────────────────────────────────────────┐
  │ <image>                                          │
  │ Question: how many goats can be spotted?         │
  │ First briefly describe what you see in this      │
  │ frame. Then state the count as a number.         │
  └────────────────────────────────────────────────┘
            ↓
   "There are eight goats in the enclosure. Count: 8"
                                              ↑
                              now we can SEE the number it believes
```

We run this on **all 177** counting questions, for **both models**, and save
**the full text of every frame** (8 frames × 177 questions = 1,416 little
descriptions per model).

### The EXACT prompt text

The prompt is built per frame. `<image>` is replaced by the model's special
image token (`DEFAULT_IMAGE_TOKEN`). `{question}` is the question text from the
dataset, e.g. *"how many goats can be spotted"*.

**Both models use the identical prompt body** (`build_reasoning_prompt` in both
scripts):

```
<image>
Question: {question}
First briefly describe what you see in this frame. Then state the count as a number.
```

The two models differ only in how that body is wrapped:

```
  TinyLLaVA — wrapped with the "phi" chat template
  ────────────────────────────────────────────────────────────
  via TextPreprocess(tokenizer, 'phi'); the <image> token is
  IMAGE_TOKEN_INDEX. One frame = one image. No system prompt added
  beyond the template default.

  MobileVLM — wrapped with conversation template "v1"
  ────────────────────────────────────────────────────────────
  conv = conv_templates["v1"].copy()
  conv.append_message(roles[0], <prompt body above>)
  conv.append_message(roles[1], None)
  full_prompt = conv.get_prompt()
  # the "v1" template prepends MobileVLM's default system message
```

> Note on wording: the EVAL_PLAN phrasing was *"briefly describe what you see,
> then give the count."* The implemented prompt says *"First briefly describe
> what you see in this frame. Then state the count as a number."* — same intent,
> and "as a number" nudges the model to emit a parseable digit/word. The
> per-frame phrasing ("in this frame") is deliberate: each frame is queried
> independently.

---

## 4. How we score "recoverable" (the "what")

For each of the 76 wrong questions, we collect every number the model said
across its 8 frames — written as a digit (`8`) **or** a word (`eight`) — and
ask one simple thing:

```
   Did the TRUE count appear ANYWHERE in the 8 frame descriptions?

        ┌─────────────────────────────────────────────┐
        │ "how many people are involved?"  true = 5    │
        ├─────────────────────────────────────────────┤
        │ frame1: "There are five people..."   → 5  ✓  │
        │ frame2: "There are four people..."   → 4     │
        │ frame3: "There are four people..."   → 4     │
        │ frame4: ...                          → 4     │
        │ ...                                          │
        │ numbers seen = {5, 4}                        │
        │ true count 5 is in the set       → RECOVERABLE│
        └─────────────────────────────────────────────┘
```

Then:

```
                  wrong questions where the true count showed up
   recoverable =  ───────────────────────────────────────────────
                          all 76 wrong questions
```

**How numbers are pulled out of the text** (`extract_numbers` in both scripts):

```
  1. Every run of digits is captured:  regex  \b(\d+)\b   →  "8", "12"
  2. Number-words zero..twenty are matched as whole words:
        one→1  two→2  three→3 ... eight→8 ... twenty→20
  3. Both are merged into one set per frame, then across all 8 frames.

  The TRUE count is itself converted to a number the same way
  (the correct answer option, e.g. option "eight" → 8).
  A question is recoverable if  true_number ∈ (numbers seen across frames).
```

**Which questions count as "wrong":** we reuse the **baseline majority-vote
result** (`results_3b_nextqa.csv` / `results_mobilevlm_3b_nextqa.csv`), matching
on `(videoID, question)`. The 76 wrong items are exactly the how-many rows where
that baseline file has `correct == 0`. Step 1 does **not** re-judge correctness —
it only checks whether the true number appears in the new reasoning text.

---

## 5. The results

```
   ┌──────────────┬──────────────┬───────────────┬────────────────┐
   │ Model        │ wrong items  │ recoverable   │ FRACTION       │
   ├──────────────┼──────────────┼───────────────┼────────────────┤
   │ TinyLLaVA-3B │     76       │     28        │    36.8%       │
   │ MobileVLM-3B │     76       │     33        │    43.4%       │
   └──────────────┴──────────────┴───────────────┴────────────────┘


   0%        15%              40%                          100%
   ├──────────┼────────────────┼────────────────────────────┤
   │  SKIP    │   borderline   │       BUILD Step 4         │
   │ Step 4   │ (ask supervisor)│                            │
   └──────────┴────────────────┴────────────────────────────┘
                      ▲                  ▲
              TinyLLaVA 36.8%      MobileVLM 43.4%
```

- **TinyLLaVA — 36.8%**: just below the 40% line. Borderline.
- **MobileVLM — 43.4%**: clears the 40% line.

**Decision (supervisor, 2026-06-19):** both are close enough to the line —
**proceed with Step 4 for both models.**

---

## 6. What the examples look like

These are real rows from the run. They show *why* a question counts as
recoverable or not.

### Recoverable — the answer was in there, the vote missed it

```
  TinyLLaVA  "how many people are involved?"   true = 5
  ────────────────────────────────────────────────────────
  frame1: "There are five people involved in the image."  → 5  ✓
  frame2: "There are four people involved in the scene."  → 4
  frame3: "There are four people involved in the scene."  → 4
  → true count 5 appeared. RECOVERABLE.


  MobileVLM  "how many people are on stage?"   true = 6
  ────────────────────────────────────────────────────────
  frame1: "4 people on stage"   → 4
  frame2: "6"                   → 6  ✓
  frame3: "6"                   → 6  ✓
  → true count 6 appeared. RECOVERABLE.
```

### Not recoverable — the model genuinely never sees it

```
  TinyLLaVA  "how many goats can be spotted?"   true = 8
  ────────────────────────────────────────────────────────
  frame1: "1"                                   → 1
  frame2: "There are four goats visible..."     → 4
  frame3: "There are four goats visible..."     → 4
  → 8 never appears. NOT recoverable (the model undercounts the flock).


  MobileVLM  "how many people are there?"   true = 6
  ────────────────────────────────────────────────────────
  frame1: "1"   → 1
  frame2: "2"   → 2
  frame3: "2"   → 2
  → 6 never appears. NOT recoverable.
```

Notice the pattern: the misses are usually **large counts** (8 goats, 6 people)
where the model consistently sees only a few. The wins are usually **small
counts** where one or two frames nail it but get out-voted.

---

## 7. What this means going forward

```
   Step 1 says:  ~4 out of every 10 wrong counting answers are "fixable" —
                 the model already said the right number somewhere.

   So Step 4 (later) will try a cheap fix WITHOUT any extra detector:
   ┌──────────────────────────────────────────────────────────┐
   │  take the numbers the model said across all 8 frames       │
   │  → combine them smartly (most common / middle / max)       │
   │  → map back to the closest answer choice                   │
   │  → see if that beats the plain majority vote               │
   └──────────────────────────────────────────────────────────┘

   Next up (Step 2 + 3): a different, more natural fix — show the model
   ALL frames at once instead of one at a time, and sweep how many frames
   it needs. See EVAL_PLAN.md.
```

---

## 8. How to reproduce these exact numbers

Everything below is deterministic — same inputs give the same outputs (greedy
decoding, fixed frame sampling). There is no random seed to set because nothing
is random.

### 8.1 Environment

Two separate uv venvs (they need conflicting library versions — see
`docs/SETUP_DANAVIS3.md` for the full restore commands):

```
  TinyLLaVA  →  .venv/                 (transformers 4.40.1, timm 0.6.13)
  MobileVLM  →  .venv-mobilevlm/       (transformers 4.33.1, timm 0.9.12, numpy 1.25.0)
```

MobileVLM also needs its source on disk with one fix already applied:
```
  MobileVLM/  (git clone https://github.com/Meituan-AutoML/MobileVLM.git)
  fix: mobilevlm/utils.py line 45  →  preprocess([image], ...)  not preprocess(image, ...)
```

### 8.2 Inputs (exact)

```
  Questions : tinyllava/data/nextqa/val_descriptive.csv   (777 rows)
              subset = rows where question.lower().startswith("how many")  → 177 rows
  Videos    : tinyllava/data/nextqa/videos/NExTVideo/<videoID>.mp4
  Baseline  : results_3b_nextqa.csv            (TinyLLaVA, for the "wrong" set)
              results_mobilevlm_3b_nextqa.csv  (MobileVLM, for the "wrong" set)
  Models    : TinyLLaVA/TinyLLaVA-3.1B   and   mtgv/MobileVLM_V2-3B
              (downloaded from HuggingFace on first run)
```

### 8.3 Frame sampling (exact, deterministic)

8 frames, evenly spaced across the whole video by frame index — **not** random:

```
  total = number of frames in the video
  indices = [ int(i * total / 8)  for i in 0..7 ]      # i = 0,1,2,...,7
  read those 8 frames, convert BGR→RGB (OpenCV cv2.VideoCapture)
```

Because the indices are a fixed formula, re-running picks the **same 8 frames**
every time. (This is also why error bars later require varying this rule — a
plain re-run has zero variance.)

### 8.4 Generation settings (exact)

```
  ┌─────────────────────┬───────────────────────┬───────────────────────┐
  │ setting             │ TinyLLaVA-3.1B        │ MobileVLM V2-3B       │
  ├─────────────────────┼───────────────────────┼───────────────────────┤
  │ precision           │ float16 (.half().cuda)│ float16 (.half().cuda)│
  │ decoding            │ greedy, do_sample=False│ greedy, do_sample=False│
  │ temperature         │ — (greedy)            │ — (greedy)            │
  │ max_new_tokens      │ 128                   │ 128                   │
  │ frames per question │ 8                     │ 8                     │
  │ chat template       │ "phi"                 │ conv template "v1"    │
  │ attn_implementation │ "eager"               │ (default)             │
  │ vocab resize        │ max(len(tok),         │ (model default)       │
  │                     │  vocab_size)+100      │                       │
  │ stop criteria       │ template separator    │ conv.sep / sep2       │
  └─────────────────────┴───────────────────────┴───────────────────────┘
```

Greedy decoding (`do_sample=False`) is what makes the run deterministic.

### 8.5 Commands

```bash
cd /data/raja/20_WORK/50_RSML/thesis-vlm-evaluation

# TinyLLaVA — GPU 0
CUDA_VISIBLE_DEVICES=0 nohup \
  .venv/bin/python3 scripts/step1_tinyllava_reasoning.py \
  > results/step1/step1_tinyllava_run.log 2>&1 &

# MobileVLM — GPU 1 (separate venv)
CUDA_VISIBLE_DEVICES=1 nohup \
  .venv-mobilevlm/bin/python3 scripts/step1_mobilevlm_reasoning.py \
  > results/step1/step1_mobilevlm_run.log 2>&1 &
```

Each run takes roughly **25–30 min** on one GTX 1080 Ti (177 questions × 8
frames). The recoverable fraction is printed at the end of each log and the
per-item flags are written to `results/step1/step1_*_wrong_annotated.csv`.

### 8.6 What each script does, end to end

```
  load model (fp16, GPU)
        │
        ▼
  read val_descriptive.csv → keep 177 "how many" rows
        │
        ▼
  for each question:
      sample 8 frames (fixed indices)
      for each frame:  build prompt → generate (greedy, 128 tok) → save text
        │
        ▼
  write results/step1/step1_<model>_reasoning.csv   (177 rows, all frame texts)
        │
        ▼
  load baseline csv → find the 76 wrong how-many items
  for each wrong item: extract numbers from its 8 texts;
                       recoverable = (true count in that set)
        │
        ▼
  print fraction + examples; write *_wrong_annotated.csv
```

### 8.7 Re-checking the fraction without re-running the models

The model outputs are saved, so the fraction is reproducible from CSV alone:

```python
import pandas as pd
df = pd.read_csv("results/step1/step1_tinyllava_reasoning_wrong_annotated.csv")
print(df["recoverable"].mean())          # → 0.368  (36.8%)
print(int(df["recoverable"].sum()), "of", len(df))   # → 28 of 76
```

---

## 9. Where the files live

```
  scripts/step1_tinyllava_reasoning.py      run script (TinyLLaVA, .venv)
  scripts/step1_mobilevlm_reasoning.py      run script (MobileVLM, .venv-mobilevlm)

  results/step1/
    step1_tinyllava_reasoning.csv           all 177 Q × 8 frame texts
    step1_tinyllava_reasoning_wrong_annotated.csv   the 76 wrong + recoverable flag
    step1_mobilevlm_reasoning.csv
    step1_mobilevlm_reasoning_wrong_annotated.csv
    step1_*_run.log                         full console logs
```

Reproduce the numbers: open the `*_wrong_annotated.csv` files — the
`recoverable` column is `True`/`False`; the fraction is just its mean.
