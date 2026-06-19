# Thesis Eval Plan — 6 Steps, Run In Order

> **Context:** evaluating **TinyLLaVA-3B** and **MobileVLM-3B** on video/aerial
> counting. Each step gates the next. **Report results after each step before
> proceeding. Do not skip ahead.**

## Where things run — danavis3 GPU workflow

**This machine is for authoring and CSV-only analysis. Every step that runs a
model lives on the `danavis3` server.** There is no local GPU.

```
  THIS MACHINE (author + local analysis)     danavis3 (GPU, 4x GTX 1080 Ti 11GB)
  ┌──────────────────────────┐               ┌──────────────────────────┐
  │ edit scripts             │  push         │ git pull                 │
  │ Step 0 + Step 5 (CSV,    │ ───raja───►   │ Steps 1-4: source .venv, │
  │   stats, plots) in       │  develop      │   python … (no SLURM)    │
  │   .venv-analysis         │ ◄──pull───    │ commit results + push    │
  │ read results ◄───────────│   results     │ results CSVs             │
  └──────────────────────────┘               └──────────────────────────┘
        Claude has direct SSH to danavis3 ("ssh danavis3") and can
        push / pull / run / read logs on BOTH ends itself.
```

- **Contract branch: `raja-develop`.** Push from here, pull on danavis3, push
  results back, pull here. Both checkouts share `origin`
  (`git@github.com:brisic03/thesis-vlm-evaluation.git`). Keep everything on this
  branch until integrating to `main`.
- **danavis3 access:** Claude has direct SSH as `ssh danavis3`. Repo path:
  `/home/raja/data/20_WORK/50_RSML/thesis-vlm-evaluation`. Claude runs the GPU
  steps itself — no manual hand-off.
- **Environments are uv-based on both machines** (NOT conda):
  - *Local (this machine):* `.venv-analysis` + `requirements-analysis.txt`
    (pandas/numpy/matplotlib/scipy, CPU only). Activate with
    `source .venv-analysis/bin/activate` before running analysis scripts.
  - *danavis3:* same uv setup; activate the repo venv, `uv pip install` from the
    GPU requirements, run with `python`.
- **Job runner: direct `python` invocation** on danavis3 — NOT SLURM. The
  existing `sbatch/` files are reference only; new runs are plain python.
- **Path portability ⚠️:** existing scripts hardcode `/home/brisic03/...`
  (model paths, `DATA_ROOT`, `VIDEO_ROOT`, VisDrone dirs, `OUT_PATH`) — that is
  another student's checkout and does NOT exist on danavis3. Before any run,
  parameterise these (env vars or argparse with a repo-relative default) so they
  resolve under `/home/raja/data/20_WORK/50_RSML/thesis-vlm-evaluation`.
- **Execution loop per GPU step:** (1) author/edit + commit, push `raja-develop`;
  (2) `ssh danavis3`, pull, activate venv, run python; (3) commit result
  CSVs/logs on danavis3, push; (4) pull here and analyse in `.venv-analysis`.

## How to run this plan

- **Sequential.** Steps run in order. Steps 0 and 1 can *kill* later steps —
  do not pre-build anything they gate.
- **Report back.** After every step, report the numbers to the supervisor
  *before* starting the next step. The gate decisions below depend on the
  actual values, not assumptions.
- **Step 0 runs locally** (pure CSV). Steps 1–4 need GPU → danavis3. Step 5 is
  local (stats on collected CSVs).
- **Two easy-to-get-wrong rules, apply everywhere:**
  1. **Vary the seed for error bars.** Deterministic re-runs of identical
     settings give *zero* variance. For any error bar / repeated run, vary the
     **frame-sampling seed**, not just re-execute the same command.
  2. **Do not assume an inflection exists.** When a step produces a curve
     (frames, size threshold), report its real shape — peak, cliff, or flat.

---

## Dependency map

```
 Step 0 ──> tells you if VisDrone counting needs numeric redesign
              │
 Step 1 ──> tells you if Step 4 is worth building
              │
 Step 2 ──> the likely headline (composite vs vote; numeric VisDrone)
              │
 Step 3 ──> frame + size sweeps (standalone findings)
              │
 Step 4 ──> only if Step 1 was positive
              │
 Step 5 ──> stats on ALL of the above, gates the writeup
```

---

## Step 0 — Characterise the VisDrone data

**No model runs. Pure CSV analysis.**

**Goal:** find out if the existing VisDrone counting/location questions are
usable or contaminated by skewed answer distributions.

**Task:** from the existing VisDrone questions CSV
(`results/visdrone/visdrone_val_questions.csv`), compute:
- (a) distribution of **counting** answers — fraction in each bucket
  `0, 1, 2, 3, "4 or more"`.
- (b) distribution of **location** answers — fraction that are `"center"`.
- (c) **majority-class baseline accuracy** for both (accuracy if a model always
  picked the single most common answer).

**Output:** two small tables — answer distribution + majority-class baseline —
for counting and location.

**Gate:** if counting answers are **>60% one bucket**, the current
multiple-choice counting result is **not defensible** → confirms we move
VisDrone counting to the numeric redesign in **Step 2**.

**Checklist**
- [x] Load `visdrone_val_questions.csv`, split by `qtype`.
- [x] Counting: value-count of `answer_text` over the 5 buckets, as fractions.
- [x] Location: fraction where `answer_text == "center"`.
- [x] Majority-class baseline = max bucket fraction, for counting and location.
- [x] Report both tables; flag whether the >60% gate fires.

**RESULTS (2026-06-19)** — script: `scripts/analysis/step0_visdrone_characterise.py`
(run locally in `.venv-analysis`). qtype counts: presence 545, counting 545,
most_frequent 525, location 545.

Counting answer distribution (n=545):

| bucket | count | fraction |
|---|---:|---:|
| 0 | 0 | 0.0% |
| 1 | 207 | 38.0% |
| 2 | 99 | 18.2% |
| 3 | 56 | 10.3% |
| 4 or more | 183 | 33.6% |

Location answer distribution (n=545):

| bucket | count | fraction |
|---|---:|---:|
| center | 251 | 46.1% |
| bottom-left | 149 | 27.3% |
| bottom-right | 132 | 24.2% |
| top-right | 7 | 1.3% |
| top-left | 6 | 1.1% |

Majority-class baseline: counting (always "1") = **38.0%**; location (always
"center") = **46.1%**.

**Gate outcome:** counting's largest bucket is 38.0% (≤60%) → **gate does NOT
fire**. This is a *one-way* trigger: a >60% skew would have made the numeric
(cardinality) redesign urgent/mandatory. It NOT firing does **not** mean we keep
MCQ — **VisDrone counting still moves to the numeric/cardinality redesign in
Step 2 Part B regardless** (that was always the plan; Steps 3 & 5 only report
MAE/RMSE, never MCQ counting accuracy). The ≤60% result simply means the
*existing* MCQ counting numbers are not auto-disqualified by skew and can still
be cited as a baseline alongside the numeric metric.

**But note for the writeup, two real concerns the numbers expose:**
- Counting bucket "0" never occurs (every counting question is about a class
  that *is* present, by construction) — so "0" is dead weight and the effective
  task is a 4-way choice, not 5-way.
- Location is degenerate: 97.6% of answers fall in just three cells
  (center/bottom-left/bottom-right); top-left and top-right together are 2.4%.
  The majority baseline (46.1%) is high, and any model accuracy on `location`
  must be read against that floor.

**RECOMMENDATION — `location`:** drop it from the reported results. With a 46.1%
always-"center" floor and two near-empty cells, a model can look competent
without locating anything, and the metric can't distinguish skill from guessing
the prior. Two defensible options if we keep it instead of dropping:
  1. **Report vs the 46.1% baseline explicitly** and only claim a finding if a
     model clears it by a margin that survives Step 5's CI/McNemar test.
  2. **Redesign**: balance the cells (sample so each location is ~equally likely)
     or collapse to a coarser, better-populated scheme (e.g. center vs edge).
Default for now: **exclude `location` from headline numbers**; keep `presence`,
`most_frequent`, and the numeric `counting` from Step 2 as the VisDrone story.

---

## Step 1 — Recoverable fraction on NExT-QA counting

**DONE (2026-06-19).**

**Goal:** the number that decides whether "symbolic repair" (Step 4) is alive.
Existing reasoning data is only 50 examples — too few. Re-generate at full scale.

**Task:**
1. Re-run **both models** with the reasoning prompt ("briefly describe what you
   see, then give the count") on **all** NExT-QA descriptive counting questions
   (the `how many` subset, ~177 per model). **Save the per-frame stated text.**
2. For every counting question the original **majority vote got wrong**, check
   whether the **correct count appears among the per-frame stated numbers**.
3. Compute:

```
   recoverable = (wrong items whose reasoning contains the true count)
                 ----------------------------------------------------
                      (total wrong counting items)
```

**Output:** the recoverable fraction (one percentage), plus a few example rows
showing reasoning text vs true count.

**Gate:**
- recoverable **> 40%** → Step 4 (symbolic repair) is worth building.
- recoverable **< 15%** → **skip Step 4**; the model is genuinely blind, not
  mis-aggregated.
- in between → report and decide with supervisor.

**Note:** the recoverable fraction *is* the faithfulness measure — no separate
faithfulness step is needed.

**RESULTS (2026-06-19)** — scripts: `scripts/step1_tinyllava_reasoning.py`,
`scripts/step1_mobilevlm_reasoning.py`. Full outputs in `results/step1/`.

| Model | Wrong how-many items | Recoverable | Fraction |
|---|---|---|---|
| TinyLLaVA-3.1B | 76 | 28 | **36.8%** |
| MobileVLM V2-3B | 76 | 33 | **43.4%** |

Gate outcomes:
- **TinyLLaVA 36.8%** → 15–40% middle zone (borderline).
- **MobileVLM 43.4%** → > 40% (Step 4 gate fires).
- **Supervisor decision (2026-06-19):** proceed with Step 4 for **both** models.

Example recoverable item (TinyLLaVA): Q "how many people are involved",
true=5 (five), frame[0] output "There are five people involved in the image."

Example non-recoverable (TinyLLaVA): Q "how many goats can be spotted",
true=8 (eight), frame[0] output "1" — model never states 8 across any frame.

**Checklist**
- [x] Identify the `how many` counting subset in the NExT-QA descriptive data.
- [x] Reasoning prompt run for TinyLLaVA over the full subset; save per-frame text.
- [x] Reasoning prompt run for MobileVLM over the full subset; save per-frame text.
- [x] Parse stated numbers per frame (word- and digit-form, e.g. "two"/"2").
- [x] Restrict to originally-wrong items; compute recoverable fraction per model.
- [x] Report fraction + 3–5 example rows; state which gate branch fires.

---

## Step 2 — Composite baseline + numeric VisDrone counting

**Goal:** evaluate the models the way they were designed (all frames in one
prompt), and fix VisDrone counting by making it numeric.

**Task — Part A (NExT-QA):** re-run both models giving **all 8 frames in a
single prompt, one answer** — on the counting subset first, then all descriptive
questions. Compare against the existing **per-frame-majority-vote** numbers.

**Task — Part B (VisDrone):** rebuild counting as **open-ended numeric** — ask
"how many Xs?", parse the number from the text reply, compute **MAE and RMSE**
against the ground-truth count. Define and log a rule for **unparseable replies**
(e.g. mark as parse-failure, report that rate separately — do not silently treat
as 0).

**Output:**
- NExT-QA accuracy table: composite vs per-frame vote (counting + all descriptive).
- VisDrone: MAE, RMSE, and parse-failure rate per model.

**Gate:** the composite-vs-vote gap is likely the **headline**. If composite
*fully* fixes counting, Step 4 becomes a cost comparison rather than a fix.

**Implementation note:** Step 2A (counting subset, composite@8) is run as part
of the combined Step 2+3 frame-sweep (see Step 3 note below). The 8-frame result
from the sweep is the Step 2A counting number.

**Checklist**
- [ ] Composite prompt: 8 frames concatenated into one prompt, single answer.
- [ ] NExT-QA composite run — counting subset, both models. *(covered by Step 3A sweep)*
- [ ] NExT-QA composite run — all descriptive, both models.
- [ ] Table: composite vs vote accuracy, per model, per subset.
- [ ] VisDrone numeric prompt ("how many Xs?"), no fixed buckets.
- [ ] Number parser + explicit parse-failure rule, logged.
- [ ] Compute MAE, RMSE, parse-failure rate per model; report.

---

## Step 3 — Frame-count sweep + object-size sweep

**Goal:** find the sweet spots — does more frames help, and how small can
objects get before counting breaks.

**Task — Sweep A (NExT-QA, frame count):** run the **composite** method with
**1, 2, 4, 8, 16** frames on the counting subset; plot accuracy vs frame count.

**Task — Sweep B (VisDrone, object size):** using the numeric counting from
Step 2, recompute MAE/RMSE keeping only objects **above a minimum bbox-size
threshold**, swept across **0.05%, 0.1%, 0.2%, 0.5%, 1%** of image area; plot
error vs threshold.

**Output:** two plots — accuracy vs frames, and error vs size threshold. Report
the shape **honestly** (peak? cliff? flat?) — do not assume an inflection exists.

**Gate:** a peak in frames, or a sharp drop in error as small objects are
excluded, each become standalone findings.

**Implementation note (Step 2+3 combined run):** Step 3A and Step 2A (counting
subset) are batched together in one pass: run composite at {1,2,4,8,16} frames
on the counting subset for both models. The 8-frame result feeds Step 2A; the
full curve feeds Step 3A. This saves a full re-run. 4 GPUs available on danavis3
(GTX 1080 Ti ×4): assign TinyLLaVA and MobileVLM to GPU 0/1 for NExT-QA and
GPU 2/3 for VisDrone to run all in parallel.

**Checklist**
- [ ] Composite runs at frames ∈ {1,2,4,8,16}, counting subset, both models.
- [ ] Plot accuracy vs frame count; describe shape.
- [ ] Recompute VisDrone MAE/RMSE at thresholds {0.05,0.1,0.2,0.5,1}% area.
- [ ] Plot error vs threshold; describe shape.
- [ ] Note: changing the threshold changes the GT count — recompute GT per threshold.

---

## Step 4 — Symbolic repair (only if Step 1 said yes)

**Goal:** test whether reading the model's own stated numbers beats voting on
letters, **without any external detector**.

**Task:** take the per-frame **stated numbers** from Step 1, aggregate with
simple fixed rules (**mode, median, max**), map the result to the answer option.
Compare against (a) the per-frame **majority vote** and (b) the **composite
baseline** from Step 2.

**Output:** accuracy table — vote vs symbolic-repair vs composite — on NExT-QA
counting.

**Gate:** if symbolic repair approaches composite, the finding is "a trivial
detector-free fix recovers most of the loss."

**Checklist** *(skip entire step if Step 1 recoverable < 15%)*
- [ ] Aggregate per-frame stated numbers via mode / median / max.
- [ ] Map aggregated count → nearest answer option.
- [ ] Table: vote vs symbolic-repair (×3 rules) vs composite, NExT-QA counting.
- [ ] State whether repair approaches composite.

---

## Step 5 — Statistics on everything

**Goal:** make every comparison defensible.

**Task:**
- Add **95% confidence intervals** (normal approximation to the binomial) to
  every accuracy number across all steps.
- For every **paired comparison on the same questions** (composite vs vote,
  repair vs vote, model vs model), run **McNemar's test** and report whether the
  difference is significant.
- For numeric VisDrone, report **MAE/RMSE with bootstrap confidence intervals**.

**Output:** all earlier tables, now with CIs and significance flags. **Explicitly
mark any difference that is *not* significant.**

**Gate:** nothing proceeds to the writeup without this — an unmarked
insignificant difference is the fastest way to get rejected.

**Checklist**
- [ ] Binomial-normal 95% CI added to every accuracy figure.
- [ ] McNemar's test on every same-question paired comparison; significance flagged.
- [ ] Bootstrap CIs on VisDrone MAE/RMSE.
- [ ] Every non-significant difference explicitly marked as such.
