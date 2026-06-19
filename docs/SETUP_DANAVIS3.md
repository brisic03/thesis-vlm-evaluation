# danavis3 GPU Setup

Machine: danavis3, 4× GTX 1080 Ti (11 GB each), GPU indices 0–3.

## Repository path

**Always work from:**
```
/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation/
```
There is also a checkout at `/home/raja/data/20_WORK/50_RSML/...` — do NOT run
`uv sync` from there; the uv.lock references the `/data/raja/...` path.

## Python environments (two separate venvs — do NOT mix)

### TinyLLaVA — `.venv/`
```
source /data/raja/20_WORK/50_RSML/thesis-vlm-evaluation/.venv/bin/activate
```
- uv-managed, editable install of `tinyllava` from repo root
- transformers==4.40.1, timm==0.6.13, torch==2.0.1+cu117
- Restore with: `uv sync` (from repo root)

### MobileVLM — `.venv-mobilevlm/`
```
source /data/raja/20_WORK/50_RSML/thesis-vlm-evaluation/.venv-mobilevlm/bin/activate
```
- Created 2026-06-19 with `uv venv --python 3.9`
- transformers==4.33.1, timm==0.9.12, numpy==1.25.0, torch==2.0.1+cu117
- Restore with:
  ```
  uv venv .venv-mobilevlm --python 3.9
  uv pip install --python .venv-mobilevlm/bin/python3 \
    torch==2.0.1 torchvision==0.15.2 transformers==4.33.1 tokenizers==0.13.3 \
    timm==0.9.12 accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.41.0 \
    sentencepiece shortuuid einops==0.6.1 einops-exts==0.0.4 \
    scipy numpy==1.25.0 opencv-python pandas tqdm pillow
  ```

## MobileVLM source

Cloned to `MobileVLM/` at repo root (gitignored).
```
git clone https://github.com/Meituan-AutoML/MobileVLM.git MobileVLM
```

**Required one-line fix** (already applied, do not revert):
`MobileVLM/mobilevlm/utils.py` line 45 — wrap single image in list for
transformers 4.33.1 compatibility:
```python
# Before (broken with transformers 4.33.1):
image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
# After (fixed):
image = image_processor.preprocess([image], return_tensors='pt')['pixel_values'][0]
```

All scripts set `sys.path.insert(0, '.../MobileVLM')` explicitly; no install needed.

## Data paths (all under repo root)

| Data | Path |
|---|---|
| NExT-QA descriptive CSV | `tinyllava/data/nextqa/val_descriptive.csv` (777 rows) |
| NExT-QA videos | `tinyllava/data/nextqa/videos/NExTVideo/<videoID>.mp4` |
| VisDrone questions | `results/visdrone/visdrone_val_questions.csv` |
| VisDrone images+labels | `/data/raja/Datasets/VisDrone/VisDrone2019-DET-val/` (548 imgs + YOLO labels; downloaded from HF `banu4prasad/VisDrone-Dataset`, outside the repo) |
| VisDrone GT counts | `results/step2/visdrone_counting_gt.csv` (re-derived, validated) |
| TinyLLaVA baseline | `results_3b_nextqa.csv` (777 rows, per-frame vote) |
| MobileVLM baseline | `results_mobilevlm_3b_nextqa.csv` (777 rows, per-frame vote) |

## Running jobs

- **No SLURM.** Plain python, background with `nohup ... &`.
- **4 GPUs available.** Use `CUDA_VISIBLE_DEVICES=N` to assign.
- Log to `results/stepN/stepN_<model>_run.log`.
- TinyLLaVA: always `.venv/bin/python3`, runs on GPU 0 by default.
- MobileVLM: always `.venv-mobilevlm/bin/python3`, use GPU 1 (or 2/3 for parallelism).

### GPU assignment convention for Steps 2+3
```
GPU 0 — TinyLLaVA  NExT-QA composite + frame sweep
GPU 1 — MobileVLM  NExT-QA composite + frame sweep
GPU 2 — TinyLLaVA  VisDrone numeric
GPU 3 — MobileVLM  VisDrone numeric
```

## Model downloads

Both models download from HuggingFace at first run (no local weights cache):
- TinyLLaVA: `TinyLLaVA/TinyLLaVA-3.1B`
- MobileVLM: `mtgv/MobileVLM_V2-3B`

## Step scripts written so far

| Script | Env | Purpose |
|---|---|---|
| `scripts/step1_tinyllava_reasoning.py` | `.venv` | Step 1 reasoning run |
| `scripts/step1_mobilevlm_reasoning.py` | `.venv-mobilevlm` | Step 1 reasoning run |

## Completed steps

- **Step 0** ✓ — VisDrone characterisation (local analysis machine)
- **Step 1** ✓ — Recoverable fraction: TinyLLaVA 36.8%, MobileVLM 43.4%
  - Gate: supervisor approved proceeding with both models into Step 4
- **Steps 2 & 3** ✓ (2026-06-19) — composite vs vote, frame sweep, VisDrone
  numeric + size sweep. Results: `docs/STEP2_3_RESULTS.md`. Headline: composite
  never beats the per-frame vote; both models are weak counters (Tiny
  under-counts/refuses, Mobile over-counts).

## Next to run

- **Step 4** — symbolic repair (aggregate Step 1 per-frame stated numbers via
  mode/median/max; compare to vote and composite). Gate from Step 1 was positive.
- **Step 5** — confidence intervals + McNemar significance + bootstrap CIs on all
  of the above (local CSV analysis).
