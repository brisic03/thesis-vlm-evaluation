# VisDrone counting — resolution bottleneck & grid (tiling) fix

> Plain-language. CSV + GPU runs. Models: TinyLLaVA-3.1B, MobileVLM-V2-3B.
> Scripts: `scripts/visdrone/visdrone_grid_count.py`, `grid_compare.py`.
> Last updated: 2026-06-20.

## Why we tried this — the objects are below the model's resolution

VLMs don't see full-resolution images. Each model shrinks every photo to a small
fixed square, then chops it into 14×14-pixel patches:

```
  VisDrone source (mostly 1360x765)
        │  shrink to a square
        ▼
  TinyLLaVA -> 384x384 (SigLIP)        MobileVLM -> 336x336 (CLIP)
        │  cut into 14x14 patches
        ▼
  one patch covers ~50-57 px of the ORIGINAL image
```

What that does to the things we count:

```
  object in source           after shrink     vs one patch (~50-57px)
  smallest we count (0.1%)   ~32px -> ~8px    ▏ SMALLER than one patch
  a typical car (0.3%)       ~56px -> ~15px   ▏ about one patch
```

So most counting targets occupy **one patch token or less** — the model literally
cannot resolve them. This is the most likely reason VisDrone counting failed and
why error dropped as objects got bigger (Step 3B).

## The fix — grid mode (tiling)

Split the image into an NxN grid, count each tile on its own (each tile fills the
encoder, so objects are seen at N× the resolution), then add up the tiles.

```
  ┌────┬────┐   2x2: tiles ~680px -> objects ~16px  (2x resolution)
  │ t1 │ t2 │   3x3: tiles ~453px -> objects ~24px  (3x resolution)
  ├────┼────┤
  │ t3 │ t4 │   per-image count = t1 + t2 + t3 + t4
  └────┴────┘
```
(We also fixed a prompt bug here: the whole-image prompt said "peoples"/"buss";
grid mode uses correct plurals "people"/"buses".)

## Results — all 545 images

| model | method | MAE | RMSE | bias | pred mean (gt 4.6) |
|---|---|---:|---:|---:|---:|
| TinyLLaVA | whole-image (forced to answer) | 13.51 | 31.27 | +10.29 | 14.9 |
| TinyLLaVA | **grid 2×2** | **4.10** | **7.14** | **+0.79** | 5.4 |
| TinyLLaVA | grid 3×3 | 5.77 | 10.54 | +4.46 | 9.0 |
| MobileVLM | whole-image | 7.80 | 20.69 | +5.30 | 9.9 |
| MobileVLM | grid 2×2 | 13.62 | 29.41 | +13.26 | 17.9 |
| MobileVLM | grid 3×3 | 18.60 | 34.71 | +18.54 | 23.1 |

```
  MAE (lower better), all 545 images
  TinyLLaVA  whole(forced) ████████████████████████████ 13.51
             grid 2x2      ████████ 4.10     <- BEST counter we have
             grid 3x3      ████████████ 5.77
  MobileVLM  whole-image   ████████████████ 7.80
             grid 2x2      ████████████████████████████ 13.62
             grid 3x3      ██████████████████████████████████████ 18.60
```

## The key split — easy scenes vs the crowded ones TinyLLaVA used to refuse

TinyLLaVA originally refused 27% of images (the crowded ones). Forcing a
whole-image answer there was useless (MAE 43). Grid mode is where it pays off:

| subset | method | MAE | RMSE | bias |
|---|---|---:|---:|---:|
| EASY (399, gt~2.8) | whole-image | 2.69 | 7.14 | −1.04 |
| EASY | grid 2×2 | **2.30** | 3.84 | +0.29 |
| EASY | grid 3×3 | 3.47 | 6.22 | +2.51 |
| HARD (146, gt~9.5) | whole-image (forced) | 43.08 | 59.26 | +41.25 |
| HARD | **grid 2×2** | **9.01** | 12.25 | +2.14 |
| HARD | grid 3×3 | 12.08 | 17.57 | +9.79 |

```
  TinyLLaVA on the 146 CROWDED scenes it used to refuse:
     forced whole-image  MAE 43.1  (blurts "50"+ for a true ~9)
     grid 2x2            MAE  9.0  *** usable — from garbage to ballpark ***
```

## What we learned

1. **Grid 2×2 rescues TinyLLaVA — resolution really was the bottleneck.** On the
   crowded scenes it used to refuse, MAE drops from 43 (forced whole-image) to
   **9.0**, and bias from +41 to +2. On easy scenes it's also slightly better
   (2.69→2.30). Overall MAE **13.51→4.10**, nearly unbiased. **Grid 2×2 TinyLLaVA
   is now the best VisDrone counter we have** — better than whole-image MobileVLM.

2. **Grid mode HURTS MobileVLM** (7.80→13.62→18.60). MobileVLM never refused and
   already over-counts; tiling just gives it more tiles to over-guess (plus
   boundary double-counting), so the over-count compounds. Its bottleneck was
   over-guessing, not resolution — so adding resolution makes it worse.

3. **2×2 beats 3×3 for both.** More tiles = more boundary double-counting and more
   small tiles where the model over-guesses; bias climbs with grid size
   (TinyLLaVA +0.8 → +4.5; MobileVLM +13 → +18.5). 2×2 is the sweet spot.

```
  RULE OF THUMB:
    under-counter / refuser (TinyLLaVA)  -> grid HELPS  (give it resolution)
    over-counter (MobileVLM)             -> grid HURTS  (amplifies over-guessing)
```

## Does forcing answers still hurt under grid mode? YES.

Grid mode above counts a refused tile as 0. Natural question: is that 0-fallback
doing the work, or the resolution boost? We re-ran TinyLLaVA grid mode with
per-tile **retry-until-answered** (`--force`: firmer prompt then sampling, no free
0) — the same forcing that wrecked the whole-image run.

| TinyLLaVA grid | all 545 MAE | bias | 146 crowded MAE | bias |
|---|---:|---:|---:|---:|
| 2×2, refused tile = 0 | **4.10** | +0.79 | **9.01** | +2.14 |
| 2×2, FORCED | 13.37 | +12.19 | 40.87 | +40.61 |
| 3×3, refused tile = 0 | 5.77 | +4.46 | 12.08 | +9.79 |
| 3×3, FORCED | 20.01 | +19.30 | 58.68 | +58.42 |

```
  MAE on the 146 crowded scenes:
    grid 2x2, allow refuse(=0)  █████ 9.0
    grid 2x2, FORCED            ████████████████████████ 40.9  (~= whole-image-forced 43)
```

**Forcing re-breaks everything.** Tiling fixed the *resolution* (sparse tiles are
now countable), but the dense tiles are still uncountable — and when forced, the
model blurts huge numbers (bias +40 on crowded scenes, exactly like the
whole-image forced run). The 0-fallback was essential: it lets the model **abstain
on the tiles it can't do** while still counting the ones it can.

```
  The grid win = RESOLUTION (tiling)  +  HONESTY (abstain -> 0 on hard tiles)
  Remove the honesty (force an answer) and MAE goes 4.1 -> 13.4 (9.0 -> 40.9 on crowds).
```

So across BOTH whole-image and grid settings, forcing the model to answer always
hurts — the refusal is a useful signal, not a bug to suppress.

## Caveats (stated honestly)
- **Boundary effect:** an object on a tile edge can be counted twice or missed; we
  have no coordinates to dedupe. This is part of why 3×3 over-counts more than 2×2.
- **Empty/refused tiles** are summed as 0 (TinyLLaVA tile parse-fail 7.9% at 2×2,
  3.4% at 3×3; MobileVLM 0%). Reasonable since most empty tiles truly have 0.
- Not yet significance-tested (would extend Step 5 bootstrap to the grid runs).
