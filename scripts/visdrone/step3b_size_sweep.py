"""
Step 3B — VisDrone object-size sweep (pure CSV analysis, no GPU).

Question: are the models' counting errors driven by tiny objects they cannot
see? We keep only ground-truth objects whose bbox area is above a threshold,
recompute the TRUE count at each threshold, and compare against the SAME model
prediction from Step 2B. Thresholds swept over {0.05, 0.1, 0.2, 0.5, 1}% of image
area. (0.1% == the 0.001 ratio used to build the base GT.)

The model prediction is fixed; only the ground truth shrinks as we raise the
threshold. If MAE drops sharply as small objects are excluded, the error was
dominated by objects too small to see.

Inputs : results/step2/visdrone_counting_gt.csv (area_ratios per question)
         results/step2/step2b_<model>_numeric.csv (pred_count per image)
Output : results/step3/step3b_size_sweep.csv  (model, threshold_pct, MAE, RMSE, n)
"""

import os
import ast
import math
import pandas as pd

REPO_ROOT = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
GT_CSV    = os.path.join(REPO_ROOT, "results/step2/visdrone_counting_gt.csv")
OUT_PATH  = os.path.join(REPO_ROOT, "results/step3/step3b_size_sweep.csv")
MODELS = {
    "TinyLLaVA":  os.path.join(REPO_ROOT, "results/step2/step2b_tinyllava_numeric.csv"),
    "MobileVLM":  os.path.join(REPO_ROOT, "results/step2/step2b_mobilevlm_numeric.csv"),
}
THRESHOLDS_PCT = [0.05, 0.1, 0.2, 0.5, 1.0]  # % of image area


def gt_at(area_ratios, thr_ratio):
    return sum(1 for a in area_ratios if a >= thr_ratio)


def main():
    gt = pd.read_csv(GT_CSV)
    gt["area_ratios"] = gt["area_ratios"].apply(ast.literal_eval)
    gt = gt.set_index("image")

    rows = []
    for model, path in MODELS.items():
        if not os.path.exists(path):
            print(f"  skip {model}: {path} not found")
            continue
        pred = pd.read_csv(path)
        pred = pred[pred["parse_ok"]].copy()  # only parseable predictions
        for thr_pct in THRESHOLDS_PCT:
            thr_ratio = thr_pct / 100.0
            errs = []
            for _, r in pred.iterrows():
                if r["image"] not in gt.index:
                    continue
                ratios = gt.loc[r["image"], "area_ratios"]
                gt_t = gt_at(ratios, thr_ratio)
                errs.append(r["pred_count"] - gt_t)
            errs = pd.Series(errs, dtype=float)
            mae = errs.abs().mean()
            rmse = math.sqrt((errs ** 2).mean())
            rows.append({
                "model": model,
                "threshold_pct": thr_pct,
                "n": len(errs),
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}\n")
    print("=== STEP 3B — VisDrone size sweep (MAE / RMSE vs min bbox size) ===")
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        print(f"\n{model}:")
        print(f"  {'thr%':>6} {'n':>5} {'MAE':>7} {'RMSE':>7}")
        for _, r in sub.iterrows():
            print(f"  {r['threshold_pct']:>6} {int(r['n']):>5} {r['MAE']:>7} {r['RMSE']:>7}")


if __name__ == "__main__":
    main()
