"""
Complement to Step 3B — error as a function of the ACTUAL size of the objects
being counted (not a GT-threshold sweep).

For each counting question we characterise the target objects by their MEDIAN
bbox area (as % of image area), bin the questions into size bands, and report
the model's MAE / signed error within each band. This directly answers: "are
images whose objects are small harder to count?"

Caveat printed below: small objects correlate with crowded scenes (more objects
-> each is smaller), so the size effect is entangled with the count effect.

Inputs : results/step2/visdrone_counting_gt.csv (area_ratios per question)
         results/step2/step2b_<model>_numeric.csv (pred_count)
Output : results/step3/step3b_error_by_objsize.csv
"""

import os
import ast
import math
import pandas as pd
import numpy as np

REPO = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
GT   = os.path.join(REPO, "results/step2/visdrone_counting_gt.csv")
OUT  = os.path.join(REPO, "results/step3/step3b_error_by_objsize.csv")
MODELS = {
    "TinyLLaVA": os.path.join(REPO, "results/step2/step2b_tinyllava_numeric.csv"),
    "MobileVLM": os.path.join(REPO, "results/step2/step2b_mobilevlm_numeric.csv"),
}
# size bands in % of image area (objects all have area >= 0.1% by construction)
BANDS = [(0.1, 0.2, "0.1-0.2%"), (0.2, 0.5, "0.2-0.5%"),
         (0.5, 1.0, "0.5-1%"), (1.0, 2.0, "1-2%"), (2.0, 1e9, ">2%")]


def main():
    gt = pd.read_csv(GT)
    gt["area_ratios"] = gt["area_ratios"].apply(ast.literal_eval)
    # median object size (% of image) among the counted objects
    gt["med_pct"] = gt["area_ratios"].apply(
        lambda r: float(np.median(r)) * 100 if len(r) else np.nan)
    gt = gt.set_index("image")

    rows = []
    for model, path in MODELS.items():
        d = pd.read_csv(path)
        d = d[d.parse_ok].copy()
        d["med_pct"] = d["image"].map(gt["med_pct"])
        d["gtc"] = d["image"].map(gt["gt_count"])
        d["err"] = d["pred_count"] - d["gtc"]
        for lo, hi, lab in BANDS:
            s = d[(d.med_pct >= lo) & (d.med_pct < hi)]
            if len(s) == 0:
                rows.append({"model": model, "size_band": lab, "n": 0,
                             "MAE": None, "signed": None, "mean_true_count": None})
                continue
            rows.append({
                "model": model, "size_band": lab, "n": len(s),
                "MAE": round(s.err.abs().mean(), 2),
                "signed": round(s.err.mean(), 2),
                "mean_true_count": round(s.gtc.mean(), 1),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved -> {OUT}\n")
    print("=== MAE / signed error by actual object size (median bbox % of image) ===")
    print("(signed: - = under-count, + = over-count;  mtc = mean TRUE count in band)\n")
    for model in df.model.unique():
        sub = df[df.model == model]
        print(f"{model}:")
        print(f"  {'size band':>10} {'n':>4} {'MAE':>6} {'signed':>7} {'mtc':>5}")
        for _, r in sub.iterrows():
            if r["n"] == 0:
                print(f"  {r['size_band']:>10} {0:>4}      -       -     -")
            else:
                print(f"  {r['size_band']:>10} {int(r['n']):>4} {r['MAE']:>6} "
                      f"{r['signed']:>+7} {r['mean_true_count']:>5}")
        print()


if __name__ == "__main__":
    main()
