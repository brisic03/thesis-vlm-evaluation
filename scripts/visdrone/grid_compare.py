"""
Compare VisDrone counting: whole-image vs grid 2x2 vs grid 3x3, per model.
For TinyLLaVA also split by the EASY images it originally answered vs the CROWDED
images it originally refused — that is where grid mode is supposed to help.
"""
import os, math
import pandas as pd

REPO = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
def load(p): return pd.read_csv(p)
def mr(d):
    e = d.pred_count - d.gt_count
    return e.abs().mean(), math.sqrt((e**2).mean()), e.mean()

orig_t = load(f"{REPO}/results/step2/step2b_tinyllava_numeric.csv")
retry_t = load(f"{REPO}/results/step2/step2b_tinyllava_numeric_retry.csv")
orig_m = load(f"{REPO}/results/step2/step2b_mobilevlm_numeric.csv")
g2t = load(f"{REPO}/results/step3/visdrone_grid2x2_tinyllava.csv")
g3t = load(f"{REPO}/results/step3/visdrone_grid3x3_tinyllava.csv")
g2m = load(f"{REPO}/results/step3/visdrone_grid2x2_mobilevlm.csv")
g3m = load(f"{REPO}/results/step3/visdrone_grid3x3_mobilevlm.csv")

rows = []
def row(model, method, d, subset_imgs=None):
    if subset_imgs is not None: d = d[d.image.isin(subset_imgs)]
    if "parse_ok" in d: d = d[d.parse_ok]
    mae, rmse, bias = mr(d)
    rows.append({"model": model, "method": method, "n": len(d),
                 "MAE": round(mae,2), "RMSE": round(rmse,2), "bias": round(bias,2),
                 "pred_mean": round(d.pred_count.mean(),1), "gt_mean": round(d.gt_count.mean(),1)})

# ---- TinyLLaVA: all 545 ----
row("TinyLLaVA", "whole-image (retry, forced)", retry_t)
row("TinyLLaVA", "grid 2x2", g2t)
row("TinyLLaVA", "grid 3x3", g3t)
# ---- MobileVLM: all 545 ----
row("MobileVLM", "whole-image", orig_m)
row("MobileVLM", "grid 2x2", g2m)
row("MobileVLM", "grid 3x3", g3m)

df = pd.DataFrame(rows)
df.to_csv(f"{REPO}/results/step3/grid_compare.csv", index=False)
print("=== ALL 545 images ===")
print(df.to_string(index=False))

# ---- per-subset for TinyLLaVA: easy(answered) vs crowded(refused) ----
easy = set(orig_t[orig_t.parse_ok].image)
hard = set(orig_t[~orig_t.parse_ok].image)
print(f"\n=== TinyLLaVA by subset  (easy={len(easy)} answered orig, hard={len(hard)} refused orig) ===")
sub = []
for label, imgs in [("EASY (orig answered)", easy), ("HARD (orig refused)", hard)]:
    for method, d in [("whole-image orig", orig_t), ("whole-image retry", retry_t),
                      ("grid 2x2", g2t), ("grid 3x3", g3t)]:
        dd = d[d.image.isin(imgs)]
        if "parse_ok" in dd: dd = dd[dd.parse_ok]
        if len(dd) == 0: continue
        mae, rmse, bias = mr(dd)
        sub.append({"subset": label, "method": method, "n": len(dd),
                    "MAE": round(mae,2), "RMSE": round(rmse,2), "bias": round(bias,2),
                    "gt_mean": round(dd.gt_count.mean(),1)})
sdf = pd.DataFrame(sub)
sdf.to_csv(f"{REPO}/results/step3/grid_compare_tinyllava_subsets.csv", index=False)
print(sdf.to_string(index=False))
