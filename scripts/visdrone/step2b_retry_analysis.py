"""
Analyse the TinyLLaVA retry-until-answered run vs the original (no-retry) run, and
recompute Step 3B (size sweeps) using the retry predictions. MobileVLM is left as
the original (it never refused). CSV-only.

Outputs:
  results/step2/step2b_retry_vs_original.csv   (headline comparison)
  results/step3/step3b_size_sweep_retry.csv    (threshold sweep, retry TinyLLaVA)
  results/step3/step3b_error_by_objsize_retry.csv
"""
import os, ast, math
import numpy as np
import pandas as pd

REPO = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
GT   = f"{REPO}/results/step2/visdrone_counting_gt.csv"
ORIG = f"{REPO}/results/step2/step2b_tinyllava_numeric.csv"
RETRY= f"{REPO}/results/step2/step2b_tinyllava_numeric_retry.csv"

def mr(df):
    e = df.pred_count - df.gt_count
    return e.abs().mean(), math.sqrt((e**2).mean()), e.mean()

def main():
    o = pd.read_csv(ORIG); r = pd.read_csv(RETRY)
    rows = []
    for name, d in [("TinyLLaVA original", o), ("TinyLLaVA retry", r)]:
        ok = d[d.parse_ok]
        mae, rmse, bias = mr(ok)
        rows.append({"run": name, "n_total": len(d), "n_answered": len(ok),
                     "parse_fail_%": round(100*(1-len(ok)/len(d)), 1),
                     "MAE": round(mae, 2), "RMSE": round(rmse, 2), "bias": round(bias, 2)})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(f"{REPO}/results/step2/step2b_retry_vs_original.csv", index=False)
    print("=== Step 2B — TinyLLaVA retry vs original ===")
    print(cmp.to_string(index=False))
    if "n_attempts" in r:
        print("\nattempts distribution (retry):",
              r.n_attempts.value_counts().sort_index().to_dict())
        print(f"answered on try 1: {(r.n_attempts==1).sum()}  needed retry: {(r.n_attempts>1).sum()}"
              f"  still failed: {(~r.parse_ok).sum()}")

    # --- on the images the ORIGINAL refused, what did retry get? ---
    refused = set(o[~o.parse_ok].image)
    rr = r[r.image.isin(refused) & r.parse_ok]
    if len(rr):
        mae, rmse, bias = mr(rr)
        print(f"\nOn the {len(refused)} images the original REFUSED, retry now answers "
              f"{len(rr)}: MAE={mae:.2f} RMSE={rmse:.2f} bias={bias:+.2f} "
              f"(gt mean {rr.gt_count.mean():.1f}, pred mean {rr.pred_count.mean():.1f})")

    # --- Step 3B threshold sweep (retry TinyLLaVA) ---
    gt = pd.read_csv(GT); gt["area_ratios"] = gt["area_ratios"].apply(ast.literal_eval)
    gt = gt.set_index("image")
    THR = [0.05, 0.1, 0.2, 0.5, 1.0]
    sw = []
    ok = r[r.parse_ok]
    for tp in THR:
        thr = tp/100.0; errs = []
        for _, row in ok.iterrows():
            if row.image not in gt.index: continue
            gtt = sum(1 for a in gt.loc[row.image, "area_ratios"] if a >= thr)
            errs.append(row.pred_count - gtt)
        e = pd.Series(errs, dtype=float)
        sw.append({"model": "TinyLLaVA-retry", "threshold_pct": tp, "n": len(e),
                   "MAE": round(e.abs().mean(), 2), "RMSE": round(math.sqrt((e**2).mean()), 2)})
    pd.DataFrame(sw).to_csv(f"{REPO}/results/step3/step3b_size_sweep_retry.csv", index=False)
    print("\n=== Step 3B threshold sweep (TinyLLaVA retry) ===")
    for s in sw: print(f"  {s['threshold_pct']:>5}%  MAE={s['MAE']:>5}  RMSE={s['RMSE']:>6}  n={s['n']}")

    # --- error by actual object size (retry TinyLLaVA) ---
    gt["med_pct"] = gt["area_ratios"].apply(lambda a: float(np.median(a))*100 if len(a) else np.nan)
    ok = ok.copy()
    ok["med_pct"] = ok.image.map(gt["med_pct"]); ok["gtc"] = ok.image.map(gt["gt_count"])
    ok["err"] = ok.pred_count - ok.gtc
    BANDS = [(0.1,0.2,"0.1-0.2%"),(0.2,0.5,"0.2-0.5%"),(0.5,1.0,"0.5-1%"),(1.0,2.0,"1-2%"),(2.0,1e9,">2%")]
    eb = []
    for lo,hi,lab in BANDS:
        s = ok[(ok.med_pct>=lo)&(ok.med_pct<hi)]
        eb.append({"model":"TinyLLaVA-retry","size_band":lab,"n":len(s),
                   "MAE": round(s.err.abs().mean(),2) if len(s) else None,
                   "signed": round(s.err.mean(),2) if len(s) else None,
                   "mean_true_count": round(s.gtc.mean(),1) if len(s) else None})
    pd.DataFrame(eb).to_csv(f"{REPO}/results/step3/step3b_error_by_objsize_retry.csv", index=False)
    print("\n=== Step 3B error by object size (TinyLLaVA retry) ===")
    for s in eb:
        print(f"  {s['size_band']:>10}  n={s['n']:>3}  MAE={s['MAE']}  signed={s['signed']}  mtc={s['mean_true_count']}")

if __name__ == "__main__":
    main()
