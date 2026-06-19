"""
Step 5 — statistics on everything (CSV-only, no GPU).

Adds, on top of the Steps 2/3/4 result CSVs:
  1. 95% confidence intervals (normal approximation to the binomial) on every
     accuracy figure.
  2. McNemar's exact test on every paired same-question comparison (composite vs
     vote, repair vs vote, model vs model). Significance flagged at p < 0.05.
  3. Bootstrap 95% CIs on the VisDrone MAE / RMSE.

Outputs (results/step5/):
  step5_accuracy_ci.csv, step5_mcnemar.csv, step5_visdrone_bootstrap.csv
and a readable printed summary that explicitly marks non-significant differences.
"""

import os
import ast
import math
from math import comb
import numpy as np
import pandas as pd

REPO = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
OUT = os.path.join(REPO, "results/step5")
ALPHA = 0.05

FILES = {
    "TinyLLaVA": {
        "vote":  f"{REPO}/results_3b_nextqa.csv",
        "sweep": f"{REPO}/results/step3/step3a_tinyllava_framesweep.csv",
        "comp8": f"{REPO}/results/step2/step2a_tinyllava_composite8.csv",
        "repair": f"{REPO}/results/step4/step4_tinyllava_repair.csv",
        "vd":    f"{REPO}/results/step2/step2b_tinyllava_numeric.csv",
    },
    "MobileVLM": {
        "vote":  f"{REPO}/results_mobilevlm_3b_nextqa.csv",
        "sweep": f"{REPO}/results/step3/step3a_mobilevlm_framesweep.csv",
        "comp8": f"{REPO}/results/step2/step2a_mobilevlm_composite8.csv",
        "repair": f"{REPO}/results/step4/step4_mobilevlm_repair.csv",
        "vd":    f"{REPO}/results/step2/step2b_mobilevlm_numeric.csv",
    },
}


def K(df):
    return (df["videoID"].astype(str) + "||" + df["question"].astype(str)).tolist()


def binom_normal_ci(k, n):
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    lo, hi = p - 1.96 * se, p + 1.96 * se
    return p, max(0.0, lo), min(1.0, hi)


def mcnemar_exact(a, b):
    """a, b: aligned 0/1 lists. Returns (b_AnotB, c_BnotA, n_disc, p_two_sided)."""
    bb = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    cc = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = bb + cc
    if n == 0:
        return bb, cc, 0, 1.0
    k = min(bb, cc)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return bb, cc, n, min(1.0, p)


def correct_map(path, col="correct", key_filter=None):
    df = pd.read_csv(path)
    if key_filter is not None:
        df = key_filter(df)
    return dict(zip(K(df), df[col].astype(int)))


def aligned(map_a, map_b):
    keys = [k for k in map_a if k in map_b]
    return [map_a[k] for k in keys], [map_b[k] for k in keys], keys


def main():
    os.makedirs(OUT, exist_ok=True)
    acc_rows, mc_rows, bs_rows = [], [], []

    # ---- gather per-method correctness maps per model ----
    M = {}
    for model, f in FILES.items():
        vote = pd.read_csv(f["vote"])
        vote_all = dict(zip(K(vote), vote["correct"].astype(int)))
        votehw = vote[vote["question"].str.lower().str.startswith("how many")]
        vote_hw = dict(zip(K(votehw), votehw["correct"].astype(int)))

        sw = pd.read_csv(f["sweep"]); sw8 = sw[sw["n_frames"] == 8]
        comp_hw = dict(zip(K(sw8), sw8["correct"].astype(int)))
        c8 = pd.read_csv(f["comp8"])
        comp_all = dict(zip(K(c8), c8["correct"].astype(int)))

        rep = pd.read_csv(f["repair"])
        repk = K(rep)
        repair = {r: dict(zip(repk, rep[f"{r}_correct"].astype(int)))
                  for r in ["mode", "median", "max"]}
        oracle = dict(zip(repk, rep["oracle_anyframe_correct"].astype(int)))
        M[model] = dict(vote_all=vote_all, vote_hw=vote_hw, comp_hw=comp_hw,
                        comp_all=comp_all, repair=repair, oracle=oracle)

        # ---- accuracy + CI for each method ----
        def add_acc(metric, subset, d):
            n = len(d); k = sum(d.values())
            p, lo, hi = binom_normal_ci(k, n)
            acc_rows.append({"model": model, "metric": metric, "subset": subset,
                             "n": n, "accuracy_%": round(100 * p, 1),
                             "ci_low_%": round(100 * lo, 1), "ci_high_%": round(100 * hi, 1)})
        add_acc("vote", "counting", vote_hw)
        add_acc("vote", "all-777", vote_all)
        add_acc("composite@8", "counting", comp_hw)
        add_acc("composite@8", "all-777", comp_all)
        for r in ["mode", "median", "max"]:
            add_acc(f"repair-{r}", "counting", repair[r])
        add_acc("oracle-bestframe", "counting", oracle)

    # ---- McNemar: paired comparisons within each model ----
    def add_mc(scope, name, ma, mb):
        a, b, keys = aligned(ma, mb)
        bb, cc, n, p = mcnemar_exact(a, b)
        accA = 100 * np.mean(a) if a else float("nan")
        accB = 100 * np.mean(b) if b else float("nan")
        mc_rows.append({"scope": scope, "comparison": name, "n_pairs": len(a),
                        "accA_%": round(accA, 1), "accB_%": round(accB, 1),
                        "deltaB-A_%": round(accB - accA, 1),
                        "discordant": n, "b(A>B)": bb, "c(B>A)": cc,
                        "p_value": round(p, 4),
                        "significant": "YES" if p < ALPHA else "no"})

    for model in FILES:
        m = M[model]
        add_mc(model, "composite@8 vs vote (counting)", m["vote_hw"], m["comp_hw"])
        add_mc(model, "composite@8 vs vote (all-777)", m["vote_all"], m["comp_all"])
        for r in ["mode", "median", "max"]:
            add_mc(model, f"repair-{r} vs vote (counting)", m["vote_hw"], m["repair"][r])
        add_mc(model, "oracle-bestframe vs vote (counting)", m["vote_hw"], m["oracle"])

    # ---- McNemar: model vs model (same questions) ----
    add_mc("TinyLLaVA vs MobileVLM", "vote (counting)",
           M["TinyLLaVA"]["vote_hw"], M["MobileVLM"]["vote_hw"])
    add_mc("TinyLLaVA vs MobileVLM", "vote (all-777)",
           M["TinyLLaVA"]["vote_all"], M["MobileVLM"]["vote_all"])
    add_mc("TinyLLaVA vs MobileVLM", "repair-max (counting)",
           M["TinyLLaVA"]["repair"]["max"], M["MobileVLM"]["repair"]["max"])

    # ---- bootstrap CIs for VisDrone MAE / RMSE ----
    rng = np.random.default_rng(42)
    B = 10000
    for model, f in FILES.items():
        d = pd.read_csv(f["vd"]); d = d[d["parse_ok"]]
        err = (d["pred_count"] - d["gt_count"]).to_numpy(dtype=float)
        n = len(err)
        idx = rng.integers(0, n, size=(B, n))
        samp = err[idx]
        maes = np.abs(samp).mean(axis=1)
        rmses = np.sqrt((samp ** 2).mean(axis=1))
        for metric, point, dist in [("MAE", np.abs(err).mean(), maes),
                                    ("RMSE", math.sqrt((err ** 2).mean()), rmses)]:
            lo, hi = np.percentile(dist, [2.5, 97.5])
            bs_rows.append({"model": model, "metric": metric, "n_scored": n,
                            "value": round(float(point), 2),
                            "ci_low": round(float(lo), 2), "ci_high": round(float(hi), 2)})

    acc = pd.DataFrame(acc_rows); acc.to_csv(f"{OUT}/step5_accuracy_ci.csv", index=False)
    mc = pd.DataFrame(mc_rows);   mc.to_csv(f"{OUT}/step5_mcnemar.csv", index=False)
    bs = pd.DataFrame(bs_rows);   bs.to_csv(f"{OUT}/step5_visdrone_bootstrap.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=" * 78)
    print("  ACCURACY with 95% CI (normal approx to binomial)")
    print("=" * 78)
    print(acc.to_string(index=False))
    print("\n" + "=" * 78)
    print("  McNEMAR paired tests  (significant = the difference is unlikely chance)")
    print("=" * 78)
    print(mc.to_string(index=False))
    print("\n" + "=" * 78)
    print("  VisDrone MAE / RMSE with bootstrap 95% CI (10k resamples)")
    print("=" * 78)
    print(bs.to_string(index=False))
    print("\nNon-significant comparisons (cannot claim a real difference):")
    for _, r in mc[mc["significant"] == "no"].iterrows():
        print(f"   - [{r['scope']}] {r['comparison']}  (Δ={r['deltaB-A_%']:+}%, p={r['p_value']})")


if __name__ == "__main__":
    main()
