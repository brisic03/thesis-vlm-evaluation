"""
Step 2A — composite @ 8 frames vs per-frame majority vote (NExT-QA).

Compares, on the SAME questions:
  - composite @ 8 frames  (counting subset from the Run A frame sweep;
                           all-777 from the Run B composite8 file)
  - per-frame majority vote baseline (results_<model>_3b_nextqa.csv)

Output: results/step2/step2a_composite_vs_vote.csv and a printed table.
"""

import os
import pandas as pd

REPO_ROOT = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
OUT_PATH  = os.path.join(REPO_ROOT, "results/step2/step2a_composite_vs_vote.csv")

MODELS = {
    "TinyLLaVA": {
        "baseline":   os.path.join(REPO_ROOT, "results_3b_nextqa.csv"),
        "sweep":      os.path.join(REPO_ROOT, "results/step3/step3a_tinyllava_framesweep.csv"),
        "composite8": os.path.join(REPO_ROOT, "results/step2/step2a_tinyllava_composite8.csv"),
    },
    "MobileVLM": {
        "baseline":   os.path.join(REPO_ROOT, "results_mobilevlm_3b_nextqa.csv"),
        "sweep":      os.path.join(REPO_ROOT, "results/step3/step3a_mobilevlm_framesweep.csv"),
        "composite8": os.path.join(REPO_ROOT, "results/step2/step2a_mobilevlm_composite8.csv"),
    },
}


def key(df):
    return df["videoID"].astype(str) + "||" + df["question"].astype(str)


def acc_on(df, keys):
    sub = df[df["_k"].isin(keys)]
    return 100.0 * sub["correct"].sum() / len(sub), len(sub)


def main():
    rows = []
    for model, paths in MODELS.items():
        base = pd.read_csv(paths["baseline"])
        base["_k"] = key(base)
        base_hw = base[base["question"].str.lower().str.startswith("how many")]

        # --- counting subset: composite@8 comes from the frame sweep (n_frames==8)
        if os.path.exists(paths["sweep"]):
            sweep = pd.read_csv(paths["sweep"])
            comp8 = sweep[sweep["n_frames"] == 8].copy()
            comp8["_k"] = key(comp8)
            common = set(comp8["_k"]) & set(base_hw["_k"])
            v_acc, _ = acc_on(base_hw, common)
            c_acc, n = acc_on(comp8, common)
            rows.append({"model": model, "subset": "counting (how-many)", "n": n,
                         "vote_acc": round(v_acc, 1), "composite8_acc": round(c_acc, 1),
                         "delta": round(c_acc - v_acc, 1)})

        # --- all 777 descriptive: composite@8 from the dedicated run
        if os.path.exists(paths["composite8"]):
            comp = pd.read_csv(paths["composite8"])
            comp = comp[comp["n_frames"] == 8].copy() if "n_frames" in comp else comp
            comp["_k"] = key(comp)
            common = set(comp["_k"]) & set(base["_k"])
            v_acc, _ = acc_on(base, common)
            c_acc, n = acc_on(comp, common)
            rows.append({"model": model, "subset": "all descriptive", "n": n,
                         "vote_acc": round(v_acc, 1), "composite8_acc": round(c_acc, 1),
                         "delta": round(c_acc - v_acc, 1)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}\n")
    print("=== STEP 2A — composite@8 vs per-frame vote ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
