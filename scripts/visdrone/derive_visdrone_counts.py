"""
Derive ground-truth integer counts for the VisDrone counting questions (Step 2B/3B).

The questions CSV only stores the MCQ bucket (0,1,2,3,"4 or more"); for MAE/RMSE
we need the exact integer count, including for the ">=4" cases. We recompute the
count straight from the YOLO label files, then VALIDATE by checking that our
recomputed bucket matches the CSV's bucket for every non-">=4" row (those are
exact). If validation passes we trust the integer count for the ">=4" rows too.

YOLO labels: "class cx cy w h" (normalized). class 0..9 == original VisDrone
category 1..10 == CLASS_NAMES[1..10]. Normalized w*h == bbox_area/image_area,
so the generator's MIN_AREA_RATIO=0.001 filter is just (w*h) >= 0.001.

Output: results/step2/visdrone_counting_gt.csv with one row per counting
question: image, image_path, target_class, gt_count, area_ratios (sorted desc,
for the Step 3B size sweep), csv_bucket, recomputed_bucket, bucket_match.
"""

import os
import re
import ast
import pandas as pd

REPO_ROOT  = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
DATA_DIR   = "/data/raja/Datasets/VisDrone/VisDrone2019-DET-val"
IMAGE_DIR  = os.path.join(DATA_DIR, "images")
LABEL_DIR  = os.path.join(DATA_DIR, "labels")
QUESTIONS  = os.path.join(REPO_ROOT, "results/visdrone/visdrone_val_questions.csv")
OUT_PATH   = os.path.join(REPO_ROOT, "results/step2/visdrone_counting_gt.csv")
MIN_AREA_RATIO = 0.001

# original VisDrone category id -> name (same as the question generator)
CLASS_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}
# name -> YOLO class id (0-based) = original-1
NAME_TO_YOLO = {name: cat - 1 for cat, name in CLASS_NAMES.items()}


def parse_target_class(question):
    """'How many cars are visible in the drone image?' -> 'car'."""
    m = re.search(r"how many (.+?) are visible", question.lower())
    if not m:
        return None
    plural = m.group(1).strip()
    # generator appended a single 's'; strip it. handle exact name match first.
    if plural in NAME_TO_YOLO:
        return plural
    if plural.endswith("s") and plural[:-1] in NAME_TO_YOLO:
        return plural[:-1]
    return None


def class_area_ratios(image_name, yolo_cls):
    """sorted-desc list of area ratios for objects of yolo_cls in this image."""
    lab = os.path.join(LABEL_DIR, os.path.splitext(image_name)[0] + ".txt")
    ratios = []
    if not os.path.exists(lab):
        return ratios
    with open(lab) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            if c != yolo_cls:
                continue
            w, h = float(parts[3]), float(parts[4])
            ratios.append(w * h)
    return sorted(ratios, reverse=True)


def to_bucket(count):
    return "4 or more" if count >= 4 else str(count)


def main():
    q = pd.read_csv(QUESTIONS)
    cnt = q[q["qtype"] == "counting"].copy()
    print(f"counting questions: {len(cnt)}")

    rows = []
    for _, r in cnt.iterrows():
        image = r["image"]
        target = parse_target_class(r["question"])
        if target is None:
            print(f"  WARN could not parse class from: {r['question']}")
            continue
        ratios = class_area_ratios(image, NAME_TO_YOLO[target])
        gt = sum(1 for x in ratios if x >= MIN_AREA_RATIO)
        rows.append({
            "image": image,
            "image_path": os.path.join(IMAGE_DIR, image),
            "target_class": target,
            "gt_count": gt,
            "area_ratios": str([round(x, 6) for x in ratios]),
            "csv_bucket": str(r["answer_text"]),
            "recomputed_bucket": to_bucket(gt),
        })

    df = pd.DataFrame(rows)
    df["bucket_match"] = df["csv_bucket"] == df["recomputed_bucket"]

    # validation: exact (non ">=4") buckets must all match
    exact = df[df["csv_bucket"] != "4 or more"]
    n_exact = len(exact)
    n_exact_match = int(exact["bucket_match"].sum())
    n_4plus = int((df["csv_bucket"] == "4 or more").sum())
    n_4plus_ok = int(((df["csv_bucket"] == "4 or more") & (df["gt_count"] >= 4)).sum())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.drop(columns=["area_ratios"]).to_csv(OUT_PATH.replace(".csv", "_preview.csv"), index=False)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} rows -> {OUT_PATH}")
    print("\n=== VALIDATION (recomputed bucket vs CSV bucket) ===")
    print(f"exact buckets (0/1/2/3): {n_exact_match}/{n_exact} match "
          f"({100.0*n_exact_match/max(n_exact,1):.1f}%)")
    print(f"'4 or more' rows: {n_4plus_ok}/{n_4plus} have recomputed count >= 4")
    print(f"\ngt_count stats: min={df['gt_count'].min()} max={df['gt_count'].max()} "
          f"mean={df['gt_count'].mean():.2f}")
    if n_exact_match < n_exact:
        print("\n  !! MISMATCHES (first 10):")
        bad = exact[~exact["bucket_match"]].head(10)
        for _, b in bad.iterrows():
            print(f"   {b['image']} {b['target_class']}: csv={b['csv_bucket']} recomputed={b['recomputed_bucket']}")


if __name__ == "__main__":
    main()
