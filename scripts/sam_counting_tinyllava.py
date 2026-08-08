import os
import cv2
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

PHASE1_RESULTS = "/home/brisic03/thesis_eval/results_3b_nextqa.csv"
VIDEO_ROOT = "/home/brisic03/NExT-QA/dataset/videos/val"
OUT_PATH = "/home/brisic03/sam_tinyllava_object_counts.csv"

SAM_CHECKPOINT = "/home/brisic03/sam_checkpoints/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

NUM_FRAMES = 8
MAX_ROWS = None

def get_frames(v_path, n=NUM_FRAMES):
    vid = cv2.VideoCapture(v_path)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        vid.release()
        return []

    indices = [int(i * total / n) for i in range(n)]
    frames = []

    for frame_pos, idx in enumerate(indices):
        vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = vid.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_pos, idx, frame))

    vid.release()
    return frames


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SAM on {device}...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    min_mask_region_area=500,
)

df = pd.read_csv(PHASE1_RESULTS)

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS)

out_data = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    v_id = str(row["videoID"])
    v_file = os.path.join(VIDEO_ROOT, f"{v_id}.mp4")

    if not os.path.exists(v_file):
        continue

    frames = get_frames(v_file)

    if not frames:
        continue

    frame_region_counts = []
    frame_area_ratios = []
    frame_small_region_counts = []

    for frame_pos, frame_idx, frame in frames:
        h, w, _ = frame.shape
        frame_area = h * w

        masks = mask_generator.generate(frame)

        region_count = len(masks)
        total_mask_area = sum(mask["area"] for mask in masks)
        area_ratio = total_mask_area / frame_area if frame_area > 0 else 0

        small_region_count = sum(
            1 for mask in masks
            if mask["area"] < 0.01 * frame_area
        )

        frame_region_counts.append(region_count)
        frame_area_ratios.append(area_ratio)
        frame_small_region_counts.append(small_region_count)

    out_data.append({
        "videoID": v_id,
        "question": row["question"],
        "prediction": row["prediction"],
        "answer": row["answer"],
        "correct": row["correct"],
        "frame_region_counts": str(frame_region_counts),
        "avg_region_count": float(np.mean(frame_region_counts)),
        "max_region_count": int(np.max(frame_region_counts)),
        "min_region_count": int(np.min(frame_region_counts)),
        "avg_total_mask_area_ratio": float(np.mean(frame_area_ratios)),
        "avg_small_region_count": float(np.mean(frame_small_region_counts)),
    })

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

out_df = pd.DataFrame(out_data)
out_df.to_csv(OUT_PATH, index=False)

print(f"Saved SAM counting results to {OUT_PATH}")
print(out_df.groupby("correct")["avg_region_count"].describe())

