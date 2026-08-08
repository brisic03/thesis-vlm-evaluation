import os
import cv2
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.models import resnet50, ResNet50_Weights

PHASE1_RESULTS = "/home/brisic03/thesis_eval/results_3b_nextqa.csv"
VIDEO_ROOT = "/home/brisic03/NExT-QA/dataset/videos/val"

SAM_CHECKPOINT = "/home/brisic03/sam_checkpoints/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

OUT_PATH = "/home/brisic03/sam_resnet_tinyllava_object_types.csv"
CROP_OUT_PATH = "/home/brisic03/sam_resnet_tinyllava_crop_labels.csv"
SUMMARY_OUT_PATH = "/home/brisic03/sam_resnet_tinyllava_label_summary.csv"

NUM_FRAMES = 8
MAX_ROWS = 20  
MAX_MASKS_PER_FRAME = 5
MIN_CROP_SIDE = 20

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

def classify_crops(crops, preprocess, model, categories, device):
    if not crops:
        return []

    batch = torch.stack([
        preprocess(Image.fromarray(crop))
        for crop in crops
    ]).to(device)

    with torch.inference_mode():
        probs = model(batch).softmax(dim=1)
        confs, idxs = probs.max(dim=1)

    return [
        (categories[idx.item()], float(conf.item()))
        for idx, conf in zip(idxs, confs)
    ]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading SAM...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    min_mask_region_area=500,
)

print("Loading ResNet50...")
weights = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights).to(device)
resnet.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]

df = pd.read_csv(PHASE1_RESULTS)

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS)

row_records = []
crop_records = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    v_id = str(row["videoID"])
    v_file = os.path.join(VIDEO_ROOT, f"{v_id}.mp4")

    if not os.path.exists(v_file):
        continue

    frames = get_frames(v_file)

    if not frames:
        continue

    all_labels = []

    for frame_pos, frame_idx, frame in frames:
        masks = mask_generator.generate(frame)

        selected_masks = sorted(
            masks,
            key=lambda m: m["area"],
            reverse=True
        )[:MAX_MASKS_PER_FRAME]

        crops = []
        crop_meta = []

        for mask_id, mask in enumerate(selected_masks):
            x, y, w, h = [int(v) for v in mask["bbox"]]

            if w < MIN_CROP_SIDE or h < MIN_CROP_SIDE:
                continue

            crop = frame[y:y + h, x:x + w]

            if crop.size == 0:
                continue

            crops.append(crop)
            crop_meta.append((mask_id, mask["area"], x, y, w, h))

        classified = classify_crops(crops, preprocess, resnet, categories, device)

        for (label, confidence), meta in zip(classified, crop_meta):
            mask_id, area, x, y, w, h = meta
            all_labels.append(label)

            crop_records.append({
                "videoID": v_id,
                "question": row["question"],
                "correct": int(row["correct"]),
                "frame_pos": frame_pos,
                "frame_idx": frame_idx,
                "mask_id": mask_id,
                "label": label,
                "confidence": confidence,
                "mask_area": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
            })

    label_counts = Counter(all_labels)
    top_labels = label_counts.most_common(10)

    row_records.append({
        "videoID": v_id,
        "question": row["question"],
        "prediction": row["prediction"],
        "answer": row["answer"],
        "correct": int(row["correct"]),
        "num_classified_crops": len(all_labels),
        "unique_label_count": len(label_counts),
        "most_common_label": top_labels[0][0] if top_labels else "none",
        "top_labels": str(top_labels),
    })

row_df = pd.DataFrame(row_records)
crop_df = pd.DataFrame(crop_records)

row_df.to_csv(OUT_PATH, index=False)
crop_df.to_csv(CROP_OUT_PATH, index=False)

if len(crop_df) > 0:
    summary_df = (
        crop_df
        .groupby(["correct", "label"])
        .agg(
            count=("label", "size"),
            avg_confidence=("confidence", "mean")
        )
        .reset_index()
        .sort_values(["correct", "count"], ascending=[True, False])
    )
    summary_df.to_csv(SUMMARY_OUT_PATH, index=False)

print(f"Saved row-level object types to {OUT_PATH}")
print(f"Saved crop-level labels to {CROP_OUT_PATH}")
print(f"Saved label summary to {SUMMARY_OUT_PATH}")
