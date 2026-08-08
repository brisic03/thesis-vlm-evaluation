import re
import sys
import os
import io
import random
import time
import torch
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from collections import Counter
from pathlib import Path

MOBILEVLM_REPO = "/home/brisic03/MobileVLM"
sys.path.insert(0, MOBILEVLM_REPO)

from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import (
    disable_torch_init,
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


MODEL_PATH = "mtgv/MobileVLM_V2-3B"
DATA_ROOT = "/home/brisic03/thesis_eval/results_mobilevlm_3b_nextqa.csv"
VIDEO_ROOT = "/home/brisic03/NExT-QA/dataset/videos/val"

OUT_PATH = "/home/brisic03/mobilevlm_reasoning_failures.csv"

CONV_MODE = "v1"
NUM_FRAMES = 8

NOISE_TYPE = None
SEVERITY = None


def get_frames(v_path, n=NUM_FRAMES):
    vid = cv2.VideoCapture(v_path)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        vid.release()
        return []

    indices = [int(i * total / n) for i in range(n)]
    frames = []

    for idx in indices:
        vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = vid.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    vid.release()
    return frames


def apply_noise(img, noise_type=None, severity=None, seed=42):
    if noise_type is None:
        return img

    if noise_type == "blur":
        arr = np.array(img)
        k = int(severity)

        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(arr, (k, k), 0)
        return Image.fromarray(blurred)

    if noise_type == "jpeg":
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if noise_type == "occlusion":
        rng = random.Random(seed)
        arr = np.array(img).copy()
        h, w, _ = arr.shape

        area_ratio = float(severity)
        occ_area = int(h * w * area_ratio)

        occ_w = int(np.sqrt(occ_area))
        occ_h = int(occ_area / max(occ_w, 1))

        occ_w = min(occ_w, w)
        occ_h = min(occ_h, h)

        x = rng.randint(0, max(w - occ_w, 0))
        y = rng.randint(0, max(h - occ_h, 0))

        arr[y:y + occ_h, x:x + occ_w] = 0
        return Image.fromarray(arr)

    return img


def build_prompt(question, options):
    opts_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

    return (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {question}\n"
        f"{opts_str}\n"
        f"Do not answer with only a letter.\n"
        f"First write one short sentence describing the visual evidence in the frame.\n"
        f"Then write the final answer as: Answer: <letter>.\n"
    )

def extract_answer_letter(text):
    match = re.search(r"(?:final\s*)?answer\s*[:\-]\s*([A-E])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"\b([A-E])\b", text.upper())
    if matches:
        return matches[-1]

    return None

def query_single_frame(frame, question, options, model, tokenizer, image_processor):
    prompt = build_prompt(question, options)

    images_tensor = process_images([frame], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)

    full_prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    input_ids = (
        tokenizer_image_token(
            full_prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(model.device)
    )

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            max_new_tokens=192,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]

    decoded = tokenizer.batch_decode(
        output_ids[:, input_token_len:],
        skip_special_tokens=True,
    )[0].strip()

    if decoded.endswith(stop_str):
        decoded = decoded[:-len(stop_str)].strip()

    pred_letter = extract_answer_letter(decoded)
    return decoded, pred_letter

def majority_vote(predictions):
    valid = [p for p in predictions if p is not None]

    if not valid:
        return "A"

    return Counter(valid).most_common(1)[0][0]


disable_torch_init()

print(f"Loading {MODEL_PATH}...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH,
    load_8bit=False,
    load_4bit=False,
)

model = model.half().cuda()

questions = pd.read_csv(DATA_ROOT)
questions = questions[questions["correct"] == 0]
questions = questions.head(50)

print(f"Columns: {questions.columns.tolist()}")
print(f"Running MobileVLM robustness test on {len(questions)} samples.")
print(f"Noise type: {NOISE_TYPE}")
print(f"Severity: {SEVERITY}")
print(f"Frames per video: {NUM_FRAMES}")

OPTION_KEYS = ["a0", "a1", "a2", "a3", "a4"]
LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

out_data = []

for i, row in tqdm(questions.iterrows(), total=len(questions)):
    v_id = str(row['videoID'])
    v_file = os.path.join(VIDEO_ROOT, f"{v_id}.mp4")

    if not os.path.exists(v_file):
        continue

    frames = get_frames(v_file)

    if not frames:
        continue

    frames = [
        apply_noise(
            frame,
            noise_type=NOISE_TYPE,
            severity=SEVERITY,
            seed=i * 1000 + j,
        )
        for j, frame in enumerate(frames)
    ]

    options = {
        LETTER_MAP[j]: str(row[key])
        for j, key in enumerate(OPTION_KEYS)
        if key in row and pd.notna(row[key])
    }

    gt_raw = row["answer"]

    if isinstance(gt_raw, (int, float)):
        gt_letter = LETTER_MAP.get(int(gt_raw), str(gt_raw))
    else:
        gt_letter = str(gt_raw).strip().upper()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    frame_outputs = [
        query_single_frame(
            frame,
            row["question"],
            options,
            model,
            tokenizer,
            image_processor,
        )
        for frame in frames
    ]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_time = time.time() - start_time

    reasoning_outputs = [item[0] for item in frame_outputs]
    frame_preds = [item[1] for item in frame_outputs]

    final_pred = majority_vote(frame_preds)

    out_data.append(
        {
            "videoID": v_id,
            "question": row["question"],
            "options": str(options),
            "baseline_prediction":row.get("prediction", ""),
            "answer": gt_letter,
            "reasoning_output": str(reasoning_outputs),
            "frame_reasoning_predictions": str(frame_preds),
            "reasoning_prediction": final_pred,
            "num_frames": NUM_FRAMES,
            "correct": int(final_pred == gt_letter),
            "inference_time_sec": inference_time,
        }
    )

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.DataFrame(out_data)
df.to_csv(OUT_PATH, index=False)

accuracy = df["correct"].mean() * 100
avg_time = df["inference_time_sec"].mean()

print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Average inference time per question: {avg_time:.2f} seconds")
print(f"Results saved in {OUT_PATH}")

