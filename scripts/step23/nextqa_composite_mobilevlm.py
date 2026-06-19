"""
Steps 2A + 3A — MobileVLM V2-3B composite-prompt NExT-QA evaluation.

Composite = all N frames in ONE prompt (N image tokens), single letter answer,
vs the per-frame-majority-vote baseline (results_mobilevlm_3b_nextqa.csv).

Usage (mirrors the TinyLLaVA script):
  python nextqa_composite_mobilevlm.py --subset counting --frames 1,2,4,8,16 \
         --out results/step3/step3a_mobilevlm_framesweep.csv
  python nextqa_composite_mobilevlm.py --subset all --frames 8 \
         --out results/step2/step2a_mobilevlm_composite8.csv
"""

import sys
import os
import re
import argparse
import torch
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT      = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
MOBILEVLM_REPO = os.path.join(REPO_ROOT, "MobileVLM")
sys.path.insert(0, MOBILEVLM_REPO)

from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import (
    disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria,
)
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "mtgv/MobileVLM_V2-3B"
DATA_ROOT  = os.path.join(REPO_ROOT, "tinyllava/data/nextqa/val_descriptive.csv")
VIDEO_ROOT = os.path.join(REPO_ROOT, "tinyllava/data/nextqa/videos/NExTVideo")
CONV_MODE  = "v1"

LETTER_MAP  = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
OPTION_KEYS = ["a0", "a1", "a2", "a3", "a4"]


def get_frames(v_path, n):
    vid = cv2.VideoCapture(v_path)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        vid.release()
        return []
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        vid.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = vid.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    vid.release()
    return frames


def build_prompt(question, options, n_frames):
    p = "".join(f"{DEFAULT_IMAGE_TOKEN}\n" for _ in range(n_frames))
    opts_str = "\n".join(f"{k}. {v}" for k, v in options.items())
    p += (
        f"These are {n_frames} frames sampled in order from one video.\n"
        f"Question: {question}\n{opts_str}\n"
        f"Answer with only the letter."
    )
    return p


def parse_letter(text, valid_letters):
    m = re.search(r"\b([A-E])\b", text.upper())
    if m and m.group(1) in valid_letters:
        return m.group(1)
    for ch in text.upper():
        if ch in valid_letters:
            return ch
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["counting", "all"], required=True)
    ap.add_argument("--frames", default="8", help="comma list, e.g. 1,2,4,8,16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test: cap #questions")
    args = ap.parse_args()
    frame_counts = [int(x) for x in args.frames.split(",")]

    print(f"Loading MobileVLM V2-3B …  frames={frame_counts}  subset={args.subset}")
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH)
    model = model.half().cuda()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    def query(frames, question, options):
        prompt = build_prompt(question, options, len(frames))
        images_tensor = process_images(frames, image_processor, model.config).to(
            model.device, dtype=torch.float16)
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)
        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=images_tensor, do_sample=False, max_new_tokens=16,
                use_cache=True, stopping_criteria=[stopping],
            )
        decoded = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if decoded.endswith(stop_str):
            decoded = decoded[: -len(stop_str)].strip()
        return decoded

    questions = pd.read_csv(DATA_ROOT)
    if args.subset == "counting":
        questions = questions[questions["question"].str.lower().str.startswith("how many")].copy()
    if args.limit:
        questions = questions.head(args.limit)
    print(f"subset rows: {len(questions)}")

    out_rows = []
    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        v_id = str(row["video"])
        v_file = os.path.join(VIDEO_ROOT, f"{v_id}.mp4")
        if not os.path.exists(v_file):
            continue
        options = {
            LETTER_MAP[j]: str(row[k])
            for j, k in enumerate(OPTION_KEYS)
            if k in row and pd.notna(row[k])
        }
        valid = set(options.keys())
        gt_raw = row["answer"]
        gt_letter = LETTER_MAP.get(int(gt_raw)) if isinstance(gt_raw, (int, float)) else str(gt_raw).strip().upper()
        for n in frame_counts:
            frames = get_frames(v_file, n)
            if not frames:
                continue
            raw = query(frames, row["question"], options)
            pred = parse_letter(raw, valid)
            out_rows.append({
                "videoID": v_id,
                "question": row["question"],
                "options": str(options),
                "n_frames": n,
                "raw_output": raw,
                "prediction": pred,
                "answer": gt_letter,
                "correct": int(pred == gt_letter),
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")
    print("\n=== accuracy by frame count ===")
    for n in frame_counts:
        sub = df[df["n_frames"] == n]
        if len(sub):
            acc = 100.0 * sub["correct"].sum() / len(sub)
            nparse = sub["prediction"].notna().sum()
            print(f"  {n:>2} frames: {acc:5.1f}%  (n={len(sub)}, parsed={nparse})")


if __name__ == "__main__":
    main()
