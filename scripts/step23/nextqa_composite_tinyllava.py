"""
Steps 2A + 3A — TinyLLaVA-3.1B composite-prompt NExT-QA evaluation.

Composite = all N frames in ONE prompt (N image tokens), single letter answer,
as opposed to the per-frame-majority-vote baseline (results_3b_nextqa.csv).

Usage:
  # Run A — frame sweep on the counting ("how many") subset
  python nextqa_composite_tinyllava.py --subset counting --frames 1,2,4,8,16 \
         --out results/step3/step3a_tinyllava_framesweep.csv
  # Run B — all 777 descriptive questions, composite @ 8 frames
  python nextqa_composite_tinyllava.py --subset all --frames 8 \
         --out results/step2/step2a_tinyllava_composite8.csv

One output row per (question, frame_count): videoID, question, options,
n_frames, prediction, answer, correct.
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

REPO_ROOT = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
sys.path.insert(0, REPO_ROOT)

from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3.1B"
DATA_ROOT  = os.path.join(REPO_ROOT, "tinyllava/data/nextqa/val_descriptive.csv")
VIDEO_ROOT = os.path.join(REPO_ROOT, "tinyllava/data/nextqa/videos/NExTVideo")

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

    print(f"Loading TinyLLaVA-3.1B …  frames={frame_counts}  subset={args.subset}")
    parts = load_pretrained_model(MODEL_PATH, attn_implementation="eager", device_map=None)
    model = tok = img_proc = None
    for p in parts:
        if hasattr(p, "parameters"):
            model = p
        elif "Tokenizer" in str(type(p)):
            tok = p
        elif "Processor" in str(type(p)) or "Image" in str(type(p)):
            img_proc = p
    model = model.half().cuda()
    model.resize_token_embeddings(max(len(tok), model.config.vocab_size) + 100)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from tinyllava.data import TextPreprocess, ImagePreprocess
    from tinyllava.utils.message import Message
    from tinyllava.utils.eval_utils import KeywordsStoppingCriteria
    text_proc  = TextPreprocess(tok, "phi")
    image_proc = ImagePreprocess(img_proc, model.config)
    stop_str   = text_proc.template.separator.apply()[1]

    def query(frames, question, options):
        prompt = build_prompt(question, options, len(frames))
        msg = Message()
        msg.add_message(prompt)
        input_ids = text_proc(msg.messages, mode="eval")["input_ids"].unsqueeze(0).to(model.device)
        imgs = torch.stack([image_proc(f) for f in frames]).to(model.device, dtype=torch.float16)
        stopping = KeywordsStoppingCriteria([stop_str], tok, input_ids)
        with torch.inference_mode():
            out = model.generate(
                input_ids, images=imgs, do_sample=False, max_new_tokens=16,
                pad_token_id=tok.pad_token_id, use_cache=True,
                stopping_criteria=[stopping],
            )
        decoded = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
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
