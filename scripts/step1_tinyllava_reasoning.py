"""
Step 1 – TinyLLaVA-3.1B reasoning run (recoverable fraction).

For every 'how many' question (177 total), query each of 8 frames with a
free-text counting prompt.  Save the full per-frame text so we can later
check whether the true count appears in the output of wrong items.
"""

import sys
import os
import re
import ast
import torch
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
from pathlib import Path

REPO_ROOT = "/data/raja/20_WORK/50_RSML/thesis-vlm-evaluation"
sys.path.insert(0, REPO_ROOT)

from tinyllava.data.template.base import Template
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.arguments import DataArguments, ModelArguments, TrainingArguments
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH  = "TinyLLaVA/TinyLLaVA-3.1B"
DATA_ROOT   = os.path.join(REPO_ROOT, "tinyllava/data/nextqa/val_descriptive.csv")
VIDEO_ROOT  = os.path.join(REPO_ROOT, "tinyllava/data/nextqa/videos/NExTVideo")
BASE_RESULTS = os.path.join(REPO_ROOT, "results_3b_nextqa.csv")
OUT_PATH    = os.path.join(REPO_ROOT, "results/step1/step1_tinyllava_reasoning.csv")
NUM_FRAMES  = 8

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def extract_numbers(text: str) -> set:
    nums = set()
    for m in re.finditer(r"\b(\d+)\b", text):
        nums.add(int(m.group(1)))
    for word, num in WORD_TO_NUM.items():
        if re.search(r"\b" + word + r"\b", text.lower()):
            nums.add(num)
    return nums


def option_to_num(text: str):
    t = str(text).strip().lower()
    if t in WORD_TO_NUM:
        return WORD_TO_NUM[t]
    try:
        return int(t)
    except ValueError:
        return None


def get_frames(v_path: str, n: int = NUM_FRAMES):
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


def build_reasoning_prompt(question: str) -> str:
    return (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {question}\n"
        f"First briefly describe what you see in this frame. "
        f"Then state the count as a number."
    )


def query_frame(frame, question, model, tok, img_proc) -> str:
    from tinyllava.data import TextPreprocess, ImagePreprocess
    from tinyllava.utils.message import Message
    from tinyllava.utils.eval_utils import KeywordsStoppingCriteria

    qs = build_reasoning_prompt(question)
    text_proc  = TextPreprocess(tok, "phi")
    image_proc = ImagePreprocess(img_proc, model.config)

    msg = Message()
    msg.add_message(qs)
    result    = text_proc(msg.messages, mode="eval")
    input_ids = result["input_ids"].unsqueeze(0).to(model.device)

    img_tensor = image_proc(frame).unsqueeze(0).to(model.device, dtype=torch.float16)

    stop_str  = text_proc.template.separator.apply()[1]
    stopping  = KeywordsStoppingCriteria([stop_str], tok, input_ids)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            images=img_tensor,
            do_sample=False,
            max_new_tokens=128,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            stopping_criteria=[stopping],
        )

    decoded = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    if decoded.endswith(stop_str):
        decoded = decoded[: -len(stop_str)].strip()
    return decoded


# ── model loading ────────────────────────────────────────────────────────────
print("Loading TinyLLaVA-3.1B …")
parts = load_pretrained_model(MODEL_PATH, attn_implementation="eager", device_map=None)
model, tok, img_proc = None, None, None
for p in parts:
    if hasattr(p, "parameters"):
        model = p
    elif "Tokenizer" in str(type(p)):
        tok = p
    elif "Processor" in str(type(p)) or "Image" in str(type(p)):
        img_proc = p

model = model.half().cuda()
target_size = max(len(tok), model.config.vocab_size) + 100
model.resize_token_embeddings(target_size)
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── data ─────────────────────────────────────────────────────────────────────
questions = pd.read_csv(DATA_ROOT)
hw = questions[questions["question"].str.lower().str.startswith("how many")].copy()
print(f"how-many subset: {len(hw)} questions")

OPTION_KEYS = ["a0", "a1", "a2", "a3", "a4"]

# ── inference ─────────────────────────────────────────────────────────────────
out_rows = []
for _, row in tqdm(hw.iterrows(), total=len(hw)):
    v_id   = str(row["video"])
    v_file = os.path.join(VIDEO_ROOT, f"{v_id}.mp4")
    if not os.path.exists(v_file):
        continue

    frames = get_frames(v_file)
    if not frames:
        continue

    options = {
        LETTER_MAP[j]: str(row[k])
        for j, k in enumerate(OPTION_KEYS)
        if k in row and pd.notna(row[k])
    }

    gt_raw = row["answer"]
    gt_letter = LETTER_MAP.get(int(gt_raw)) if isinstance(gt_raw, (int, float)) else str(gt_raw).strip().upper()
    gt_text   = options.get(gt_letter, "")
    gt_num    = option_to_num(gt_text)

    frame_texts = [query_frame(f, row["question"], model, tok, img_proc) for f in frames]

    out_rows.append({
        "videoID":     v_id,
        "question":    row["question"],
        "options":     str(options),
        "answer_letter": gt_letter,
        "answer_text": gt_text,
        "answer_num":  gt_num,
        "frame_reasoning": str(frame_texts),
    })

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df = pd.DataFrame(out_rows)
df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} rows → {OUT_PATH}")

# ── recoverable fraction ──────────────────────────────────────────────────────
base = pd.read_csv(BASE_RESULTS)
base_hw = base[base["question"].str.lower().str.startswith("how many")].copy()
base_hw["videoID"] = base_hw["videoID"].astype(str)
wrong_set = set(
    (str(r["videoID"]), r["question"])
    for _, r in base_hw[base_hw["correct"] == 0].iterrows()
)
print(f"\nTotal wrong how-many (baseline): {len(wrong_set)}")

df["videoID"] = df["videoID"].astype(str)
df["is_wrong"] = df.apply(lambda r: (r["videoID"], r["question"]) in wrong_set, axis=1)
wrong_df = df[df["is_wrong"]].copy()

def any_frame_has_count(row):
    if row["answer_num"] is None:
        return False
    texts = ast.literal_eval(row["frame_reasoning"])
    all_nums = set()
    for t in texts:
        all_nums |= extract_numbers(str(t))
    return row["answer_num"] in all_nums

wrong_df["recoverable"] = wrong_df.apply(any_frame_has_count, axis=1)
n_wrong = len(wrong_df)
n_recov = wrong_df["recoverable"].sum()
pct     = 100.0 * n_recov / n_wrong if n_wrong > 0 else 0.0

print(f"\n=== STEP 1 RESULTS – TinyLLaVA-3.1B ===")
print(f"Total wrong how-many items:       {n_wrong}")
print(f"Recoverable (true count in text): {n_recov}")
print(f"Recoverable fraction:             {pct:.1f}%")

# save annotated wrong-items CSV
wrong_out = OUT_PATH.replace(".csv", "_wrong_annotated.csv")
wrong_df.to_csv(wrong_out, index=False)
print(f"Annotated wrong items → {wrong_out}")

# example rows
examples = wrong_df[wrong_df["recoverable"]].head(5)
print("\n--- Example recoverable items ---")
for _, r in examples.iterrows():
    texts = ast.literal_eval(r["frame_reasoning"])
    print(f"  Q: {r['question']}")
    print(f"  True count: {r['answer_num']} ({r['answer_text']})")
    print(f"  Frame[0] text: {texts[0][:120]}")
    print()
