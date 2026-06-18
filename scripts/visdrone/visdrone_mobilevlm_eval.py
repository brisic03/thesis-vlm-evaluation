# -*- coding: utf-8 -*-

import sys
import os
import time
import re
import torch
import pandas as pd

from PIL import Image
from tqdm import tqdm
from pathlib import Path

sys.path.append("/home/brisic03/MobileVLM")

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
DATA_ROOT = "/home/brisic03/visdrone_val_questions.csv"
OUT_PATH = "/home/brisic03/visdrone_mobilevlm_results.csv"
CONV_MODE = "v1"

MAX_ROWS = None


def build_prompt(question, options):
    opts_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

    return (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {question}\n"
        f"{opts_str}\n"
        f"Answer with only the letter of the correct option (A, B, C, D or E)."
    )


def extract_answer_letter(text):
    text = str(text).strip()

    match = re.search(r"answer\s*[:\-]?\s*([ABCDE])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([ABCDE])\b", text.upper())
    if match:
        return match.group(1)

    for char in text.upper():
        if char in ("A", "B", "C", "D", "E"):
            return char

    return None


def query_single_image(image, question, options, model, tokenizer, image_processor):
    prompt = build_prompt(question, options)

    images_tensor = process_images([image], image_processor, model.config).to(
        model.device,
        dtype=torch.float16,
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
            max_new_tokens=64,
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

    return extract_answer_letter(decoded), decoded


disable_torch_init()

print(f"Loading {MODEL_PATH}...")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH,
    load_8bit=False,
    load_4bit=False,
)

model = model.half().cuda()

questions = pd.read_csv(DATA_ROOT)

if MAX_ROWS is not None:
    questions = questions.head(MAX_ROWS)

print(f"Columns: {questions.columns.tolist()}")
print(f"Running MobileVLM on {len(questions)} VisDrone questions...")

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
OPTION_KEYS = ["a0", "a1", "a2", "a3", "a4"]

out_data = []

for i, row in tqdm(questions.iterrows(), total=len(questions)):
    image_path = str(row["image_path"])

    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")

    options = {
        LETTER_MAP[j]: str(row[key])
        for j, key in enumerate(OPTION_KEYS)
        if key in row and pd.notna(row[key])
    }

    gt_letter = str(row["answer_letter"]).strip().upper()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    pred, raw_output = query_single_image(
        image,
        row["question"],
        options,
        model,
        tokenizer,
        image_processor,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_time = time.time() - start_time

    out_data.append({
        "image": row["image"],
        "image_path": image_path,
        "qtype": row["qtype"],
        "question": row["question"],
        "options": str(options),
        "prediction": pred,
        "raw_output": raw_output,
        "answer": gt_letter,
        "answer_text": row["answer_text"],
        "correct": int(pred == gt_letter),
        "inference_time_sec": inference_time,
    })

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

df = pd.DataFrame(out_data)
df.to_csv(OUT_PATH, index=False)

accuracy = df["correct"].mean() * 100
avg_time = df["inference_time_sec"].mean()

print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Average inference time: {avg_time:.2f}s")
print("\nAccuracy by question type:")
print(df.groupby("qtype")["correct"].mean() * 100)
print(f"\nResults saved in {OUT_PATH}")

