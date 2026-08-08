import sys
import os
import time
import re
import torch
import pandas as pd

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/brisic03/TinyLLaVA_Factory")
from tinyllava.data.template.base import Template
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.arguments import *
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3.1B"
DATA_ROOT = "/home/brisic03/visdrone_val_questions.csv"
IMAGE_ROOT = "/home/brisic03/VisDrone/VisDrone2019-DET-val/images"
OUT_PATH = "/home/brisic03/visdrone_tinyllava_results.csv"

MAX_ROWS = None

def build_prompt(question, options):
    opts_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

    prompt = (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {question}\n"
        f"{opts_str}\n"
        f"Answer with only the letter of the correct option (A, B, C, D or E)."
    )

    return prompt

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

def query_single_image(image, question, options, model, tok, img_proc):
    from tinyllava.data import TextPreprocess, ImagePreprocess
    from tinyllava.utils.message import Message
    from tinyllava.utils.eval_utils import KeywordsStoppingCriteria

    qs = build_prompt(question, options)

    text_processor = TextPreprocess(tok, "phi")
    image_processor = ImagePreprocess(img_proc, model.config)

    msg = Message()
    msg.add_message(qs)

    result = text_processor(msg.messages, mode="eval")
    input_ids = result["input_ids"].unsqueeze(0).to(model.device)

    images_tensor = image_processor(image)
    images_tensor = images_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)

    stop_str = text_processor.template.separator.apply()[1]
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tok, input_ids)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    decoded = tok.batch_decode(output, skip_special_tokens=True)[0].strip()

    if decoded.endswith(stop_str):
        decoded = decoded[:-len(stop_str)].strip()

    return extract_answer_letter(decoded), decoded


print("Loading TinyLLaVA-3.1B...")

results = load_pretrained_model(
    MODEL_PATH,
    attn_implementation="eager",
    device_map=None,
)

model, tok, img_proc = None, None, None

for item in results:
    if hasattr(item, "parameters"):
        model = item
    elif "Tokenizer" in str(type(item)):
        tok = item
    elif "Processor" in str(type(item)) or "Image" in str(type(item)):
        img_proc = item

model = model.half().cuda()

target_size = max(len(tok), model.config.vocab_size) + 100

for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
    break

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Syncing vocab to {target_size}...")

model.resize_token_embeddings(target_size)

questions = pd.read_csv(DATA_ROOT)

if MAX_ROWS is not None:
    questions = questions.head(MAX_ROWS)

print(f"Columns: {questions.columns.tolist()}")
print(f"Running TinyLLaVA on {len(questions)} VisDrone questions...")

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
        tok,
        img_proc,
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
