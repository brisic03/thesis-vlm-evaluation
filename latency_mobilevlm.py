import sys
import os
import torch
import cv2
import time
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "mtgv/MobileVLM_V2-3B"
DATA_ROOT  = "/home/brisic03/dataset/nextqa/val_descriptive.csv"
VIDEO_ROOT = "/home/brisic03/NExT-QA/dataset/videos/val"
CONV_MODE  = "v1"
NUM_FRAMES = 8
N_SAMPLES  = 50

def get_frames(v_path, n=NUM_FRAMES):
    vid = cv2.VideoCapture(v_path)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
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

def build_prompt(question, options):
    opts_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
    return (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {question}\n"
        f"{opts_str}\n"
        f"Answer with only the letter of the correct option (A, B, C, D, or E)."
    )

def measure_latency(frame, question, options, model, tokenizer, image_processor):
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
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            max_new_tokens=64,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start

disable_torch_init()
print(f"Loading {MODEL_PATH}...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, load_8bit=False, load_4bit=False
)
model = model.half().cuda()

questions = pd.read_csv(DATA_ROOT).head(N_SAMPLES)
OPTION_KEYS = ['a0', 'a1', 'a2', 'a3', 'a4']
LETTER_MAP  = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

latencies = []

for i, row in tqdm(questions.iterrows(), total=len(questions)):
    v_id   = str(row['video'])
    v_file = os.path.join(VIDEO_ROOT, f"{v_id}.mp4")
    if not os.path.exists(v_file):
        continue

    frames = get_frames(v_file)
    if not frames:
        continue

    options = {
        LETTER_MAP[j]: str(row[key])
        for j, key in enumerate(OPTION_KEYS)
        if key in row and pd.notna(row[key])
    }

    latency = measure_latency(frames[0], row['question'], options, model, tokenizer, image_processor)
    latencies.append(latency)

latencies = np.array(latencies)
print(f"\n── MobileVLM 3B Inference Latency ──")
print(f"Samples measured: {len(latencies)}")
print(f"Mean latency:     {latencies.mean():.3f}s")
print(f"Std latency:      {latencies.std():.3f}s")
print(f"Min latency:      {latencies.min():.3f}s")
print(f"Max latency:      {latencies.max():.3f}s")
print(f"Median latency:   {np.median(latencies):.3f}s")

pd.DataFrame({'latency_s': latencies}).to_csv(
    '/home/brisic03/thesis_eval/latency_mobilevlm_3b.csv', index=False
)
print("Saved to /home/brisic03/thesis_eval/latency_mobilevlm_3b.csv")
