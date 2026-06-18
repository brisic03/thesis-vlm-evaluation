import sys
import os
import torch
import cv2
import time
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/brisic03/TinyLLaVA_Factory")
from tinyllava.data import TextPreprocess, ImagePreprocess
from tinyllava.utils.message import Message
from tinyllava.utils.eval_utils import KeywordsStoppingCriteria
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3.1B"
DATA_ROOT  = "/home/brisic03/dataset/nextqa/val_descriptive.csv"
VIDEO_ROOT = "/home/brisic03/NExT-QA/dataset/videos/val"
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

def measure_latency(frame, question, options, model, tok, img_proc):
    qs = build_prompt(question, options)
    text_processor = TextPreprocess(tok, 'phi')
    image_processor = ImagePreprocess(img_proc, model.config)

    msg = Message()
    msg.add_message(qs)
    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids'].unsqueeze(0).to(model.device)

    images_tensor = image_processor(frame)
    images_tensor = images_tensor.unsqueeze(0).to(model.device, dtype=torch.float32)

    stop_str = text_processor.template.separator.apply()[1]
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tok, input_ids)

    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start

print("Loading TinyLLaVA-3.1B...")
results = load_pretrained_model(MODEL_PATH, attn_implementation="eager", device_map=None)
model, tok, img_proc = None, None, None
for item in results:
    if hasattr(item, 'parameters'):
        model = item
    elif "Tokenizer" in str(type(item)):
        tok = item
    elif "Processor" in str(type(item)) or "Image" in str(type(item)):
        img_proc = item

model = model.half().cuda()
target_size = max(len(tok), model.config.vocab_size) + 100
model.resize_token_embeddings(target_size)

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

    latency = measure_latency(frames[0], row['question'], options, model, tok, img_proc)
    latencies.append(latency)

latencies = np.array(latencies)
print(f"\n── TinyLLaVA 3B Inference Latency ──")
print(f"Samples measured: {len(latencies)}")
print(f"Mean latency:     {latencies.mean():.3f}s")
print(f"Std latency:      {latencies.std():.3f}s")
print(f"Min latency:      {latencies.min():.3f}s")
print(f"Max latency:      {latencies.max():.3f}s")
print(f"Median latency:   {np.median(latencies):.3f}s")

pd.DataFrame({'latency_s': latencies}).to_csv(
    '/home/brisic03/thesis_eval/latency_tinyllava_3b.csv', index=False
)
print("Saved to /home/brisic03/thesis_eval/latency_tinyllava_3b.csv")
