import sys
import os
import torch
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, "/home/brisic03/TinyLLaVA_Factory")
from tinyllava.data.template.base import Template
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.arguments import *
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3.1B"
DATA_ROOT = "/home/brisic03/dataset/nextqa/val_descriptive.csv"
VIDEO_ROOT = "/home/brisic03/NExT-QA/dataset/videos/val"
OUT_PATH = "/home/brisic03/thesis_eval/results_3b_nextqa.csv"
NUM_FRAMES = 8

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
    prompt = (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {question}\n"
        f"{opts_str}\n"
        f"Answer with only the letter of the correct option (A, B, C, D or E)."
    )
    return prompt

def query_single_frame(frame, question, options, model, tok, img_proc):
    from tinyllava.data import TextPreprocess, ImagePreprocess
    from tinyllava.utils.message import Message
    from tinyllava.utils.eval_utils import KeywordsStoppingCriteria

    qs = build_prompt(question, options)
    text_processor = TextPreprocess(tok, 'phi')
    image_processor = ImagePreprocess(img_proc, model.config)

    msg = Message()
    msg.add_message(qs)
    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids'].unsqueeze(0).to(model.device)

    images_tensor = image_processor(frame)
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
            stopping_criteria=[stopping_criteria]
        )

    decoded = tok.batch_decode(output, skip_special_tokens=True)[0].strip()
    if decoded.endswith(stop_str):
        decoded = decoded[:-len(stop_str)].strip()

    for char in decoded.upper():
        if char in ('A', 'B', 'C', 'D', 'E', ):
            return char
    return None

def majority_vote(predictions):
    valid = [p for p in predictions if p is not None]
    if not valid:
        return 'A'
    return Counter(valid).most_common(1)[0][0]

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
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
    break
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Syncing vocab to {target_size}...")
model.resize_token_embeddings(target_size)

questions = pd.read_csv(DATA_ROOT)
print(f"Columns:  {questions.columns.tolist()}")
print(f"Running inference on {len(questions)} samples with {NUM_FRAMES} frames each...")

OPTION_KEYS = ['a0', 'a1', 'a2', 'a3', 'a4']
LETTER_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

out_data = []

for i, row in tqdm(questions.iterrows(), total=len(questions)):
    v_id = str(row['video'])
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

    gt_raw = row['answer']
    if isinstance(gt_raw, (int, float)):
        gt_letter = LETTER_MAP.get(int(gt_raw), str(gt_raw))
    else:
        gt_letter = str(gt_raw).strip().upper()

    frame_preds = [
       query_single_frame(frame, row['question'], options, model, tok, img_proc)
       for frame in frames
    ]
    final_pred = majority_vote(frame_preds)

    out_data.append({
        'videoID': v_id,
        'question': row['question'],
        'options': str(options),
        'frame_preds': str(frame_preds),
        'prediction': final_pred,
        'answer': gt_letter,
        'correct': int(final_pred == gt_letter)
    })

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df = pd.DataFrame(out_data)
df.to_csv(OUT_PATH, index=False)

accuracy = df['correct'].mean() * 100
print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Results saved in {OUT_PATH}")


