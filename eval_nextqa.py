import sys
import os
import torch
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from download_nextqa import download_annotations, download_videos
download_annotations()
download_videos()

from tinyllava.data.template.base import Template
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.arguments import *
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

MODEL_PATH = "TinyLLaVA/TinyLLaVA-3.1B"
DATA_ROOT = os.path.join(os.path.dirname(__file__), "tinyllava", "data", "nextqa", "val_descriptive.csv")
VIDEO_ROOT = os.path.join(os.path.dirname(__file__), "tinyllava", "data", "nextqa", "videos", "NExTVideo")
OUT_PATH = os.path.join(os.path.dirname(__file__), "results_3b_nextqa.csv")

def get_frames(v_path, n=8):
	vid = cv2.VideoCapture(v_path)
	total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	if total <= 0: return []
	step = total // n
	frames = []
	for i in range(n):
	    vid.set(cv2.CAP_PROP_POS_FRAMES, i * step)
	    success, frame = vid.read()
	    if success:
	      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	      frames.append(Image.fromarray(frame))
	vid.release()
	return frames

print("Loading TinyLLaVA-3.1B...")
results = load_pretrained_model(MODEL_PATH, attn_implementation="eager")

model, tok, img_proc = None, None, None
for item in results:
   if hasattr(item, 'cuda'):
      model = item
   elif "Tokenizer" in str(type(item)):
      tok = item
   elif "Processor" in str(type(item)) or "Image" in str(type(item)):
      img_proc = item

print("Moving model to GPU (float16)...")
model = model.half().cuda()


questions = pd.read_csv(DATA_ROOT)
print(f"DEBUG: Available columns are: {questions.columns.tolist()}")
out_data = []
print(f"Running inference on {len(questions)} samples...")

for i, row in tqdm(questions.iterrows(), total=len(questions)):
	v_id = str(row['video'])
	v_file = os.path.join(VIDEO_ROOT, f"{v_id}.mp4")
	if not os.path.exists(v_file):
	    continue

	video_frames = get_frames(v_file)
	if not video_frames:
	     continue

	pixel_values = [img_proc.preprocess(f, return_tensors='pt')['pixel_values'][0] for f in video_frames]
	pixel_values = torch.stack(pixel_values).cuda()

	qs = row['question']
	prompt = DEFAULT_IMAGE_TOKEN + "\n" + qs

	input_ids = Template.tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
	images_tensor = pixel_values.to(device=model.device)

	with torch.inference_mode():
	    output = model.generate(
		input_ids,
		images=images_tensor,
		do_sample=False,
		max_new_tokens=64,
		use_cache=True
	   )

	res = tok.decode(output[0], skip_special_tokens=True).strip()

	out_data.append({
	    'videoID': v_id,
	    'question': qs,
	    'prediction': res,
	    'answer': row ['answer']
	})

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
pd.DataFrame(out_data).to_csv(OUT_PATH, index=False)
print(f"Done. Results saved in {OUT_PATH}")


